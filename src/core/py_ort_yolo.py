"""ORT-based YOLOv8 inference wrapper, ultralytics-compatible API.

Cel: zastapic `from ultralytics import YOLO; YOLO('model.onnx')` direct ORT
session z DirectML EP (Ryzen AI bundle daje 14 ms na iGPU vs 26 ms C++).

Ultralytics z .onnx defaultnie idzie na CPU EP (400 ms) — niezaleznie od
device parametru. Trzeba bypass.

Zwraca obiekty kompatybilne z ultralytics Result:
- result.boxes.xyxy.cpu().numpy() — bboxes [N, 4]
- result.boxes.conf.cpu().numpy() — confidences [N]
- result.boxes.cls.cpu().numpy() — class IDs [N]
- result.boxes.id — None (brak ByteTrack — MTT bedzie matchowac samo)
- len(result.boxes) — N

Uzycie:
    model = PyOrtYOLO('data/weights/v7_best_fp16_imgsz1280.onnx', imgsz=1280)
    results = model.track(frame, conf=0.20, classes=[0,1,2])
    boxes = results[0].boxes
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


class _FakeNumpyTensor:
    """Mimic torch.Tensor API for compat: .cpu().numpy() / .tolist() / __len__."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def cpu(self) -> "_FakeNumpyTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr

    def tolist(self) -> list:
        return self._arr.tolist()

    def __len__(self) -> int:
        return len(self._arr) if self._arr.ndim > 0 else 0

    def __getitem__(self, idx):
        return self._arr[idx]


class _FakeBoxes:
    """Mimic ultralytics Boxes object."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = _FakeNumpyTensor(xyxy.astype(np.float32))
        self.conf = _FakeNumpyTensor(conf.astype(np.float32))
        self.cls = _FakeNumpyTensor(cls.astype(np.int32))
        self.id = None  # No ByteTrack — MTT assigns own track_id

    def __len__(self) -> int:
        return len(self.xyxy)


class _FakeResult:
    """Mimic ultralytics Result object."""

    def __init__(self, boxes: _FakeBoxes, orig_shape: tuple):
        self.boxes = boxes
        self.orig_shape = orig_shape


def _preprocess(frame_bgr: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, int, int]:
    """Letterbox + BGR->RGB + /255 + NCHW. Returns (blob, scale, pad_x, pad_y).

    Uses cv2.dnn.blobFromImage (same as C++ pipeline) — szybsze niż manual
    transpose+normalize w numpy.
    """
    h, w = frame_bgr.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pad_x = (imgsz - new_w) // 2
    pad_y = (imgsz - new_h) // 2

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    # cv2.dnn.blobFromImage: BGR uint8 -> /255 -> swapRB (-> RGB) -> NCHW float32
    blob = cv2.dnn.blobFromImage(padded, 1.0 / 255.0, (imgsz, imgsz),
                                  mean=None, swapRB=True, crop=False)
    return blob, scale, pad_x, pad_y


def _decode_yolov8(out: np.ndarray, conf_thresh: float, allowed_classes: Optional[set]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLOv8 head output.

    out shape: [1, 4+nc, num_anchors] — channels: [cx, cy, w, h, cls0, cls1, ...]
    Returns (xyxy [N,4], conf [N], cls [N]) in INPUT pixel coords (1280×1280).
    """
    assert out.ndim == 3 and out.shape[0] == 1, f"unexpected shape {out.shape}"
    dims = out.shape[1]
    n_anchors = out.shape[2]
    n_classes = dims - 4
    assert n_classes > 0

    # Transpose to [num_anchors, 4+nc] for easier indexing
    arr = out[0].T  # [n_anchors, 4+nc]

    # Best class per anchor
    cls_scores = arr[:, 4:4 + n_classes]
    best_cls = np.argmax(cls_scores, axis=1)
    best_conf = cls_scores[np.arange(len(arr)), best_cls]

    # Filter by conf
    mask = best_conf >= conf_thresh
    if allowed_classes is not None:
        cls_mask = np.isin(best_cls, list(allowed_classes))
        mask = mask & cls_mask
    if not mask.any():
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32)

    arr_kept = arr[mask]
    conf_kept = best_conf[mask]
    cls_kept = best_cls[mask]

    cx = arr_kept[:, 0]
    cy = arr_kept[:, 1]
    w = arr_kept[:, 2]
    h = arr_kept[:, 3]
    xyxy = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]).astype(np.float32)
    return xyxy, conf_kept.astype(np.float32), cls_kept.astype(np.int32)


def _nms_per_class(xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, iou_thresh: float = 0.45) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class NMS using OpenCV (well-tested, no torch dependency)."""
    if len(xyxy) == 0:
        return xyxy, conf, cls
    keep_idx = []
    for c in np.unique(cls):
        mask = cls == c
        idxs = np.where(mask)[0]
        boxes_c = xyxy[idxs]
        scores_c = conf[idxs]
        # OpenCV NMSBoxes wants [x, y, w, h]
        xywh = np.column_stack([boxes_c[:, 0], boxes_c[:, 1],
                                 boxes_c[:, 2] - boxes_c[:, 0],
                                 boxes_c[:, 3] - boxes_c[:, 1]])
        kept = cv2.dnn.NMSBoxes(xywh.tolist(), scores_c.tolist(), 0.0, iou_thresh)
        if len(kept) > 0:
            kept = np.array(kept).flatten()
            keep_idx.extend(idxs[kept].tolist())
    if not keep_idx:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int32)
    keep_idx = np.array(keep_idx)
    return xyxy[keep_idx], conf[keep_idx], cls[keep_idx]


def _unletterbox(xyxy: np.ndarray, scale: float, pad_x: int, pad_y: int,
                  orig_w: int, orig_h: int) -> np.ndarray:
    """Convert bboxes from input space (1280) to original image space."""
    if len(xyxy) == 0:
        return xyxy
    out = xyxy.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / scale
    out[:, 0] = np.clip(out[:, 0], 0, orig_w)
    out[:, 1] = np.clip(out[:, 1], 0, orig_h)
    out[:, 2] = np.clip(out[:, 2], 0, orig_w)
    out[:, 3] = np.clip(out[:, 3], 0, orig_h)
    return out


class PyOrtYOLO:
    """ORT + DirectML wrapper, mimicking ultralytics YOLO predict/track API."""

    def __init__(self, model_path: str, imgsz: int = 1280,
                 default_conf: float = 0.20, iou_thresh: float = 0.45,
                 device_id: int = 0):
        import onnxruntime as ort
        self.model_path = model_path
        self.imgsz = imgsz
        self.default_conf = default_conf
        self.iou_thresh = iou_thresh

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = [("DmlExecutionProvider", {"device_id": device_id}),
                     "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, sess_options=sess_opts,
                                            providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        actual_providers = self.session.get_providers()
        print(f"PyOrtYOLO loaded: {Path(model_path).name}, imgsz={imgsz}, "
              f"providers={actual_providers}")
        self.last_inference_ms = 0.0

    def _run_inference(self, frame: np.ndarray, conf: float,
                       allowed_classes: Optional[set]) -> _FakeResult:
        h, w = frame.shape[:2]
        blob, scale, pad_x, pad_y = _preprocess(frame, self.imgsz)
        t0 = time.time()
        out = self.session.run(None, {self.input_name: blob})
        self.last_inference_ms = (time.time() - t0) * 1000.0
        xyxy_in, confs, classes = _decode_yolov8(out[0], conf, allowed_classes)
        xyxy_in, confs, classes = _nms_per_class(xyxy_in, confs, classes, self.iou_thresh)
        xyxy_orig = _unletterbox(xyxy_in, scale, pad_x, pad_y, w, h)
        boxes = _FakeBoxes(xyxy_orig, confs, classes)
        return _FakeResult(boxes, frame.shape)

    def predict(self, source, conf: Optional[float] = None,
                imgsz: Optional[int] = None, classes: Optional[list] = None,
                verbose: bool = False, **kwargs) -> List[_FakeResult]:
        """Mimic ultralytics model.predict()."""
        if isinstance(source, np.ndarray):
            frames = [source]
        else:
            raise TypeError(f"PyOrtYOLO.predict source must be np.ndarray, got {type(source)}")
        c = conf if conf is not None else self.default_conf
        allowed = set(classes) if classes else None
        return [self._run_inference(f, c, allowed) for f in frames]

    def track(self, source, persist: bool = True, tracker: Optional[str] = None,
              conf: Optional[float] = None, imgsz: Optional[int] = None,
              classes: Optional[list] = None, verbose: bool = False, **kwargs) -> List[_FakeResult]:
        """Mimic ultralytics model.track() — but BEZ ByteTrack.

        MTT (multi_target_tracker) bedzie samo matchowac detekcje przez Kalman+IoU.
        Track ID przypisuje MTT, nie ByteTrack — jest to OK bo C++ pipeline tez tak robi.
        """
        return self.predict(source, conf=conf, imgsz=imgsz, classes=classes,
                            verbose=verbose, **kwargs)
