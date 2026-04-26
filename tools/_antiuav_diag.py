"""Diagnose dlaczego v4@640 nie wykrywa nic na anti_uav_rgb.mp4.

Wczytuje 1 klatke, robi raw ONNX inference, pokazuje top-5 confidence
zeby zobaczyc czy w ogole sa jakies detekcje (filtrowane przez 0.05) czy
model w ogole nie reaguje na ten typ drone'a.
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

import onnxruntime as ort


def main() -> int:
    video = r"C:\Users\pwyso\drone-tracker-system\artifacts\test_videos\anti_uav_rgb.mp4"
    model_path = r"C:\Users\pwyso\drone-tracker-system\data\weights\v4_best_fp16_imgsz640.onnx"

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("ERROR: cannot read frame")
        return 1
    print(f"Frame shape: {frame.shape} dtype: {frame.dtype}")

    out_dir = Path(r"C:\Users\pwyso\drone-tracker-system\artifacts\v4_validation\antiuav_diag")
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "antiuav_frame100.png"), frame)
    print(f"Saved: {out_dir / 'antiuav_frame100.png'}")

    sess = ort.InferenceSession(model_path,
                                 providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    print(f"Model input: {inp_name} shape={inp_shape}")

    imgsz = 640
    h, w = frame.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    pad_x, pad_y = (imgsz - nw) // 2, (imgsz - nh) // 2
    resized = cv2.resize(frame, (nw, nh))
    padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    padded[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, (imgsz, imgsz), swapRB=True).astype(np.float32)

    out = sess.run(None, {inp_name: blob})[0]
    print(f"Output shape: {out.shape} dtype: {out.dtype}")

    data = out[0]
    n_classes = data.shape[0] - 4
    print(f"Num classes: {n_classes} anchors: {data.shape[1]}")

    scores = data[4:, :].max(axis=0)
    print(f"Score stats: min={scores.min():.4f} max={scores.max():.4f} mean={scores.mean():.4f}")
    print(f"Detections >= 0.05: {(scores >= 0.05).sum()}")
    print(f"Detections >= 0.10: {(scores >= 0.10).sum()}")
    print(f"Detections >= 0.20: {(scores >= 0.20).sum()}")

    idx = np.argsort(-scores)[:8]
    print("\nTop 8 detections (cx,cy,w,h sa w 640x640 letterboxed coords):")
    for i in idx:
        cx, cy, ww, hh = data[0, i], data[1, i], data[2, i], data[3, i]
        sc = scores[i]
        # Unproject do oryginalnej klatki
        ox = (cx - pad_x) / scale
        oy = (cy - pad_y) / scale
        ow_ = ww / scale
        oh_ = hh / scale
        print(f"  conf={sc:.4f}  box(orig)=cx={ox:.0f} cy={oy:.0f} w={ow_:.0f} h={oh_:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
