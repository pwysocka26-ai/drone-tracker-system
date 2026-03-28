from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import time
import cv2


@dataclass
class DetectionTrack:
    track_id: int
    cls_id: int
    conf: float
    bbox_xyxy: Tuple[float, float, float, float]
    center_xy: Tuple[float, float]
    timestamp: float


@dataclass
class TargetMessage:
    timestamp: float
    selected_track_id: Optional[int]
    bbox_xyxy: Optional[Tuple[float, float, float, float]]
    center_xy: Optional[Tuple[float, float]]
    confidence: float
    mode: str


class AutoFramer:
    def __init__(self, out_w: int, out_h: int, margin: float = 0.28, smooth: float = 0.82):
        self.out_w = out_w
        self.out_h = out_h
        self.margin = margin
        self.smooth = smooth
        self.aspect = out_w / out_h
        self.state: Optional[Tuple[float, float, float, float]] = None

    def update(self, box: Optional[Tuple[float, float, float, float]], frame_w: int, frame_h: int):
        if box is None:
            if self.state is None:
                self.state = (0.0, 0.0, float(frame_w), float(frame_h))
            return self.state

        x1, y1, x2, y2 = box
        bw = max(2.0, x2 - x1) * (1.0 + 2.0 * self.margin)
        bh = max(2.0, y2 - y1) * (1.0 + 2.0 * self.margin)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if bw / bh > self.aspect:
            bh = bw / self.aspect
        else:
            bw = bh * self.aspect

        tx1 = max(0.0, min(frame_w - bw, cx - bw / 2.0))
        ty1 = max(0.0, min(frame_h - bh, cy - bh / 2.0))
        tx2 = min(float(frame_w), tx1 + bw)
        ty2 = min(float(frame_h), ty1 + bh)

        target = (tx1, ty1, tx2, ty2)

        if self.state is None:
            self.state = target
        else:
            a = self.smooth
            self.state = tuple(a * s + (1.0 - a) * t for s, t in zip(self.state, target))

        return self.state


def clamp_queue_put(q, item):
    try:
        q.put_nowait(item)
    except Exception:
        try:
            q.get_nowait()
        except Exception:
            pass
        try:
            q.put_nowait(item)
        except Exception:
            pass


def drain_queue_latest(q):
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except Exception:
            break
    return latest


def choose_primary_track(tracks: Sequence[DetectionTrack], preferred_track_id=None):
    if preferred_track_id is not None:
        for t in tracks:
            if t.track_id == preferred_track_id:
                return t
    return max(tracks, key=lambda t: t.conf, default=None)


def group_box(tracks: Sequence[DetectionTrack]):
    if not tracks:
        return None
    xs1 = [t.bbox_xyxy[0] for t in tracks]
    ys1 = [t.bbox_xyxy[1] for t in tracks]
    xs2 = [t.bbox_xyxy[2] for t in tracks]
    ys2 = [t.bbox_xyxy[3] for t in tracks]
    return min(xs1), min(ys1), max(xs2), max(ys2)


def crop_frame(frame, box, out_size):
    out_w, out_h = out_size
    if box is None:
        return cv2.resize(frame, (out_w, out_h))
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return cv2.resize(frame, (out_w, out_h))
    return cv2.resize(roi, (out_w, out_h))


def annotate_target(frame, target, color, prefix):
    out = frame.copy()
    if target and target.bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in target.bbox_xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{prefix} ID {target.selected_track_id} {target.mode} {target.confidence:.2f}"
        cv2.putText(out, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def annotate_tracks(frame, tracks: Sequence[DetectionTrack], color, prefix):
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{prefix} {t.track_id} {t.conf:.2f}", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out
