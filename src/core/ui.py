import cv2
import numpy as np
from typing import List, Optional, Tuple

from core.models import TrackedObject, TargetMessage


def clamp_box(x1, y1, x2, y2, w, h):
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        x1 -= (x2 - w)
        x2 = w
    if y2 > h:
        y1 -= (y2 - h)
        y2 = h
    return max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))


def crop_to_16_9(frame, center=None, scale: float = 2.5, out_size=(780, 360)):
    out_w, out_h = out_size
    h, w = frame.shape[:2]
    aspect = out_w / out_h

    if center is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = center

    if (w / h) > aspect:
        crop_h = h / scale
        crop_w = crop_h * aspect
    else:
        crop_w = w / scale
        crop_h = crop_w / aspect

    crop_w = int(max(80, min(w, crop_w)))
    crop_h = int(max(80, min(h, crop_h)))

    x1 = int(cx - crop_w / 2)
    y1 = int(cy - crop_h / 2)
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(frame, out_size)
    return cv2.resize(crop, out_size, interpolation=cv2.INTER_LINEAR)


def crop_group(frame, tracks: List[TrackedObject], out_size=(780, 360)):
    if not tracks:
        return crop_to_16_9(frame, None, 1.0, out_size)

    xs1 = [t.bbox_xyxy[0] for t in tracks]
    ys1 = [t.bbox_xyxy[1] for t in tracks]
    xs2 = [t.bbox_xyxy[2] for t in tracks]
    ys2 = [t.bbox_xyxy[3] for t in tracks]

    gx1 = min(xs1)
    gy1 = min(ys1)
    gx2 = max(xs2)
    gy2 = max(ys2)

    cx = (gx1 + gx2) / 2.0
    cy = (gy1 + gy2) / 2.0
    gw = max(60.0, gx2 - gx1)
    gh = max(60.0, gy2 - gy1)

    h, w = frame.shape[:2]
    aspect = out_size[0] / out_size[1]

    crop_w = gw * 4.5
    crop_h = gh * 4.5

    if crop_w / crop_h < aspect:
        crop_w = crop_h * aspect
    else:
        crop_h = crop_w / aspect

    crop_w = min(w, crop_w)
    crop_h = min(h, crop_h)

    x1 = int(cx - crop_w / 2.0)
    y1 = int(cy - crop_h / 2.0)
    x2 = int(cx + crop_w / 2.0)
    y2 = int(cy + crop_h / 2.0)
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(frame, out_size)
    return cv2.resize(crop, out_size, interpolation=cv2.INTER_LINEAR)


def draw_tracks(frame, tracks: List[TrackedObject], selected_id: Optional[int]):
    out = frame.copy()
    for tr in tracks:
        x1, y1, x2, y2 = [int(v) for v in tr.bbox_xyxy]
        cx, cy = [int(v) for v in tr.center_xy]
        color = (0, 255, 255) if tr.track_id == selected_id else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(out, (cx, cy), 4, color, -1)
        cv2.putText(out, f"ID {tr.track_id}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return out


def draw_narrow(frame, target: Optional[TargetMessage], out_size=(780, 360)):
    if target is None:
        panel = crop_to_16_9(frame, None, 1.5, out_size)
        cv2.putText(panel, "BRAK CELU", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return panel

    panel = crop_to_16_9(frame, target.center_xy, 3.0, out_size)
    cv2.putText(panel, f"TARGET ID {target.target_id}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.line(panel, (out_size[0] // 2, 0), (out_size[0] // 2, out_size[1]), (0, 255, 255), 1)
    cv2.line(panel, (0, out_size[1] // 2), (out_size[0], out_size[1] // 2), (0, 255, 255), 1)
    return panel


def add_title(panel, title):
    cv2.rectangle(panel, (0, 0), (430, 56), (0, 0, 0), -1)
    cv2.putText(panel, title, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return panel
