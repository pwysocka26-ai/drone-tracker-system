from __future__ import annotations

import cv2
import numpy as np
from numpy import ndarray
from typing import List, Tuple


def tighten_bbox(bbox, frame_shape=None, min_size=12):
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    area = bw * bh

    if area < 250.0:
        scale = 0.88
    elif area < 900.0:
        scale = 0.78
    else:
        scale = 0.65

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(float(min_size), bw * float(scale))
    h = max(float(min_size), bh * float(scale))

    aspect = w / max(1.0, h)
    if aspect < 0.30:
        w = h * 0.30
    elif aspect > 4.50:
        h = w / 4.50

    nx1 = int(cx - w / 2.0)
    ny1 = int(cy - h / 2.0)
    nx2 = int(cx + w / 2.0)
    ny2 = int(cy + h / 2.0)

    if frame_shape is not None:
        fh, fw = frame_shape[:2]
        nx1 = max(0, min(fw - 1, nx1))
        ny1 = max(0, min(fh - 1, ny1))
        nx2 = max(0, min(fw - 1, nx2))
        ny2 = max(0, min(fh - 1, ny2))

    return nx1, ny1, nx2, ny2


def fit_panel_full(frame, out_size):
    out_w, out_h = out_size
    if frame is None or frame.size == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def add_title(panel, title):
    cv2.rectangle(panel, (0, 0), (440, 56), (0, 0, 0), -1)
    cv2.putText(panel, title, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return panel


def draw_tracks(frame, tracks, selected_id):
    out = frame.copy()
    for tr in tracks:
        x1, y1, x2, y2 = tighten_bbox(tr.bbox_xyxy, out.shape, min_size=12)
        cx, cy = [int(v) for v in tr.center_xy]
        if getattr(tr, 'is_active_target', False):
            color = (0, 255, 0)
        elif getattr(tr, 'is_valid_target', False):
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(out, (cx, cy), 4, color, -1)
        cv2.putText(
            out,
            f'ID {tr.track_id} {tr.confidence:.2f}',
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
    return out


def render_wide_panels(
    frame: ndarray,
    tracks: list,
    visible_sorted: list,
    *,
    selected_id,
    manual_lock: bool,
    lock_age: int,
    hold_count: int,
    pan_speed: float,
    tilt_speed: float,
    handoff_missed: int,
    handoff_max_gap: int,
    last_backend: str,
    conf: float,
    imgsz: int,
    last_yolo_boxes: int,
    last_det_tracks: int,
    drop_streak: int,
    wide_fov_deg: float,
    narrow_min_fov_deg: float,
    narrow_max_fov_deg: float,
    narrow_max_zoom: float,
    narrow_target_fill: float,
) -> Tuple[ndarray, ndarray]:
    wide_program = fit_panel_full(frame, (780, 360))
    debug_frame = draw_tracks(frame, tracks, selected_id)
    for idx, tr in enumerate(visible_sorted[:9], start=1):
        x1, y1, x2, y2 = tighten_bbox(tr.bbox_xyxy, debug_frame.shape, min_size=12)
        label = f'[{idx}] ID {tr.track_id}' if getattr(tr, 'is_confirmed', False) else f'[{idx}] ID {tr.track_id}?'
        cv2.putText(debug_frame, label, (x1, max(30, y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    wide_debug = fit_panel_full(debug_frame, (1560, 450))

    lock_mode = 'AUTO' if not manual_lock else 'MANUAL'
    cv2.putText(wide_debug, f'LOCK MODE: {lock_mode}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(wide_debug, f'SELECTED ID: {selected_id}', (20, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(wide_debug, f'HOLD COUNT: {hold_count}', (20, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(wide_debug, f'LOCK AGE: {lock_age}', (20, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(wide_debug, f'PAN SPD: {pan_speed:.1f}  TILT SPD: {tilt_speed:.1f}', (20, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    confirmed_count = sum(1 for t in tracks if getattr(t, 'is_confirmed', False))
    cv2.putText(wide_debug, f'MULTI TRACKS: {len(tracks)}  CONFIRMED: {confirmed_count}', (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(
        wide_debug,
        f'YOLO [{last_backend}]  conf={conf:.2f} imgsz={imgsz}  BOXES: {last_yolo_boxes}  DETS: {last_det_tracks}  DROP: {drop_streak}',
        (20, 316),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )
    cv2.putText(wide_debug, 'CORE MODE: PRIMARY TARGET', (20, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(wide_debug, f'WIDE 1920x1080 FOV {wide_fov_deg:.1f}   NARROW 1920x1080 FOV {narrow_min_fov_deg:.2f}-{narrow_max_fov_deg:.1f}   MAXZ {narrow_max_zoom:.1f}x  FILL {narrow_target_fill:.0f}%', (20, 388), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    auto_text = 'AUTO PICK ENABLED' if not manual_lock else 'AUTO PICK DISABLED'
    cv2.putText(wide_debug, auto_text, (20, 424), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(wide_debug, f'HANDOFF MISS: {handoff_missed}  MAX GAP: {handoff_max_gap}', (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    wide_program = add_title(wide_program, 'WIDE PROGRAM')
    wide_debug = add_title(wide_debug, 'WIDE DEBUG')
    return wide_program, wide_debug
