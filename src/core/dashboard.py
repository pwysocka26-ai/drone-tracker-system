from __future__ import annotations

import cv2
import numpy as np
from numpy import ndarray
from typing import List, Tuple

from core.lock_pipeline import STATE_REFINE, STATE_LOCKED


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


def crop_to_16_9(frame, center=None, scale=2.5, out_size=(780, 360), return_meta=False):
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
        resized = cv2.resize(frame, out_size)
        if return_meta:
            return resized, (0, 0, w, h)
        return resized

    resized = cv2.resize(crop, out_size, interpolation=cv2.INTER_LINEAR)
    if return_meta:
        return resized, (x1, y1, x2, y2)
    return resized


def crop_group(frame, tracks, out_size=(780, 360)):
    if not tracks:
        return crop_to_16_9(frame, None, 1.0, out_size)

    xs1 = [t.bbox_xyxy[0] for t in tracks]
    ys1 = [t.bbox_xyxy[1] for t in tracks]
    xs2 = [t.bbox_xyxy[2] for t in tracks]
    ys2 = [t.bbox_xyxy[3] for t in tracks]

    gx1, gy1, gx2, gy2 = min(xs1), min(ys1), max(xs2), max(ys2)
    cx = (gx1 + gx2) / 2.0
    cy = (gy1 + gy2) / 2.0
    gw = max(60.0, gx2 - gx1)
    gh = max(60.0, gy2 - gy1)

    h, w = frame.shape[:2]
    aspect = out_size[0] / out_size[1]

    crop_w = gw * 3.2
    crop_h = gh * 3.2
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


def draw_target_on_narrow(narrow_frame, crop_rect, display_bbox, display_center, display_no='?'):
    if display_bbox is None or display_center is None:
        return narrow_frame

    x1, y1, x2, y2 = tighten_bbox(display_bbox, min_size=14)
    cx1, cy1, cx2, cy2 = crop_rect
    crop_w = max(1, cx2 - cx1)
    crop_h = max(1, cy2 - cy1)
    nh, nw = narrow_frame.shape[:2]

    nx1 = int((x1 - cx1) * nw / crop_w)
    ny1 = int((y1 - cy1) * nh / crop_h)
    nx2 = int((x2 - cx1) * nw / crop_w)
    ny2 = int((y2 - cy1) * nh / crop_h)

    nx1 = max(0, min(nw - 1, nx1))
    ny1 = max(0, min(nh - 1, ny1))
    nx2 = max(0, min(nw - 1, nx2))
    ny2 = max(0, min(nh - 1, ny2))

    dcx = int((display_center[0] - cx1) * nw / crop_w)
    dcy = int((display_center[1] - cy1) * nh / crop_h)
    dcx = max(0, min(nw - 1, dcx))
    dcy = max(0, min(nh - 1, dcy))

    if nx2 > nx1 and ny2 > ny1:
        cv2.rectangle(narrow_frame, (nx1, ny1), (nx2, ny2), (0, 255, 255), 2)
        cv2.circle(narrow_frame, (dcx, dcy), 4, (0, 255, 255), -1)
        cv2.putText(
            narrow_frame,
            f'TRACKED TARGET [{display_no}]',
            (max(10, nx1), max(28, ny1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
    return narrow_frame


def render_narrow_panel(
    frame: ndarray,
    smooth_center,
    smooth_zoom: float,
    display_center,
    display_bbox,
    effective_track,
    narrow_tracker,
    selected_id,
    hold_count: int,
    edge_limit_active: bool,
    pipeline_state,
    lock_info: dict,
    narrow_min_zoom: float,
    narrow_max_zoom: float,
) -> tuple:
    if smooth_center is not None:
        smooth_zoom = max(narrow_min_zoom, min(narrow_max_zoom, float(smooth_zoom)))
        narrow_output, narrow_crop_rect = crop_to_16_9(frame, smooth_center, smooth_zoom, (780, 360), return_meta=True)
        label_track_id = getattr(effective_track, 'track_id', getattr(narrow_tracker, 'owner_id', selected_id))
        label = f'TARGET ID {label_track_id}' if effective_track is not None else f'TRACK HOLD ID {label_track_id}'
        real_pan_err = 0.0
        real_tilt_err = 0.0

        if display_center is not None:
            cx1, cy1, cx2, cy2 = narrow_crop_rect
            crop_w = max(1, cx2 - cx1)
            crop_h = max(1, cy2 - cy1)
            target_nx = (display_center[0] - cx1) * 780.0 / crop_w
            target_ny = (display_center[1] - cy1) * 360.0 / crop_h
            real_pan_err = target_nx - 390.0
            real_tilt_err = target_ny - 180.0

        measured_center_lock = abs(real_pan_err) < 16 and abs(real_tilt_err) < 16
        center_lock = bool(lock_info.get('ui_truthful_lock')) if pipeline_state in (STATE_REFINE, STATE_LOCKED) else measured_center_lock

        cv2.putText(narrow_output, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(narrow_output, f'PAN ERR {real_pan_err:.1f}  TILT ERR {real_tilt_err:.1f}', (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(narrow_output, f'ZOOM {smooth_zoom:.1f}x', (20, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(narrow_output, f'ZOOM MODE: TARGET FILL', (20, 294), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(narrow_output, f'HOLD {hold_count}', (20, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        center_lock_text = 'CENTER LOCK ON' if (center_lock and display_center is not None) else 'CENTER LOCK OFF'
        cv2.putText(narrow_output, center_lock_text, (20, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if edge_limit_active:
            cv2.putText(narrow_output, 'EDGE LIMIT COMP', (20, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        if display_bbox is not None and display_center is not None:
            disp_id = getattr(effective_track, 'track_id', getattr(narrow_tracker, 'owner_id', selected_id or '?'))
            narrow_output = draw_target_on_narrow(narrow_output, narrow_crop_rect, display_bbox, display_center, disp_id)
        cross_color = (0, 255, 0) if center_lock else (0, 255, 255)
        cv2.line(narrow_output, (390, 0), (390, 360), cross_color, 1)
        cv2.line(narrow_output, (0, 180), (780, 180), cross_color, 1)
    else:
        center_lock = False
        narrow_output, narrow_crop_rect = crop_to_16_9(frame, None, 1.7, (780, 360), return_meta=True)
        cv2.putText(narrow_output, 'BRAK CELU', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        real_pan_err = 0.0
        real_tilt_err = 0.0

    return narrow_output, narrow_crop_rect, real_pan_err, real_tilt_err, center_lock
