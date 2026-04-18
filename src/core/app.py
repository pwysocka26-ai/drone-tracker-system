import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import inspect
from core.run_reporting import generate_run_reports
from ultralytics import YOLO

from core.dashboard import render_wide_panels, add_title, tighten_bbox
from core.target_manager import TargetManager
from core.narrow_tracker import NarrowTracker
from core.multi_target_tracker import MultiTargetTracker
from core.local_target_tracker import LocalTargetTracker
from core.lock_pipeline import LockPipeline, STATE_LOCKED, STATE_REFINE, STATE_ACQUIRE, STATE_REACQUIRE, STATE_HOLD
from core.telemetry import TelemetryLogger


class Track:
    def __init__(self, track_id, bbox_xyxy, center_xy, confidence, raw_id=None):
        self.track_id = int(track_id)
        self.raw_id = int(raw_id if raw_id is not None else track_id)
        self.bbox_xyxy = bbox_xyxy
        self.center_xy = center_xy
        self.confidence = float(confidence)

        self.velocity_xy = (0.0, 0.0)
        self.age = 1
        self.hits = 1
        self.missed_frames = 0
        self.is_confirmed = False
        self.is_valid_target = True
        self.is_active_target = False


class DisplayBoxSmoother:
    def __init__(self, center_alpha=0.78, size_alpha=0.82, max_center_step=42.0, max_size_step=24.0):
        self.center_alpha = float(center_alpha)
        self.size_alpha = float(size_alpha)
        self.max_center_step = float(max_center_step)
        self.max_size_step = float(max_size_step)
        self.center = None
        self.size = None
        self.track_id = None

    def reset(self):
        self.center = None
        self.size = None
        self.track_id = None

    def update(self, track):
        if track is None:
            return None, None

        x1, y1, x2, y2 = track.bbox_xyxy
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        if self.track_id != int(track.track_id):
            self.center = (cx, cy)
            self.size = (w, h)
            self.track_id = int(track.track_id)
        else:
            pcx, pcy = self.center
            pw, ph = self.size

            dx = cx - pcx
            dy = cy - pcy
            if abs(dx) > self.max_center_step:
                cx = pcx + self.max_center_step * (1 if dx > 0 else -1)
            if abs(dy) > self.max_center_step:
                cy = pcy + self.max_center_step * (1 if dy > 0 else -1)

            dw = w - pw
            dh = h - ph
            if abs(dw) > self.max_size_step:
                w = pw + self.max_size_step * (1 if dw > 0 else -1)
            if abs(dh) > self.max_size_step:
                h = ph + self.max_size_step * (1 if dh > 0 else -1)

            scx = self.center_alpha * pcx + (1.0 - self.center_alpha) * cx
            scy = self.center_alpha * pcy + (1.0 - self.center_alpha) * cy
            sw = self.size_alpha * pw + (1.0 - self.size_alpha) * w
            sh = self.size_alpha * ph + (1.0 - self.size_alpha) * h

            self.center = (scx, scy)
            self.size = (sw, sh)

        scx, scy = self.center
        sw, sh = self.size
        return (scx, scy), (scx - 0.5 * sw, scy - 0.5 * sh, scx + 0.5 * sw, scy + 0.5 * sh)


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


def parse_tracks(result, frame_shape):
    h, w = frame_shape[:2]
    tracks = []

    boxes = getattr(result, 'boxes', None)
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return tracks

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else [0.0] * len(xyxy)
    raw_ids = boxes.id.cpu().numpy().astype(int).tolist() if boxes.id is not None else list(range(1, len(xyxy) + 1))

    for box, conf, raw_id in zip(xyxy, confs, raw_ids):
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh
        cy = (y1 + y2) / 2.0
        aspect = bw / max(1.0, bh)

        if float(conf) < 0.08:
            continue
        if cy > h * 0.82:
            continue
        if bw < 8.0 or bh < 8.0:
            continue
        if area < 60.0:
            continue
        if area > (w * h * 0.035):
            continue
        if aspect < 0.20 or aspect > 5.00:
            continue

        tracks.append(
            Track(
                track_id=int(raw_id),
                raw_id=int(raw_id),
                bbox_xyxy=(x1, y1, x2, y2),
                center_xy=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                confidence=float(conf),
            )
        )

    return tracks


class NarrowHandoffState:
    def __init__(self):
        self.track = None
        self.center = None
        self.bbox = None
        self.zoom = 1.8
        self.missed = 9999
        self.age = 9999
        self.last_good_center = None
        self.last_good_bbox = None
        self.last_good_zoom = 1.8
        self.gap_len = 0
        self.max_gap_len = 0

    def reset(self):
        self.track = None
        self.center = None
        self.bbox = None
        self.zoom = 1.8
        self.missed = 9999
        self.age = 9999
        self.last_good_center = None
        self.last_good_bbox = None
        self.last_good_zoom = 1.8
        self.gap_len = 0
        self.max_gap_len = 0

    def update_from_track(self, tr, zoom=None):
        self.track = tr
        self.center = tuple(float(v) for v in tr.center_xy)
        self.bbox = tuple(float(v) for v in tr.bbox_xyxy)
        if zoom is not None:
            self.zoom = float(zoom)
            self.last_good_zoom = float(zoom)
        self.last_good_center = self.center
        self.last_good_bbox = self.bbox
        self.missed = 0
        self.age = 0
        self.gap_len = 0

    def mark_missed(self):
        self.missed += 1
        self.age += 1
        self.gap_len += 1
        self.max_gap_len = max(self.max_gap_len, self.gap_len)


def _bbox_size(bbox):
    x1, y1, x2, y2 = bbox
    return (max(1.0, x2 - x1), max(1.0, y2 - y1))


def _distance(a, b):
    if a is None or b is None:
        return float('inf')
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return (dx * dx + dy * dy) ** 0.5


def _bbox_similarity(a, b):
    if a is None or b is None:
        return 0.0
    aw, ah = _bbox_size(a)
    bw, bh = _bbox_size(b)
    dw = abs(aw - bw) / max(aw, bw)
    dh = abs(ah - bh) / max(ah, bh)
    return max(0.0, 1.0 - 0.5 * (dw + dh))


def _choose_soft_handoff_track(tracks, selected_id, handoff_state, radius_px=140.0):
    if not tracks:
        return None

    if selected_id is not None:
        for tr in tracks:
            if int(getattr(tr, 'track_id', -1)) == int(selected_id):
                return tr

    anchor = handoff_state.last_good_center or handoff_state.center
    if anchor is None:
        return None

    best = None
    best_score = -1e9
    for tr in tracks:
        dist = _distance(tuple(tr.center_xy), anchor)
        if dist > radius_px:
            continue

        conf = float(getattr(tr, 'confidence', 0.0) or 0.0)
        sim = _bbox_similarity(getattr(tr, 'bbox_xyxy', None), handoff_state.last_good_bbox or handoff_state.bbox)
        score = conf * 8.0 + sim * 4.0 - dist / max(1.0, radius_px)
        if best is None or score > best_score:
            best = tr
            best_score = score
    return best


def _blend_track_with_handoff(tr, handoff_state, center_alpha=0.76, size_alpha=0.84):
    if tr is None:
        return None
    ref_center = handoff_state.last_good_center or handoff_state.center
    ref_bbox = handoff_state.last_good_bbox or handoff_state.bbox
    if ref_center is None or ref_bbox is None:
        return tr

    hx, hy = ref_center
    tx, ty = tr.center_xy
    tb = tr.bbox_xyxy

    bw, bh = _bbox_size(tb)
    hw, hh = _bbox_size(ref_bbox)

    smx = center_alpha * hx + (1.0 - center_alpha) * tx
    smy = center_alpha * hy + (1.0 - center_alpha) * ty
    smw = size_alpha * hw + (1.0 - size_alpha) * bw
    smh = size_alpha * hh + (1.0 - size_alpha) * bh

    tr.center_xy = (smx, smy)
    tr.bbox_xyxy = (smx - smw * 0.5, smy - smh * 0.5, smx + smw * 0.5, smy + smh * 0.5)
    return tr




def _make_track_from_bbox(track_id, bbox_xyxy, confidence, raw_id=None, frame_shape=None):
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    if frame_shape is not None:
        fh, fw = frame_shape[:2]
        x1 = max(0.0, min(float(fw - 1), x1))
        y1 = max(0.0, min(float(fh - 1), y1))
        x2 = max(0.0, min(float(fw - 1), x2))
        y2 = max(0.0, min(float(fh - 1), y2))
    return Track(
        track_id=int(track_id),
        raw_id=int(raw_id if raw_id is not None else track_id),
        bbox_xyxy=(x1, y1, x2, y2),
        center_xy=((x1 + x2) * 0.5, (y1 + y2) * 0.5),
        confidence=float(confidence),
    )


def _bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter / max(1.0, area_a + area_b - inter))


def _build_reacquire_roi(frame_shape, handoff_state, predicted_center=None, expand=4.8, min_size=280, max_size=1200):
    fh, fw = frame_shape[:2]
    ref_bbox = getattr(handoff_state, "last_good_bbox", None) or getattr(handoff_state, "bbox", None)
    ref_center = predicted_center or getattr(handoff_state, "last_good_center", None) or getattr(handoff_state, "center", None)

    if ref_center is None and ref_bbox is not None:
        ref_center = ((float(ref_bbox[0]) + float(ref_bbox[2])) * 0.5, (float(ref_bbox[1]) + float(ref_bbox[3])) * 0.5)
    if ref_center is None:
        return None

    if ref_bbox is not None:
        bw, bh = _bbox_size(ref_bbox)
    else:
        bw = bh = max(40.0, float(min_size) * 0.20)

    roi_w = max(float(min_size), bw * float(expand))
    roi_h = max(float(min_size), bh * float(expand))
    aspect = 16.0 / 9.0
    if roi_w / max(1.0, roi_h) < aspect:
        roi_w = roi_h * aspect
    else:
        roi_h = roi_w / aspect

    roi_w = min(float(max_size), max(120.0, roi_w))
    roi_h = min(float(max_size), max(120.0, roi_h))

    x1 = int(ref_center[0] - roi_w * 0.5)
    y1 = int(ref_center[1] - roi_h * 0.5)
    x2 = int(ref_center[0] + roi_w * 0.5)
    y2 = int(ref_center[1] + roi_h * 0.5)
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, fw, fh)
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return (x1, y1, x2, y2)


def _predict_tracks_in_roi(model, frame, roi_rect, conf, imgsz, classes):
    if roi_rect is None:
        return []
    x1, y1, x2, y2 = [int(v) for v in roi_rect]
    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return []

    results = model.predict(
        source=crop,
        conf=float(conf),
        imgsz=int(imgsz),
        classes=classes,
        verbose=False,
    )
    crop_tracks = parse_tracks(results[0], crop.shape)
    mapped = []
    for idx, tr in enumerate(crop_tracks, start=1):
        bx1, by1, bx2, by2 = tr.bbox_xyxy
        mapped.append(
            Track(
                track_id=int(getattr(tr, "track_id", idx)),
                raw_id=int(900000 + idx),
                bbox_xyxy=(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1),
                center_xy=(tr.center_xy[0] + x1, tr.center_xy[1] + y1),
                confidence=float(getattr(tr, "confidence", 0.0) or 0.0),
            )
        )
    return mapped


def _merge_track_lists(primary_tracks, secondary_tracks, iou_thresh=0.16, center_thresh=48.0):
    merged = list(primary_tracks or [])
    for cand in secondary_tracks or []:
        duplicate_idx = None
        duplicate_score = None
        for idx, base in enumerate(merged):
            if _bbox_iou(base.bbox_xyxy, cand.bbox_xyxy) >= float(iou_thresh):
                duplicate_idx = idx
                duplicate_score = float(getattr(base, "confidence", 0.0) or 0.0)
                break
            if _distance(base.center_xy, cand.center_xy) <= float(center_thresh):
                duplicate_idx = idx
                duplicate_score = float(getattr(base, "confidence", 0.0) or 0.0)
                break
        if duplicate_idx is None:
            merged.append(cand)
        elif float(getattr(cand, "confidence", 0.0) or 0.0) > float(duplicate_score or 0.0):
            merged[duplicate_idx] = cand
    return merged


def _estimate_zoom_for_track(frame_shape, track, current_zoom, max_zoom=2.4):
    fh, fw = frame_shape[:2]
    tx, ty = track.center_xy
    bw, bh = _bbox_size(track.bbox_xyxy)

    desired_crop_w = max(bw * 9.0, 180.0)
    desired_crop_h = max(bh * 9.0, 110.0)

    aspect = 780.0 / 360.0
    if desired_crop_w / desired_crop_h < aspect:
        desired_crop_w = desired_crop_h * aspect
    else:
        desired_crop_h = desired_crop_w / aspect

    margin_x = min(tx, fw - tx)
    margin_y = min(ty, fh - ty)
    edge_crop_w = max(80.0, margin_x * 2.0)
    edge_crop_h = max(80.0, margin_y * 2.0)
    if edge_crop_w / edge_crop_h < aspect:
        edge_crop_h = edge_crop_w / aspect
    else:
        edge_crop_w = edge_crop_h * aspect

    crop_w = max(desired_crop_w, min(edge_crop_w, fw))
    crop_h = max(desired_crop_h, min(edge_crop_h, fh))
    crop_w = min(crop_w, fw)
    crop_h = min(crop_h, fh)

    if (fw / fh) > aspect:
        req_zoom = fh / max(1.0, crop_h)
    else:
        req_zoom = fw / max(1.0, crop_w)

    req_zoom = max(1.0, min(max_zoom, req_zoom))

    max_step_up = 0.12
    max_step_down = 0.08
    if req_zoom > current_zoom:
        req_zoom = min(req_zoom, current_zoom + max_step_up)
    else:
        req_zoom = max(req_zoom, current_zoom - max_step_down)

    return req_zoom


def _apply_center_slew_limit(prev_center, next_center, max_step=24.0):
    if prev_center is None or next_center is None:
        return next_center
    px, py = prev_center
    nx, ny = next_center
    dx = nx - px
    dy = ny - py
    if abs(dx) > max_step:
        nx = px + max_step * (1 if dx > 0 else -1)
    if abs(dy) > max_step:
        ny = py + max_step * (1 if dy > 0 else -1)
    return (nx, ny)



def _resize_full_frame(frame, out_size):
    return cv2.resize(frame, out_size, interpolation=cv2.INTER_LINEAR)


def _safe_start_recording(out_name, record_fps, record_frame_size):
    for codec_name in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(str(out_name), cv2.VideoWriter_fourcc(*codec_name), record_fps, record_frame_size)
        if writer.isOpened():
            return writer, codec_name
        try:
            writer.release()
        except Exception:
            pass
    return None, None


def _build_target_manager_compat(tracker_cfg):
    desired = {
        "reacquire_radius_auto": float(tracker_cfg.get("reacquire_radius_auto", 135.0)),
        "reacquire_radius_manual": float(tracker_cfg.get("reacquire_radius_manual", 220.0)),
        "sticky_frames": int(tracker_cfg.get("sticky_frames", 18)),
        "switch_margin": float(tracker_cfg.get("switch_margin", 0.30)),
        "switch_dwell": int(tracker_cfg.get("switch_dwell", 5)),
        "switch_cooldown": int(tracker_cfg.get("switch_cooldown", 6)),
        "switch_persist": int(tracker_cfg.get("switch_persist", 3)),
        "max_select_missed": int(tracker_cfg.get("max_select_missed", 2)),
        "min_start_conf": float(tracker_cfg.get("min_start_conf", 0.10)),
        "min_start_hits": int(tracker_cfg.get("min_start_hits", 2)),
        "min_confirmed_conf": float(tracker_cfg.get("min_confirmed_conf", 0.10)),
        "min_hold_frames": int(tracker_cfg.get("min_hold_frames", 5)),
        "predicted_dist_px": float(tracker_cfg.get("predicted_dist_px", 95.0)),
        "raw_id_bonus": float(tracker_cfg.get("raw_id_bonus", 1.4)),
        "current_target_bonus": float(tracker_cfg.get("current_target_bonus", 4.0)),
        "selection_freeze_frames": int(tracker_cfg.get("target_selection_freeze_frames", 8)),
        "hard_keep_missed": int(tracker_cfg.get("hard_keep_missed", 1)),
        "hard_keep_conf": float(tracker_cfg.get("hard_keep_conf", 0.18)),
        "hard_switch_min_gain": float(tracker_cfg.get("hard_switch_min_gain", 1.10)),
        "owner_switch_min_gap_px": float(tracker_cfg.get("owner_switch_min_gap_px", 22.0)),
        "degraded_switch_persist": int(tracker_cfg.get("degraded_switch_persist", 2)),
        "healthy_switch_persist": int(tracker_cfg.get("healthy_switch_persist", 4)),
    }
    try:
        sig = inspect.signature(TargetManager.__init__)
        allowed = set(sig.parameters.keys()) - {"self"}
        desired = {k: v for k, v in desired.items() if k in allowed}
        return TargetManager(**desired)
    except Exception:
        return TargetManager()


def _narrow_update_compat(narrow_tracker, frame, active_track, tracks=None, manual_switch=False):
    try:
        sig = inspect.signature(narrow_tracker.update)
        kwargs = {}
        if "tracks" in sig.parameters:
            kwargs["tracks"] = tracks
        if "manual_switch" in sig.parameters:
            kwargs["manual_switch"] = manual_switch
        result = narrow_tracker.update(frame, active_track, **kwargs)
    except TypeError:
        result = narrow_tracker.update(frame, active_track)

    if not isinstance(result, tuple):
        raise ValueError("NarrowTracker.update returned unsupported result type")

    if len(result) == 7:
        return result
    if len(result) == 6:
        predicted_center, smooth_center, smooth_zoom, hold_count, pan_speed, tilt_speed = result
        return predicted_center, smooth_center, smooth_zoom, hold_count, pan_speed, tilt_speed, active_track

    raise ValueError(f"NarrowTracker.update returned unexpected tuple length: {len(result)}")


def _make_wide_snapshot(selected_track, target_manager, frame_id):
    if selected_track is None:
        return SimpleNamespace(
            valid=False,
            track_id=None,
            quality_score=0.0,
            track_age=0,
            missed_count=9999,
            center_xy=None,
            bbox_xyxy=None,
            owner_changed=False,
            reason='no_track',
            is_large_target=False,
            is_huge_outlier=False,
        )

    bbox = tuple(float(v) for v in getattr(selected_track, 'bbox_xyxy', (0, 0, 0, 0)))
    x1, y1, x2, y2 = bbox
    area = max(1.0, (x2 - x1) * (y2 - y1))
    conf = float(getattr(selected_track, 'confidence', 0.0) or 0.0)
    hits = int(getattr(selected_track, 'hits', 0) or 0)
    missed = int(getattr(selected_track, 'missed_frames', 0) or 0)
    quality_score = min(1.0, max(0.0, conf * 0.72 + min(0.28, hits * 0.04) - missed * 0.08))
    return SimpleNamespace(
        valid=True,
        track_id=int(getattr(selected_track, 'track_id', -1)),
        quality_score=quality_score,
        track_age=hits,
        missed_count=missed,
        center_xy=tuple(float(v) for v in getattr(selected_track, 'center_xy', (0.0, 0.0))),
        bbox_xyxy=bbox,
        owner_changed=(getattr(target_manager, 'last_switch_frame', -10**9) == int(frame_id)),
        reason='auto_tracking' if not getattr(target_manager, 'manual_lock', False) else 'manual_lock',
        is_large_target=area >= 1100.0,
        is_huge_outlier=False,
    )


class OwnerSyncGate:
    def __init__(self, sync_frames=5, max_current_missed=1, min_candidate_conf=0.16):
        self.sync_frames = int(sync_frames)
        self.max_current_missed = int(max_current_missed)
        self.min_candidate_conf = float(min_candidate_conf)
        self.pending_id = None
        self.pending_count = 0

    def reset(self):
        self.pending_id = None
        self.pending_count = 0

    def _find(self, tracks, tid):
        if tid is None:
            return None
        for tr in tracks or []:
            if int(getattr(tr, "track_id", -1)) == int(tid):
                return tr
        return None

    def maybe_sync(self, target_manager, candidate_track, tracks):
        if getattr(target_manager, "manual_lock", False):
            self.reset()
            return None
        if candidate_track is None:
            self.reset()
            return None

        candidate_id = int(getattr(candidate_track, "track_id", -1))
        if candidate_id < 0:
            self.reset()
            return None

        current_id = getattr(target_manager, "selected_id", None)
        if current_id is not None and int(current_id) == candidate_id:
            self.reset()
            return None

        current_track = self._find(tracks, current_id)
        current_degraded = (
            current_track is None
            or int(getattr(current_track, "missed_frames", 0) or 0) > self.max_current_missed
            or float(getattr(current_track, "confidence", 0.0) or 0.0) < self.min_candidate_conf
        )
        cand_ok = float(getattr(candidate_track, "confidence", 0.0) or 0.0) >= self.min_candidate_conf

        if not current_degraded or not cand_ok:
            self.reset()
            return None

        if self.pending_id == candidate_id:
            self.pending_count += 1
        else:
            self.pending_id = candidate_id
            self.pending_count = 1

        if self.pending_count >= self.sync_frames:
            target_manager.selected_id = candidate_id
            target_manager.lock_age = 0
            if hasattr(target_manager, "freeze_to"):
                try:
                    target_manager.freeze_to(candidate_id, max(6, int(getattr(target_manager, "selection_freeze_frames", 6))))
                except Exception:
                    pass
            self.reset()
            return candidate_id
        return None

def run_app(config):
    mode = config.get('mode', 'video')
    if mode != 'video':
        print('Ta wersja jest przygotowana do testow wideo.')
        return

    video_cfg = config.get('video') or {}
    yolo_cfg = config.get('yolo') or {}
    tracker_cfg = config.get('tracker') or {}
    narrow_cfg = config.get('narrow') or {}
    handoff_cfg = config.get('handoff') or {}
    control_cfg = config.get('narrow_control') or {}
    camera_cfg = config.get('camera') or {}
    wide_camera = camera_cfg.get('wide') or {}
    narrow_camera = camera_cfg.get('narrow') or {}

    source = video_cfg.get('source', 'video.mp4')
    record_on_start = bool(video_cfg.get('record_on_start', True))
    model_name = yolo_cfg.get('model', 'yolov8n.pt')
    backend = str(yolo_cfg.get('backend', 'track')).lower().strip()
    tracker_name = yolo_cfg.get('tracker', 'config/bytetrack_fast_acquire.yaml')
    conf = float(yolo_cfg.get('conf', 0.12))
    imgsz = int(yolo_cfg.get('imgsz', 960))
    classes = yolo_cfg.get('classes', [4])
    inference_every = int(yolo_cfg.get('inference_every', 1))

    search_fallback = bool(yolo_cfg.get('search_fallback', True))
    search_conf = float(yolo_cfg.get('search_conf', max(0.05, conf * 0.5)))
    search_imgsz = int(yolo_cfg.get('search_imgsz', max(imgsz, 1280)))
    search_interval = int(yolo_cfg.get('search_interval', 1))

    roi_cfg = config.get('roi_search', {})
    roi_search_enabled = bool(roi_cfg.get('enabled', True))
    roi_search_conf = float(roi_cfg.get('conf', max(0.05, search_conf * 0.85)))
    roi_search_imgsz = int(roi_cfg.get('imgsz', max(search_imgsz, 1920)))
    roi_search_interval = int(roi_cfg.get('interval', 1))
    roi_search_required_drop = int(roi_cfg.get('required_drop', 1))
    roi_search_expand = float(roi_cfg.get('expand', 4.8))
    roi_search_min_size = int(roi_cfg.get('min_size', 280))
    roi_search_max_size = int(roi_cfg.get('max_size', 1200))
    roi_merge_iou = float(roi_cfg.get('merge_iou', 0.16))
    roi_merge_center_px = float(roi_cfg.get('merge_center_px', 48.0))

    local_cfg = config.get('local_tracker', {})
    local_tracker_enabled = bool(local_cfg.get('enabled', True))
    local_tracker_max_lost = int(local_cfg.get('max_lost_frames', 20))
    local_tracker_min_score = float(local_cfg.get('min_score', 0.55))
    local_tracker_reacquire_radius = float(local_cfg.get('reacquire_radius_px', 240.0))

    soft_active_max_missed = int(handoff_cfg.get('soft_active_max_missed', 4))
    handoff_reacquire_radius = float(handoff_cfg.get('handoff_reacquire_radius', 165.0))
    handoff_hold_frames = int(handoff_cfg.get('handoff_hold_frames', 10))

    crop_max_step_px = float(control_cfg.get('crop_max_step_px', 24.0))
    crop_snap_deadband_px = float(control_cfg.get('crop_snap_deadband_px', 14.0))

    wide_fov_deg = float(wide_camera.get('fov_deg', 34.5))
    narrow_min_fov_deg = float(narrow_camera.get('min_fov_deg', narrow_camera.get('fov_min_deg', 1.77)))
    narrow_max_fov_deg = float(narrow_camera.get('max_fov_deg', narrow_camera.get('fov_max_deg', 49.0)))
    narrow_min_zoom = float(narrow_camera.get('min_zoom', 1.0))
    narrow_max_zoom = float(narrow_camera.get('max_zoom', 19.5))
    narrow_target_fill = float(narrow_camera.get('target_fill_percent', narrow_camera.get('screen_fill', 0.18)))
    if narrow_target_fill <= 1.0:
        narrow_target_fill *= 100.0

    effective_tracker_cfg = dict(tracker_cfg)
    effective_tracker_cfg['sticky_frames'] = max(int(effective_tracker_cfg.get('sticky_frames', 10)), 18)
    effective_tracker_cfg['switch_margin'] = max(float(effective_tracker_cfg.get('switch_margin', 0.15)), 0.30)
    effective_tracker_cfg['switch_dwell'] = max(int(effective_tracker_cfg.get('switch_dwell', 3)), 5)
    effective_tracker_cfg['switch_cooldown'] = max(int(effective_tracker_cfg.get('switch_cooldown', 2)), 6)
    effective_tracker_cfg['switch_persist'] = max(int(effective_tracker_cfg.get('switch_persist', 1)), 3)
    effective_tracker_cfg['raw_id_bonus'] = max(float(effective_tracker_cfg.get('raw_id_bonus', 0.8)), 1.4)
    effective_tracker_cfg['current_target_bonus'] = max(float(effective_tracker_cfg.get('current_target_bonus', 1.4)), 4.0)
    effective_tracker_cfg['target_selection_freeze_frames'] = max(int(effective_tracker_cfg.get('target_selection_freeze_frames', 4)), 8)
    effective_tracker_cfg['hard_keep_missed'] = int(effective_tracker_cfg.get('hard_keep_missed', 1))
    effective_tracker_cfg['hard_keep_conf'] = float(effective_tracker_cfg.get('hard_keep_conf', 0.18))
    effective_tracker_cfg['hard_switch_min_gain'] = float(effective_tracker_cfg.get('hard_switch_min_gain', 1.10))
    effective_tracker_cfg['owner_switch_min_gap_px'] = float(effective_tracker_cfg.get('owner_switch_min_gap_px', 22.0))
    effective_tracker_cfg['degraded_switch_persist'] = max(int(effective_tracker_cfg.get('degraded_switch_persist', 2)), 2)
    effective_tracker_cfg['healthy_switch_persist'] = max(int(effective_tracker_cfg.get('healthy_switch_persist', 4)), 4)

    model = YOLO(model_name)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'Nie moge otworzyc pliku: {source}')
        return

    multi_tracker = MultiTargetTracker(
        max_missed_frames=int(tracker_cfg.get('max_missed_frames', 24)),
        confirm_hits=int(tracker_cfg.get('confirm_hits', 2)),
        max_center_distance=float(tracker_cfg.get('max_center_distance', 180.0)),
        min_iou_for_match=float(tracker_cfg.get('min_iou_for_match', 0.01)),
        velocity_alpha=float(tracker_cfg.get('velocity_alpha', 0.65)),
        history_size=int(tracker_cfg.get('history_size', 12)),
        use_kalman=bool(tracker_cfg.get('use_kalman', True)),
        kalman_process_noise=float(tracker_cfg.get('kalman_process_noise', 0.03)),
        kalman_measurement_noise=float(tracker_cfg.get('kalman_measurement_noise', 0.20)),
    )
    target_manager = _build_target_manager_compat(effective_tracker_cfg)
    owner_sync_gate = OwnerSyncGate(
        sync_frames=int(effective_tracker_cfg.get('owner_sync_frames', 5)),
        max_current_missed=int(effective_tracker_cfg.get('hard_keep_missed', 1)),
        min_candidate_conf=float(effective_tracker_cfg.get('hard_keep_conf', 0.18)),
    )
    narrow_tracker = NarrowTracker(hold_frames=int(narrow_cfg.get('hold_frames', 80)))
    handoff_state = NarrowHandoffState()
    handoff_state.zoom = float(narrow_min_zoom)
    handoff_state.last_good_zoom = float(narrow_min_zoom)
    local_tracker = LocalTargetTracker(max_lost_frames=local_tracker_max_lost) if local_tracker_enabled else None
    local_tracker_owner_id = None
    lock_pipeline = LockPipeline(config)

    display_box_smoother = DisplayBoxSmoother(
        center_alpha=float(control_cfg.get('display_center_alpha', 0.78)),
        size_alpha=float(control_cfg.get('display_size_alpha', 0.82)),
        max_center_step=float(control_cfg.get('display_max_center_step', 42.0)),
        max_size_step=float(control_cfg.get('display_max_size_step', 24.0)),
    )

    window_name = 'Drone Tracker Multiview'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)

    base_artifacts_dir = Path("artifacts")
    runs_dir = base_artifacts_dir / "runs"
    latest_dir = base_artifacts_dir / "latest"
    runs_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = runs_dir / run_id
    images_dir = run_dir / "images"
    video_dir = run_dir / "video"
    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    recording = False
    video_writer = None
    record_fps = 30.0
    record_frame_size = None
    auto_record_pending = bool(record_on_start)

    telemetry_enabled = False
    telemetry = None

    prev_wide_owner_id = None
    prev_narrow_owner_id = None
    prev_center_lock = None
    prev_lock_state = None
    auto_keyframe_last = {}

    def start_recording():
        nonlocal recording, video_writer, auto_record_pending
        if recording and video_writer is not None:
            auto_record_pending = False
            return True
        out_name = video_dir / "tracker_analysis.mp4"
        if record_frame_size is None:
            auto_record_pending = True
            print('REC INFO: waiting for first dashboard frame to determine output size')
            return False
        video_writer, codec_name = _safe_start_recording(out_name, record_fps, record_frame_size)
        if video_writer is None:
            print('REC ERROR: cannot open output file')
            recording = False
            return False
        recording = True
        auto_record_pending = False
        print(f'REC START [{codec_name}]: {out_name}')
        return True

    def stop_recording():
        nonlocal recording, video_writer, auto_record_pending
        if not recording and video_writer is None:
            auto_record_pending = False
            return
        recording = False
        auto_record_pending = False
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        print('REC STOP')

    def start_telemetry():
        nonlocal telemetry_enabled, telemetry
        if telemetry_enabled and telemetry is not None:
            return
        telemetry = TelemetryLogger(run_name=run_id, fps=record_fps, run_dir=run_dir)
        telemetry_enabled = True
        print(f'METRICS START: {telemetry.path}')

    def generate_reports_if_possible(path_obj):
        if path_obj is None:
            return
        try:
            reports = generate_run_reports(
                telemetry_path=path_obj,
                shot_dir=images_dir,
                output_dir=run_dir,
                fps=record_fps if record_fps > 0 else 30.0,
                run_id=run_id,
                latest_dir=latest_dir,
                video_dir=video_dir,
            )
            print(f'RUN SUMMARY: {reports.summary_path}')
            print(f'METRICS CSV: {reports.metrics_path}')
            print(f'TIMELINE: {reports.timeline_path}')
            print(f'KEYFRAMES: {reports.keyframes_path}')
            print(f'MANIFEST MD: {reports.manifest_md_path}')
            print(f'MANIFEST JSON: {reports.manifest_json_path}')
        except Exception as exc:
            print(f'RUN REPORT ERROR: {exc}')

    def stop_telemetry():
        nonlocal telemetry_enabled, telemetry
        telemetry_enabled = False
        if telemetry is not None:
            print(f'METRICS STOP: {telemetry.path}')
            path_obj = telemetry.path
            telemetry.close()
            telemetry = None
            generate_reports_if_possible(path_obj)

    def save_screenshot(dashboard, wide_frame=None, narrow_frame=None, tag=None):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        suffix = f'_{tag}' if tag else ''
        dashboard_path = images_dir / f'dashboard_{stamp}{suffix}.png'
        cv2.imwrite(str(dashboard_path), dashboard)
        if wide_frame is not None:
            cv2.imwrite(str(images_dir / f'wide_{stamp}{suffix}.png'), wide_frame)
        if narrow_frame is not None:
            cv2.imwrite(str(images_dir / f'narrow_{stamp}{suffix}.png'), narrow_frame)
        print(f'SHOT SAVED: {dashboard_path}')

    def auto_capture_keyframe(tag, frame_idx, dashboard, wide_frame=None, narrow_frame=None):
        nonlocal auto_keyframe_last
        cooldown = 12
        last_frame = auto_keyframe_last.get(tag, -10000)
        if frame_idx - last_frame < cooldown:
            return
        save_screenshot(dashboard, wide_frame, narrow_frame, tag=tag)
        auto_keyframe_last[tag] = frame_idx

    frame_id = 0
    tracks = []
    last_yolo_boxes = 0
    last_det_tracks = 0
    last_backend = '-'
    drop_streak = 0

    if auto_record_pending:
        start_recording()
    start_telemetry()

    try:
        while True:
            stable_owner_track = None
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                multi_tracker.reset()
                target_manager.set_auto_mode()
                owner_sync_gate.reset()
                narrow_tracker.reset()
                handoff_state.reset()
                if local_tracker is not None:
                    local_tracker.reset()
                local_tracker_owner_id = None
                lock_pipeline.reset()
                display_box_smoother.reset()
                ret, frame = cap.read()
                if not ret:
                    break

            frame_id += 1
            predicted_center = narrow_tracker.kalman.predict()

            if frame_id % max(1, inference_every) == 0 or not tracks:
                if backend == 'predict':
                    results = model.predict(source=frame, conf=conf, imgsz=imgsz, classes=classes, verbose=False)
                    last_backend = 'predict'
                else:
                    results = model.track(
                        source=frame,
                        persist=True,
                        tracker=tracker_name,
                        conf=conf,
                        imgsz=imgsz,
                        classes=classes,
                        verbose=False,
                    )
                    last_backend = 'track'

                result = results[0]
                boxes = getattr(result, 'boxes', None)
                last_yolo_boxes = int(len(boxes)) if (boxes is not None and getattr(boxes, 'xyxy', None) is not None) else 0
                det_tracks = parse_tracks(result, frame.shape)

                need_search = search_fallback and (not det_tracks) and (frame_id % max(1, search_interval) == 0)
                if need_search:
                    search_results = model.predict(
                        source=frame,
                        conf=search_conf,
                        imgsz=search_imgsz,
                        classes=classes,
                        verbose=False,
                    )
                    search_result = search_results[0]
                    search_boxes = getattr(search_result, 'boxes', None)
                    search_box_count = int(len(search_boxes)) if (search_boxes is not None and getattr(search_boxes, 'xyxy', None) is not None) else 0
                    search_tracks = parse_tracks(search_result, frame.shape)
                    if search_tracks:
                        det_tracks = search_tracks
                        last_backend = 'predict-search'
                        last_yolo_boxes = search_box_count
                    else:
                        last_yolo_boxes = max(last_yolo_boxes, search_box_count)

                roi_tracks = []
                need_roi_search = (
                    roi_search_enabled
                    and handoff_state.last_good_center is not None
                    and (frame_id % max(1, roi_search_interval) == 0)
                    and ((not det_tracks) or drop_streak >= roi_search_required_drop)
                )
                if need_roi_search:
                    dynamic_expand = roi_search_expand * (1.0 + min(1.2, 0.10 * max(0, drop_streak)))
                    roi_rect = _build_reacquire_roi(
                        frame.shape,
                        handoff_state,
                        predicted_center=predicted_center,
                        expand=dynamic_expand,
                        min_size=roi_search_min_size,
                        max_size=roi_search_max_size,
                    )
                    roi_tracks = _predict_tracks_in_roi(
                        model,
                        frame,
                        roi_rect,
                        conf=roi_search_conf,
                        imgsz=roi_search_imgsz,
                        classes=classes,
                    )
                    if roi_tracks:
                        det_tracks = _merge_track_lists(
                            det_tracks,
                            roi_tracks,
                            iou_thresh=roi_merge_iou,
                            center_thresh=roi_merge_center_px,
                        )
                        last_backend = 'roi-search' if last_backend == 'track' and last_yolo_boxes == 0 else f'{last_backend}+roi'
                        last_yolo_boxes = max(last_yolo_boxes, len(roi_tracks))

                last_det_tracks = int(len(det_tracks))
                drop_streak = (drop_streak + 1) if last_det_tracks == 0 else 0
                tracks = multi_tracker.update(det_tracks, frame.shape)

            visible_sorted = sorted(tracks, key=lambda t: t.track_id)
            confirmed_tracks = [t for t in tracks if getattr(t, 'is_confirmed', False)]
            selection_tracks = confirmed_tracks if confirmed_tracks else tracks

            target_manager.update(selection_tracks, predicted_center, frame.shape)
            selected_track = target_manager.find_active_track(selection_tracks)

            lock_info = lock_pipeline.update(frame.shape, selected_track, tracks, target_manager.selected_id)
            pipeline_soft_track = lock_info.get('soft_track')
            pipeline_state = lock_info.get('state')
            if lock_info.get('predicted_center') is not None:
                predicted_center = lock_info.get('predicted_center')

            active_track = selected_track
            if pipeline_soft_track is not None:
                same_owner_soft = (
                    selected_track is not None
                    and int(getattr(pipeline_soft_track, 'track_id', -1)) == int(getattr(selected_track, 'track_id', -2))
                )
                soft_allowed_state = pipeline_state in (STATE_ACQUIRE, STATE_REFINE, STATE_REACQUIRE, STATE_HOLD)
                if same_owner_soft:
                    active_track = pipeline_soft_track
                elif active_track is None and soft_allowed_state:
                    active_track = pipeline_soft_track
            if active_track is not None and int(getattr(active_track, 'missed_frames', 0) or 0) > soft_active_max_missed:
                active_track = None

            if selected_track is not None and int(getattr(selected_track, 'missed_frames', 0) or 0) <= soft_active_max_missed:
                handoff_state.update_from_track(selected_track, zoom=handoff_state.zoom)
            else:
                handoff_state.mark_missed()

            soft_track = active_track
            reused_last_good = False

            if local_tracker is not None:
                if selected_track is not None and int(getattr(selected_track, 'missed_frames', 0) or 0) <= 1:
                    sid = int(getattr(selected_track, 'track_id', -1))
                    if (not local_tracker.is_active()) or local_tracker_owner_id != sid:
                        if local_tracker.initialize(frame, tuple(int(v) for v in selected_track.bbox_xyxy)):
                            local_tracker_owner_id = sid
                elif target_manager.selected_id is None and handoff_state.last_good_bbox is None:
                    local_tracker.reset()
                    local_tracker_owner_id = None

            if soft_track is None and handoff_state.missed <= handoff_hold_frames:
                reacquired = _choose_soft_handoff_track(
                    tracks,
                    target_manager.selected_id,
                    handoff_state,
                    handoff_reacquire_radius,
                )
                if reacquired is not None:
                    soft_track = _blend_track_with_handoff(reacquired, handoff_state)
                    handoff_state.update_from_track(soft_track, zoom=handoff_state.zoom)
                elif local_tracker is not None and local_tracker.is_active() and target_manager.selected_id is not None:
                    local_result = local_tracker.update(frame)
                    if (
                        local_result.bbox is not None
                        and local_result.center is not None
                        and (
                            local_result.ok
                            or float(local_result.score) >= local_tracker_min_score
                        )
                        and _distance(local_result.center, handoff_state.last_good_center or local_result.center) <= local_tracker_reacquire_radius
                    ):
                        soft_track = _make_track_from_bbox(
                            target_manager.selected_id,
                            local_result.bbox,
                            confidence=max(0.12, float(local_result.score) * 0.35),
                            raw_id=target_manager.selected_id,
                            frame_shape=frame.shape,
                        )
                        soft_track = _blend_track_with_handoff(soft_track, handoff_state, center_alpha=0.68, size_alpha=0.78)
                    elif handoff_state.last_good_center is not None and handoff_state.last_good_bbox is not None:
                        soft_track = Track(
                            track_id=int(target_manager.selected_id if target_manager.selected_id is not None else -1),
                            raw_id=int(target_manager.selected_id if target_manager.selected_id is not None else -1),
                            bbox_xyxy=handoff_state.last_good_bbox,
                            center_xy=handoff_state.last_good_center,
                            confidence=0.0,
                        )
                        reused_last_good = True
                elif handoff_state.last_good_center is not None and handoff_state.last_good_bbox is not None:
                    soft_track = Track(
                        track_id=int(target_manager.selected_id if target_manager.selected_id is not None else -1),
                        raw_id=int(target_manager.selected_id if target_manager.selected_id is not None else -1),
                        bbox_xyxy=handoff_state.last_good_bbox,
                        center_xy=handoff_state.last_good_center,
                        confidence=0.0,
                    )
                    reused_last_good = True

            for tr in tracks:
                tr.is_active_target = False
                tr.is_valid_target = bool(
                    getattr(tr, 'is_confirmed', False)
                    or getattr(tr, 'hits', 0) >= 2
                )
            if soft_track is not None:
                soft_track.is_active_target = True

            requested_track = soft_track

            predicted_center, smooth_center, smooth_zoom, hold_count, pan_speed, tilt_speed, narrow_owner_track = _narrow_update_compat(
                narrow_tracker,
                frame,
                requested_track,
                tracks=tracks,
                manual_switch=False,
            )

            # Wide/app request stays authoritative. Narrow-owned track is only
            # allowed as fallback during hold/reacquire when wide has no live track.
            effective_track = requested_track
            if effective_track is None and narrow_owner_track is not None:
                effective_track = narrow_owner_track

            current_wide_track = target_manager.find_active_track(selection_tracks or tracks)
            if (
                not target_manager.manual_lock
                and current_wide_track is not None
            ):
                current_missed = int(getattr(current_wide_track, "missed_frames", 0) or 0)
                current_conf = float(getattr(current_wide_track, "confidence", 0.0) or 0.0)
                if current_missed <= 1 and current_conf >= 0.08:
                    target_manager.freeze_to(current_wide_track.track_id, frames=10)

            display_center = None
            display_bbox = None
            edge_limit_active = False
            if effective_track is not None:
                display_center, display_bbox = display_box_smoother.update(effective_track)
                tx, ty = display_center if display_center is not None else effective_track.center_xy
                if smooth_center is None:
                    smooth_center = (tx, ty)
                pan_err = tx - smooth_center[0]
                tilt_err = ty - smooth_center[1]

                if target_manager.manual_lock:
                    smooth_center = (tx, ty)
                    pan_speed = pan_err
                    tilt_speed = tilt_err
                else:
                    alpha = 0.12
                    cx = smooth_center[0] + alpha * pan_err
                    cy = smooth_center[1] + alpha * tilt_err
                    refined_center = lock_info.get('refined_center')
                    if refined_center is not None and pipeline_state in (STATE_ACQUIRE, STATE_REFINE, STATE_LOCKED):
                        cx, cy = refined_center
                    elif abs(pan_err) < crop_snap_deadband_px and abs(tilt_err) < crop_snap_deadband_px:
                        cx, cy = tx, ty
                    smooth_center = _apply_center_slew_limit(smooth_center, (cx, cy), max_step=crop_max_step_px)
                    pan_speed = pan_err * alpha
                    tilt_speed = tilt_err * alpha

                if reused_last_good:
                    smooth_center = handoff_state.last_good_center
                    smooth_zoom = handoff_state.last_good_zoom
                    display_center = handoff_state.last_good_center
                    display_bbox = handoff_state.last_good_bbox
                else:
                    handoff_state.last_good_zoom = float(smooth_zoom)
                    handoff_state.zoom = float(smooth_zoom)
                    handoff_state.last_good_center = (float(tx), float(ty))
                    handoff_state.last_good_bbox = tuple(float(v) for v in display_bbox) if display_bbox is not None else handoff_state.last_good_bbox

                edge_limit_active = bool(getattr(narrow_tracker, "zoom_mode", "") != "TARGET_FILL")
            else:
                pan_speed = 0.0
                tilt_speed = 0.0
                display_box_smoother.reset()
                if handoff_state.missed <= handoff_hold_frames and handoff_state.last_good_center is not None:
                    smooth_center = handoff_state.last_good_center
                    smooth_zoom = handoff_state.last_good_zoom
                    display_center = handoff_state.last_good_center
                    display_bbox = handoff_state.last_good_bbox

            if smooth_center is not None:
                smooth_zoom = max(narrow_min_zoom, min(narrow_max_zoom, float(smooth_zoom)))
                narrow_output, narrow_crop_rect = crop_to_16_9(frame, smooth_center, smooth_zoom, (780, 360), return_meta=True)
                label_track_id = getattr(effective_track, 'track_id', getattr(narrow_tracker, 'owner_id', target_manager.selected_id))
                label = f'TARGET ID {label_track_id}' if effective_track is not None else f'TRACK HOLD ID {label_track_id}'
                real_pan_err = 0.0
                real_tilt_err = 0.0
                center_lock = False

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
                    disp_id = getattr(effective_track, 'track_id', getattr(narrow_tracker, 'owner_id', target_manager.selected_id or '?'))
                    narrow_output = draw_target_on_narrow(narrow_output, narrow_crop_rect, display_bbox, display_center, disp_id)
                cross_color = (0, 255, 0) if center_lock else (0, 255, 255)
                cv2.line(narrow_output, (390, 0), (390, 360), cross_color, 1)
                cv2.line(narrow_output, (0, 180), (780, 180), cross_color, 1)
            else:
                center_lock = False
                narrow_output, narrow_crop_rect = crop_to_16_9(frame, None, 1.7, (780, 360), return_meta=True)
                cv2.putText(narrow_output, 'BRAK CELU', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            wide_program, wide_debug = render_wide_panels(
                frame, tracks, visible_sorted,
                selected_id=target_manager.selected_id,
                manual_lock=target_manager.manual_lock,
                lock_age=target_manager.lock_age,
                hold_count=hold_count,
                pan_speed=pan_speed,
                tilt_speed=tilt_speed,
                handoff_missed=handoff_state.missed,
                handoff_max_gap=handoff_state.max_gap_len,
                last_backend=last_backend,
                conf=conf,
                imgsz=imgsz,
                last_yolo_boxes=last_yolo_boxes,
                last_det_tracks=last_det_tracks,
                drop_streak=drop_streak,
                wide_fov_deg=wide_fov_deg,
                narrow_min_fov_deg=narrow_min_fov_deg,
                narrow_max_fov_deg=narrow_max_fov_deg,
                narrow_max_zoom=narrow_max_zoom,
                narrow_target_fill=narrow_target_fill,
            )
            narrow_output = add_title(narrow_output, 'NARROW OUTPUT')
            dashboard = cv2.vconcat([cv2.hconcat([wide_program, narrow_output]), wide_debug])

            current_wide_owner_id = target_manager.selected_id
            current_narrow_owner_id = getattr(effective_track, 'track_id', None) if effective_track is not None else None
            current_lock_state = 'TRACKING' if effective_track is not None else 'REACQUIRE'
            current_center_lock = bool(center_lock)

            if prev_wide_owner_id is not None and current_wide_owner_id != prev_wide_owner_id and current_wide_owner_id is not None:
                auto_capture_keyframe('owner_switch', frame_id, dashboard, wide_debug, narrow_output)

            if prev_narrow_owner_id is not None and current_narrow_owner_id is None:
                auto_capture_keyframe('lock_lost', frame_id, dashboard, wide_debug, narrow_output)

            if prev_lock_state is not None and current_lock_state == 'REACQUIRE' and prev_lock_state != 'REACQUIRE':
                auto_capture_keyframe('reacquire', frame_id, dashboard, wide_debug, narrow_output)

            if prev_center_lock is True and current_center_lock is False:
                auto_capture_keyframe('center_lock_off', frame_id, dashboard, wide_debug, narrow_output)

            prev_wide_owner_id = current_wide_owner_id
            prev_narrow_owner_id = current_narrow_owner_id
            prev_lock_state = current_lock_state
            prev_center_lock = current_center_lock

            if record_frame_size is None:
                record_frame_size = (int(dashboard.shape[1]), int(dashboard.shape[0]))

            if auto_record_pending and video_writer is None and record_frame_size is not None:
                start_recording()

            if recording:
                cv2.circle(dashboard, (1510, 30), 8, (0, 0, 255), -1)
                cv2.putText(dashboard, 'REC', (1450, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if video_writer is not None:
                    safe_dashboard = np.ascontiguousarray(dashboard)
                    if safe_dashboard.dtype != np.uint8:
                        safe_dashboard = safe_dashboard.astype(np.uint8)
                    if safe_dashboard.shape[1] != record_frame_size[0] or safe_dashboard.shape[0] != record_frame_size[1]:
                        safe_dashboard = cv2.resize(safe_dashboard, record_frame_size, interpolation=cv2.INTER_LINEAR)
                    video_writer.write(safe_dashboard)

            if telemetry_enabled:
                cv2.circle(dashboard, (1510, 58), 8, (255, 255, 0), -1)
                cv2.putText(dashboard, 'METRICS', (1350, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(
                dashboard,
                'R=REC  T=METRICS  S=SHOT  0=AUTO  1-9=SLOT  ,/.=PREV/NEXT',
                (860, 800),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            if telemetry_enabled and telemetry is not None:
                active_bbox = list(effective_track.bbox_xyxy) if effective_track is not None else None
                active_area = None
                if effective_track is not None:
                    bw = max(0.0, float(effective_track.bbox_xyxy[2]) - float(effective_track.bbox_xyxy[0]))
                    bh = max(0.0, float(effective_track.bbox_xyxy[3]) - float(effective_track.bbox_xyxy[1]))
                    active_area = bw * bh

                telemetry.log_frame(
                    frame_idx=frame_id,
                    mode=('MANUAL' if target_manager.manual_lock else 'AUTO'),
                    selected_id=target_manager.selected_id,
                    active_track=effective_track,
                    tracks=tracks,
                    narrow_center=smooth_center,
                    center_lock=center_lock,
                    drift_gate_open=(soft_track is None and smooth_center is not None),
                    wide_owner_id=target_manager.selected_id,
                    narrow_owner_id=(getattr(effective_track, 'track_id', None) if effective_track is not None else None),
                    pending_owner_id=getattr(lock_pipeline.context, 'steering_target_id', None),
                    lock_state=str(pipeline_state or ('TRACKING' if requested_track is not None else ('HOLD' if effective_track is not None else 'REACQUIRE'))),
                    wide_owner_quality=(float(getattr(effective_track, 'confidence', 0.0)) if effective_track is not None else 0.0),
                    geometry_score=(max(0.0, 1.0 - ((abs(real_pan_err) + abs(real_tilt_err)) / 260.0)) if display_center is not None else 0.0),
                    edge_active=edge_limit_active,
                    edge_limit_active=edge_limit_active,
                    owner_missed_frames=handoff_state.missed,
                    yolo_boxes=last_yolo_boxes,
                    yolo_dets=last_det_tracks,
                    yolo_drop=drop_streak,
                    owner_reason='manual_lock' if target_manager.manual_lock else str(getattr(lock_pipeline.context, 'lock_loss_reason', '') or 'auto_tracking'),
                    handoff_reject_reason=str(getattr(lock_pipeline.context, 'lock_loss_reason', '') or ''),
                    manual_lock=bool(target_manager.manual_lock),
                    active_track_bbox=active_bbox,
                    active_track_area=active_area,
                    handoff_gap_len=handoff_state.gap_len,
                    handoff_max_gap_len=handoff_state.max_gap_len,
                    measurement_support=float(lock_info.get('measurement_support', 0.0) or 0.0),
                    identity_desync=bool(lock_info.get('identity_desync', False)),
                    identity_desync_frames=int(lock_info.get('identity_desync_frames', 0) or 0),
                    ui_truthful_lock=bool(lock_info.get('ui_truthful_lock', False)),
                )

            cv2.imshow(window_name, dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key in (ord('r'), ord('R')):
                if recording or auto_record_pending:
                    stop_recording()
                else:
                    auto_record_pending = True
                    start_recording()
            elif key in (ord('t'), ord('T')):
                stop_telemetry() if telemetry_enabled else start_telemetry()
            elif key in (ord('s'), ord('S')):
                save_screenshot(dashboard, wide_debug, narrow_output)
            elif key == ord('0'):
                target_manager.set_auto_mode()
                owner_sync_gate.reset()
                narrow_tracker.reset()
                handoff_state.reset()
                if local_tracker is not None:
                    local_tracker.reset()
                local_tracker_owner_id = None
                lock_pipeline.reset()
                display_box_smoother.reset()
            elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')):
                idx = int(chr(key)) - 1
                cand = visible_sorted
                if 0 <= idx < len(cand):
                    tr = cand[idx]
                    target_manager.set_manual_target(tr.track_id)
                    owner_sync_gate.reset()
                    narrow_tracker.reset()
                    handoff_state.reset()
                    lock_pipeline.reset()
                    display_box_smoother.reset()
                    narrow_tracker.kalman.init_state(tr.center_xy[0], tr.center_xy[1])
                    handoff_state.update_from_track(tr, zoom=handoff_state.zoom)
            elif key in (ord(','), ord('.')):
                cand = visible_sorted
                if cand:
                    cur_idx = 0
                    if target_manager.selected_id is not None:
                        for i, t in enumerate(cand):
                            if t.track_id == target_manager.selected_id:
                                cur_idx = i
                                break
                    step = -1 if key == ord(',') else 1
                    tr = cand[(cur_idx + step) % len(cand)]
                    target_manager.set_manual_target(tr.track_id)
                    owner_sync_gate.reset()
                    narrow_tracker.reset()
                    handoff_state.reset()
                    lock_pipeline.reset()
                    display_box_smoother.reset()
                    narrow_tracker.kalman.init_state(tr.center_xy[0], tr.center_xy[1])
                    handoff_state.update_from_track(tr, zoom=handoff_state.zoom)
    except Exception as exc:
        import sys, traceback
        print(f"CRASH at frame {frame_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        if telemetry is not None:
            telemetry.log_crash(frame_idx=frame_id, exc=exc)
        raise
    finally:
        if telemetry is not None:
            path_obj = telemetry.path
            telemetry.close()
            telemetry = None
            generate_reports_if_possible(path_obj)
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        cap.release()
        cv2.destroyAllWindows()