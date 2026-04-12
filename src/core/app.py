import cv2
from datetime import datetime
from pathlib import Path
from core.run_reporting import generate_run_reports
from ultralytics import YOLO

from core.target_manager import TargetManager
from core.narrow_tracker import NarrowTracker
from core.multi_target_tracker import MultiTargetTracker
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

    source = video_cfg.get('source', 'video.mp4')
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

    soft_active_max_missed = int(handoff_cfg.get('soft_active_max_missed', 4))
    handoff_reacquire_radius = float(handoff_cfg.get('handoff_reacquire_radius', 165.0))
    handoff_hold_frames = int(handoff_cfg.get('handoff_hold_frames', 10))

    crop_max_step_px = float(control_cfg.get('crop_max_step_px', 24.0))
    crop_snap_deadband_px = float(control_cfg.get('crop_snap_deadband_px', 14.0))

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
    target_manager = TargetManager(
        reacquire_radius_auto=float(tracker_cfg.get('reacquire_radius_auto', 135.0)),
        reacquire_radius_manual=float(tracker_cfg.get('reacquire_radius_manual', 220.0)),
        sticky_frames=int(tracker_cfg.get('sticky_frames', 22)),
        switch_margin=float(tracker_cfg.get('switch_margin', 0.36)),
        switch_dwell=int(tracker_cfg.get('switch_dwell', 6)),
        switch_cooldown=int(tracker_cfg.get('switch_cooldown', 7)),
        switch_persist=int(tracker_cfg.get('switch_persist', 2)),
        max_select_missed=int(tracker_cfg.get('max_select_missed', 2)),
        min_start_conf=float(tracker_cfg.get('min_start_conf', 0.10)),
        min_start_hits=int(tracker_cfg.get('min_start_hits', 2)),
        min_confirmed_conf=float(tracker_cfg.get('min_confirmed_conf', 0.10)),
        min_hold_frames=int(tracker_cfg.get('min_hold_frames', 5)),
        predicted_dist_px=float(tracker_cfg.get('predicted_dist_px', 95.0)),
        raw_id_bonus=float(tracker_cfg.get('raw_id_bonus', 1.8)),
        current_target_bonus=float(tracker_cfg.get('current_target_bonus', 2.6)),
        selection_freeze_frames=int(tracker_cfg.get('target_selection_freeze_frames', 4)),
    )
    narrow_tracker = NarrowTracker(hold_frames=int(narrow_cfg.get('hold_frames', 80)))
    handoff_state = NarrowHandoffState()
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
    record_frame_size = (1560, 810)

    telemetry_enabled = False
    telemetry = None

    prev_wide_owner_id = None
    prev_narrow_owner_id = None
    prev_center_lock = None
    prev_lock_state = None
    auto_keyframe_last = {}

    def start_recording():
        nonlocal recording, video_writer
        if recording and video_writer is not None:
            return
        out_name = video_dir / "tracker_analysis.mp4"
        video_writer = cv2.VideoWriter(str(out_name), cv2.VideoWriter_fourcc(*'mp4v'), record_fps, record_frame_size)
        if not video_writer.isOpened():
            print('REC ERROR: cannot open output file')
            video_writer = None
            recording = False
            return
        recording = True
        print(f'REC START: {out_name}')

    def stop_recording():
        nonlocal recording, video_writer
        if not recording and video_writer is None:
            return
        recording = False
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

    start_recording()
    start_telemetry()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                multi_tracker.reset()
                target_manager.set_auto_mode()
                narrow_tracker.reset()
                handoff_state.reset()
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

                last_det_tracks = int(len(det_tracks))
                drop_streak = (drop_streak + 1) if last_det_tracks == 0 else 0
                tracks = multi_tracker.update(det_tracks, frame.shape)

            visible_sorted = sorted(tracks, key=lambda t: t.track_id)
            confirmed_tracks = [t for t in tracks if getattr(t, 'is_confirmed', False)]
            selection_tracks = confirmed_tracks if confirmed_tracks else tracks

            target_manager.update(selection_tracks, predicted_center, frame.shape)
            selected_track = target_manager.find_active_track(tracks)

            active_track = selected_track
            if active_track is not None and int(getattr(active_track, 'missed_frames', 0) or 0) > soft_active_max_missed:
                active_track = None

            if selected_track is not None and int(getattr(selected_track, 'missed_frames', 0) or 0) <= soft_active_max_missed:
                handoff_state.update_from_track(selected_track, zoom=handoff_state.zoom)
            else:
                handoff_state.mark_missed()

            soft_track = active_track
            reused_last_good = False

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

            predicted_center, smooth_center, smooth_zoom, hold_count, _, _ = narrow_tracker.update(frame, soft_track)

            display_center = None
            display_bbox = None
            edge_limit_active = False
            if soft_track is not None:
                display_center, display_bbox = display_box_smoother.update(soft_track)
                tx, ty = display_center if display_center is not None else soft_track.center_xy
                if smooth_center is None:
                    smooth_center = (tx, ty)
                pan_err = tx - smooth_center[0]
                tilt_err = ty - smooth_center[1]

                if target_manager.manual_lock:
                    smooth_center = (tx, ty)
                    pan_speed = pan_err
                    tilt_speed = tilt_err
                else:
                    alpha = 0.18
                    cx = smooth_center[0] + alpha * pan_err
                    cy = smooth_center[1] + alpha * tilt_err
                    if abs(pan_err) < crop_snap_deadband_px and abs(tilt_err) < crop_snap_deadband_px:
                        cx, cy = tx, ty
                    smooth_center = _apply_center_slew_limit(smooth_center, (cx, cy), max_step=crop_max_step_px)
                    pan_speed = pan_err * alpha
                    tilt_speed = tilt_err * alpha

                target_zoom = _estimate_zoom_for_track(frame.shape, soft_track, handoff_state.zoom, max_zoom=2.4)
                handoff_state.zoom = target_zoom
                smooth_zoom = target_zoom

                if reused_last_good:
                    smooth_center = handoff_state.last_good_center
                    smooth_zoom = handoff_state.last_good_zoom
                    display_center = handoff_state.last_good_center
                    display_bbox = handoff_state.last_good_bbox
                else:
                    handoff_state.last_good_zoom = smooth_zoom

                edge_limit_active = abs(smooth_zoom - target_zoom) < 1e-6
            else:
                pan_speed = 0.0
                tilt_speed = 0.0
                display_box_smoother.reset()
                if handoff_state.missed <= handoff_hold_frames and handoff_state.last_good_center is not None:
                    smooth_center = handoff_state.last_good_center
                    smooth_zoom = handoff_state.last_good_zoom
                    display_center = handoff_state.last_good_center
                    display_bbox = handoff_state.last_good_bbox

            wide_program = crop_group(frame, tracks, (780, 360))
            debug_frame = draw_tracks(frame, tracks, target_manager.selected_id)
            for idx, tr in enumerate(visible_sorted[:9], start=1):
                x1, y1, x2, y2 = tighten_bbox(tr.bbox_xyxy, debug_frame.shape, min_size=12)
                label = f'[{idx}] ID {tr.track_id}' if getattr(tr, 'is_confirmed', False) else f'[{idx}] ID {tr.track_id}?'
                cv2.putText(debug_frame, label, (x1, max(30, y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            wide_debug = crop_group(debug_frame, tracks, (1560, 450))

            if smooth_center is not None:
                narrow_output, narrow_crop_rect = crop_to_16_9(frame, smooth_center, smooth_zoom, (780, 360), return_meta=True)
                label = f'TARGET ID {target_manager.selected_id}' if soft_track is not None else f'TRACK HOLD ID {target_manager.selected_id}'
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
                    center_lock = abs(real_pan_err) < 16 and abs(real_tilt_err) < 16

                cv2.putText(narrow_output, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(narrow_output, f'PAN ERR {real_pan_err:.1f}  TILT ERR {real_tilt_err:.1f}', (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(narrow_output, f'ZOOM {smooth_zoom:.1f}x', (20, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(narrow_output, f'HOLD {hold_count}', (20, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                center_lock_text = 'CENTER LOCK ON' if (center_lock and display_center is not None) else 'CENTER LOCK OFF'
                cv2.putText(narrow_output, center_lock_text, (20, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                if edge_limit_active:
                    cv2.putText(narrow_output, 'EDGE LIMIT COMP', (20, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                if display_bbox is not None and display_center is not None:
                    disp_id = soft_track.track_id if soft_track is not None else (target_manager.selected_id or '?')
                    narrow_output = draw_target_on_narrow(narrow_output, narrow_crop_rect, display_bbox, display_center, disp_id)
                cross_color = (0, 255, 0) if center_lock else (0, 255, 255)
                cv2.line(narrow_output, (390, 0), (390, 360), cross_color, 1)
                cv2.line(narrow_output, (0, 180), (780, 180), cross_color, 1)
            else:
                center_lock = False
                narrow_output, narrow_crop_rect = crop_to_16_9(frame, None, 1.7, (780, 360), return_meta=True)
                cv2.putText(narrow_output, 'BRAK CELU', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            lock_mode = 'AUTO' if not target_manager.manual_lock else 'MANUAL'
            cv2.putText(wide_debug, f'LOCK MODE: {lock_mode}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(wide_debug, f'SELECTED ID: {target_manager.selected_id}', (20, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(wide_debug, f'HOLD COUNT: {hold_count}', (20, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(wide_debug, f'LOCK AGE: {target_manager.lock_age}', (20, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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
            cv2.putText(wide_debug, 'SUPPORT TRACKS: DEBUG ONLY', (20, 388), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            auto_text = 'AUTO PICK ENABLED' if not target_manager.manual_lock else 'AUTO PICK DISABLED'
            cv2.putText(wide_debug, auto_text, (20, 424), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(wide_debug, f'HANDOFF MISS: {handoff_state.missed}  MAX GAP: {handoff_state.max_gap_len}', (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            wide_program = add_title(wide_program, 'WIDE PROGRAM')
            narrow_output = add_title(narrow_output, 'NARROW OUTPUT')
            wide_debug = add_title(wide_debug, 'WIDE DEBUG')
            dashboard = cv2.vconcat([cv2.hconcat([wide_program, narrow_output]), wide_debug])

            current_wide_owner_id = target_manager.selected_id
            current_narrow_owner_id = soft_track.track_id if soft_track is not None else None
            current_lock_state = 'TRACKING' if soft_track is not None else 'REACQUIRE'
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

            if recording:
                cv2.circle(dashboard, (1510, 30), 8, (0, 0, 255), -1)
                cv2.putText(dashboard, 'REC', (1450, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if video_writer is not None:
                    video_writer.write(dashboard)

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
                active_bbox = list(soft_track.bbox_xyxy) if soft_track is not None else None
                active_area = None
                if soft_track is not None:
                    bw = max(0.0, float(soft_track.bbox_xyxy[2]) - float(soft_track.bbox_xyxy[0]))
                    bh = max(0.0, float(soft_track.bbox_xyxy[3]) - float(soft_track.bbox_xyxy[1]))
                    active_area = bw * bh

                telemetry.log_frame(
                    frame_idx=frame_id,
                    mode=('MANUAL' if target_manager.manual_lock else 'AUTO'),
                    selected_id=target_manager.selected_id,
                    active_track=soft_track,
                    tracks=tracks,
                    narrow_center=smooth_center,
                    center_lock=center_lock,
                    drift_gate_open=(soft_track is None and smooth_center is not None),
                    wide_owner_id=target_manager.selected_id,
                    narrow_owner_id=(soft_track.track_id if soft_track is not None else None),
                    pending_owner_id=None,
                    lock_state=('TRACKING' if soft_track is not None else 'REACQUIRE'),
                    wide_owner_quality=(float(getattr(soft_track, 'confidence', 0.0)) if soft_track is not None else 0.0),
                    geometry_score=1.0 if center_lock else 0.0,
                    edge_active=edge_limit_active,
                    edge_limit_active=edge_limit_active,
                    owner_missed_frames=handoff_state.missed,
                    yolo_boxes=last_yolo_boxes,
                    yolo_dets=last_det_tracks,
                    yolo_drop=drop_streak,
                    owner_reason='manual_lock' if target_manager.manual_lock else 'auto_tracking',
                    handoff_reject_reason='',
                    manual_lock=bool(target_manager.manual_lock),
                    active_track_bbox=active_bbox,
                    active_track_area=active_area,
                    handoff_gap_len=handoff_state.gap_len,
                    handoff_max_gap_len=handoff_state.max_gap_len,
                )

            cv2.imshow(window_name, dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key in (ord('r'), ord('R')):
                stop_recording() if recording else start_recording()
            elif key in (ord('t'), ord('T')):
                stop_telemetry() if telemetry_enabled else start_telemetry()
            elif key in (ord('s'), ord('S')):
                save_screenshot(dashboard, wide_debug, narrow_output)
            elif key == ord('0'):
                target_manager.set_auto_mode()
                narrow_tracker.reset()
                handoff_state.reset()
                display_box_smoother.reset()
            elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')):
                idx = int(chr(key)) - 1
                cand = visible_sorted
                if 0 <= idx < len(cand):
                    tr = cand[idx]
                    target_manager.set_manual_target(tr.track_id)
                    narrow_tracker.reset()
                    handoff_state.reset()
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
                    narrow_tracker.reset()
                    handoff_state.reset()
                    display_box_smoother.reset()
                    narrow_tracker.kalman.init_state(tr.center_xy[0], tr.center_xy[1])
                    handoff_state.update_from_track(tr, zoom=handoff_state.zoom)
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