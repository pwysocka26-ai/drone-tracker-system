import cv2
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

from core.target_manager import TargetManager
from core.narrow_tracker import NarrowTracker
from core.multi_target_tracker import MultiTargetTracker
from core.telemetry import TelemetryLogger
from core.lock_pipeline import LockPipeline


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
    def __init__(self, center_alpha=0.70, size_alpha=0.74, max_center_step=44.0, max_size_step=26.0):
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

    if area < 180.0:
        scale = 0.60
    elif area < 700.0:
        scale = 0.56
    else:
        scale = 0.54

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


def run_app(config):
    mode = config.get('mode', 'video')
    if mode != 'video':
        print('Ta wersja jest przygotowana do testow wideo.')
        return

    video_cfg = config.get('video') or {}
    yolo_cfg = config.get('yolo') or {}
    tracker_cfg = config.get('tracker') or {}
    narrow_cfg = config.get('narrow') or {}
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
    )
    narrow_tracker = NarrowTracker(hold_frames=int(narrow_cfg.get('hold_frames', 80)))
    narrow_tracker.last_zoom = None
    lock_pipeline = LockPipeline(config)
    display_box_smoother = DisplayBoxSmoother(
        center_alpha=float(control_cfg.get('display_center_alpha', 0.70)),
        size_alpha=float(control_cfg.get('display_size_alpha', 0.74)),
        max_center_step=float(control_cfg.get('display_max_center_step', 44.0)),
        max_size_step=float(control_cfg.get('display_max_size_step', 26.0)),
    )

    window_name = 'Drone Tracker Multiview'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)

    recording = False
    video_writer = None
    record_fps = 30.0
    record_frame_size = (1560, 810)

    telemetry_enabled = False
    telemetry = None

    def start_recording():
        nonlocal recording, video_writer
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_name = f'tracker_analysis_{stamp}.mp4'
        video_writer = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), record_fps, record_frame_size)
        if not video_writer.isOpened():
            print('REC ERROR: cannot open output file')
            video_writer = None
            recording = False
            return
        recording = True
        print(f'REC START: {out_name}')

    def stop_recording():
        nonlocal recording, video_writer
        recording = False
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception as exc:
                print(f'REC STOP WARN: {exc}')
            finally:
                video_writer = None
        print('REC STOP')

    def start_telemetry():
        nonlocal telemetry_enabled, telemetry
        if telemetry is not None:
            try:
                telemetry.close()
            except Exception:
                pass
            telemetry = None
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f'run_{stamp}'
        telemetry = TelemetryLogger(run_name=run_name, fps=record_fps)
        telemetry_enabled = True
        path = getattr(telemetry, 'path', '(unknown)')
        print(f'METRICS START: {path}')

    def stop_telemetry():
        nonlocal telemetry_enabled, telemetry
        telemetry_enabled = False
        if telemetry is not None:
            path = getattr(telemetry, 'path', '(unknown)')
            print(f'METRICS STOP: {path}')
            try:
                telemetry.close()
            except Exception as exc:
                print(f'METRICS STOP WARN: {exc}')
            finally:
                telemetry = None

    screenshot_dir = None

    def ensure_screenshot_dir():
        nonlocal screenshot_dir
        if screenshot_dir is None:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            screenshot_dir = Path(f'tracker_shots_{stamp}')
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            print(f'SHOT DIR: {screenshot_dir}')
        return screenshot_dir

    def save_screenshot(dashboard, wide_frame=None, narrow_frame=None):
        shot_dir = ensure_screenshot_dir()
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        dashboard_path = shot_dir / f'dashboard_{stamp}.png'
        cv2.imwrite(str(dashboard_path), dashboard)
        if wide_frame is not None:
            cv2.imwrite(str(shot_dir / f'wide_{stamp}.png'), wide_frame)
        if narrow_frame is not None:
            cv2.imwrite(str(shot_dir / f'narrow_{stamp}.png'), narrow_frame)
        print(f'SHOT SAVED: {dashboard_path}')

    frame_id = 0
    tracks = []
    last_yolo_boxes = 0
    last_det_tracks = 0
    last_backend = '-'
    drop_streak = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                multi_tracker.reset()
                target_manager.set_auto_mode()
                narrow_tracker.reset()
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

                last_det_tracks = int(len(det_tracks))
                drop_streak = (drop_streak + 1) if last_det_tracks == 0 else 0
                tracks = multi_tracker.update(det_tracks, frame.shape)

            visible_sorted = sorted(tracks, key=lambda t: t.track_id)
            confirmed_tracks = [t for t in tracks if getattr(t, 'is_confirmed', False)]
            selection_tracks = confirmed_tracks if confirmed_tracks else tracks

            target_manager.update(selection_tracks, predicted_center, frame.shape)
            selected_track = target_manager.find_active_track(tracks)

            pipeline_out = lock_pipeline.update(
                frame_shape=frame.shape,
                selected_track=selected_track,
                tracks=tracks,
                selected_id=target_manager.selected_id,
            )

            soft_track = pipeline_out['soft_track']
            refined_center = pipeline_out['refined_center']
            refined_bbox = pipeline_out['refined_bbox']
            pipeline_state = pipeline_out['state']
            local_lock_score = pipeline_out['local_lock_score']
            lock_confidence = pipeline_out['lock_confidence']
            jump_rejected = pipeline_out['jump_rejected']
            anchor_jump_px = pipeline_out['anchor_jump_px']
            handoff_ready_score = pipeline_out['handoff_ready_score']
            handoff_ready_streak = pipeline_out['handoff_ready_streak']
            wide_center_jitter = pipeline_out['wide_center_jitter']
            wide_bbox_jitter = pipeline_out['wide_bbox_jitter']
            lock_loss_reason = pipeline_out['lock_loss_reason']

            for tr in tracks:
                tr.is_active_target = False
                tr.is_valid_target = bool(
                    getattr(tr, 'is_confirmed', False)
                    or getattr(tr, 'hits', 0) >= 2
                )
            if soft_track is not None:
                soft_track.is_active_target = True

            steering_track = soft_track
            if refined_center is not None and steering_track is not None:
                steering_track = Track(
                    track_id=int(getattr(soft_track, 'track_id', -1)),
                    raw_id=int(getattr(soft_track, 'raw_id', -1)),
                    bbox_xyxy=refined_bbox if refined_bbox is not None else soft_track.bbox_xyxy,
                    center_xy=refined_center,
                    confidence=float(getattr(soft_track, 'confidence', 0.0) or 0.0),
                )

            predicted_center, smooth_center, smooth_zoom, hold_count, pan_speed, tilt_speed, narrow_state, jump_limited = narrow_tracker.update(frame, steering_track)

            display_center = None
            display_bbox = None
            if soft_track is not None:
                display_center, display_bbox = display_box_smoother.update(soft_track)
            else:
                display_box_smoother.reset()

            wide_program = crop_group(frame, tracks, (780, 360))
            debug_frame = draw_tracks(frame, tracks, target_manager.selected_id)
            for idx, tr in enumerate(visible_sorted[:9], start=1):
                x1, y1, x2, y2 = tighten_bbox(tr.bbox_xyxy, debug_frame.shape, min_size=12)
                label = f'[{idx}] ID {tr.track_id}' if getattr(tr, 'is_confirmed', False) else f'[{idx}] ID {tr.track_id}?'
                cv2.putText(debug_frame, label, (x1, max(30, y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            wide_debug = crop_group(debug_frame, tracks, (1560, 450))

            if smooth_center is not None:
                narrow_output, narrow_crop_rect = crop_to_16_9(frame, smooth_center, smooth_zoom, (780, 360), return_meta=True)
                is_proxy_track = bool(getattr(steering_track, 'is_proxy', False)) if steering_track is not None else False
                label = f'TRACK HOLD ID {target_manager.selected_id}' if is_proxy_track else (f'TARGET ID {target_manager.selected_id}' if soft_track is not None else f'TRACK HOLD ID {target_manager.selected_id}')
                real_pan_err = 0.0
                real_tilt_err = 0.0
                center_lock = False

                if steering_track is not None:
                    cx1, cy1, cx2, cy2 = narrow_crop_rect
                    crop_w = max(1, cx2 - cx1)
                    crop_h = max(1, cy2 - cy1)
                    target_nx = (steering_track.center_xy[0] - cx1) * 780.0 / crop_w
                    target_ny = (steering_track.center_xy[1] - cy1) * 360.0 / crop_h
                    real_pan_err = target_nx - 390.0
                    real_tilt_err = target_ny - 180.0
                    radial_error = float((real_pan_err * real_pan_err + real_tilt_err * real_tilt_err) ** 0.5)
                    dynamic_lock_radius = 22.0 if (is_proxy_track or narrow_state == 'HOLD' or pipeline_state in ('REACQUIRE', 'HOLD')) else 16.0
                    center_lock = (
                        radial_error <= dynamic_lock_radius
                        and (
                            local_lock_score >= 0.72
                            or lock_confidence >= 0.68
                            or is_proxy_track
                            or hold_count > 0
                        )
                    )
                else:
                    radial_error = 0.0
                radial_error = float((real_pan_err * real_pan_err + real_tilt_err * real_tilt_err) ** 0.5)

                cv2.putText(narrow_output, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(narrow_output, f'PAN ERR {real_pan_err:.1f}  TILT ERR {real_tilt_err:.1f}', (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(narrow_output, f'ZOOM {smooth_zoom:.1f}x', (20, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(narrow_output, f'HOLD {hold_count}', (20, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(narrow_output, f'NARROW {narrow_state} / PIPE {pipeline_state}', (20, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
                cv2.putText(narrow_output, f'LOCK SCORE {local_lock_score:.2f}  CONF {lock_confidence:.2f}', (20, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
                if jump_limited or jump_rejected:
                    cv2.putText(narrow_output, f'JUMP LIMIT {anchor_jump_px:.1f}px', (20, 294), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                center_lock_text = 'CENTER LOCK ON' if (center_lock and steering_track is not None) else 'CENTER LOCK OFF'
                cv2.putText(narrow_output, center_lock_text, (20, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

                overlay_track = soft_track if soft_track is not None else steering_track
                overlay_bbox = display_bbox if (display_bbox is not None and not bool(getattr(overlay_track, 'is_proxy', False))) else (getattr(overlay_track, 'bbox_xyxy', None) if overlay_track is not None else None)
                overlay_center = display_center if (display_center is not None and not bool(getattr(overlay_track, 'is_proxy', False))) else (getattr(overlay_track, 'center_xy', None) if overlay_track is not None else None)
                if overlay_bbox is not None and overlay_center is not None:
                    disp_id = getattr(overlay_track, 'track_id', None) if overlay_track is not None else (target_manager.selected_id or '?')
                    narrow_output = draw_target_on_narrow(narrow_output, narrow_crop_rect, overlay_bbox, overlay_center, disp_id)

                cross_color = (0, 255, 0) if center_lock else (0, 255, 255)
                cv2.line(narrow_output, (390, 0), (390, 360), cross_color, 1)
                cv2.line(narrow_output, (0, 180), (780, 180), cross_color, 1)
            else:
                center_lock = False
                real_pan_err = None
                real_tilt_err = None
                radial_error = None
                narrow_output, narrow_crop_rect = crop_to_16_9(frame, None, 1.8, (780, 360), return_meta=True)
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
            cv2.putText(wide_debug, f'HANDOFF SCORE: {handoff_ready_score:.2f}  STREAK: {handoff_ready_streak}', (20, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(wide_debug, f'WIDE JITTER: center {wide_center_jitter:.1f} px  bbox {wide_bbox_jitter:.3f}', (20, 388), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            cv2.putText(wide_debug, f'PIPE STATE: {pipeline_state}  LOSS: {lock_loss_reason or "-"}', (20, 424), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)

            wide_program = add_title(wide_program, 'WIDE PROGRAM')
            narrow_output = add_title(narrow_output, 'NARROW OUTPUT')
            wide_debug = add_title(wide_debug, 'WIDE DEBUG')
            dashboard = cv2.vconcat([cv2.hconcat([wide_program, narrow_output]), wide_debug])

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
                telemetry.log_frame(
                    frame_idx=frame_id,
                    mode=('MANUAL' if target_manager.manual_lock else 'AUTO'),
                    selected_id=target_manager.selected_id,
                    active_track=soft_track,
                    tracks=tracks,
                    narrow_center=smooth_center,
                    center_lock=center_lock,
                    drift_gate_open=(soft_track is None and smooth_center is not None),
                    steering_target_id=pipeline_out['steering_target_id'],
                    lock_state=pipeline_state,
                    pan_error_px=real_pan_err,
                    tilt_error_px=real_tilt_err,
                    radial_error_px=radial_error,
                    zoom=smooth_zoom,
                    jump_limited=(jump_limited or jump_rejected),
                    handoff_ready_score=handoff_ready_score,
                    handoff_ready_streak=handoff_ready_streak,
                    wide_center_jitter=wide_center_jitter,
                    wide_bbox_jitter=wide_bbox_jitter,
                    local_lock_score=local_lock_score,
                    lock_confidence=lock_confidence,
                    anchor_jump_px=anchor_jump_px,
                    jump_rejected=jump_rejected,
                    lock_loss_reason=lock_loss_reason,
                    pipeline_state=pipeline_state,
                    owner_track_id=pipeline_out.get('owner_track_id'),
                    owner_raw_id=pipeline_out.get('owner_raw_id'),
                    owner_strength=pipeline_out.get('owner_strength'),
                    ownership_confidence=pipeline_out.get('ownership_confidence'),
                    identity_consistency_score=pipeline_out.get('identity_consistency_score'),
                    ownership_score=pipeline_out.get('ownership_score'),
                    pending_track_id=pipeline_out.get('pending_track_id'),
                    pending_frames=pipeline_out.get('pending_frames'),
                    transfer_reason=pipeline_out.get('transfer_reason'),
                    ownership_reject_reason=pipeline_out.get('ownership_reject_reason'),
                    jump_risk=pipeline_out.get('jump_risk'),
                    active_track_id=(soft_track.track_id if soft_track is not None else None),
                    active_track_missed=(soft_track.missed_frames if soft_track is not None else None),
                    active_track_conf=(soft_track.confidence if soft_track is not None else None),
                    center_delta_px=(float(((smooth_center[0] - predicted_center[0]) ** 2 + (smooth_center[1] - predicted_center[1]) ** 2) ** 0.5) if (smooth_center is not None and predicted_center is not None) else None),
                    zoom_delta=(
                        float(smooth_zoom - last_zoom_val)
                        if (
                            smooth_zoom is not None
                            and (last_zoom_val := getattr(narrow_tracker, 'last_zoom', None)) is not None
                        )
                        else None
                    ),
                )
                if smooth_zoom is not None:
                    narrow_tracker.last_zoom = float(smooth_zoom)

            cv2.imshow(window_name, dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key in (ord('r'), ord('R')):
                try:
                    stop_recording() if recording else start_recording()
                except Exception as exc:
                    recording = False
                    if video_writer is not None:
                        try:
                            video_writer.release()
                        except Exception:
                            pass
                        video_writer = None
                    print(f'REC TOGGLE ERROR: {exc}')
            elif key in (ord('t'), ord('T')):
                try:
                    stop_telemetry() if telemetry_enabled else start_telemetry()
                except Exception as exc:
                    telemetry_enabled = False
                    if telemetry is not None:
                        try:
                            telemetry.close()
                        except Exception:
                            pass
                        telemetry = None
                    print(f'METRICS TOGGLE ERROR: {exc}')
            elif key in (ord('s'), ord('S')):
                save_screenshot(dashboard, wide_debug, narrow_output)
            elif key == ord('0'):
                target_manager.set_auto_mode()
                narrow_tracker.reset()
                lock_pipeline.reset()
            elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')):
                idx = int(chr(key)) - 1
                cand = visible_sorted
                if 0 <= idx < len(cand):
                    tr = cand[idx]
                    target_manager.set_manual_target(tr.track_id)
                    narrow_tracker.reset()
                    lock_pipeline.reset()
                    display_box_smoother.reset()
                    narrow_tracker.kalman.init_state(tr.center_xy[0], tr.center_xy[1])
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
                    lock_pipeline.reset()
                    narrow_tracker.kalman.init_state(tr.center_xy[0], tr.center_xy[1])
    finally:
        if telemetry is not None:
            telemetry.close()
        if video_writer is not None:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
