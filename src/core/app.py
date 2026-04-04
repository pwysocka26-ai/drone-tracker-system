import cv2
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

from core.target_manager import TargetManager
from core.narrow_tracker import NarrowTracker
from core.multi_target_tracker import MultiTargetTracker


class Track:
    def __init__(self, track_id, bbox_xyxy, center_xy, confidence, raw_id=None):
        self.track_id = int(track_id)
        self.raw_id = int(raw_id if raw_id is not None else track_id)
        self.bbox_xyxy = bbox_xyxy
        self.center_xy = center_xy
        self.confidence = float(confidence)


def tighten_bbox(bbox, scale=0.65, min_size=12):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(float(min_size), (x2 - x1) * float(scale))
    h = max(float(min_size), (y2 - y1) * float(scale))
    nx1 = int(cx - w / 2.0)
    ny1 = int(cy - h / 2.0)
    nx2 = int(cx + w / 2.0)
    ny2 = int(cy + h / 2.0)
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
        x1, y1, x2, y2 = tighten_bbox(tr.bbox_xyxy, scale=0.65)
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


def draw_target_on_narrow(narrow_frame, crop_rect, track, display_no='?'):
    if track is None:
        return narrow_frame

    x1, y1, x2, y2 = tighten_bbox(track.bbox_xyxy, scale=0.60)
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

    if nx2 > nx1 and ny2 > ny1:
        cv2.rectangle(narrow_frame, (nx1, ny1), (nx2, ny2), (0, 255, 255), 2)
        cv2.circle(narrow_frame, ((nx1 + nx2) // 2, (ny1 + ny2) // 2), 4, (0, 255, 255), -1)
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
        reacquire_radius_auto=float(tracker_cfg.get('reacquire_radius_auto', 160.0)),
        reacquire_radius_manual=float(tracker_cfg.get('reacquire_radius_manual', 260.0)),
        sticky_frames=int(tracker_cfg.get('sticky_frames', 22)),
        switch_margin=float(tracker_cfg.get('switch_margin', 0.18)),
        switch_dwell=int(tracker_cfg.get('switch_dwell', 4)),
        switch_cooldown=int(tracker_cfg.get('switch_cooldown', 6)),
        max_select_missed=int(tracker_cfg.get('max_select_missed', 1)),
        min_start_conf=float(tracker_cfg.get('min_start_conf', 0.10)),
    )
    narrow_tracker = NarrowTracker(hold_frames=int(narrow_cfg.get('hold_frames', 80)))

    window_name = 'Drone Tracker Multiview'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)

    recording = False
    video_writer = None
    record_fps = 30.0
    record_frame_size = (1560, 810)

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
            video_writer.release()
            video_writer = None
        print('REC STOP')

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

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            multi_tracker.reset()
            target_manager.set_auto_mode()
            narrow_tracker.reset()
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
        active_track = target_manager.find_active_track(tracks)
        if active_track is not None and int(getattr(active_track, 'missed_frames', 0) or 0) > 1:
            active_track = None

        for tr in tracks:
            tr.is_active_target = False
            tr.is_valid_target = bool(
                getattr(tr, 'is_confirmed', False)
                or getattr(tr, 'hits', 0) >= 2
            )
        if active_track is not None:
            active_track.is_active_target = True

        predicted_center, smooth_center, smooth_zoom, hold_count, _, _ = narrow_tracker.update(frame, active_track)

        edge_limit_active = False
        if active_track is not None:
            tx, ty = active_track.center_xy
            if smooth_center is None:
                smooth_center = (tx, ty)
            pan_err = tx - smooth_center[0]
            tilt_err = ty - smooth_center[1]

            if target_manager.manual_lock:
                cx, cy = tx, ty
                smooth_center = (cx, cy)
                pan_speed = pan_err
                tilt_speed = tilt_err
            else:
                alpha = 0.24
                snap_px = 18
                cx = smooth_center[0] + alpha * pan_err
                cy = smooth_center[1] + alpha * tilt_err
                if abs(pan_err) < snap_px and abs(tilt_err) < snap_px:
                    cx, cy = tx, ty
                smooth_center = (cx, cy)
                pan_speed = pan_err * alpha
                tilt_speed = tilt_err * alpha

            fh, fw = frame.shape[:2]
            aspect = 780.0 / 360.0
            margin_x = min(tx, fw - tx)
            margin_y = min(ty, fh - ty)
            max_crop_w = max(80.0, margin_x * 2.0)
            max_crop_h = max(80.0, margin_y * 2.0)
            if max_crop_w / max_crop_h < aspect:
                max_crop_h = max_crop_w / aspect
            else:
                max_crop_w = max_crop_h * aspect

            if (fw / fh) > aspect:
                required_zoom = fh / max_crop_h
            else:
                required_zoom = fw / max_crop_w
            required_zoom = max(1.0, min(2.4, required_zoom))
            if required_zoom > smooth_zoom:
                smooth_zoom = required_zoom
                edge_limit_active = True
        else:
            pan_speed = 0.0
            tilt_speed = 0.0

        if active_track is not None and smooth_center is not None:
            tx, ty = active_track.center_xy
            if target_manager.manual_lock or (abs(tx - smooth_center[0]) < 120 and abs(ty - smooth_center[1]) < 120):
                smooth_center = (tx, ty)

        wide_program = crop_group(frame, tracks, (780, 360))
        debug_frame = draw_tracks(frame, tracks, target_manager.selected_id)
        for idx, tr in enumerate(visible_sorted[:9], start=1):
            x1, y1, x2, y2 = tighten_bbox(tr.bbox_xyxy, scale=0.65)
            label = f'[{idx}] ID {tr.track_id}' if getattr(tr, 'is_confirmed', False) else f'[{idx}] ID {tr.track_id}?'
            cv2.putText(debug_frame, label, (x1, max(30, y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        wide_debug = crop_group(debug_frame, tracks, (1560, 450))

        if smooth_center is not None:
            narrow_output, narrow_crop_rect = crop_to_16_9(frame, smooth_center, smooth_zoom, (780, 360), return_meta=True)
            label = f'TARGET ID {target_manager.selected_id}' if active_track is not None else f'TRACK HOLD ID {target_manager.selected_id}'
            real_pan_err = 0.0
            real_tilt_err = 0.0
            center_lock = False

            if active_track is not None:
                cx1, cy1, cx2, cy2 = narrow_crop_rect
                crop_w = max(1, cx2 - cx1)
                crop_h = max(1, cy2 - cy1)
                target_nx = (active_track.center_xy[0] - cx1) * 780.0 / crop_w
                target_ny = (active_track.center_xy[1] - cy1) * 360.0 / crop_h
                real_pan_err = target_nx - 390.0
                real_tilt_err = target_ny - 180.0
                center_lock = abs(real_pan_err) < 12 and abs(real_tilt_err) < 12

            cv2.putText(narrow_output, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(narrow_output, f'PAN ERR {real_pan_err:.1f}  TILT ERR {real_tilt_err:.1f}', (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(narrow_output, f'ZOOM {smooth_zoom:.1f}x', (20, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(narrow_output, f'HOLD {hold_count}', (20, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            center_lock_text = 'CENTER LOCK ON' if (center_lock and active_track is not None) else 'CENTER LOCK OFF'
            cv2.putText(narrow_output, center_lock_text, (20, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if edge_limit_active:
                cv2.putText(narrow_output, 'EDGE LIMIT COMP', (20, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            if active_track is not None:
                narrow_output = draw_target_on_narrow(narrow_output, narrow_crop_rect, active_track, active_track.track_id)
            cross_color = (0, 255, 0) if center_lock else (0, 255, 255)
            cv2.line(narrow_output, (390, 0), (390, 360), cross_color, 1)
            cv2.line(narrow_output, (0, 180), (780, 180), cross_color, 1)
        else:
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

        wide_program = add_title(wide_program, 'WIDE PROGRAM')
        narrow_output = add_title(narrow_output, 'NARROW OUTPUT')
        wide_debug = add_title(wide_debug, 'WIDE DEBUG')
        dashboard = cv2.vconcat([cv2.hconcat([wide_program, narrow_output]), wide_debug])

        if recording:
            cv2.circle(dashboard, (1510, 30), 8, (0, 0, 255), -1)
            cv2.putText(dashboard, 'REC', (1450, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if video_writer is not None:
                video_writer.write(dashboard)

        cv2.putText(dashboard, 'R=REC  S=SHOT  0=AUTO  1-9=SLOT  ,/.=PREV/NEXT', (980, 800), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window_name, dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key in (ord('r'), ord('R')):
            stop_recording() if recording else start_recording()
        elif key in (ord('s'), ord('S')):
            save_screenshot(dashboard, wide_debug, narrow_output)
        elif key == ord('0'):
            target_manager.set_auto_mode()
            narrow_tracker.reset()
        elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')):
            idx = int(chr(key)) - 1
            cand = visible_sorted
            if 0 <= idx < len(cand):
                tr = cand[idx]
                target_manager.set_manual_target(tr.track_id)
                narrow_tracker.reset()
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
                narrow_tracker.kalman.init_state(tr.center_xy[0], tr.center_xy[1])

    cap.release()
    cv2.destroyAllWindows()
