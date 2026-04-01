
import cv2
from ultralytics import YOLO

from core.target_manager import TargetManager
from core.narrow_tracker import NarrowTracker
from core.stable_registry import StableTargetRegistry
from core.target_filter import TargetFilter
from core.head_motion_test import HeadMotionTestMode
from core.optical_flow_fallback import NarrowOpticalFlowFallback


class Track:
    def __init__(self, track_id, bbox_xyxy, center_xy, confidence, raw_id=None):
        self.track_id = int(track_id)
        self.raw_id = int(raw_id if raw_id is not None else track_id)
        self.bbox_xyxy = bbox_xyxy
        self.center_xy = center_xy
        self.confidence = float(confidence)


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


def add_title(panel, title):
    cv2.rectangle(panel, (0, 0), (440, 56), (0, 0, 0), -1)
    cv2.putText(panel, title, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return panel


def draw_tracks(frame, tracks, selected_id):
    out = frame.copy()
    for tr in tracks:
        x1, y1, x2, y2 = [int(v) for v in tr.bbox_xyxy]
        cx, cy = [int(v) for v in tr.center_xy]
        if getattr(tr, "is_active_target", False):
            color = (0, 255, 0)
        elif getattr(tr, "is_valid_target", False):
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(out, (cx, cy), 4, color, -1)
        cv2.putText(
            out,
            f"ID {tr.track_id} {tr.confidence:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
    return out


def draw_target_on_narrow(narrow_frame, crop_rect, track, display_no="?"):
    if track is None:
        return narrow_frame

    x1, y1, x2, y2 = track.bbox_xyxy
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
            f"TRACKED TARGET [{display_no}]",
            (max(10, nx1), max(28, ny1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    return narrow_frame


def draw_predicted_box_on_narrow(narrow_frame, crop_rect, center_xy, size_xy, display_no="?"):
    if center_xy is None or size_xy is None:
        return narrow_frame

    cx1, cy1, cx2, cy2 = crop_rect
    crop_w = max(1, cx2 - cx1)
    crop_h = max(1, cy2 - cy1)

    nh, nw = narrow_frame.shape[:2]

    cx, cy = center_xy
    bw, bh = size_xy
    bw = max(14.0, float(bw))
    bh = max(14.0, float(bh))

    x1 = cx - bw / 2.0
    y1 = cy - bh / 2.0
    x2 = cx + bw / 2.0
    y2 = cy + bh / 2.0

    nx1 = int((x1 - cx1) * nw / crop_w)
    ny1 = int((y1 - cy1) * nh / crop_h)
    nx2 = int((x2 - cx1) * nw / crop_w)
    ny2 = int((y2 - cy1) * nh / crop_h)

    nx1 = max(0, min(nw - 1, nx1))
    ny1 = max(0, min(nh - 1, ny1))
    nx2 = max(0, min(nw - 1, nx2))
    ny2 = max(0, min(nh - 1, ny2))

    if nx2 > nx1 and ny2 > ny1:
        color = (255, 255, 0)
        cv2.rectangle(narrow_frame, (nx1, ny1), (nx2, ny2), color, 2)
        cv2.circle(narrow_frame, ((nx1 + nx2) // 2, (ny1 + ny2) // 2), 4, color, -1)
        cv2.putText(
            narrow_frame,
            f"PREDICTED [{display_no}]",
            (max(10, nx1), max(28, ny1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
    return narrow_frame


def parse_tracks(result, frame_shape):
    h, w = frame_shape[:2]
    tracks = []

    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return tracks

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else [0.0] * len(xyxy)
    raw_ids = boxes.id.cpu().numpy().astype(int).tolist() if boxes.id is not None else list(range(1, len(xyxy) + 1))

    for box, conf, raw_id in zip(xyxy, confs, raw_ids):
        x1, y1, x2, y2 = [float(v) for v in box]
        cy = (y1 + y2) / 2.0

        if cy > h * 0.92:
            continue

        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh
        if area > (w * h * 0.08):
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


def _choose_reacquire_track(tracks, ref_center, frame_shape, selected_id=None, ref_size=None):
    if ref_center is None or not tracks:
        return None

    fw = frame_shape[1]
    fh = frame_shape[0]
    best_track = None
    best_score = 1e18
    ref_area = None
    if ref_size is not None:
        ref_area = max(1.0, float(ref_size[0]) * float(ref_size[1]))

    for tr in tracks:
        if not getattr(tr, "is_valid_target", True):
            continue

        x1, y1, x2, y2 = tr.bbox_xyxy
        cx, cy = tr.center_xy
        area = max(1.0, (x2 - x1) * (y2 - y1))
        conf = float(getattr(tr, "confidence", 0.0))

        dx = cx - ref_center[0]
        dy = cy - ref_center[1]
        dist2 = dx * dx + dy * dy

        if area < 8.0:
            continue

        size_penalty = 0.0
        if ref_area is not None:
            size_penalty = abs(area - ref_area) / max(ref_area, 1.0) * 2500.0

        persistence_bonus = 4000.0 if selected_id is not None and tr.track_id == selected_id else 0.0
        score = dist2 + size_penalty - conf * 5000.0 - persistence_bonus

        if score < best_score:
            best_score = score
            best_track = tr

    max_dist2 = (0.22 * max(fw, fh)) ** 2
    if best_track is not None and best_score < max_dist2 + 2500.0:
        return best_track
    return None


def _make_guided_roi(frame_shape, ref_center, ref_size=None, expand=3.4, velocity=(0.0, 0.0), hold_count=0):
    if ref_center is None:
        return None

    h, w = frame_shape[:2]
    cx, cy = ref_center
    vx, vy = velocity
    vel_mag = (vx * vx + vy * vy) ** 0.5
    dynamic_expand = expand + min(2.0, vel_mag * 0.08) + min(1.2, max(0, hold_count) * 0.03)

    if ref_size is None:
        bw = max(180.0, w * 0.20)
        bh = max(100.0, h * 0.20)
    else:
        bw = max(90.0, float(ref_size[0]) * dynamic_expand)
        bh = max(90.0, float(ref_size[1]) * dynamic_expand)

    lookahead = min(40.0, vel_mag * 1.5)
    if vel_mag > 1e-6:
        cx += (vx / vel_mag) * lookahead
        cy += (vy / vel_mag) * lookahead

    aspect = 16.0 / 9.0
    if bw / bh < aspect:
        bw = bh * aspect
    else:
        bh = bw / aspect

    x1 = int(cx - bw / 2.0)
    y1 = int(cy - bh / 2.0)
    x2 = int(cx + bw / 2.0)
    y2 = int(cy + bh / 2.0)
    return clamp_box(x1, y1, x2, y2, w, h)


def _run_detector(model, frame, tracker_name, conf, imgsz, classes, roi_rect=None):
    if roi_rect is None:
        results = model.track(
            source=frame,
            persist=True,
            tracker=tracker_name,
            conf=conf,
            imgsz=imgsz,
            classes=classes,
            verbose=False,
        )
        return parse_tracks(results[0], frame.shape), False, None

    x1, y1, x2, y2 = roi_rect
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return [], True, roi_rect

    results = model.track(
        source=roi,
        persist=True,
        tracker=tracker_name,
        conf=max(0.05, conf * 0.75),
        imgsz=imgsz,
        classes=classes,
        verbose=False,
    )

    local_tracks = parse_tracks(results[0], roi.shape)
    tracks = []
    for tr in local_tracks:
        bx1, by1, bx2, by2 = tr.bbox_xyxy
        gx1 = bx1 + x1
        gy1 = by1 + y1
        gx2 = bx2 + x1
        gy2 = by2 + y1
        gcx = (gx1 + gx2) / 2.0
        gcy = (gy1 + gy2) / 2.0
        tracks.append(
            Track(
                track_id=tr.track_id,
                raw_id=tr.raw_id,
                bbox_xyxy=(gx1, gy1, gx2, gy2),
                center_xy=(gcx, gcy),
                confidence=tr.confidence,
            )
        )
    return tracks, True, roi_rect



def _matches_identity(track, ref_center, ref_size, frame_shape):
    if track is None or ref_center is None:
        return False

    fw = frame_shape[1]
    fh = frame_shape[0]
    cx, cy = track.center_xy
    dx = cx - ref_center[0]
    dy = cy - ref_center[1]
    dist = (dx * dx + dy * dy) ** 0.5

    max_dist = 0.12 * max(fw, fh)
    if dist > max_dist:
        return False

    if ref_size is not None:
        x1, y1, x2, y2 = track.bbox_xyxy
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        ref_w = max(1.0, float(ref_size[0]))
        ref_h = max(1.0, float(ref_size[1]))

        wr = bw / ref_w
        hr = bh / ref_h

        if not (0.45 <= wr <= 2.2 and 0.45 <= hr <= 2.2):
            return False

    return True


def run_app(config):
    last_target_center = None
    last_target_size = None
    last_flow_center = None
    prev_center = None
    velocity = (0.0, 0.0)
    display_target_id = None
    display_id_hold = 0
    DISPLAY_ID_HOLD_MAX = 180

    wide_hold_frames = 0
    WIDE_HOLD_MAX = 90
    narrow_fallback_hold = 0
    NARROW_FALLBACK_MAX = 45
    FLOW_HOLD_MAX = 100
    flow_hold_frames = 0
    track_state = "DETECTED_LOCK"
    roi_search_active = False

    mode = config.get("mode", "video")
    if mode != "video":
        print("Ta wersja jest przygotowana do testow wideo.")
        return

    video_cfg = config.get("video") or {}
    yolo_cfg = config.get("yolo") or {}

    source = video_cfg.get("source", "video.mp4")
    model_name = yolo_cfg.get("model", "yolov8n.pt")
    tracker_name = yolo_cfg.get("tracker", "bytetrack.yaml")
    conf = float(yolo_cfg.get("conf", 0.15))
    imgsz = int(yolo_cfg.get("imgsz", 640))
    classes = yolo_cfg.get("classes", [4])
    inference_every = int(yolo_cfg.get("inference_every", 1))

    model = YOLO(model_name)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Nie moge otworzyc pliku: {source}")
        return

    stable_registry = StableTargetRegistry(max_missing=25, match_distance=140.0, min_iou=0.01)
    target_manager = TargetManager(reacquire_radius_auto=100.0, reacquire_radius_manual=140.0, sticky_frames=75)
    narrow_tracker = NarrowTracker(hold_frames=140)
    target_filter = TargetFilter()
    head_motion_test = HeadMotionTestMode()
    optical_flow_fallback = NarrowOpticalFlowFallback()

    FLOW_MIN_POINTS = 8
    MAX_V = 20.0

    window_name = "Drone Tracker Multiview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)

    frame_id = 0
    tracks = []
    roi_debug_rect = None
    roi_debug_on = False

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            stable_registry.reset()
            target_manager.set_auto_mode()
            narrow_tracker.reset()
            target_filter.reset()
            optical_flow_fallback.reset()
            last_target_center = None
            last_target_size = None
            last_flow_center = None
            prev_center = None
            velocity = (0.0, 0.0)
            wide_hold_frames = 0
            narrow_fallback_hold = 0
            flow_hold_frames = 0
            track_state = "DETECTED_LOCK"
            roi_search_active = False
            ret, frame = cap.read()
            if not ret:
                break

        frame_id += 1

        if frame_id % max(1, inference_every) == 0 or not tracks:
            guided_ref_center = last_target_center
            guided_ref_size = last_target_size

            roi_rect = None
            if target_manager.selected_id is not None and guided_ref_center is not None:
                roi_rect = _make_guided_roi(
                    frame.shape,
                    guided_ref_center,
                    guided_ref_size,
                    expand=4.0,
                    velocity=velocity,
                    hold_count=hold_count,
                )

            roi_tracks = []
            used_roi = False
            roi_debug_rect = None
            roi_debug_on = False
            roi_search_active = False

            if roi_rect is not None:
                roi_tracks, used_roi, roi_debug_rect = _run_detector(
                    model, frame, tracker_name, conf, imgsz, classes, roi_rect=roi_rect
                )
                roi_debug_on = used_roi
                roi_search_active = used_roi

            if roi_tracks:
                det_tracks = roi_tracks
            else:
                det_tracks, _, _ = _run_detector(
                    model, frame, tracker_name, conf, imgsz, classes, roi_rect=None
                )
                roi_debug_on = False
                roi_debug_rect = None

            tracks = stable_registry.update(det_tracks)
            tracks = target_filter.update(tracks, frame.shape)

        visible_sorted = sorted(tracks, key=lambda t: t.track_id)

        fh, fw = frame.shape[:2]
        frame_cx = fw / 2.0
        frame_cy = fh / 2.0

        for tr in tracks:
            tx, ty = tr.center_xy
            dist_to_center = ((tx - frame_cx) ** 2 + (ty - frame_cy) ** 2) ** 0.5
            center_bonus = max(0.0, 1.0 - dist_to_center / max(1.0, fw * 0.5))

            valid_score = float(getattr(tr, "target_score", 0.0))
            conf_score = float(getattr(tr, "confidence", 0.0))
            persistence_bonus = 0.35 if target_manager.selected_id is not None and tr.track_id == target_manager.selected_id else 0.0
            size_bonus = 0.0
            if last_target_size is not None:
                ref_area = max(1.0, float(last_target_size[0]) * float(last_target_size[1]))
                x1, y1, x2, y2 = tr.bbox_xyxy
                area = max(1.0, (x2 - x1) * (y2 - y1))
                size_bonus = max(0.0, 1.0 - abs(area - ref_area) / max(ref_area, 1.0)) * 0.25

            tr.selection_priority = (
                valid_score * 0.45
                + conf_score * 0.20
                + center_bonus * 0.20
                + persistence_bonus
                + size_bonus
            )

        predicted_center = narrow_tracker.kalman.predict()

        candidate_tracks = [t for t in tracks if getattr(t, "is_valid_target", False)]
        if candidate_tracks:
            candidate_tracks = sorted(candidate_tracks, key=lambda t: getattr(t, "selection_priority", 0.0), reverse=True)
            ordered_tracks = candidate_tracks + [t for t in tracks if not getattr(t, "is_valid_target", False)]
        else:
            ordered_tracks = tracks

        target_manager.update(ordered_tracks, predicted_center, frame.shape)
        active_track = target_manager.find_active_track(tracks)

        if active_track is None and last_target_center is not None and wide_hold_frames > 0:
            reacquired = _choose_reacquire_track(
                tracks,
                last_target_center,
                frame.shape,
                target_manager.selected_id,
                last_target_size,
            )
            if reacquired is not None:
                target_manager.selected_id = reacquired.track_id
                active_track = reacquired
            else:
                wide_hold_frames -= 1

        if (
            active_track is None
            and target_manager.selected_id is not None
            and flow_hold_frames <= 0
            and target_manager.lock_age > 120
        ):
            target_manager.set_auto_mode()
            narrow_tracker.reset()
            optical_flow_fallback.reset()
            track_state = "LOST"

        for tr in tracks:
            tr.is_active_target = False
            tr.is_valid_target = getattr(tr, "is_valid_target", True)

        if active_track is not None:
            active_track.is_active_target = True
            last_target_center = active_track.center_xy
            x1, y1, x2, y2 = active_track.bbox_xyxy
            last_target_size = (x2 - x1, y2 - y1)
            wide_hold_frames = WIDE_HOLD_MAX
            flow_hold_frames = FLOW_HOLD_MAX
            track_state = "DETECTED_LOCK"

            if display_target_id is None:
                display_target_id = active_track.track_id
                display_id_hold = DISPLAY_ID_HOLD_MAX
            else:
                same_identity = _matches_identity(
                    active_track,
                    last_target_center,
                    last_target_size,
                    frame.shape,
                )
                if same_identity and display_id_hold > 0:
                    display_id_hold = DISPLAY_ID_HOLD_MAX
                else:
                    display_target_id = active_track.track_id
                    display_id_hold = DISPLAY_ID_HOLD_MAX
        else:
            if display_id_hold > 0:
                display_id_hold -= 1
            else:
                display_target_id = None

        predicted_center, smooth_center, smooth_zoom, hold_count, _, _ = narrow_tracker.update(frame, active_track)

        edge_limit_active = False

        if active_track is not None:
            tx, ty = active_track.center_xy

            if smooth_center is None:
                smooth_center = (tx, ty)

            pan_err = tx - smooth_center[0]
            tilt_err = ty - smooth_center[1]

            if target_manager.manual_lock:
                cx = tx
                cy = ty
                smooth_center = (cx, cy)
                pan_speed = pan_err
                tilt_speed = tilt_err

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

                required_zoom = max(1.0, min(8.0, required_zoom))

                if required_zoom > smooth_zoom:
                    smooth_zoom = required_zoom
                    edge_limit_active = True

            else:
                alpha = 0.28
                snap_px = 18

                cx = smooth_center[0] + alpha * pan_err
                cy = smooth_center[1] + alpha * tilt_err

                if abs(pan_err) < snap_px and abs(tilt_err) < snap_px:
                    cx = tx
                    cy = ty

                smooth_center = (cx, cy)
                pan_speed = pan_err * alpha
                tilt_speed = tilt_err * alpha
        else:
            pan_speed = 0.0
            tilt_speed = 0.0

        dx_test, dy_test = head_motion_test.update()
        dx_test = max(-3.0, min(3.0, dx_test))
        dy_test = max(-2.0, min(2.0, dy_test))
        pan_speed += dx_test
        tilt_speed += dy_test

        wide_program = cv2.resize(frame, (780, 360), interpolation=cv2.INTER_LINEAR)

        debug_frame = draw_tracks(frame, tracks, target_manager.selected_id)
        if roi_debug_on and roi_debug_rect is not None:
            rx1, ry1, rx2, ry2 = roi_debug_rect
            cv2.rectangle(debug_frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
            cv2.putText(debug_frame, "ROI SEARCH", (rx1, max(24, ry1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        for tr in visible_sorted:
            x1, y1, x2, y2 = [int(v) for v in tr.bbox_xyxy]
            cv2.putText(
                debug_frame,
                f"[{tr.track_id}]",
                (x1, max(30, y1 - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2,
            )

        wide_debug = cv2.resize(debug_frame, (1560, 450), interpolation=cv2.INTER_LINEAR)

        flow_debug_on = False
        flow_points_debug = 0
        flow_points = None
        flow_center = None

        fallback_center = None
        if smooth_center is not None:
            fallback_center = smooth_center
            narrow_fallback_hold = NARROW_FALLBACK_MAX
        elif active_track is not None:
            fallback_center = active_track.center_xy
            narrow_fallback_hold = NARROW_FALLBACK_MAX
        elif narrow_fallback_hold > 0 and last_target_center is not None:
            narrow_fallback_hold -= 1
            fallback_center = last_target_center

        predicted_box_center = None
        predicted_box_size = last_target_size

        if fallback_center is not None:
            if active_track is not None:
                try:
                    optical_flow_fallback.init_from_bbox(frame, active_track.bbox_xyxy)
                except Exception:
                    pass
                crop_center = fallback_center
                last_flow_center = fallback_center
                prev_center = fallback_center
                velocity = (0.0, 0.0)
            else:
                flow_ok, flow_center, flow_points_debug, flow_points = optical_flow_fallback.update(frame)
                if flow_ok and flow_center is not None:
                    crop_center = flow_center
                    smooth_center = flow_center
                    flow_debug_on = True
                    track_state = "FLOW_HOLD"
                    flow_hold_frames = FLOW_HOLD_MAX
                    last_flow_center = flow_center

                    center = flow_center
                    if prev_center is not None and flow_points_debug > FLOW_MIN_POINTS:
                        vx = center[0] - prev_center[0]
                        vy = center[1] - prev_center[1]
                        vx = max(min(vx, MAX_V), -MAX_V)
                        vy = max(min(vy, MAX_V), -MAX_V)
                        velocity = (vx * 0.7, vy * 0.7)
                    prev_center = center
                    predicted_box_center = center
                else:
                    if flow_points_debug == 0:
                        velocity = (velocity[0] * 0.9, velocity[1] * 0.9)

                    if prev_center is not None:
                        k = min(max(1, hold_count), 8)
                        px = prev_center[0] + velocity[0] * k
                        py = prev_center[1] + velocity[1] * k
                        crop_center = (float(px), float(py))
                        smooth_center = crop_center
                        predicted_box_center = crop_center
                    else:
                        crop_center = fallback_center
                        predicted_box_center = fallback_center

                    if flow_hold_frames > 0:
                        flow_hold_frames -= 1
                        track_state = "FLOW_HOLD"
                    else:
                        track_state = "LOST"

            narrow_output, narrow_crop_rect = crop_to_16_9(
                frame, crop_center, smooth_zoom, (780, 360), return_meta=True
            )

            shown_id = display_target_id if display_target_id is not None else target_manager.selected_id
            label = f"TARGET ID {shown_id}" if active_track is not None else f"TRACK HOLD ID {shown_id}"

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
                center_lock = (abs(real_pan_err) < 12 and abs(real_tilt_err) < 12)

            cv2.putText(narrow_output, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(narrow_output, f"PAN ERR {real_pan_err:.1f}  TILT ERR {real_tilt_err:.1f}", (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(narrow_output, f"ZOOM {smooth_zoom:.1f}x", (20, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(narrow_output, f"HOLD {hold_count}", (20, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(narrow_output, f"STATE {track_state}", (20, 206), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            center_lock_text = "CENTER LOCK ON" if (center_lock and active_track is not None) else "CENTER LOCK OFF"
            cv2.putText(narrow_output, center_lock_text, (20, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if edge_limit_active:
                cv2.putText(narrow_output, "EDGE LIMIT COMP", (20, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.putText(
                narrow_output,
                f"FLOW: {'ON' if flow_debug_on else 'OFF'}  PTS: {flow_points_debug}",
                (20, 294),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if active_track is not None:
                narrow_output = draw_target_on_narrow(
                    narrow_output,
                    narrow_crop_rect,
                    active_track,
                    shown_id if shown_id is not None else active_track.track_id,
                )
            else:
                if predicted_box_center is not None and predicted_box_size is not None and track_state == "FLOW_HOLD":
                    narrow_output = draw_predicted_box_on_narrow(
                        narrow_output,
                        narrow_crop_rect,
                        predicted_box_center,
                        predicted_box_size,
                        shown_id if shown_id is not None else "?",
                    )

                if flow_debug_on and flow_center is not None:
                    cx1, cy1, cx2, cy2 = narrow_crop_rect
                    fx, fy = flow_center
                    rx = int(fx - cx1)
                    ry = int(fy - cy1)
                    if 0 <= rx < 780 and 0 <= ry < 360:
                        cv2.circle(narrow_output, (rx, ry), 6, (255, 255, 0), -1)
                    if flow_points is not None:
                        for pt in flow_points:
                            px = int(pt[0] - cx1)
                            py = int(pt[1] - cy1)
                            if 0 <= px < 780 and 0 <= py < 360:
                                cv2.circle(narrow_output, (px, py), 2, (255, 255, 0), -1)

            cross_color = (0, 255, 0) if center_lock else (0, 255, 255)
            cv2.line(narrow_output, (390, 0), (390, 360), cross_color, 1)
            cv2.line(narrow_output, (0, 180), (780, 180), cross_color, 1)
        else:
            narrow_output, narrow_crop_rect = crop_to_16_9(frame, None, 1.7, (780, 360), return_meta=True)
            cv2.putText(narrow_output, "BRAK CELU", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(
                narrow_output,
                "FLOW: OFF  PTS: 0",
                (20, 294),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            track_state = "LOST"

        lock_mode = "AUTO" if not target_manager.manual_lock else "MANUAL"
        cv2.putText(wide_debug, f"LOCK MODE: {lock_mode}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        shown_id = display_target_id if display_target_id is not None else target_manager.selected_id
        cv2.putText(wide_debug, f"SELECTED ID: {shown_id}", (20, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"HOLD COUNT: {hold_count}", (20, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"LOCK AGE: {target_manager.lock_age}", (20, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"PAN SPD: {pan_speed:.1f}  TILT SPD: {tilt_speed:.1f}", (20, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        auto_text = "AUTO PICK ENABLED" if not target_manager.manual_lock else "AUTO PICK DISABLED"
        cv2.putText(wide_debug, auto_text, (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(
            wide_debug,
            f"HEAD TEST: {'ON' if head_motion_test.enabled else 'OFF'}",
            (20, 316),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            wide_debug,
            f"TRACK STATE: {track_state}",
            (20, 352),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        wide_program = add_title(wide_program, "WIDE PROGRAM")
        narrow_output = add_title(narrow_output, "NARROW OUTPUT")
        wide_debug = add_title(wide_debug, "WIDE DEBUG")

        dashboard = cv2.vconcat([cv2.hconcat([wide_program, narrow_output]), wide_debug])
        cv2.imshow(window_name, dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("m") or key == ord("M"):
            head_motion_test.toggle()
            print(f"HEAD TEST MODE: {'ON' if head_motion_test.enabled else 'OFF'}")
        elif key == ord("0"):
            target_manager.set_auto_mode()
            narrow_tracker.reset()
            optical_flow_fallback.reset()
            last_flow_center = None
            prev_center = None
            velocity = (0.0, 0.0)
            flow_hold_frames = 0
            track_state = "LOST"
        elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7"), ord("8"), ord("9")):
            wanted = int(chr(key))
            tr = next((t for t in tracks if t.track_id == wanted), None)
            if tr is not None:
                target_manager.set_manual_target(tr.track_id)
                narrow_tracker.reset()
                optical_flow_fallback.reset()
                narrow_tracker.kalman.init_state(tr.center_xy[0], tr.center_xy[1])
                narrow_tracker.smooth_center = tr.center_xy
                last_target_center = tr.center_xy
                x1, y1, x2, y2 = tr.bbox_xyxy
                last_target_size = (x2 - x1, y2 - y1)
                prev_center = tr.center_xy
                velocity = (0.0, 0.0)
                flow_hold_frames = FLOW_HOLD_MAX
                track_state = "DETECTED_LOCK"

    cap.release()
    cv2.destroyAllWindows()
