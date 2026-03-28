import cv2
import numpy as np

from ultralytics import YOLO


class Track:
    def __init__(self, track_id, bbox_xyxy, center_xy, confidence):
        self.track_id = int(track_id)
        self.bbox_xyxy = bbox_xyxy
        self.center_xy = center_xy
        self.confidence = float(confidence)


class SimpleKalman2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            np.float32,
        )
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.12
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def reset(self):
        self.initialized = False

    def init_state(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return None
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def correct(self, x, y):
        if not self.initialized:
            self.init_state(x, y)
            return (x, y)
        m = np.array([[x], [y]], np.float32)
        est = self.kf.correct(m)
        return float(est[0, 0]), float(est[1, 0])


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


def crop_to_16_9(frame, center=None, scale=2.5, out_size=(780, 360)):
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


def crop_group(frame, tracks, out_size=(780, 360)):
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
        x1, y1, x2, y2 = [int(v) for v in tr.bbox_xyxy]
        cx, cy = [int(v) for v in tr.center_xy]
        color = (0, 255, 255) if tr.track_id == selected_id else (0, 255, 0)
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


def parse_tracks(result, frame_shape):
    h, w = frame_shape[:2]
    tracks = []

    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return tracks

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else [0.0] * len(xyxy)
    ids = boxes.id.cpu().numpy().astype(int).tolist() if boxes.id is not None else list(range(1, len(xyxy) + 1))

    for box, conf, tid in zip(xyxy, confs, ids):
        x1, y1, x2, y2 = [float(v) for v in box]

        cy = (y1 + y2) / 2.0
        if cy > h * 0.78:
            continue

        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh
        if area > (w * h * 0.03):
            continue

        tracks.append(
            Track(
                track_id=int(tid),
                bbox_xyxy=(x1, y1, x2, y2),
                center_xy=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                confidence=float(conf),
            )
        )

    tracks.sort(key=lambda t: t.track_id)
    return tracks


def choose_auto_target(tracks, selected_id, predicted_center, lock_age):
    if not tracks:
        return selected_id

    for tr in tracks:
        if tr.track_id == selected_id:
            return tr.track_id

    if selected_id is not None and lock_age < 60:
        return selected_id

    if predicted_center is not None:
        candidates = []
        for t in tracks:
            dist = np.hypot(t.center_xy[0] - predicted_center[0], t.center_xy[1] - predicted_center[1])
            candidates.append((dist, -t.confidence, t))
        candidates.sort(key=lambda x: (x[0], x[1]))
        best_dist, _, best_track = candidates[0]
        if best_dist <= 85.0:
            return best_track.track_id
        return selected_id

    def score(tr):
        x1, y1, x2, y2 = tr.bbox_xyxy
        area = max(1.0, (x2 - x1) * (y2 - y1))
        return (y1, -tr.confidence, -area)

    return min(tracks, key=score).track_id


def desired_zoom(frame, track):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = track.bbox_xyxy
    tw = max(1.0, x2 - x1)
    th = max(1.0, y2 - y1)
    rel = max(tw / w, th / h)
    z = 0.055 / max(rel, 0.012)
    return float(np.clip(z, 2.0, 3.4))


def run_app(config):
    mode = config.get("mode", "video")
    if mode != "video":
        print("Ta wersja YOLO jest przygotowana do testow wideo.")
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

    window_name = "Drone Tracker Multiview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)

    selected_id = None
    manual_lock = False

    kalman = SimpleKalman2D()
    predicted_center = None
    smooth_center = None
    smooth_zoom = 2.35

    hold_count = 0
    lock_age = 9999

    frame_id = 0
    tracks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        frame_id += 1

        if frame_id % max(1, inference_every) == 0 or not tracks:
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker_name,
                conf=conf,
                imgsz=imgsz,
                classes=classes,
                verbose=False,
            )
            result = results[0]
            tracks = parse_tracks(result, frame.shape)

        predicted_center = kalman.predict()

        if not manual_lock:
            selected_id = choose_auto_target(tracks, selected_id, predicted_center, lock_age)

        active_track = None
        for tr in tracks:
            if tr.track_id == selected_id:
                active_track = tr
                break

        if active_track is None and predicted_center is not None and tracks:
            candidates = []
            for t in tracks:
                dist = np.hypot(t.center_xy[0] - predicted_center[0], t.center_xy[1] - predicted_center[1])
                candidates.append((dist, -t.confidence, t))
            candidates.sort(key=lambda x: (x[0], x[1]))
            best_dist, _, best_track = candidates[0]
            if best_dist <= 85.0:
                active_track = best_track
                selected_id = best_track.track_id

        if active_track is not None:
            corrected = kalman.correct(active_track.center_xy[0], active_track.center_xy[1])
            predicted_center = corrected
            hold_count = 0
            lock_age = 0

            if smooth_center is None:
                smooth_center = corrected
            else:
                a = 0.975
                smooth_center = (
                    a * smooth_center[0] + (1.0 - a) * corrected[0],
                    a * smooth_center[1] + (1.0 - a) * corrected[1],
                )

            dz = desired_zoom(frame, active_track)
            smooth_zoom = 0.985 * smooth_zoom + 0.015 * dz
        else:
            hold_count += 1
            lock_age += 1

            if predicted_center is not None:
                if smooth_center is None:
                    smooth_center = predicted_center
                else:
                    a_hold = 0.985
                    smooth_center = (
                        a_hold * smooth_center[0] + (1.0 - a_hold) * predicted_center[0],
                        a_hold * smooth_center[1] + (1.0 - a_hold) * predicted_center[1],
                    )

            if hold_count > 120:
                selected_id = None
                predicted_center = None
                smooth_center = None
                smooth_zoom = 2.35
                hold_count = 0
                lock_age = 9999
                kalman.reset()

        wide_program = crop_group(frame, tracks, (780, 360))
        wide_debug = crop_group(draw_tracks(frame, tracks, selected_id), tracks, (1560, 450))

        if smooth_center is not None:
            narrow_output = crop_to_16_9(frame, smooth_center, smooth_zoom, (780, 360))
            pan_err = smooth_center[0] - frame.shape[1] / 2.0
            tilt_err = smooth_center[1] - frame.shape[0] / 2.0

            if active_track is not None:
                label = f"TARGET ID {selected_id}"
            else:
                label = f"TRACK HOLD ID {selected_id}"

            cv2.putText(narrow_output, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(narrow_output, f"PAN ERR {pan_err:.1f}  TILT ERR {tilt_err:.1f}", (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(narrow_output, f"ZOOM {smooth_zoom:.1f}x", (20, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(narrow_output, f"HOLD {hold_count}", (20, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.line(narrow_output, (390, 0), (390, 360), (0, 255, 255), 1)
            cv2.line(narrow_output, (0, 180), (780, 180), (0, 255, 255), 1)
        else:
            narrow_output = crop_to_16_9(frame, None, 1.7, (780, 360))
            cv2.putText(narrow_output, "BRAK CELU", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        lock_mode = "AUTO" if not manual_lock else "MANUAL"
        cv2.putText(wide_debug, f"LOCK MODE: {lock_mode}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"SELECTED ID: {selected_id}", (20, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"HOLD COUNT: {hold_count}", (20, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"LOCK AGE: {lock_age}", (20, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        wide_program = add_title(wide_program, "WIDE PROGRAM")
        narrow_output = add_title(narrow_output, "NARROW OUTPUT")
        wide_debug = add_title(wide_debug, "WIDE DEBUG")

        dashboard = cv2.vconcat([cv2.hconcat([wide_program, narrow_output]), wide_debug])
        cv2.imshow(window_name, dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("0"):
            manual_lock = False
            selected_id = None
            predicted_center = None
            smooth_center = None
            smooth_zoom = 2.35
            hold_count = 0
            lock_age = 9999
            kalman.reset()
        elif key in (ord("1"), ord("2"), ord("3")):
            visible = sorted(tracks, key=lambda t: t.center_xy[0])
            idx = int(chr(key)) - 1
            if idx < len(visible):
                manual_lock = True
                selected_id = visible[idx].track_id
                start_xy = visible[idx].center_xy
                kalman.reset()
                kalman.init_state(start_xy[0], start_xy[1])
                predicted_center = start_xy
                smooth_center = start_xy
                hold_count = 0
                lock_age = 0

    cap.release()
    cv2.destroyAllWindows()
