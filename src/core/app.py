import cv2
import numpy as np

from ultralytics import YOLO
from core.target_manager import TargetManager
from core.narrow_tracker import NarrowTracker


class Track:
    def __init__(self, track_id, bbox_xyxy, center_xy, confidence):
        self.track_id = int(track_id)
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

        tracks.append(Track(
            track_id=int(tid),
            bbox_xyxy=(x1, y1, x2, y2),
            center_xy=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            confidence=float(conf),
        ))

    tracks.sort(key=lambda t: t.center_xy[0])
    return tracks


def run_app(config):
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

    target_manager = TargetManager(reacquire_radius_auto=85.0, reacquire_radius_manual=120.0, sticky_frames=60)
    narrow_tracker = NarrowTracker(hold_frames=120)

    window_name = "Drone Tracker Multiview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)

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

        visible_sorted = sorted(tracks, key=lambda t: t.center_xy[0])[:9]
        display_map = {tr.track_id: i for i, tr in enumerate(visible_sorted, start=1)}

        predicted_center = narrow_tracker.kalman.predict()

        if not target_manager.manual_lock:
            target_manager.update(tracks, predicted_center, frame.shape)
        else:
            # w MANUAL próbujemy odzyskać ten sam cel, ale nie przełączamy się samoczynnie na "losowy"
            target_manager.update(tracks, predicted_center, frame.shape)

        active_track = target_manager.find_active_track(tracks)

        predicted_center, smooth_center, smooth_zoom, hold_count, pan_speed, tilt_speed = narrow_tracker.update(frame, active_track)

        wide_program = crop_group(frame, tracks, (780, 360))

        debug_frame = draw_tracks(frame, tracks, target_manager.selected_id)
        for i, tr in enumerate(visible_sorted, start=1):
            x1, y1, x2, y2 = [int(v) for v in tr.bbox_xyxy]
            cv2.putText(
                debug_frame,
                f"[{i}]",
                (x1, max(30, y1 - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2,
            )

        wide_debug = crop_group(debug_frame, tracks, (1560, 450))

        if smooth_center is not None:
            narrow_output, narrow_crop_rect = crop_to_16_9(frame, smooth_center, smooth_zoom, (780, 360), return_meta=True)

            pan_err = smooth_center[0] - frame.shape[1] / 2.0
            tilt_err = smooth_center[1] - frame.shape[0] / 2.0

            label = f"TARGET ID {target_manager.selected_id}" if active_track is not None else f"TRACK HOLD ID {target_manager.selected_id}"
            cv2.putText(narrow_output, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(narrow_output, f"PAN ERR {pan_err:.1f}  TILT ERR {tilt_err:.1f}", (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(narrow_output, f"ZOOM {smooth_zoom:.1f}x", (20, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(narrow_output, f"HOLD {hold_count}", (20, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if active_track is not None:
                display_no = display_map.get(active_track.track_id, "?")
                narrow_output = draw_target_on_narrow(narrow_output, narrow_crop_rect, active_track, display_no)

            cv2.line(narrow_output, (390, 0), (390, 360), (0, 255, 255), 1)
            cv2.line(narrow_output, (0, 180), (780, 180), (0, 255, 255), 1)
        else:
            narrow_output, narrow_crop_rect = crop_to_16_9(frame, None, 1.7, (780, 360), return_meta=True)
            cv2.putText(narrow_output, "BRAK CELU", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        lock_mode = "AUTO" if not target_manager.manual_lock else "MANUAL"
        cv2.putText(wide_debug, f"LOCK MODE: {lock_mode}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"SELECTED ID: {target_manager.selected_id}", (20, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"HOLD COUNT: {hold_count}", (20, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"LOCK AGE: {target_manager.lock_age}", (20, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"PAN SPD: {pan_speed:.1f}  TILT SPD: {tilt_speed:.1f}", (20, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        wide_program = add_title(wide_program, "WIDE PROGRAM")
        narrow_output = add_title(narrow_output, "NARROW OUTPUT")
        wide_debug = add_title(wide_debug, "WIDE DEBUG")

        dashboard = cv2.vconcat([cv2.hconcat([wide_program, narrow_output]), wide_debug])
        cv2.imshow(window_name, dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("0"):
            target_manager.set_auto_mode()
            narrow_tracker.reset()
        elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7"), ord("8"), ord("9")):
            idx = int(chr(key)) - 1
            if idx < len(visible_sorted):
                tr = visible_sorted[idx]
                target_manager.set_manual_target(tr.track_id)
                narrow_tracker.reset()
                narrow_tracker.kalman.init_state(tr.center_xy[0], tr.center_xy[1])
                narrow_tracker.smooth_center = tr.center_xy

    cap.release()
    cv2.destroyAllWindows()
