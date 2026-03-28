import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

from sim.simulator import FormationSimulator, draw_sim_frame, synthesize_tracks


@dataclass
class Track:
    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    center_xy: Tuple[float, float]
    confidence: float
    age: int = 0


@dataclass
class AppArgs:
    mode: str
    fps: int
    width: int
    height: int
    drones: int
    video_path: str


def build_args(config):
    sim = config.get("simulation") or {}
    video_cfg = config.get("video") or {}
    return AppArgs(
        mode=config.get("mode", "video"),
        fps=config.get("fps", 30),
        width=sim.get("width", 1920),
        height=sim.get("height", 1080),
        drones=min(3, sim.get("drones", 3)),
        video_path=video_cfg.get("source", "video.mp4"),
    )


class StableTracker:
    def __init__(self, max_age=16, match_dist=160):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.match_dist = match_dist

    def update(self, detections: List[Track]) -> List[Track]:
        new_tracks = {}
        used_ids = set()
        results = []

        for det in detections:
            cx, cy = det.center_xy
            best_id = None
            best_dist = 1e9

            for tid, st in self.tracks.items():
                if tid in used_ids:
                    continue
                px, py = st["center"]
                dist = float(np.hypot(cx - px, cy - py))
                if dist < best_dist and dist < self.match_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                sx, sy = cx, cy
            else:
                px, py = self.tracks[best_id]["center"]
                sx = 0.78 * px + 0.22 * cx
                sy = 0.78 * py + 0.22 * cy

            tr = Track(
                track_id=best_id,
                bbox_xyxy=det.bbox_xyxy,
                center_xy=(sx, sy),
                confidence=det.confidence,
                age=0,
            )
            new_tracks[best_id] = {
                "center": (sx, sy),
                "bbox": det.bbox_xyxy,
                "confidence": det.confidence,
                "age": 0,
            }
            used_ids.add(best_id)
            results.append(tr)

        for tid, st in self.tracks.items():
            if tid not in new_tracks and st["age"] < self.max_age:
                hold = Track(
                    track_id=tid,
                    bbox_xyxy=st["bbox"],
                    center_xy=st["center"],
                    confidence=max(0.2, st["confidence"] - 0.04 * (st["age"] + 1)),
                    age=st["age"] + 1,
                )
                new_tracks[tid] = {
                    "center": st["center"],
                    "bbox": st["bbox"],
                    "confidence": hold.confidence,
                    "age": hold.age,
                }
                results.append(hold)

        self.tracks = new_tracks
        results.sort(key=lambda t: t.track_id)
        return results[:3]


def detect_dark_flying_objects(frame, max_targets=3) -> List[Track]:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, th1 = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY_INV)
    _, th2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(th1, th2)

    sky_limit = int(h * 0.78)
    sky_mask = np.zeros_like(mask)
    sky_mask[:sky_limit, :] = 255
    mask = cv2.bitwise_and(mask, sky_mask)

    small = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small)
    mask = cv2.dilate(mask, small, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4 or area > 900:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 2 or bh < 2:
            continue

        ratio = bw / float(max(1, bh))
        if ratio < 0.18 or ratio > 9.0:
            continue

        cx = x + bw / 2.0
        cy = y + bh / 2.0
        score = area + max(0, (sky_limit - cy) * 0.02)
        candidates.append((score, x, y, bw, bh, cx, cy))

    candidates.sort(reverse=True, key=lambda x: x[0])

    out = []
    kept = []
    min_dist = 26

    for score, x, y, bw, bh, cx, cy in candidates:
        if any(np.hypot(cx - px, cy - py) < min_dist for px, py in kept):
            continue
        kept.append((cx, cy))
        conf = min(0.99, max(0.45, score / 120.0))
        out.append(
            Track(
                track_id=0,
                bbox_xyxy=(x, y, x + bw, y + bh),
                center_xy=(cx, cy),
                confidence=conf,
                age=0,
            )
        )
        if len(out) >= max_targets:
            break

    return out


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
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return int(x1), int(y1), int(x2), int(y2)


def crop_to_16_9(frame, center=None, scale=1.0, out_size=(780, 360)):
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

    xs1 = [t.bbox_xyxy[0] for t in tracks if t.bbox_xyxy is not None]
    ys1 = [t.bbox_xyxy[1] for t in tracks if t.bbox_xyxy is not None]
    xs2 = [t.bbox_xyxy[2] for t in tracks if t.bbox_xyxy is not None]
    ys2 = [t.bbox_xyxy[3] for t in tracks if t.bbox_xyxy is not None]

    if not xs1:
        return crop_to_16_9(frame, None, 1.0, out_size)

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


def add_title(panel, title):
    cv2.rectangle(panel, (0, 0), (430, 56), (0, 0, 0), -1)
    cv2.putText(panel, title, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return panel


def draw_debug(frame, tracks, selected_id):
    debug = frame.copy()
    for t in tracks:
        if t.bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
        cx, cy = [int(v) for v in t.center_xy]
        color = (0, 255, 255) if t.track_id == selected_id else (0, 255, 0)
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        cv2.circle(debug, (cx, cy), 4, color, -1)
        cv2.putText(debug, f"DRON {t.track_id} {t.confidence:.2f}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return debug


def auto_select_target(tracks, previous_target_id, previous_center):
    if not tracks:
        return previous_target_id

    for t in tracks:
        if t.track_id == previous_target_id:
            return previous_target_id

    if previous_center is not None:
        best = min(tracks, key=lambda t: np.hypot(t.center_xy[0] - previous_center[0], t.center_xy[1] - previous_center[1]))
        return best.track_id

    best = min(tracks, key=lambda t: t.center_xy[1])
    return best.track_id


def desired_zoom_from_target(frame, target: Track):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = target.bbox_xyxy
    tw = max(1.0, x2 - x1)
    th = max(1.0, y2 - y1)
    rel = max(tw / w, th / h)
    desired = 0.06 / max(rel, 0.012)
    return float(np.clip(desired, 2.0, 5.0))


def get_frame_and_tracks(args, cap, tracker, sim):
    if args.mode == "video":
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                return None, []
        detections = detect_dark_flying_objects(frame)
        tracks = tracker.update(detections)

        # filtr: tylko małe obiekty w górze kadru
        filtered = []
        h, w = frame.shape[:2]
        for t in tracks:
            if t.bbox_xyxy is None:
                continue
            x1, y1, x2, y2 = t.bbox_xyxy
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            cy = (y1 + y2) / 2.0

            if area > 0.01 * w * h:
                continue
            if cy > 0.65 * h:
                continue

            filtered.append(t)

        return frame, filtered

    drones, phase, sim_t = sim.step()
    frame = draw_sim_frame(args.width, args.height, drones, phase, sim_t)
    sim_tracks = synthesize_tracks(drones, 0.01, 5.0, (88, 68))
    tracks = []
    for t in sim_tracks:
        tracks.append(
            Track(
                track_id=t.track_id,
                bbox_xyxy=t.bbox_xyxy,
                center_xy=t.center_xy,
                confidence=t.conf,
                age=0,
            )
        )
    return frame, tracks


def run_app(config):
    args = build_args(config)

    cap = None
    sim = None
    tracker = StableTracker(max_age=20, match_dist=90)

    if args.mode == "video":
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            print(f"Nie moge otworzyc pliku: {args.video_path}")
            return
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps and source_fps > 0:
            args.fps = int(source_fps)
        print(f"VIDEO MODE STARTED: {args.video_path}")
    else:
        sim = FormationSimulator(args.width, args.height, args.drones, args.fps)
        print("SIM MODE STARTED")

    target_id = 1
    smooth_center: Optional[Tuple[float, float]] = None
    previous_center: Optional[Tuple[float, float]] = None
    smooth_zoom = 3.0
    lost_count = 0

    window_name = "Drone Tracker Multiview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)

    while True:
        frame, tracks = get_frame_and_tracks(args, cap, tracker, sim)
        if frame is None:
            break

        target_id = auto_select_target(tracks, target_id, previous_center)

        selected = None
        for t in tracks:
            if t.track_id == target_id:
                selected = t
                break

        wide_program = crop_group(frame, tracks, (780, 360))
        wide_debug = crop_group(draw_debug(frame, tracks, target_id), tracks, (1560, 450))

        if selected is not None:
            cx, cy = selected.center_xy

            if smooth_center is None:
                smooth_center = (cx, cy)
            else:
                a_center = 0.90
                smooth_center = (
                    a_center * smooth_center[0] + (1.0 - a_center) * cx,
                    a_center * smooth_center[1] + (1.0 - a_center) * cy,
                )

            previous_center = smooth_center
            lost_count = 0

            desired_zoom = desired_zoom_from_target(frame, selected)
            smooth_zoom = 0.94 * smooth_zoom + 0.06 * desired_zoom

            narrow_output = crop_to_16_9(frame, smooth_center, smooth_zoom, (780, 360))
            cv2.putText(narrow_output, f"TRACKING DRON {target_id}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(narrow_output, f"ZOOM {smooth_zoom:.1f}x", (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.line(narrow_output, (390, 0), (390, 360), (0, 255, 255), 1)
            cv2.line(narrow_output, (0, 180), (780, 180), (0, 255, 255), 1)
        else:
            lost_count += 1
            if smooth_center is not None and lost_count < 30:
                narrow_output = crop_to_16_9(frame, smooth_center, smooth_zoom, (780, 360))
                cv2.putText(narrow_output, f"TRACK HOLD DRON {target_id}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            else:
                smooth_center = None
                previous_center = None
                smooth_zoom = 3.0
                narrow_output = crop_to_16_9(frame, None, 2.0, (780, 360))
                cv2.putText(narrow_output, "BRAK CELU", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        wide_program = add_title(wide_program, "WIDE PROGRAM")
        narrow_output = add_title(narrow_output, "NARROW OUTPUT")
        wide_debug = add_title(wide_debug, "WIDE DEBUG")

        dashboard = np.vstack([np.hstack([wide_program, narrow_output]), wide_debug])
        cv2.imshow(window_name, dashboard)

        key = cv2.waitKey(max(1, int(1000 / max(1, args.fps)))) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("1"):
            target_id = 1
            smooth_center = None
            previous_center = None
        elif key == ord("2"):
            target_id = 2
            smooth_center = None
            previous_center = None
        elif key == ord("3"):
            target_id = 3
            smooth_center = None
            previous_center = None

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
