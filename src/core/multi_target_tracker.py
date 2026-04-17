from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple
import math


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]

RAW_ID_BONUS = 0.60
BBOX_ALPHA = 0.72
SIZE_JUMP_LIMIT = 1.35


class SimpleKalman2D:
    """
    Lightweight constant-velocity Kalman filter without external dependencies.
    State: [x, y, vx, vy]
    """
    def __init__(self, process_noise: float = 0.03, measurement_noise: float = 0.20) -> None:
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.initialized = False
        self.x = [0.0, 0.0, 0.0, 0.0]
        self.P = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def clone_for_copy(self) -> "SimpleKalman2D":
        k = SimpleKalman2D(self.q, self.r)
        k.initialized = self.initialized
        k.x = list(self.x)
        k.P = [list(row) for row in self.P]
        return k

    def init_state(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        self.x = [float(x), float(y), float(vx), float(vy)]
        self.P = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]
        self.initialized = True

    def predict(self) -> Point:
        if not self.initialized:
            return (0.0, 0.0)
        self.x = [
            self.x[0] + self.x[2],
            self.x[1] + self.x[3],
            self.x[2],
            self.x[3],
        ]
        F = [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        Ft = list(zip(*F))
        FP = [[sum(F[i][k] * self.P[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
        self.P = [[sum(FP[i][k] * Ft[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
        for i in range(4):
            self.P[i][i] += self.q
        return (self.x[0], self.x[1])

    def correct(self, mx: float, my: float) -> Point:
        if not self.initialized:
            self.init_state(mx, my)
            return (mx, my)

        yx = mx - self.x[0]
        yy = my - self.x[1]
        s00 = self.P[0][0] + self.r
        s01 = self.P[0][1]
        s10 = self.P[1][0]
        s11 = self.P[1][1] + self.r
        det = s00 * s11 - s01 * s10
        if abs(det) < 1e-9:
            return (self.x[0], self.x[1])
        invS = [
            [ s11 / det, -s01 / det],
            [-s10 / det,  s00 / det],
        ]
        K = [[
            self.P[i][0] * invS[0][0] + self.P[i][1] * invS[1][0],
            self.P[i][0] * invS[0][1] + self.P[i][1] * invS[1][1],
        ] for i in range(4)]

        self.x = [
            self.x[i] + K[i][0] * yx + K[i][1] * yy
            for i in range(4)
        ]

        KH = [
            [K[i][0], K[i][1], 0.0, 0.0]
            for i in range(4)
        ]
        I_minus_KH = [[(1.0 if i == j else 0.0) - KH[i][j] for j in range(4)] for i in range(4)]
        self.P = [[sum(I_minus_KH[i][k] * self.P[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
        return (self.x[0], self.x[1])

    @property
    def velocity(self) -> Point:
        return (self.x[2], self.x[3])


@dataclass
class MultiTrackState:
    track_id: int
    raw_id: int
    bbox_xyxy: BBox
    center_xy: Point
    confidence: float

    velocity_xy: Point = (0.0, 0.0)
    age: int = 1
    hits: int = 1
    missed_frames: int = 0
    is_confirmed: bool = False

    is_valid_target: bool = True
    is_active_target: bool = False
    selection_priority: float = 0.0
    target_score: float = 0.0

    history: List[Point] = field(default_factory=list)
    predicted_center_xy: Optional[Point] = None
    updated_this_frame: bool = False
    _kalman: Optional[SimpleKalman2D] = field(default=None, repr=False, compare=False)

    def copy_for_output(self) -> "MultiTrackState":
        return MultiTrackState(
            track_id=self.track_id,
            raw_id=self.raw_id,
            bbox_xyxy=self.bbox_xyxy,
            center_xy=self.center_xy,
            confidence=self.confidence,
            velocity_xy=self.velocity_xy,
            age=self.age,
            hits=self.hits,
            missed_frames=self.missed_frames,
            is_confirmed=self.is_confirmed,
            is_valid_target=self.is_valid_target,
            is_active_target=self.is_active_target,
            selection_priority=self.selection_priority,
            target_score=self.target_score,
            history=list(self.history),
            predicted_center_xy=self.predicted_center_xy,
            updated_this_frame=self.updated_this_frame,
            _kalman=self._kalman.clone_for_copy() if self._kalman is not None else None,
        )


class MultiTargetTracker:
    def __init__(
        self,
        max_missed_frames: int = 20,
        confirm_hits: int = 3,
        max_center_distance: float = 130.0,
        min_iou_for_match: float = 0.01,
        velocity_alpha: float = 0.65,
        history_size: int = 12,
        use_kalman: bool = True,
        kalman_process_noise: float = 0.03,
        kalman_measurement_noise: float = 0.20,
    ) -> None:
        self.max_missed_frames = int(max_missed_frames)
        self.confirm_hits = int(confirm_hits)
        self.max_center_distance = float(max_center_distance)
        self.min_iou_for_match = float(min_iou_for_match)
        self.velocity_alpha = float(velocity_alpha)
        self.history_size = int(history_size)
        self.use_kalman = bool(use_kalman)
        self.kalman_process_noise = float(kalman_process_noise)
        self.kalman_measurement_noise = float(kalman_measurement_noise)

        self._tracks: List[MultiTrackState] = []
        self._next_id: int = 1

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    def tracks(self) -> List[MultiTrackState]:
        return [tr.copy_for_output() for tr in self._tracks]

    def update(self, detections: Iterable[object], frame_shape: Optional[Sequence[int]] = None) -> List[MultiTrackState]:
        dets = [self._normalize_detection(det) for det in detections]
        for tr in self._tracks:
            tr.updated_this_frame = False
        predicted_centers = [self._predict_center(tr) for tr in self._tracks]
        matches, unmatched_tracks, unmatched_dets = self._greedy_match(self._tracks, predicted_centers, dets)

        for track_idx, det_idx in matches:
            tr = self._tracks[track_idx]
            det = dets[det_idx]
            self._apply_detection_to_track(tr, det)

        for track_idx in unmatched_tracks:
            self._mark_track_missed(self._tracks[track_idx])

        for det_idx in unmatched_dets:
            self._tracks.append(self._spawn_track(dets[det_idx]))

        self._tracks = [tr for tr in self._tracks if tr.missed_frames <= self.max_missed_frames]
        self._tracks.sort(key=lambda t: t.track_id)
        return [tr.copy_for_output() for tr in self._tracks]

    @staticmethod
    def _bbox_center(bbox: BBox) -> Point:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @staticmethod
    def _bbox_size(bbox: BBox) -> Point:
        x1, y1, x2, y2 = bbox
        return (max(1.0, x2 - x1), max(1.0, y2 - y1))

    @staticmethod
    def _distance(a: Point, b: Point) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
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
        union = area_a + area_b - inter
        return inter / max(1.0, union)

    def _smooth_bbox(self, old_bbox: BBox, new_bbox: BBox) -> BBox:
        ox1, oy1, ox2, oy2 = old_bbox
        nx1, ny1, nx2, ny2 = new_bbox
        ow, oh = self._bbox_size(old_bbox)
        nw, nh = self._bbox_size(new_bbox)
        alpha = max(BBOX_ALPHA, 0.84) if (nw > ow * SIZE_JUMP_LIMIT or nh > oh * SIZE_JUMP_LIMIT) else BBOX_ALPHA
        return (
            alpha * ox1 + (1.0 - alpha) * nx1,
            alpha * oy1 + (1.0 - alpha) * ny1,
            alpha * ox2 + (1.0 - alpha) * nx2,
            alpha * oy2 + (1.0 - alpha) * ny2,
        )

    def _normalize_detection(self, det: object) -> MultiTrackState:
        bbox = tuple(float(v) for v in getattr(det, "bbox_xyxy"))
        center = tuple(float(v) for v in getattr(det, "center_xy", self._bbox_center(bbox)))
        confidence = float(getattr(det, "confidence", 0.0) or 0.0)
        raw_id = int(getattr(det, "track_id", getattr(det, "raw_id", -1)) or -1)
        return MultiTrackState(track_id=-1, raw_id=raw_id, bbox_xyxy=bbox, center_xy=center, confidence=confidence)

    def _predict_center(self, tr: MultiTrackState) -> Point:
        if self.use_kalman and tr._kalman is not None:
            pred = tr._kalman.predict()
            tr.predicted_center_xy = pred
            return pred
        pred = (tr.center_xy[0] + tr.velocity_xy[0], tr.center_xy[1] + tr.velocity_xy[1])
        tr.predicted_center_xy = pred
        return pred

    def _size_penalty(self, bbox_a: BBox, bbox_b: BBox) -> float:
        wa, ha = self._bbox_size(bbox_a)
        wb, hb = self._bbox_size(bbox_b)
        dw = abs(wa - wb) / max(wa, wb)
        dh = abs(ha - hb) / max(ha, hb)
        return 0.5 * (dw + dh)

    def _motion_penalty(self, tr: MultiTrackState, det: MultiTrackState) -> float:
        ref_center = tr.predicted_center_xy or tr.center_xy
        implied_vx = det.center_xy[0] - ref_center[0]
        implied_vy = det.center_xy[1] - ref_center[1]
        dvx = implied_vx - tr.velocity_xy[0]
        dvy = implied_vy - tr.velocity_xy[1]
        return min(2.0, math.hypot(dvx, dvy) / 45.0)

    def _match_score(self, tr: MultiTrackState, predicted_center: Point, det: MultiTrackState) -> Optional[float]:
        dist = self._distance(predicted_center, det.center_xy)
        if dist > self.max_center_distance:
            return None
        iou = self._iou(tr.bbox_xyxy, det.bbox_xyxy)
        size_penalty = self._size_penalty(tr.bbox_xyxy, det.bbox_xyxy)
        motion_penalty = self._motion_penalty(tr, det)
        if iou < self.min_iou_for_match and size_penalty > 0.75:
            return None
        score = dist + 60.0 * size_penalty + 30.0 * motion_penalty - 25.0 * iou - 10.0 * det.confidence
        if det.raw_id is not None and tr.raw_id is not None and int(det.raw_id) == int(tr.raw_id):
            score -= RAW_ID_BONUS
        return score

    def _greedy_match(self, tracks: Sequence[MultiTrackState], predicted_centers: Sequence[Point], dets: Sequence[MultiTrackState]):
        pairs = []
        for ti, tr in enumerate(tracks):
            for di, det in enumerate(dets):
                score = self._match_score(tr, predicted_centers[ti], det)
                if score is not None:
                    pairs.append((score, ti, di))
        pairs.sort(key=lambda x: x[0])

        matched_tracks = set()
        matched_dets = set()
        matches = []
        for _, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            matches.append((ti, di))

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(dets)) if i not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets

    def _make_kalman(self, center: Point, velocity: Point = (0.0, 0.0)) -> Optional[SimpleKalman2D]:
        if not self.use_kalman:
            return None
        k = SimpleKalman2D(self.kalman_process_noise, self.kalman_measurement_noise)
        k.init_state(center[0], center[1], velocity[0], velocity[1])
        return k

    def _apply_detection_to_track(self, tr: MultiTrackState, det: MultiTrackState) -> None:
        old_center = tr.center_xy
        smoothed_bbox = self._smooth_bbox(tr.bbox_xyxy, det.bbox_xyxy)
        measured_center = self._bbox_center(smoothed_bbox)

        if self.use_kalman:
            if tr._kalman is None:
                tr._kalman = self._make_kalman(old_center, tr.velocity_xy)
            corrected_center = tr._kalman.correct(measured_center[0], measured_center[1]) if tr._kalman is not None else measured_center
            new_center = corrected_center
            vx, vy = tr._kalman.velocity if tr._kalman is not None else tr.velocity_xy
            tr.predicted_center_xy = (new_center[0] + vx, new_center[1] + vy)
        else:
            new_center = measured_center
            measured_vx = new_center[0] - old_center[0]
            measured_vy = new_center[1] - old_center[1]
            vx = self.velocity_alpha * tr.velocity_xy[0] + (1.0 - self.velocity_alpha) * measured_vx
            vy = self.velocity_alpha * tr.velocity_xy[1] + (1.0 - self.velocity_alpha) * measured_vy
            tr.predicted_center_xy = (new_center[0] + vx, new_center[1] + vy)

        tr.bbox_xyxy = smoothed_bbox
        tr.center_xy = new_center
        tr.confidence = det.confidence
        tr.raw_id = det.raw_id
        tr.velocity_xy = (vx, vy)
        tr.age += 1
        tr.hits += 1
        tr.missed_frames = 0
        tr.is_confirmed = tr.hits >= self.confirm_hits
        tr.updated_this_frame = True

        tr.history.append(tr.center_xy)
        if len(tr.history) > self.history_size:
            tr.history = tr.history[-self.history_size:]

    def _mark_track_missed(self, tr: MultiTrackState) -> None:
        tr.age += 1
        tr.missed_frames += 1
        old_center = tr.center_xy

        if self.use_kalman and tr._kalman is not None:
            pred = tr._kalman.predict()
            dx = pred[0] - old_center[0]
            dy = pred[1] - old_center[1]
            tr.center_xy = pred
            tr.predicted_center_xy = pred
            tr.velocity_xy = tr._kalman.velocity
        else:
            dx = tr.velocity_xy[0]
            dy = tr.velocity_xy[1]
            tr.center_xy = (tr.center_xy[0] + dx, tr.center_xy[1] + dy)
            tr.predicted_center_xy = tr.center_xy
            tr.velocity_xy = (tr.velocity_xy[0] * 0.92, tr.velocity_xy[1] * 0.92)

        x1, y1, x2, y2 = tr.bbox_xyxy
        tr.bbox_xyxy = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

        tr.history.append(tr.center_xy)
        if len(tr.history) > self.history_size:
            tr.history = tr.history[-self.history_size:]

    def _spawn_track(self, det: MultiTrackState) -> MultiTrackState:
        tr = det.copy_for_output()
        tr.track_id = self._next_id
        tr.updated_this_frame = True
        tr.age = 1
        tr.hits = 1
        tr.missed_frames = 0
        tr.is_confirmed = self.confirm_hits <= 1
        tr.velocity_xy = (0.0, 0.0)
        tr.history = [tr.center_xy]
        tr._kalman = self._make_kalman(tr.center_xy, (0.0, 0.0))
        tr.predicted_center_xy = tr.center_xy
        self._next_id += 1
        return tr
