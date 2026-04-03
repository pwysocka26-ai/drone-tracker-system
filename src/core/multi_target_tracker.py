from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple
import math


BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass
class MultiTrackState:
    """
    Stabilny stan pojedynczego tracka w warstwie multi-target.

    Pola są celowo podobne do obecnej klasy Track z app.py, żeby integracja
    była prosta:
    - track_id
    - raw_id
    - bbox_xyxy
    - center_xy
    - confidence

    Dodatkowo:
    - velocity_xy
    - age
    - hits
    - missed_frames
    - is_confirmed
    """
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

    # pola kompatybilne z istniejącym pipeline
    is_valid_target: bool = True
    is_active_target: bool = False
    selection_priority: float = 0.0
    target_score: float = 0.0

    history: List[Point] = field(default_factory=list)

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
        )


class MultiTargetTracker:
    """
    Lekki tracker wielu obiektów do podpięcia pod obecny projekt.

    Założenie:
    - dostaje listę detekcji z jednej klatki
    - każda detekcja ma pola kompatybilne z obecną klasą Track:
        track_id / raw_id / bbox_xyxy / center_xy / confidence
    - utrzymuje stabilne tracki niezależnie od chwilowych zaników detekcji
    - zwraca listę MultiTrackState

    Strategia dopasowania:
    1. przewidywanie pozycji tracka z velocity
    2. scoring: distance + size similarity + IoU
    3. greedy assignment
    """

    def __init__(
        self,
        max_missed_frames: int = 20,
        confirm_hits: int = 3,
        max_center_distance: float = 130.0,
        min_iou_for_match: float = 0.01,
        velocity_alpha: float = 0.65,
        history_size: int = 12,
    ) -> None:
        self.max_missed_frames = int(max_missed_frames)
        self.confirm_hits = int(confirm_hits)
        self.max_center_distance = float(max_center_distance)
        self.min_iou_for_match = float(min_iou_for_match)
        self.velocity_alpha = float(velocity_alpha)
        self.history_size = int(history_size)

        self._tracks: List[MultiTrackState] = []
        self._next_id: int = 1

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 1

    def tracks(self) -> List[MultiTrackState]:
        return [tr.copy_for_output() for tr in self._tracks]

    def update(
        self,
        detections: Iterable[object],
        frame_shape: Optional[Sequence[int]] = None,
    ) -> List[MultiTrackState]:
        """
        Aktualizuje stan trackerów na podstawie detekcji z bieżącej klatki.
        """
        dets = [self._normalize_detection(det) for det in detections]

        predicted_centers = [self._predict_center(tr) for tr in self._tracks]
        matches, unmatched_tracks, unmatched_dets = self._greedy_match(
            self._tracks, predicted_centers, dets
        )

        for track_idx, det_idx in matches:
            tr = self._tracks[track_idx]
            det = dets[det_idx]
            self._apply_detection_to_track(tr, det)

        for track_idx in unmatched_tracks:
            tr = self._tracks[track_idx]
            self._mark_track_missed(tr)

        for det_idx in unmatched_dets:
            self._tracks.append(self._spawn_track(dets[det_idx]))

        self._tracks = [
            tr for tr in self._tracks if tr.missed_frames <= self.max_missed_frames
        ]
        self._tracks.sort(key=lambda t: t.track_id)

        return [tr.copy_for_output() for tr in self._tracks]

    @staticmethod
    def _bbox_center(bbox: BBox) -> Point:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @staticmethod
    def _bbox_size(bbox: BBox) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (max(1.0, x2 - x1), max(1.0, y2 - y1))

    @staticmethod
    def _bbox_area(bbox: BBox) -> float:
        w, h = MultiTargetTracker._bbox_size(bbox)
        return w * h

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

        union = MultiTargetTracker._bbox_area(a) + MultiTargetTracker._bbox_area(b) - inter
        if union <= 0.0:
            return 0.0
        return inter / union

    def _normalize_detection(self, det: object) -> MultiTrackState:
        bbox_xyxy = tuple(float(v) for v in getattr(det, "bbox_xyxy"))
        center_xy = tuple(float(v) for v in getattr(det, "center_xy"))
        confidence = float(getattr(det, "confidence", 0.0))
        raw_id = int(getattr(det, "raw_id", getattr(det, "track_id", -1)))

        return MultiTrackState(
            track_id=-1,
            raw_id=raw_id,
            bbox_xyxy=bbox_xyxy,  # type: ignore[arg-type]
            center_xy=center_xy,  # type: ignore[arg-type]
            confidence=confidence,
        )

    def _predict_center(self, tr: MultiTrackState) -> Point:
        return (
            tr.center_xy[0] + tr.velocity_xy[0],
            tr.center_xy[1] + tr.velocity_xy[1],
        )

    def _size_penalty(self, bbox_a: BBox, bbox_b: BBox) -> float:
        wa, ha = self._bbox_size(bbox_a)
        wb, hb = self._bbox_size(bbox_b)

        dw = abs(wa - wb) / max(wa, wb)
        dh = abs(ha - hb) / max(ha, hb)
        return 0.5 * (dw + dh)

    def _match_score(
        self,
        tr: MultiTrackState,
        predicted_center: Point,
        det: MultiTrackState,
    ) -> Optional[float]:
        dist = self._distance(predicted_center, det.center_xy)
        if dist > self.max_center_distance:
            return None

        iou = self._iou(tr.bbox_xyxy, det.bbox_xyxy)
        size_penalty = self._size_penalty(tr.bbox_xyxy, det.bbox_xyxy)

        if iou < self.min_iou_for_match and size_penalty > 0.75:
            return None

        return dist + 60.0 * size_penalty - 25.0 * iou - 10.0 * det.confidence

    def _greedy_match(
        self,
        tracks: Sequence[MultiTrackState],
        predicted_centers: Sequence[Point],
        detections: Sequence[MultiTrackState],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        candidate_pairs: List[Tuple[float, int, int]] = []

        for ti, tr in enumerate(tracks):
            pred = predicted_centers[ti]
            for di, det in enumerate(detections):
                score = self._match_score(tr, pred, det)
                if score is not None:
                    candidate_pairs.append((score, ti, di))

        candidate_pairs.sort(key=lambda x: x[0])

        used_tracks = set()
        used_dets = set()
        matches: List[Tuple[int, int]] = []

        for _, ti, di in candidate_pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            matches.append((ti, di))

        unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]

        return matches, unmatched_tracks, unmatched_dets

    def _apply_detection_to_track(self, tr: MultiTrackState, det: MultiTrackState) -> None:
        old_center = tr.center_xy
        new_center = det.center_xy

        measured_vx = new_center[0] - old_center[0]
        measured_vy = new_center[1] - old_center[1]

        vx = (
            self.velocity_alpha * tr.velocity_xy[0]
            + (1.0 - self.velocity_alpha) * measured_vx
        )
        vy = (
            self.velocity_alpha * tr.velocity_xy[1]
            + (1.0 - self.velocity_alpha) * measured_vy
        )

        tr.bbox_xyxy = det.bbox_xyxy
        tr.center_xy = det.center_xy
        tr.confidence = det.confidence
        tr.raw_id = det.raw_id

        tr.velocity_xy = (vx, vy)
        tr.age += 1
        tr.hits += 1
        tr.missed_frames = 0
        tr.is_confirmed = tr.hits >= self.confirm_hits

        tr.history.append(tr.center_xy)
        if len(tr.history) > self.history_size:
            tr.history = tr.history[-self.history_size:]

    def _mark_track_missed(self, tr: MultiTrackState) -> None:
        tr.age += 1
        tr.missed_frames += 1

        tr.center_xy = (
            tr.center_xy[0] + tr.velocity_xy[0],
            tr.center_xy[1] + tr.velocity_xy[1],
        )

        x1, y1, x2, y2 = tr.bbox_xyxy
        tr.bbox_xyxy = (
            x1 + tr.velocity_xy[0],
            y1 + tr.velocity_xy[1],
            x2 + tr.velocity_xy[0],
            y2 + tr.velocity_xy[1],
        )

        tr.velocity_xy = (tr.velocity_xy[0] * 0.92, tr.velocity_xy[1] * 0.92)

        tr.history.append(tr.center_xy)
        if len(tr.history) > self.history_size:
            tr.history = tr.history[-self.history_size:]

    def _spawn_track(self, det: MultiTrackState) -> MultiTrackState:
        tr = det.copy_for_output()
        tr.track_id = self._next_id
        tr.age = 1
        tr.hits = 1
        tr.missed_frames = 0
        tr.is_confirmed = self.confirm_hits <= 1
        tr.velocity_xy = (0.0, 0.0)
        tr.history = [tr.center_xy]

        self._next_id += 1
        return tr
