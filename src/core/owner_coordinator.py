from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass
class OwnerDecision:
    committed_id: Optional[int]
    candidate_id: Optional[int]
    track: Optional[object]
    mode: str
    reason: str
    used_memory: bool
    just_switched: bool
    missed: int
    center: Optional[Point]
    bbox: Optional[BBox]


class OwnerCoordinator:
    """
    Decouples wide-channel candidate selection from narrow-channel committed owner.

    TargetManager may suggest a candidate every frame, but the coordinator only commits
    a new owner after a short stability window, and it keeps memory of the committed owner
    across short detector gaps or raw-id churn.
    """

    def __init__(
        self,
        commit_stable_frames: int = 4,
        protect_frames: int = 18,
        memory_hold_frames: int = 20,
        reacquire_radius: float = 185.0,
        bbox_similarity_min: float = 0.58,
        candidate_score_margin: float = 0.10,
        healthy_confidence_floor: float = 0.16,
        lost_confidence_floor: float = 0.08,
        edge_margin_ratio: float = 0.18,
        switch_edge_guard: bool = True,
    ) -> None:
        self.commit_stable_frames = int(commit_stable_frames)
        self.protect_frames = int(protect_frames)
        self.memory_hold_frames = int(memory_hold_frames)
        self.reacquire_radius = float(reacquire_radius)
        self.bbox_similarity_min = float(bbox_similarity_min)
        self.candidate_score_margin = float(candidate_score_margin)
        self.healthy_confidence_floor = float(healthy_confidence_floor)
        self.lost_confidence_floor = float(lost_confidence_floor)
        self.edge_margin_ratio = float(edge_margin_ratio)
        self.switch_edge_guard = bool(switch_edge_guard)

        self.committed_id: Optional[int] = None
        self.committed_raw_id: Optional[int] = None
        self.memory_center: Optional[Point] = None
        self.memory_bbox: Optional[BBox] = None
        self.missed: int = 9999
        self.candidate_id: Optional[int] = None
        self.candidate_count: int = 0
        self.protected_left: int = 0
        self.last_reason: str = "init"

    def reset(self) -> None:
        self.committed_id = None
        self.committed_raw_id = None
        self.memory_center = None
        self.memory_bbox = None
        self.missed = 9999
        self.candidate_id = None
        self.candidate_count = 0
        self.protected_left = 0
        self.last_reason = "reset"

    @staticmethod
    def _distance(a: Optional[Point], b: Optional[Point]) -> float:
        if a is None or b is None:
            return float("inf")
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _bbox_size(bbox: Optional[BBox]) -> Tuple[float, float]:
        if bbox is None:
            return (1.0, 1.0)
        x1, y1, x2, y2 = bbox
        return (max(1.0, float(x2) - float(x1)), max(1.0, float(y2) - float(y1)))

    def _bbox_similarity(self, a: Optional[BBox], b: Optional[BBox]) -> float:
        if a is None or b is None:
            return 0.0
        aw, ah = self._bbox_size(a)
        bw, bh = self._bbox_size(b)
        dw = abs(aw - bw) / max(aw, bw)
        dh = abs(ah - bh) / max(ah, bh)
        return max(0.0, 1.0 - 0.55 * (dw + dh))

    def _edge_track(self, tr: object, frame_shape: Sequence[int]) -> bool:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = [float(v) for v in getattr(tr, "bbox_xyxy", (0, 0, 0, 0))]
        mx = float(w) * self.edge_margin_ratio
        my = float(h) * self.edge_margin_ratio
        return x1 < mx or y1 < my or x2 > (w - mx) or y2 > (h - my)

    def _score_candidate(self, tr: object, anchor: Optional[Point]) -> float:
        conf = float(getattr(tr, "confidence", 0.0) or 0.0)
        dist = self._distance(getattr(tr, "center_xy", None), anchor)
        sim = self._bbox_similarity(getattr(tr, "bbox_xyxy", None), self.memory_bbox)
        score = conf * 8.0 + sim * 4.0
        if dist != float("inf"):
            score += max(-3.0, 2.0 - dist / max(1.0, self.reacquire_radius))
        rid = getattr(tr, "raw_id", None)
        if self.committed_raw_id is not None and rid is not None and int(rid) == int(self.committed_raw_id):
            score += 1.8
        return score

    def _make_memory_track(self) -> Optional[object]:
        if self.committed_id is None or self.memory_center is None or self.memory_bbox is None:
            return None

        class MemoryTrack:
            pass

        tr = MemoryTrack()
        tr.track_id = int(self.committed_id)
        tr.raw_id = int(self.committed_raw_id if self.committed_raw_id is not None else self.committed_id)
        tr.bbox_xyxy = tuple(float(v) for v in self.memory_bbox)
        tr.center_xy = tuple(float(v) for v in self.memory_center)
        tr.confidence = 0.01
        tr.missed_frames = int(self.missed)
        tr.is_confirmed = True
        tr.is_valid_target = True
        tr.is_active_target = True
        return tr

    def _find_exact(self, tracks: List[object]) -> Optional[object]:
        if self.committed_id is None:
            return None
        for tr in tracks:
            if int(getattr(tr, "track_id", -1)) == int(self.committed_id):
                return tr
        return None

    def _find_raw(self, tracks: List[object]) -> Optional[object]:
        if self.committed_raw_id is None:
            return None
        matches = [
            tr for tr in tracks
            if getattr(tr, "raw_id", None) is not None and int(getattr(tr, "raw_id")) == int(self.committed_raw_id)
        ]
        if not matches:
            return None
        anchor = self.memory_center
        return min(matches, key=lambda tr: self._distance(getattr(tr, "center_xy", None), anchor))

    def _find_geometry(self, tracks: List[object], radius: Optional[float] = None) -> Optional[object]:
        anchor = self.memory_center
        if anchor is None:
            return None
        radius_px = float(self.reacquire_radius if radius is None else radius)
        candidates: List[object] = []
        for tr in tracks:
            conf = float(getattr(tr, "confidence", 0.0) or 0.0)
            if conf < self.lost_confidence_floor:
                continue
            dist = self._distance(getattr(tr, "center_xy", None), anchor)
            if dist > radius_px:
                continue
            sim = self._bbox_similarity(getattr(tr, "bbox_xyxy", None), self.memory_bbox)
            if sim < self.bbox_similarity_min:
                continue
            candidates.append(tr)
        if not candidates:
            return None
        return max(candidates, key=lambda tr: self._score_candidate(tr, anchor))

    def _resolve_committed_track(self, tracks: List[object]) -> Optional[object]:
        exact = self._find_exact(tracks)
        if exact is not None:
            return exact
        raw = self._find_raw(tracks)
        if raw is not None:
            return raw
        return self._find_geometry(tracks)

    def _candidate_track(self, tracks: List[object], manager_selected_id: Optional[int], anchor: Optional[Point]) -> Optional[object]:
        if manager_selected_id is not None:
            for tr in tracks:
                if int(getattr(tr, "track_id", -1)) == int(manager_selected_id):
                    return tr
        if not tracks:
            return None
        viable = [tr for tr in tracks if float(getattr(tr, "confidence", 0.0) or 0.0) >= self.lost_confidence_floor]
        if not viable:
            return None
        return max(viable, key=lambda tr: self._score_candidate(tr, anchor))

    def _commit(self, tr: object, reason: str) -> None:
        self.committed_id = int(getattr(tr, "track_id"))
        rid = getattr(tr, "raw_id", None)
        self.committed_raw_id = int(rid) if rid is not None else int(self.committed_id)
        self.memory_center = tuple(float(v) for v in getattr(tr, "center_xy"))
        self.memory_bbox = tuple(float(v) for v in getattr(tr, "bbox_xyxy"))
        self.missed = 0
        self.candidate_id = None
        self.candidate_count = 0
        self.protected_left = self.protect_frames
        self.last_reason = reason

    def update(
        self,
        tracks: List[object],
        manager_selected_id: Optional[int],
        frame_shape: Sequence[int],
        manual_lock: bool = False,
    ) -> OwnerDecision:
        tracks = list(tracks or [])
        just_switched = False
        anchor = self.memory_center

        committed_track = self._resolve_committed_track(tracks)
        if committed_track is not None:
            self.committed_id = int(getattr(committed_track, "track_id"))
            rid = getattr(committed_track, "raw_id", None)
            if rid is not None:
                self.committed_raw_id = int(rid)
            self.memory_center = tuple(float(v) for v in getattr(committed_track, "center_xy"))
            self.memory_bbox = tuple(float(v) for v in getattr(committed_track, "bbox_xyxy"))
            self.missed = 0
        else:
            self.missed += 1

        if self.protected_left > 0:
            self.protected_left -= 1

        if manual_lock and manager_selected_id is not None:
            forced = self._candidate_track(tracks, manager_selected_id, anchor)
            if forced is not None:
                if self.committed_id != int(getattr(forced, "track_id")):
                    just_switched = True
                self._commit(forced, "manual_lock")
                return OwnerDecision(
                    committed_id=self.committed_id,
                    candidate_id=self.candidate_id,
                    track=forced,
                    mode="TRACKING",
                    reason="manual_lock",
                    used_memory=False,
                    just_switched=just_switched,
                    missed=self.missed,
                    center=self.memory_center,
                    bbox=self.memory_bbox,
                )

        candidate_track = self._candidate_track(tracks, manager_selected_id, anchor)
        candidate_id = int(getattr(candidate_track, "track_id")) if candidate_track is not None else None

        current_conf = float(getattr(committed_track, "confidence", 0.0) or 0.0) if committed_track is not None else 0.0
        current_healthy = committed_track is not None and current_conf >= self.healthy_confidence_floor and self.missed == 0
        current_edge = self._edge_track(committed_track, frame_shape) if committed_track is not None else False

        if candidate_id is not None and candidate_id != self.committed_id:
            candidate_conf = float(getattr(candidate_track, "confidence", 0.0) or 0.0)
            candidate_edge = self._edge_track(candidate_track, frame_shape)
            allow_edge_switch = (not self.switch_edge_guard) or (not candidate_edge) or (not current_healthy)
            improvement_ok = candidate_conf >= (current_conf + self.candidate_score_margin) or (not current_healthy)
            protected_ok = (self.protected_left <= 0) or (not current_healthy)
            if allow_edge_switch and improvement_ok and protected_ok:
                if self.candidate_id == candidate_id:
                    self.candidate_count += 1
                else:
                    self.candidate_id = candidate_id
                    self.candidate_count = 1
                required = max(2, self.commit_stable_frames - 1) if not current_healthy else self.commit_stable_frames
                if self.candidate_count >= required:
                    self._commit(candidate_track, "stable_switch")
                    committed_track = candidate_track
                    just_switched = True
            else:
                self.candidate_id = None
                self.candidate_count = 0
        else:
            self.candidate_id = candidate_id
            self.candidate_count = 0 if candidate_id is None else self.candidate_count

        if self.committed_id is None and candidate_track is not None:
            if self.candidate_id == candidate_id:
                self.candidate_count += 1
            else:
                self.candidate_id = candidate_id
                self.candidate_count = 1
            if self.candidate_count >= max(2, self.commit_stable_frames - 1):
                self._commit(candidate_track, "bootstrap")
                committed_track = candidate_track
                just_switched = True

        if committed_track is not None:
            mode = "TRACKING"
            reason = self.last_reason if just_switched else ("healthy_keep" if current_healthy else "soft_keep")
            return OwnerDecision(
                committed_id=self.committed_id,
                candidate_id=self.candidate_id,
                track=committed_track,
                mode=mode,
                reason=reason,
                used_memory=False,
                just_switched=just_switched,
                missed=self.missed,
                center=self.memory_center,
                bbox=self.memory_bbox,
            )

        if self.missed <= self.memory_hold_frames and self.memory_center is not None and self.memory_bbox is not None:
            return OwnerDecision(
                committed_id=self.committed_id,
                candidate_id=self.candidate_id,
                track=self._make_memory_track(),
                mode="HOLD",
                reason="memory_hold",
                used_memory=True,
                just_switched=False,
                missed=self.missed,
                center=self.memory_center,
                bbox=self.memory_bbox,
            )

        return OwnerDecision(
            committed_id=self.committed_id,
            candidate_id=self.candidate_id,
            track=None,
            mode="REACQUIRE",
            reason="memory_expired",
            used_memory=False,
            just_switched=False,
            missed=self.missed,
            center=self.memory_center,
            bbox=self.memory_bbox,
        )
