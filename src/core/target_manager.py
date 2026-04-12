from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


Point = Tuple[float, float]


@dataclass
class WideOwnerSnapshot:
    valid: bool = False
    track_id: Optional[int] = None
    switch_seq: int = 0
    frame_id: int = 0
    timestamp_ms: int = 0
    bbox_xyxy: Optional[tuple[float, float, float, float]] = None
    center_xy: Optional[Point] = None
    velocity_xy: Point = (0.0, 0.0)
    confidence: float = 0.0
    track_age: int = 0
    missed_count: int = 0
    quality_score: float = 0.0
    area_ratio: float = 0.0
    relative_area_ratio: float = 1.0
    is_large_target: bool = False
    is_huge_outlier: bool = False
    owner_changed: bool = False
    prev_track_id: Optional[int] = None
    reason: str = ""


class TargetManager:
    def __init__(
        self,
        reacquire_radius_auto=135.0,
        reacquire_radius_manual=220.0,
        sticky_frames=22,
        switch_margin=0.36,
        switch_dwell=6,
        switch_cooldown=7,
        switch_persist=2,
        max_select_missed=2,
        min_start_conf=0.10,
        min_start_hits=2,
        min_confirmed_conf=0.10,
        min_hold_frames=5,
        predicted_dist_px=95.0,
        raw_id_bonus=1.8,
        current_target_bonus=4.2,
        selection_freeze_frames=8,
        hard_keep_missed=1,
        hard_keep_conf=0.18,
        hard_switch_min_gain=1.10,
        proactive_degrade_missed=2,
        proactive_degrade_quality=0.58,
        proactive_switch_persist=1,
        proactive_switch_cooldown=1,
        neighbor_shuffle_radius_px=120.0,
        neighbor_bbox_similarity_min=0.55,
        neighbor_confidence_floor=0.12,
        neighbor_score_gain=0.18,
        large_target_area_ratio=0.0016,
        huge_target_area_ratio=0.0100,
        max_area_outlier_ratio=2.8,
        huge_area_outlier_ratio=6.0,
        startup_stabilization_frames=45,
        startup_outlier_ratio=2.3,
        startup_min_conf_for_outlier=0.24,
        suppress_huge_outlier_owner=True,
        startup_min_hits=4,
        startup_switch_bonus_margin=0.90,
    ):
        self.selected_id = None
        self.manual_lock = False

        self.reacquire_radius_auto = float(reacquire_radius_auto)
        self.reacquire_radius_manual = float(reacquire_radius_manual)
        self.sticky_frames = int(sticky_frames)
        self.switch_margin = float(switch_margin)
        self.switch_dwell = int(switch_dwell)
        self.switch_cooldown = int(switch_cooldown)
        self.switch_persist = int(switch_persist)
        self.max_select_missed = int(max_select_missed)
        self.min_start_conf = float(min_start_conf)
        self.min_start_hits = int(min_start_hits)
        self.min_confirmed_conf = float(min_confirmed_conf)
        self.min_hold_frames = int(min_hold_frames)
        self.predicted_dist_px = float(predicted_dist_px)
        self.raw_id_bonus = float(raw_id_bonus)
        self.current_target_bonus = float(current_target_bonus)
        self.selection_freeze_frames = int(selection_freeze_frames)
        self.hard_keep_missed = int(hard_keep_missed)
        self.hard_keep_conf = float(hard_keep_conf)
        self.hard_switch_min_gain = float(hard_switch_min_gain)
        self.proactive_degrade_missed = int(proactive_degrade_missed)
        self.proactive_degrade_quality = float(proactive_degrade_quality)
        self.proactive_switch_persist = int(proactive_switch_persist)
        self.proactive_switch_cooldown = int(proactive_switch_cooldown)
        self.neighbor_shuffle_radius_px = float(neighbor_shuffle_radius_px)
        self.neighbor_bbox_similarity_min = float(neighbor_bbox_similarity_min)
        self.neighbor_confidence_floor = float(neighbor_confidence_floor)
        self.neighbor_score_gain = float(neighbor_score_gain)
        self.large_target_area_ratio = float(large_target_area_ratio)
        self.huge_target_area_ratio = float(huge_target_area_ratio)
        self.max_area_outlier_ratio = float(max_area_outlier_ratio)
        self.huge_area_outlier_ratio = float(huge_area_outlier_ratio)
        self.startup_stabilization_frames = int(startup_stabilization_frames)
        self.startup_outlier_ratio = float(startup_outlier_ratio)
        self.startup_min_conf_for_outlier = float(startup_min_conf_for_outlier)
        self.suppress_huge_outlier_owner = bool(suppress_huge_outlier_owner)
        self.startup_min_hits = int(startup_min_hits)
        self.startup_switch_bonus_margin = float(startup_switch_bonus_margin)

        self.lock_age = 9999
        self.frame_id = 0
        self.last_switch_frame = -999999
        self.pending_id = None
        self.pending_count = 0

        self.last_selected_center = None
        self.last_selected_raw_id = None
        self.selection_freeze_id = None
        self.selection_freeze_left = 0

        self.owner_switch_seq = 0
        self.last_snapshot = WideOwnerSnapshot()

    def reset(self):
        self.selected_id = None
        self.manual_lock = False
        self.lock_age = 9999
        self.frame_id = 0
        self.last_switch_frame = -999999
        self.pending_id = None
        self.pending_count = 0
        self.last_selected_center = None
        self.last_selected_raw_id = None
        self.selection_freeze_id = None
        self.selection_freeze_left = 0
        self.owner_switch_seq = 0
        self.last_snapshot = WideOwnerSnapshot()

    def set_auto_mode(self):
        self.manual_lock = False
        self.selected_id = None
        self.lock_age = 9999
        self.pending_id = None
        self.pending_count = 0
        self.last_snapshot = WideOwnerSnapshot(frame_id=self.frame_id)

    def set_manual_target(self, tid):
        self.manual_lock = True
        self.selected_id = int(tid)
        self.lock_age = 0
        self.last_switch_frame = self.frame_id
        self.pending_id = None
        self.pending_count = 0
        self.selection_freeze_id = None
        self.selection_freeze_left = 0
        self.owner_switch_seq += 1

    def freeze_to(self, tid, frames=None):
        if tid is None:
            return
        self.selection_freeze_id = int(tid)
        self.selection_freeze_left = int(self.selection_freeze_frames if frames is None else frames)

    def clear_freeze(self):
        self.selection_freeze_id = None
        self.selection_freeze_left = 0

    def find_active_track(self, tracks):
        if self.selected_id is None:
            return None
        for tr in tracks or []:
            if int(getattr(tr, "track_id", -1)) == int(self.selected_id):
                return tr
        return None

    def _distance(self, a: Optional[Point], b: Optional[Point]) -> float:
        if a is None or b is None:
            return float("inf")
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _score_center(self, tr):
        return tuple(getattr(tr, "predicted_center_xy", None) or tr.center_xy)

    @staticmethod
    def _track_area(tr) -> float:
        x1, y1, x2, y2 = getattr(tr, "bbox_xyxy", (0.0, 0.0, 0.0, 0.0))
        return max(1.0, float(x2 - x1) * float(y2 - y1))

    def _median_confirmed_area(self, tracks) -> float:
        areas = []
        for tr in tracks or []:
            if bool(getattr(tr, "is_confirmed", False)):
                areas.append(self._track_area(tr))
        if not areas:
            for tr in tracks or []:
                areas.append(self._track_area(tr))
        if not areas:
            return 1.0
        areas.sort()
        mid = len(areas) // 2
        if len(areas) % 2:
            return float(areas[mid])
        return float(0.5 * (areas[mid - 1] + areas[mid]))

    def _relative_area_ratio(self, tr, reference_area: Optional[float]) -> float:
        if reference_area is None or reference_area <= 1e-6:
            return 1.0
        return float(self._track_area(tr) / max(1.0, float(reference_area)))

    def _area_ratio(self, tr, frame_shape=None) -> float:
        if frame_shape is None:
            return 0.0
        h, w = frame_shape[:2]
        return float(self._track_area(tr) / max(1.0, float(w * h)))

    def _is_large_target(self, tr, frame_shape=None, reference_area: Optional[float] = None) -> bool:
        return self._area_ratio(tr, frame_shape) >= self.large_target_area_ratio or self._relative_area_ratio(tr, reference_area) >= 1.8

    def _is_huge_outlier(self, tr, frame_shape=None, reference_area: Optional[float] = None) -> bool:
        area_ratio = self._area_ratio(tr, frame_shape)
        relative_ratio = self._relative_area_ratio(tr, reference_area)
        aspect = max(1e-6, float(getattr(tr, "bbox_xyxy", (0, 0, 1, 1))[2] - getattr(tr, "bbox_xyxy", (0, 0, 1, 1))[0])) / max(1e-6, float(getattr(tr, "bbox_xyxy", (0, 0, 1, 1))[3] - getattr(tr, "bbox_xyxy", (0, 0, 1, 1))[1]))
        elongated = aspect < 0.62 or aspect > 1.85
        return relative_ratio >= self.huge_area_outlier_ratio or (area_ratio >= self.huge_target_area_ratio and elongated)


    def _is_startup_outlier(self, tr, frame_shape=None, reference_area: Optional[float] = None) -> bool:
        relative_ratio = self._relative_area_ratio(tr, reference_area)
        conf = float(getattr(tr, "confidence", 0.0) or 0.0)
        hits = int(getattr(tr, "hits", 0) or 0)
        return (
            self.frame_id <= self.startup_stabilization_frames
            and relative_ratio >= self.startup_outlier_ratio
            and conf < self.startup_min_conf_for_outlier
            and hits < max(self.min_start_hits + 2, 5)
        )

    def _should_reject_owner_candidate(self, tr, frame_shape=None, reference_area: Optional[float] = None) -> bool:
        huge_outlier = self._is_huge_outlier(tr, frame_shape=frame_shape, reference_area=reference_area)
        startup_outlier = self._is_startup_outlier(tr, frame_shape=frame_shape, reference_area=reference_area)
        conf = float(getattr(tr, "confidence", 0.0) or 0.0)
        missed = int(getattr(tr, "missed_frames", 0) or 0)
        hits = int(getattr(tr, "hits", 0) or 0)
        if huge_outlier and self.suppress_huge_outlier_owner and conf < 0.40:
            return True
        if startup_outlier:
            return True
        if missed >= 2 and conf < 0.18 and hits < 6:
            return True
        return False

    def _eligible_tracks(self, tracks):
        out = []
        for tr in tracks or []:
            missed = int(getattr(tr, "missed_frames", 0) or 0)
            if missed > self.max_select_missed:
                continue

            conf = float(getattr(tr, "confidence", 0.0) or 0.0)
            confirmed = bool(getattr(tr, "is_confirmed", False))
            hits = int(getattr(tr, "hits", 0) or 0)
            is_current = self.selected_id is not None and int(getattr(tr, "track_id", -1)) == int(self.selected_id)

            if self.selected_id is None:
                if conf < self.min_start_conf:
                    continue
                if (not confirmed) and hits < self.min_start_hits:
                    continue
            else:
                if (not confirmed) and (not is_current) and conf < self.min_confirmed_conf:
                    continue

            if self._should_reject_owner_candidate(tr, reference_area=self._median_confirmed_area(tracks)):
                continue
            out.append(tr)
        return out

    def _score(self, tr, frame_shape, predicted_center=None, is_current=False, reference_area=None):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = tr.bbox_xyxy
        x, y = tr.center_xy
        conf = float(getattr(tr, "confidence", 0.0) or 0.0)
        hits = int(getattr(tr, "hits", 0) or 0)
        missed = int(getattr(tr, "missed_frames", 0) or 0)
        confirmed = bool(getattr(tr, "is_confirmed", False))
        raw_id = getattr(tr, "raw_id", None)

        area = max(1.0, (x2 - x1) * (y2 - y1))
        area_norm = area / max(1.0, float(w * h))
        relative_area = self._relative_area_ratio(tr, reference_area)
        huge_outlier = self._is_huge_outlier(tr, frame_shape=frame_shape, reference_area=reference_area)

        cx = x / max(1.0, w)
        cy = y / max(1.0, h)
        center_dist2 = (cx - 0.5) ** 2 + (cy - 0.5) ** 2

        score = 0.0
        score += conf * 10.0
        score += min(area_norm * 220.0, 2.0)
        if relative_area > self.max_area_outlier_ratio:
            score -= min(5.5, (relative_area - self.max_area_outlier_ratio) * 1.1)
        elif relative_area >= 1.5:
            score += min(0.55, (relative_area - 1.5) * 0.18)
        if huge_outlier:
            score -= 3.8
        if self._is_startup_outlier(tr, frame_shape=frame_shape, reference_area=reference_area):
            score -= 2.6
        score += max(-1.0, 1.0 - center_dist2 * 3.0)
        score += min(1.4, hits * 0.22)
        if confirmed:
            score += 1.8
        score -= missed * 1.8

        if is_current:
            score += self.current_target_bonus
            if missed >= 2:
                score -= 1.2 * float(missed - 1)
            if conf < 0.20:
                score -= (0.20 - conf) * 9.0
            if missed >= 4 and conf < 0.16:
                score -= 2.4

        anchor = predicted_center or self.last_selected_center
        if anchor is not None:
            score_center = self._score_center(tr)
            dist = self._distance(score_center, anchor)
            score += max(-3.5, 2.2 - dist / max(1.0, self.predicted_dist_px))

        if self.last_selected_raw_id is not None and raw_id is not None:
            if int(raw_id) == int(self.last_selected_raw_id):
                score += self.raw_id_bonus

        return score


    @staticmethod
    def _bbox_similarity(a, b) -> float:
        if a is None or b is None:
            return 0.0
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        aw = max(1.0, ax2 - ax1)
        ah = max(1.0, ay2 - ay1)
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)
        dw = abs(aw - bw) / max(aw, bw)
        dh = abs(ah - bh) / max(ah, bh)
        return max(0.0, 1.0 - 0.5 * (dw + dh))

    @staticmethod
    def _velocity_alignment(a, b) -> float:
        if a is None or b is None:
            return 0.5
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        an = math.hypot(ax, ay)
        bn = math.hypot(bx, by)
        if an < 1e-6 or bn < 1e-6:
            return 0.5
        cosv = (ax * bx + ay * by) / max(1e-6, an * bn)
        return max(0.0, min(1.0, 0.5 + 0.5 * cosv))

    def _find_neighbor_shuffle_candidate(self, active, candidates, frame_shape, predicted_center=None):
        if active is None:
            return None
        active_center = self._score_center(active)
        active_bbox = getattr(active, 'bbox_xyxy', None)
        active_vel = getattr(active, 'velocity_xy', (0.0, 0.0)) or (0.0, 0.0)
        reference_area = self._median_confirmed_area(candidates)
        active_score = self._score(active, frame_shape, predicted_center, is_current=True, reference_area=reference_area)
        best = None
        best_score = -1e9
        for tr in candidates:
            tid = int(getattr(tr, 'track_id', -1))
            if tid == int(getattr(active, 'track_id', -1)):
                continue
            conf = float(getattr(tr, 'confidence', 0.0) or 0.0)
            if conf < self.neighbor_confidence_floor:
                continue
            dist = self._distance(self._score_center(tr), active_center)
            if dist > self.neighbor_shuffle_radius_px:
                continue
            bbox_sim = self._bbox_similarity(getattr(tr, 'bbox_xyxy', None), active_bbox)
            if bbox_sim < self.neighbor_bbox_similarity_min:
                continue
            vel_align = self._velocity_alignment(getattr(tr, 'velocity_xy', (0.0, 0.0)) or (0.0, 0.0), active_vel)
            score = self._score(tr, frame_shape, predicted_center, is_current=False, reference_area=reference_area)
            formation_bonus = bbox_sim * 1.4 + vel_align * 0.8 - dist / max(1.0, self.neighbor_shuffle_radius_px)
            total = score + formation_bonus
            if total > best_score:
                best = tr
                best_score = total
        if best is None:
            return None
        if best_score < (active_score + self.neighbor_score_gain):
            return None
        return best

    def compute_owner_quality(self, tr, frame_shape=None, reference_area: Optional[float] = None) -> float:
        conf = float(getattr(tr, "confidence", 0.0) or 0.0)
        hits = int(getattr(tr, "hits", getattr(tr, "age", 0)) or 0)
        missed = int(getattr(tr, "missed_frames", 0) or 0)
        vel = getattr(tr, "velocity_xy", (0.0, 0.0)) or (0.0, 0.0)
        speed = math.hypot(float(vel[0]), float(vel[1]))
        age_norm = max(0.0, min(1.0, hits / 8.0))
        missed_penalty = max(0.0, min(1.0, missed / 3.0))
        motion_stability = max(0.0, min(1.0, 1.0 - speed / 80.0))
        visibility = 1.0
        area_ratio = self._area_ratio(tr, frame_shape)
        relative_area = self._relative_area_ratio(tr, reference_area)
        huge_outlier = self._is_huge_outlier(tr, frame_shape=frame_shape, reference_area=reference_area)
        x1, y1, x2, y2 = getattr(tr, "bbox_xyxy", (0.0, 0.0, 1.0, 1.0))
        bw = max(1.0, float(x2) - float(x1))
        bh = max(1.0, float(y2) - float(y1))
        aspect = bw / max(1.0, bh)
        shape_score = max(0.0, 1.0 - min(1.0, abs(math.log(max(1e-6, aspect))) / math.log(2.4)))
        scale_score = 0.72
        if area_ratio >= self.large_target_area_ratio:
            scale_score = 0.92
        if area_ratio >= self.huge_target_area_ratio:
            scale_score = 0.68
        if frame_shape is not None:
            h, w = frame_shape[:2]
            x, y = getattr(tr, "center_xy", (0.0, 0.0))
            nx = float(x) / max(1.0, float(w))
            ny = float(y) / max(1.0, float(h))
            edge_penalty = max(0.0, max(abs(nx - 0.5) * 2.0, abs(ny - 0.5) * 2.0) - 0.55)
            visibility = max(0.0, 1.0 - edge_penalty)
        outlier_penalty = 0.0
        if relative_area > self.max_area_outlier_ratio:
            outlier_penalty = min(1.0, (relative_area - self.max_area_outlier_ratio) / max(1.0, self.huge_area_outlier_ratio - self.max_area_outlier_ratio))
        if huge_outlier:
            outlier_penalty = max(outlier_penalty, 0.95)
        if self._is_startup_outlier(tr, frame_shape=frame_shape, reference_area=reference_area):
            outlier_penalty = max(outlier_penalty, 0.90)
        quality = (
            0.31 * max(0.0, min(1.0, conf))
            + 0.24 * age_norm
            + 0.16 * motion_stability
            + 0.15 * visibility
            + 0.08 * shape_score
            + 0.10 * scale_score
            - 0.24 * missed_penalty
            - 0.26 * outlier_penalty
        )
        return max(0.0, min(1.0, quality))

    def build_wide_owner_snapshot(self, tracks, frame_shape=None, frame_id=None) -> WideOwnerSnapshot:
        reference_area = self._median_confirmed_area(tracks)
        track = self.find_active_track(tracks)
        if track is None:
            self.last_snapshot = WideOwnerSnapshot(
                valid=False,
                track_id=self.selected_id,
                switch_seq=self.owner_switch_seq,
                frame_id=self.frame_id if frame_id is None else int(frame_id),
                reason="no_active_track",
            )
            return self.last_snapshot

        vel = getattr(track, "velocity_xy", (0.0, 0.0)) or (0.0, 0.0)
        snapshot = WideOwnerSnapshot(
            valid=True,
            track_id=int(getattr(track, "track_id", -1)),
            switch_seq=self.owner_switch_seq,
            frame_id=self.frame_id if frame_id is None else int(frame_id),
            timestamp_ms=int((self.frame_id if frame_id is None else int(frame_id)) * (1000 / 30.0)),
            bbox_xyxy=tuple(float(v) for v in getattr(track, "bbox_xyxy", (0, 0, 0, 0))),
            center_xy=tuple(float(v) for v in getattr(track, "center_xy", (0.0, 0.0))),
            velocity_xy=(float(vel[0]), float(vel[1])),
            confidence=float(getattr(track, "confidence", 0.0) or 0.0),
            track_age=int(getattr(track, "hits", getattr(track, "age", 0)) or 0),
            missed_count=int(getattr(track, "missed_frames", 0) or 0),
            quality_score=self.compute_owner_quality(track, frame_shape=frame_shape, reference_area=reference_area),
            area_ratio=self._area_ratio(track, frame_shape),
            relative_area_ratio=self._relative_area_ratio(track, reference_area),
            is_large_target=self._is_large_target(track, frame_shape=frame_shape, reference_area=reference_area),
            is_huge_outlier=self._is_huge_outlier(track, frame_shape=frame_shape, reference_area=reference_area),
            reason="manual" if self.manual_lock else "auto",
        )
        prev = self.last_snapshot.track_id if self.last_snapshot is not None else None
        snapshot.prev_track_id = prev
        snapshot.owner_changed = prev is not None and snapshot.track_id != prev
        if not self.manual_lock and snapshot.owner_changed and self.last_switch_frame == snapshot.frame_id:
            snapshot.reason = "auto_shuffle"
        self.last_snapshot = snapshot
        return snapshot

    def update(self, tracks, predicted_center, frame_shape):
        self.frame_id += 1
        tracks = list(tracks or [])
        prev_selected_id = self.selected_id
        if not tracks:
            self.lock_age += 1
            self.build_wide_owner_snapshot([], frame_shape=frame_shape, frame_id=self.frame_id)
            return self.selected_id

        active = self.find_active_track(tracks)
        if active is not None:
            self.last_selected_center = self._score_center(active)
            rid = getattr(active, "raw_id", None)
            if rid is not None:
                self.last_selected_raw_id = int(rid)

        if self.manual_lock:
            self.lock_age = 0 if active is not None else self.lock_age + 1
            self.build_wide_owner_snapshot(tracks, frame_shape=frame_shape, frame_id=self.frame_id)
            return self.selected_id

        if self.selection_freeze_left > 0 and self.selection_freeze_id is not None:
            frozen = None
            for tr in tracks:
                if int(getattr(tr, "track_id", -1)) == int(self.selection_freeze_id):
                    frozen = tr
                    break
            self.selection_freeze_left = max(0, self.selection_freeze_left - 1)
            if frozen is not None:
                self.selected_id = int(getattr(frozen, "track_id", self.selection_freeze_id))
                self.last_selected_center = self._score_center(frozen)
                rid = getattr(frozen, "raw_id", None)
                if rid is not None:
                    self.last_selected_raw_id = int(rid)
                self.pending_id = None
                self.pending_count = 0
                self.lock_age += 1
                self.build_wide_owner_snapshot(tracks, frame_shape=frame_shape, frame_id=self.frame_id)
                return self.selected_id
            if self.selection_freeze_left == 0:
                self.clear_freeze()

        candidates = self._eligible_tracks(tracks)
        if not candidates:
            self.lock_age += 1
            self.build_wide_owner_snapshot(tracks, frame_shape=frame_shape, frame_id=self.frame_id)
            return self.selected_id

        anchor = predicted_center or self.last_selected_center

        if self.selected_id is None:
            reference_area = self._median_confirmed_area(candidates)
            best = max(candidates, key=lambda tr: self._score(tr, frame_shape, predicted_center, reference_area=reference_area))
            if self.pending_id == int(best.track_id):
                self.pending_count += 1
            else:
                self.pending_id = int(best.track_id)
                self.pending_count = 1

            required_startup_persist = 3 if self.frame_id <= self.startup_stabilization_frames else 2
            enough_hits = int(getattr(best, 'hits', 0) or 0) >= (self.startup_min_hits if self.frame_id <= self.startup_stabilization_frames else self.min_start_hits)
            if self.pending_count >= required_startup_persist and enough_hits:
                self.selected_id = int(best.track_id)
                self.lock_age = 0
                self.last_switch_frame = self.frame_id
                self.pending_id = None
                self.pending_count = 0
                self.last_selected_center = self._score_center(best)
                rid = getattr(best, "raw_id", None)
                if rid is not None:
                    self.last_selected_raw_id = int(rid)
            if prev_selected_id != self.selected_id and self.selected_id is not None:
                self.owner_switch_seq += 1
            self.build_wide_owner_snapshot(tracks, frame_shape=frame_shape, frame_id=self.frame_id)
            return self.selected_id

        if active is None and anchor is not None:
            close = []
            for tr in candidates:
                dist = self._distance(self._score_center(tr), anchor)
                if dist <= self.reacquire_radius_auto:
                    close.append(tr)

            if close:
                reference_area = self._median_confirmed_area(close)
                best = max(close, key=lambda tr: self._score(tr, frame_shape, predicted_center, reference_area=reference_area))
                self.selected_id = int(best.track_id)
                self.lock_age = 0
                self.last_switch_frame = self.frame_id
                self.pending_id = None
                self.pending_count = 0
                self.last_selected_center = self._score_center(best)
                rid = getattr(best, "raw_id", None)
                if rid is not None:
                    self.last_selected_raw_id = int(rid)
                if prev_selected_id != self.selected_id:
                    self.owner_switch_seq += 1
                self.build_wide_owner_snapshot(tracks, frame_shape=frame_shape, frame_id=self.frame_id)
                return self.selected_id

            self.lock_age += 1
            self.build_wide_owner_snapshot(tracks, frame_shape=frame_shape, frame_id=self.frame_id)
            return self.selected_id

        if active is None:
            self.lock_age += 1
            self.build_wide_owner_snapshot(tracks, frame_shape=frame_shape, frame_id=self.frame_id)
            return self.selected_id

        reference_area = self._median_confirmed_area(candidates)
        current_score = self._score(active, frame_shape, predicted_center, is_current=True, reference_area=reference_area)
        best = max(
            candidates,
            key=lambda tr: self._score(
                tr,
                frame_shape,
                predicted_center,
                is_current=(int(getattr(tr, "track_id", -1)) == int(self.selected_id)),
                reference_area=reference_area,
            ),
        )

        if int(best.track_id) == int(self.selected_id):
            self.pending_id = None
            self.pending_count = 0
            self.lock_age += 1
            self.build_wide_owner_snapshot(tracks, frame_shape=frame_shape, frame_id=self.frame_id)
            return self.selected_id

        best_score = self._score(best, frame_shape, predicted_center, is_current=False, reference_area=reference_area)

        current_conf = float(getattr(active, "confidence", 0.0) or 0.0)
        current_missed = int(getattr(active, "missed_frames", 0) or 0)
        current_center = self._score_center(active)
        best_center = self._score_center(best)

        current_healthy = (
            bool(getattr(active, "is_confirmed", False))
            and current_missed <= self.hard_keep_missed
            and current_conf >= self.hard_keep_conf
        )

        if current_healthy:
            dist_best_to_anchor = self._distance(best_center, anchor)
            dist_current_to_anchor = self._distance(current_center, anchor)
            score_gap = best_score - current_score
            same_raw = (
                self.last_selected_raw_id is not None
                and getattr(best, "raw_id", None) is not None
                and int(getattr(best, "raw_id")) == int(self.last_selected_raw_id)
            )

            startup_guard = self.frame_id <= self.startup_stabilization_frames
            required_gain = self.hard_switch_min_gain + (self.startup_switch_bonus_margin if startup_guard else 0.0)
            if (
                score_gap < required_gain
                and dist_best_to_anchor >= dist_current_to_anchor * (0.88 if startup_guard else 0.92)
                and not same_raw
            ):
                self.pending_id = None
                self.pending_count = 0
                self.lock_age += 1
                self.build_wide_owner_snapshot(tracks, frame_shape=frame_shape, frame_id=self.frame_id)
                return self.selected_id

        current_quality = self.compute_owner_quality(active, frame_shape=frame_shape, reference_area=reference_area)
        current_degraded = (
            (current_missed >= 3)
            or (current_conf < 0.16)
            or (current_missed >= self.proactive_degrade_missed and current_quality <= self.proactive_degrade_quality)
        )
        proactive_degraded = (
            current_missed >= self.proactive_degrade_missed
            or current_quality <= self.proactive_degrade_quality
            or current_conf < max(self.hard_keep_conf, 0.22)
        )

        shuffle_candidate = None
        if proactive_degraded:
            shuffle_candidate = self._find_neighbor_shuffle_candidate(active, candidates, frame_shape, predicted_center)
            if shuffle_candidate is not None:
                best = shuffle_candidate
                best_score = self._score(best, frame_shape, predicted_center, is_current=False, reference_area=reference_area)
                best_center = self._score_center(best)

        near_anchor = self._distance(best_center, anchor) <= min(self.reacquire_radius_auto, self.predicted_dist_px)
        if proactive_degraded and shuffle_candidate is not None:
            near_anchor = True

        margin_scale = 0.35 if current_degraded else 1.0
        if proactive_degraded and shuffle_candidate is not None:
            margin_scale = 0.15
        margin = self.switch_margin * margin_scale
        ratio_ok = best_score > (current_score * (1.01 if current_degraded else 1.08))
        if proactive_degraded and shuffle_candidate is not None:
            ratio_ok = best_score > (current_score * 0.99)
        dwell_ok = self.lock_age >= (1 if current_degraded else self.min_hold_frames)
        if proactive_degraded and shuffle_candidate is not None:
            dwell_ok = self.lock_age >= 0
        cooldown_ok = (self.frame_id - self.last_switch_frame) >= (1 if current_degraded else self.switch_cooldown)
        if proactive_degraded and shuffle_candidate is not None:
            cooldown_ok = (self.frame_id - self.last_switch_frame) >= self.proactive_switch_cooldown

        need_switch = (
            near_anchor
            and best_score > (current_score + margin)
            and ratio_ok
            and dwell_ok
            and cooldown_ok
        )

        if need_switch:
            if self.pending_id == int(best.track_id):
                self.pending_count += 1
            else:
                self.pending_id = int(best.track_id)
                self.pending_count = 1

            required_persist = self.proactive_switch_persist if (proactive_degraded and shuffle_candidate is not None) else (1 if current_degraded else max(3, self.switch_persist))
            if self.frame_id <= self.startup_stabilization_frames and not current_degraded:
                required_persist = max(required_persist, 4)
            if self.pending_count >= required_persist:
                self.selected_id = int(best.track_id)
                self.lock_age = 0
                self.last_switch_frame = self.frame_id
                self.pending_id = None
                self.pending_count = 0
                self.last_selected_center = self._score_center(best)
                rid = getattr(best, "raw_id", None)
                if rid is not None:
                    self.last_selected_raw_id = int(rid)
                self.freeze_to(self.selected_id, self.selection_freeze_frames)
                if prev_selected_id != self.selected_id:
                    self.owner_switch_seq += 1
                self.build_wide_owner_snapshot(tracks, frame_shape=frame_shape, frame_id=self.frame_id)
                return self.selected_id
        else:
            self.pending_id = None
            self.pending_count = 0

        self.lock_age += 1
        if prev_selected_id != self.selected_id and self.selected_id is not None:
            self.owner_switch_seq += 1
        self.build_wide_owner_snapshot(tracks, frame_shape=frame_shape, frame_id=self.frame_id)
        return self.selected_id
