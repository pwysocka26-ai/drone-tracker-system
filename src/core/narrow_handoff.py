from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import math


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass
class EdgeLimitStatus:
    active: bool = False
    severity: float = 0.0
    duration_frames: int = 0
    escape_possible: bool = True


@dataclass
class GeometryStatus:
    score: float = 0.0
    ok: bool = False
    break_frames: int = 0


@dataclass
class NarrowHandoffDecision:
    accepted_precheck: bool = False
    commit_switch: bool = False
    candidate_id: Optional[int] = None
    reject_reason: str = "NONE"
    quality_score: float = 0.0
    geometry_score: float = 0.0
    quality_ok: bool = False
    geometry_ok: bool = False
    cooldown_ok: bool = False
    current_lock_should_break: bool = False


class NarrowRuntimeState:
    def __init__(self) -> None:
        self.lock_state: str = "IDLE"
        self.narrow_owner_id: Optional[int] = None
        self.pending_owner_id: Optional[int] = None
        self.pending_owner_stable_frames: int = 0
        self.pending_owner_seen_frames: int = 0
        self.last_accepted_switch_frame: int = -10**9
        self.current_geometry_score: float = 0.0
        self.current_lock_confidence: float = 0.0
        self.edge_limit_active: bool = False
        self.edge_limit_severity: float = 0.0
        self.edge_limit_duration_frames: int = 0
        self.center_lock_on: bool = False
        self.geometry_break_frames: int = 0
        self.forced_exit_reason: str = ""
        self.last_reject_reason: str = "NONE"
        self.last_geometry_score: float = 0.0
        self.last_quality_score: float = 0.0
        self.last_wide_owner_id: Optional[int] = None
        self.last_decision: Optional[NarrowHandoffDecision] = None
        self.owner_track_present: bool = False
        self.owner_missed_frames: int = 9999
        self.narrow_blind_streak: int = 0
        self.last_release_reason: str = ""
        self.post_commit_grace_frames_remaining: int = 0
        self.sticky_owner_id: Optional[int] = None
        self.sticky_hold_frames_remaining: int = 0
        self.soft_loss_streak: int = 0

    def reset(self) -> None:
        self.__init__()


class NarrowHandoffGate:
    def __init__(
        self,
        min_quality: float = 0.46,
        min_track_age: int = 2,
        max_missed: int = 1,
        geometry_threshold: float = 0.42,
        switch_cooldown_frames: int = 12,
        current_lock_keep_threshold: float = 0.70,
        center_lock_edge_exit_frames: int = 4,
        center_lock_geometry_break_frames: int = 3,
        center_lock_geometry_break_threshold: float = 0.38,
        large_target_quality_relax: float = 0.18,
        large_target_geometry_boost: float = 0.80,
        large_target_keep_current_geometry: float = 0.82,
        large_target_min_switch_geometry: float = 0.88,
    ) -> None:
        self.min_quality = float(min_quality)
        self.min_track_age = int(min_track_age)
        self.max_missed = int(max_missed)
        self.geometry_threshold = float(geometry_threshold)
        self.switch_cooldown_frames = int(switch_cooldown_frames)
        self.current_lock_keep_threshold = float(current_lock_keep_threshold)
        self.center_lock_edge_exit_frames = int(center_lock_edge_exit_frames)
        self.center_lock_geometry_break_frames = int(center_lock_geometry_break_frames)
        self.center_lock_geometry_break_threshold = float(center_lock_geometry_break_threshold)
        self.large_target_quality_relax = float(large_target_quality_relax)
        self.large_target_geometry_boost = float(large_target_geometry_boost)
        self.large_target_keep_current_geometry = float(large_target_keep_current_geometry)
        self.large_target_min_switch_geometry = float(large_target_min_switch_geometry)

    @staticmethod
    def _bbox_similarity(a: Optional[BBox], b: Optional[BBox]) -> float:
        if a is None or b is None:
            return 0.0
        aw = max(1.0, float(a[2]) - float(a[0]))
        ah = max(1.0, float(a[3]) - float(a[1]))
        bw = max(1.0, float(b[2]) - float(b[0]))
        bh = max(1.0, float(b[3]) - float(b[1]))
        dw = abs(aw - bw) / max(aw, bw)
        dh = abs(ah - bh) / max(ah, bh)
        return max(0.0, 1.0 - 0.5 * (dw + dh))

    @staticmethod
    def _distance(a: Optional[Point], b: Optional[Point]) -> float:
        if a is None or b is None:
            return float("inf")
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def _geometry_score(self, runtime: NarrowRuntimeState, wide_snapshot: object, handoff_state: object) -> float:
        wide_center = getattr(wide_snapshot, "center_xy", None)
        wide_bbox = getattr(wide_snapshot, "bbox_xyxy", None)
        ref_center = getattr(handoff_state, "last_good_center", None) or wide_center
        ref_bbox = getattr(handoff_state, "last_good_bbox", None) or wide_bbox
        dist = self._distance(wide_center, ref_center)
        dist_score = max(0.0, 1.0 - dist / 220.0)
        bbox_score = self._bbox_similarity(wide_bbox, ref_bbox)
        temporal_score = 1.0
        if runtime.narrow_owner_id is not None and getattr(wide_snapshot, "track_id", None) != runtime.narrow_owner_id:
            temporal_score = 0.90
        return max(0.0, min(1.0, 0.55 * dist_score + 0.35 * bbox_score + 0.10 * temporal_score))

    def evaluate(
        self,
        runtime: NarrowRuntimeState,
        wide_snapshot: object,
        handoff_state: object,
        frame_id: int,
    ) -> NarrowHandoffDecision:
        d = NarrowHandoffDecision()
        d.candidate_id = getattr(wide_snapshot, "track_id", None)
        d.quality_score = float(getattr(wide_snapshot, "quality_score", 0.0) or 0.0)
        d.geometry_score = self._geometry_score(runtime, wide_snapshot, handoff_state)
        track_age = getattr(wide_snapshot, "track_age", 0)
        if track_age is None:
            track_age = 0
        missed_count = getattr(wide_snapshot, "missed_count", 99)
        if missed_count is None:
            missed_count = 99
        is_large_target = bool(getattr(wide_snapshot, "is_large_target", False))
        is_huge_outlier = bool(getattr(wide_snapshot, "is_huge_outlier", False))
        relaxed_min_quality = self.min_quality
        if is_large_target and not is_huge_outlier:
            relaxed_min_quality = max(0.24, self.min_quality - self.large_target_quality_relax)
        d.quality_ok = (
            bool(getattr(wide_snapshot, "valid", False))
            and d.quality_score >= relaxed_min_quality
            and int(track_age) >= self.min_track_age
            and int(missed_count) <= (self.max_missed + (1 if is_large_target and not is_huge_outlier else 0))
        )
        d.geometry_ok = d.geometry_score >= self.geometry_threshold
        if is_large_target and not is_huge_outlier and d.geometry_score >= self.large_target_geometry_boost:
            d.geometry_ok = True
            d.quality_ok = d.quality_ok or d.quality_score >= max(0.24, relaxed_min_quality - 0.06)
        d.cooldown_ok = (frame_id - runtime.last_accepted_switch_frame) >= self.switch_cooldown_frames

        current_lock_still_good = (
            runtime.narrow_owner_id is not None
            and runtime.owner_track_present
            and runtime.owner_missed_frames <= 1
            and runtime.narrow_blind_streak <= 0
            and runtime.center_lock_on
            and runtime.current_geometry_score >= self.current_lock_keep_threshold
            and not runtime.edge_limit_active
        )
        proactive_shuffle = bool(
            getattr(wide_snapshot, 'owner_changed', False)
            and str(getattr(wide_snapshot, 'reason', '')) == 'auto_shuffle'
        )
        if is_large_target and not is_huge_outlier and runtime.center_lock_on:
            if runtime.current_geometry_score >= self.large_target_keep_current_geometry and d.geometry_score < self.large_target_min_switch_geometry:
                d.reject_reason = 'CURRENT_LOCK_STILL_GOOD'
                return d
        proactive_shuffle_ok = proactive_shuffle and (
            runtime.owner_missed_frames >= 1
            or runtime.current_geometry_score < (self.current_lock_keep_threshold + 0.12)
            or d.quality_score >= (self.current_lock_keep_threshold + 0.08)
        )
        d.current_lock_should_break = (
            runtime.center_lock_on
            and runtime.edge_limit_active
            and runtime.edge_limit_duration_frames >= self.center_lock_edge_exit_frames
            and runtime.geometry_break_frames >= self.center_lock_geometry_break_frames
            and runtime.current_geometry_score < self.center_lock_geometry_break_threshold
        )

        if not bool(getattr(wide_snapshot, "valid", False)):
            d.reject_reason = "WIDE_INVALID"
            return d
        if d.candidate_id is None:
            d.reject_reason = "NO_CANDIDATE"
            return d
        if runtime.narrow_owner_id is not None and int(d.candidate_id) == int(runtime.narrow_owner_id):
            d.reject_reason = "SAME_AS_CURRENT"
            return d
        if not d.quality_ok:
            d.reject_reason = "LOW_QUALITY"
            return d
        if bool(getattr(wide_snapshot, "is_huge_outlier", False)):
            d.reject_reason = "HUGE_OUTLIER"
            return d
        if not d.geometry_ok:
            d.reject_reason = "GEOMETRY_MISMATCH"
            return d
        if is_large_target and not is_huge_outlier and d.geometry_score < self.large_target_min_switch_geometry and runtime.narrow_owner_id is not None:
            d.reject_reason = "GEOMETRY_MISMATCH"
            return d
        if not d.cooldown_ok:
            d.reject_reason = "SWITCH_COOLDOWN"
            return d
        if current_lock_still_good and not d.current_lock_should_break and not proactive_shuffle_ok:
            d.reject_reason = "CURRENT_LOCK_STILL_GOOD"
            return d

        d.accepted_precheck = True
        d.reject_reason = "NONE"
        return d


class NarrowHandoffController:
    def __init__(
        self,
        min_stable_frames: int = 4,
        hold_frames: int = 10,
        soft_active_max_missed: int = 4,
        reacquire_radius: float = 165.0,
        post_commit_grace_frames: int = 8,
        sticky_hold_frames: int = 12,
        pending_keepalive_frames: int = 2,
        tracking_keep_quality: float = 0.42,
        pending_cancel_quality: float = 0.38,
        **gate_kwargs,
    ) -> None:
        self.runtime = NarrowRuntimeState()
        self.min_stable_frames = int(min_stable_frames)
        self.hold_frames = int(hold_frames)
        self.soft_active_max_missed = int(soft_active_max_missed)
        self.reacquire_radius = float(reacquire_radius)
        self.post_commit_grace_frames = int(post_commit_grace_frames)
        self.sticky_hold_frames = int(sticky_hold_frames)
        self.pending_keepalive_frames = int(pending_keepalive_frames)
        self.tracking_keep_quality = float(tracking_keep_quality)
        self.pending_cancel_quality = float(pending_cancel_quality)
        self.gate = NarrowHandoffGate(**gate_kwargs)

    def reset(self) -> None:
        self.runtime.reset()

    def force_owner(self, track_id: Optional[int], frame_id: int) -> None:
        self.runtime.narrow_owner_id = None if track_id is None else int(track_id)
        self.runtime.pending_owner_id = None
        self.runtime.pending_owner_seen_frames = 0
        self.runtime.pending_owner_stable_frames = 0
        self.runtime.lock_state = "TRACKING" if track_id is not None else "IDLE"
        self.runtime.last_accepted_switch_frame = int(frame_id)
        self.runtime.forced_exit_reason = ""
        self.runtime.post_commit_grace_frames_remaining = self.post_commit_grace_frames if track_id is not None else 0
        self.runtime.sticky_owner_id = None if track_id is None else int(track_id)
        self.runtime.sticky_hold_frames_remaining = self.sticky_hold_frames if track_id is not None else 0
        self.runtime.soft_loss_streak = 0

    def note_track_state(self, owner_track: Optional[object], det_track_count: int) -> None:
        self.runtime.owner_track_present = owner_track is not None
        if owner_track is not None:
            self.runtime.owner_missed_frames = int(getattr(owner_track, "missed_frames", 0) or 0)
        else:
            self.runtime.owner_missed_frames = 9999

        if self.runtime.post_commit_grace_frames_remaining > 0:
            self.runtime.post_commit_grace_frames_remaining -= 1
        if self.runtime.sticky_hold_frames_remaining > 0:
            self.runtime.sticky_hold_frames_remaining -= 1

        blind_watchdog_enabled = (
            self.runtime.lock_state == "TRACKING"
            and self.runtime.narrow_owner_id is not None
            and self.runtime.post_commit_grace_frames_remaining <= 0
        )
        if not blind_watchdog_enabled:
            self.runtime.narrow_blind_streak = 0
            self.runtime.soft_loss_streak = 0
            return

        blind_input = bool(int(det_track_count) <= 0 or owner_track is None or self.runtime.owner_missed_frames > 1)
        if blind_input:
            self.runtime.narrow_blind_streak += 1
            self.runtime.soft_loss_streak += 1
        else:
            self.runtime.narrow_blind_streak = 0
            self.runtime.soft_loss_streak = 0

    def force_reacquire(self, reason: str, frame_id: Optional[int] = None) -> None:
        self.runtime.center_lock_on = False
        self.runtime.lock_state = "REACQUIRE"
        self.runtime.forced_exit_reason = str(reason)
        self.runtime.last_release_reason = str(reason)
        self.runtime.narrow_owner_id = None
        self.runtime.pending_owner_id = None
        self.runtime.pending_owner_seen_frames = 0
        self.runtime.pending_owner_stable_frames = 0
        self.runtime.current_lock_confidence = 0.0
        self.runtime.current_geometry_score = 0.0
        self.runtime.geometry_break_frames = 0
        self.runtime.post_commit_grace_frames_remaining = 0
        self.runtime.narrow_blind_streak = 0
        self.runtime.soft_loss_streak = 0
        if frame_id is not None:
            self.runtime.last_accepted_switch_frame = min(self.runtime.last_accepted_switch_frame, int(frame_id) - self.gate.switch_cooldown_frames)

    def _restore_sticky_owner(self, wide_snapshot: object) -> None:
        candidate_id = getattr(wide_snapshot, "track_id", None)
        quality = float(getattr(wide_snapshot, "quality_score", 0.0) or 0.0)
        if (
            self.runtime.narrow_owner_id is None
            and self.runtime.sticky_owner_id is not None
            and self.runtime.sticky_hold_frames_remaining > 0
            and candidate_id is not None
            and int(candidate_id) == int(self.runtime.sticky_owner_id)
            and quality >= self.tracking_keep_quality
        ):
            self.runtime.narrow_owner_id = int(self.runtime.sticky_owner_id)
            self.runtime.lock_state = "TRACKING"
            self.runtime.pending_owner_id = None
            self.runtime.pending_owner_seen_frames = 0
            self.runtime.pending_owner_stable_frames = 0

    def _should_keep_pending_alive(self, decision: NarrowHandoffDecision, wide_snapshot: object) -> bool:
        if self.runtime.pending_owner_id is None:
            return False
        candidate_id = getattr(wide_snapshot, "track_id", None)
        quality = float(getattr(wide_snapshot, "quality_score", 0.0) or 0.0)
        return (
            candidate_id is not None
            and int(candidate_id) == int(self.runtime.pending_owner_id)
            and decision.geometry_score >= self.gate.geometry_threshold
            and quality >= self.pending_cancel_quality
            and self.runtime.pending_owner_seen_frames < (self.min_stable_frames + self.pending_keepalive_frames)
        )

    def get_active_track(self, tracks: Sequence[object], max_missed: Optional[int] = None) -> Optional[object]:
        owner_id = self.runtime.narrow_owner_id
        if owner_id is None and self.runtime.sticky_owner_id is not None and self.runtime.sticky_hold_frames_remaining > 0:
            owner_id = self.runtime.sticky_owner_id
        if owner_id is None:
            return None
        allowed_missed = self.soft_active_max_missed if max_missed is None else int(max_missed)
        for tr in tracks or []:
            if int(getattr(tr, "track_id", -1)) == int(owner_id):
                if int(getattr(tr, "missed_frames", 0) or 0) <= allowed_missed:
                    return tr
                return None
        return None

    def update_wide(self, wide_snapshot: object, handoff_state: object, frame_id: int) -> NarrowHandoffDecision:
        self.runtime.last_wide_owner_id = getattr(wide_snapshot, "track_id", None)
        self._restore_sticky_owner(wide_snapshot)
        decision = self.gate.evaluate(self.runtime, wide_snapshot, handoff_state, frame_id)
        self.runtime.last_decision = decision
        self.runtime.last_reject_reason = decision.reject_reason
        self.runtime.last_geometry_score = decision.geometry_score
        self.runtime.last_quality_score = decision.quality_score

        if decision.reject_reason == "SAME_AS_CURRENT" and self.runtime.narrow_owner_id is not None:
            self.runtime.lock_state = "TRACKING"
            self.runtime.center_lock_on = self.runtime.center_lock_on
            self.runtime.pending_owner_id = None
            self.runtime.pending_owner_seen_frames = 0
            self.runtime.pending_owner_stable_frames = 0
            if decision.quality_score >= self.tracking_keep_quality:
                self.runtime.sticky_owner_id = self.runtime.narrow_owner_id
                self.runtime.sticky_hold_frames_remaining = self.sticky_hold_frames
            return decision

        if decision.current_lock_should_break:
            self.runtime.center_lock_on = False
            self.runtime.forced_exit_reason = "EDGE_GEOMETRY_BREAK"
            self.runtime.lock_state = "REACQUIRE"
            self.runtime.narrow_owner_id = None

        candidate_id = decision.candidate_id
        if decision.accepted_precheck and candidate_id is not None:
            if self.runtime.pending_owner_id == int(candidate_id):
                self.runtime.pending_owner_seen_frames += 1
                if decision.quality_ok and decision.geometry_ok:
                    self.runtime.pending_owner_stable_frames += 1
            else:
                self.runtime.pending_owner_id = int(candidate_id)
                self.runtime.pending_owner_seen_frames = 1
                self.runtime.pending_owner_stable_frames = 1
            self.runtime.lock_state = "PENDING_HANDOFF"

            if self.runtime.pending_owner_stable_frames >= self.min_stable_frames:
                self.runtime.narrow_owner_id = int(candidate_id)
                self.runtime.pending_owner_id = None
                self.runtime.pending_owner_seen_frames = 0
                self.runtime.pending_owner_stable_frames = 0
                self.runtime.last_accepted_switch_frame = int(frame_id)
                self.runtime.lock_state = "TRACKING"
                self.runtime.forced_exit_reason = ""
                self.runtime.post_commit_grace_frames_remaining = self.post_commit_grace_frames
                self.runtime.narrow_blind_streak = 0
                self.runtime.soft_loss_streak = 0
                self.runtime.sticky_owner_id = int(candidate_id)
                self.runtime.sticky_hold_frames_remaining = self.sticky_hold_frames
                decision.commit_switch = True
        else:
            if self.runtime.lock_state == "PENDING_HANDOFF":
                if self._should_keep_pending_alive(decision, wide_snapshot):
                    self.runtime.pending_owner_seen_frames += 1
                else:
                    self.runtime.pending_owner_id = None
                    self.runtime.pending_owner_seen_frames = 0
                    self.runtime.pending_owner_stable_frames = 0
                    self.runtime.lock_state = "TRACKING" if self.runtime.narrow_owner_id is not None else "IDLE"
            elif self.runtime.narrow_owner_id is None and self.runtime.lock_state not in ("REACQUIRE", "HOLD_LAST_GOOD"):
                if self.runtime.sticky_owner_id is not None and self.runtime.sticky_hold_frames_remaining > 0:
                    self.runtime.narrow_owner_id = int(self.runtime.sticky_owner_id)
                    self.runtime.lock_state = "TRACKING"
                else:
                    self.runtime.lock_state = "IDLE"
        return decision

    def update_measurements(
        self,
        frame_shape: Sequence[int],
        crop_rect: Optional[Sequence[float]],
        display_center: Optional[Point],
        center_lock: bool,
        real_pan_err: float,
        real_tilt_err: float,
        zoom_limited: bool,
    ) -> EdgeLimitStatus:
        self.runtime.center_lock_on = bool(center_lock)
        geom = max(0.0, 1.0 - math.hypot(float(real_pan_err), float(real_tilt_err)) / 260.0)
        self.runtime.current_geometry_score = geom
        self.runtime.current_lock_confidence = 0.75 * self.runtime.current_lock_confidence + 0.25 * geom
        if geom < self.gate.center_lock_geometry_break_threshold:
            self.runtime.geometry_break_frames += 1
        else:
            self.runtime.geometry_break_frames = 0

        edge_active = False
        severity = 0.0
        if frame_shape is not None and crop_rect is not None and display_center is not None:
            fh, fw = frame_shape[:2]
            x1, y1, x2, y2 = [float(v) for v in crop_rect]
            cx, cy = float(display_center[0]), float(display_center[1])
            crop_w = max(1.0, x2 - x1)
            crop_h = max(1.0, y2 - y1)
            nx = (cx - x1) / crop_w
            ny = (cy - y1) / crop_h
            target_near_edge = nx < 0.16 or nx > 0.84 or ny < 0.16 or ny > 0.84
            crop_hits_frame_edge = x1 <= 1.0 or y1 <= 1.0 or x2 >= (fw - 1.0) or y2 >= (fh - 1.0)
            severity = max(abs(nx - 0.5) * 2.0, abs(ny - 0.5) * 2.0)
            edge_active = bool(target_near_edge and crop_hits_frame_edge and zoom_limited)

        self.runtime.edge_limit_active = edge_active
        self.runtime.edge_limit_severity = severity
        if edge_active:
            self.runtime.edge_limit_duration_frames += 1
        else:
            self.runtime.edge_limit_duration_frames = 0

        if (
            self.runtime.center_lock_on
            and self.runtime.edge_limit_active
            and self.runtime.edge_limit_duration_frames >= self.gate.center_lock_edge_exit_frames
            and self.runtime.geometry_break_frames >= self.gate.center_lock_geometry_break_frames
            and self.runtime.current_geometry_score < self.gate.center_lock_geometry_break_threshold
        ):
            self.runtime.center_lock_on = False
            self.runtime.forced_exit_reason = "EDGE_GEOMETRY_BREAK"
            self.runtime.lock_state = "REACQUIRE"
            self.runtime.narrow_owner_id = None
            self.runtime.pending_owner_id = None
            self.runtime.pending_owner_seen_frames = 0
            self.runtime.pending_owner_stable_frames = 0

        return EdgeLimitStatus(
            active=self.runtime.edge_limit_active,
            severity=self.runtime.edge_limit_severity,
            duration_frames=self.runtime.edge_limit_duration_frames,
            escape_possible=not self.runtime.edge_limit_active,
        )
