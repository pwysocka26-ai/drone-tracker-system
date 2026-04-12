from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Optional, Sequence
import math


STATE_IDLE = "IDLE"
STATE_WIDE_TRACKING = "WIDE_TRACKING"
STATE_HANDOFF_READY = "HANDOFF_READY"
STATE_ACQUIRE = "ACQUIRE"
STATE_REFINE = "REFINE"
STATE_LOCKED = "LOCKED"
STATE_HOLD = "HOLD"
STATE_REACQUIRE = "REACQUIRE"
STATE_LOST = "LOST"


@dataclass
class WideReadinessSnapshot:
    track_id: int | None = None
    conf: float = 0.0
    hits: int = 0
    missed_frames: int = 0
    center: tuple[float, float] | None = None
    bbox: tuple[float, float, float, float] | None = None
    center_jitter_px: float = 0.0
    bbox_size_jitter: float = 0.0
    velocity_consistency: float = 0.0
    readiness_score: float = 0.0
    ready_streak: int = 0
    is_ready: bool = False


@dataclass
class LocalLockContext:
    selected_id: int | None = None
    steering_target_id: int | None = None
    state: str = STATE_IDLE
    raw_center: tuple[float, float] | None = None
    raw_bbox: tuple[float, float, float, float] | None = None
    refined_center: tuple[float, float] | None = None
    refined_bbox: tuple[float, float, float, float] | None = None
    predicted_center: tuple[float, float] | None = None
    lock_confidence: float = 0.0
    local_lock_score: float = 0.0
    handoff_ready_score: float = 0.0
    handoff_ready_streak: int = 0
    acquire_frames: int = 0
    refine_frames: int = 0
    lock_frames: int = 0
    hold_frames: int = 0
    reacquire_frames: int = 0
    jump_rejected: bool = False
    last_jump_px: float = 0.0
    active_track_missed: int = 999
    measurement_support: float = 0.0
    identity_desync_frames: int = 0
    identity_desync: bool = False
    ui_truthful_lock: bool = False
    last_good_center: tuple[float, float] | None = None
    last_good_bbox: tuple[float, float, float, float] | None = None
    last_good_zoom: float = 2.8
    lock_loss_reason: str | None = None


class RefinedAnchorFilter:
    def __init__(self, position_alpha: float = 0.42, bbox_alpha: float = 0.35, reject_jump_px: float = 90.0, soften_jump_px: float = 36.0):
        self.position_alpha = float(position_alpha)
        self.bbox_alpha = float(bbox_alpha)
        self.reject_jump_px = float(reject_jump_px)
        self.soften_jump_px = float(soften_jump_px)
        self.refined_center: tuple[float, float] | None = None
        self.refined_bbox: tuple[float, float, float, float] | None = None
        self.velocity: tuple[float, float] = (0.0, 0.0)

    def reset(self) -> None:
        self.refined_center = None
        self.refined_bbox = None
        self.velocity = (0.0, 0.0)

    def predict(self) -> tuple[float, float] | None:
        if self.refined_center is None:
            return None
        return (
            self.refined_center[0] + self.velocity[0],
            self.refined_center[1] + self.velocity[1],
        )

    def update(self, measured_center, measured_bbox, confidence: float, state: str):
        predicted = self.predict()
        if measured_center is None:
            return {
                "refined_center": self.refined_center,
                "refined_bbox": self.refined_bbox,
                "predicted_center": predicted,
                "jump_px": 0.0,
                "jump_rejected": False,
                "anchor_confidence": 0.0,
            }

        if self.refined_center is None:
            self.refined_center = tuple(measured_center)
            self.refined_bbox = tuple(measured_bbox) if measured_bbox is not None else None
            return {
                "refined_center": self.refined_center,
                "refined_bbox": self.refined_bbox,
                "predicted_center": self.refined_center,
                "jump_px": 0.0,
                "jump_rejected": False,
                "anchor_confidence": float(confidence),
            }

        base = predicted if predicted is not None else self.refined_center
        dx = float(measured_center[0]) - float(base[0])
        dy = float(measured_center[1]) - float(base[1])
        jump_px = math.hypot(dx, dy)

        state_alpha = self.position_alpha
        if state == STATE_ACQUIRE:
            state_alpha = 0.56
        elif state == STATE_REFINE:
            state_alpha = 0.46
        elif state == STATE_LOCKED:
            state_alpha = 0.32
        elif state == STATE_REACQUIRE:
            state_alpha = 0.40

        jump_rejected = False
        if jump_px > self.reject_jump_px and state in (STATE_LOCKED, STATE_REFINE):
            jump_rejected = True
            next_center = base
        elif jump_px > self.soften_jump_px:
            state_alpha *= 0.45
            next_center = (
                base[0] + state_alpha * dx,
                base[1] + state_alpha * dy,
            )
        else:
            next_center = (
                base[0] + state_alpha * dx,
                base[1] + state_alpha * dy,
            )

        vx = next_center[0] - self.refined_center[0]
        vy = next_center[1] - self.refined_center[1]
        self.velocity = (0.72 * self.velocity[0] + 0.28 * vx, 0.72 * self.velocity[1] + 0.28 * vy)
        self.refined_center = next_center

        if measured_bbox is not None:
            if self.refined_bbox is None:
                self.refined_bbox = tuple(measured_bbox)
            else:
                ox1, oy1, ox2, oy2 = self.refined_bbox
                nx1, ny1, nx2, ny2 = measured_bbox
                a = self.bbox_alpha
                self.refined_bbox = (
                    ox1 + a * (nx1 - ox1),
                    oy1 + a * (ny1 - oy1),
                    ox2 + a * (nx2 - ox2),
                    oy2 + a * (ny2 - oy2),
                )

        anchor_conf = max(0.0, min(1.0, float(confidence) * (0.4 if jump_rejected else 1.0) * max(0.0, 1.0 - jump_px / max(1.0, self.reject_jump_px * 1.5))))
        return {
            "refined_center": self.refined_center,
            "refined_bbox": self.refined_bbox,
            "predicted_center": predicted,
            "jump_px": jump_px,
            "jump_rejected": jump_rejected,
            "anchor_confidence": anchor_conf,
        }


class LockPipeline:
    def __init__(self, config: dict):
        self.cfg = config or {}
        hcfg = self.cfg.get("handoff") or {}
        self.ready_frames = int(hcfg.get("handoff_ready_frames", 5))
        self.hold_frames = int(hcfg.get("handoff_hold_frames", 10))
        self.desync_guard_frames = int(hcfg.get("desync_guard_frames", 2))
        self.active_missed_soft_frames = int(hcfg.get("active_missed_soft_frames", 2))
        self.active_missed_hard_frames = int(hcfg.get("active_missed_hard_frames", 5))
        self.ui_truth_lock_support_min = float(hcfg.get("ui_truth_lock_support_min", 0.55))
        self.ui_truth_max_missed = int(hcfg.get("ui_truth_max_missed", 2))
        self.reacquire_radius = float(hcfg.get("handoff_reacquire_radius", 165.0))
        self.context = LocalLockContext()
        self.readiness = WideReadinessSnapshot()
        self.anchor_filter = RefinedAnchorFilter(
            position_alpha=float(hcfg.get("anchor_position_alpha", 0.42)),
            bbox_alpha=float(hcfg.get("anchor_bbox_alpha", 0.35)),
            reject_jump_px=float(hcfg.get("anchor_reject_jump_px", 90.0)),
            soften_jump_px=float(hcfg.get("anchor_soften_jump_px", 36.0)),
        )
        self._center_hist: Deque[tuple[float, float]] = deque(maxlen=8)
        self._size_hist: Deque[tuple[float, float]] = deque(maxlen=8)

    def reset(self) -> None:
        self.context = LocalLockContext()
        self.readiness = WideReadinessSnapshot()
        self.anchor_filter.reset()
        self._center_hist.clear()
        self._size_hist.clear()

    @staticmethod
    def _bbox_size(bbox):
        x1, y1, x2, y2 = bbox
        return (max(1.0, x2 - x1), max(1.0, y2 - y1))

    @staticmethod
    def _distance(a, b):
        if a is None or b is None:
            return float("inf")
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def _update_wide_readiness(self, selected_track, frame_shape):
        r = self.readiness
        if selected_track is None:
            r.ready_streak = 0
            r.is_ready = False
            r.readiness_score = 0.0
            return r

        center = tuple(float(v) for v in selected_track.center_xy)
        bbox = tuple(float(v) for v in selected_track.bbox_xyxy)
        size = self._bbox_size(bbox)
        self._center_hist.append(center)
        self._size_hist.append(size)

        center_jitter = 0.0
        if len(self._center_hist) >= 2:
            dists = [self._distance(self._center_hist[i - 1], self._center_hist[i]) for i in range(1, len(self._center_hist))]
            center_jitter = sum(dists) / max(1, len(dists))

        size_jitter = 0.0
        if len(self._size_hist) >= 2:
            diffs = []
            for i in range(1, len(self._size_hist)):
                pw, ph = self._size_hist[i - 1]
                cw, ch = self._size_hist[i]
                diffs.append(0.5 * (abs(cw - pw) / max(1.0, pw) + abs(ch - ph) / max(1.0, ph)))
            size_jitter = sum(diffs) / max(1, len(diffs))

        conf = float(getattr(selected_track, "confidence", 0.0) or 0.0)
        hits = int(getattr(selected_track, "hits", 0) or 0)
        missed = int(getattr(selected_track, "missed_frames", 0) or 0)
        h, w = frame_shape[:2]
        edge_margin = min(center[0], center[1], w - center[0], h - center[1])

        score = 0.0
        score += min(1.0, conf / 0.25) * 0.35
        score += min(1.0, hits / 5.0) * 0.20
        score += max(0.0, 1.0 - center_jitter / 16.0) * 0.25
        score += max(0.0, 1.0 - size_jitter / 0.18) * 0.10
        score += max(0.0, min(1.0, edge_margin / 30.0)) * 0.10
        if missed > 0:
            score *= 0.2

        ready = score >= 0.72 and missed == 0 and (hits >= 3 or bool(getattr(selected_track, "is_confirmed", False)))
        r.track_id = int(getattr(selected_track, "track_id", -1))
        r.conf = conf
        r.hits = hits
        r.missed_frames = missed
        r.center = center
        r.bbox = bbox
        r.center_jitter_px = center_jitter
        r.bbox_size_jitter = size_jitter
        r.velocity_consistency = max(0.0, 1.0 - center_jitter / 24.0)
        r.readiness_score = score
        r.ready_streak = r.ready_streak + 1 if ready else 0
        r.is_ready = ready and r.ready_streak >= self.ready_frames
        return r

    def _choose_measurement_track(self, selected_track, tracks):
        state = self.context.state
        if selected_track is not None and self.readiness.is_ready:
            return selected_track

        anchor = self.context.last_good_center or self.context.refined_center or self.readiness.center
        if anchor is None:
            return None

        best = None
        best_score = -1e9
        for tr in tracks or []:
            center = tuple(float(v) for v in tr.center_xy)
            dist = self._distance(center, anchor)
            if dist > self.reacquire_radius:
                continue
            conf = float(getattr(tr, "confidence", 0.0) or 0.0)
            score = conf * 4.0 - dist / max(1.0, self.reacquire_radius)
            if state == STATE_LOCKED and self.context.last_good_bbox is not None:
                bw, bh = self._bbox_size(tuple(float(v) for v in tr.bbox_xyxy))
                lw, lh = self._bbox_size(self.context.last_good_bbox)
                size_penalty = 0.5 * (abs(bw - lw) / max(1.0, lw) + abs(bh - lh) / max(1.0, lh))
                score -= size_penalty * 2.0
            if best is None or score > best_score:
                best = tr
                best_score = score
        return best

    def _compute_local_lock_score(self, measurement_track, anchor_confidence: float, jump_px: float):
        if measurement_track is None or self.context.refined_center is None:
            return 0.0
        center = tuple(float(v) for v in measurement_track.center_xy)
        dist = self._distance(center, self.context.refined_center)
        conf = float(getattr(measurement_track, "confidence", 0.0) or 0.0)
        missed = int(getattr(measurement_track, "missed_frames", 0) or 0)
        score = 0.0
        score += max(0.0, 1.0 - dist / 24.0) * 0.45
        score += min(1.0, conf / 0.25) * 0.20
        score += anchor_confidence * 0.25
        score += max(0.0, 1.0 - jump_px / 48.0) * 0.10
        if missed > 0:
            score *= 0.3
        return max(0.0, min(1.0, score))

    def _compute_measurement_support(self, measurement_track, selected_track):
        if measurement_track is None:
            return 0.0
        conf = float(getattr(measurement_track, "confidence", 0.0) or 0.0)
        missed = int(getattr(measurement_track, "missed_frames", 0) or 0)
        bbox = getattr(measurement_track, "bbox_xyxy", None)
        area = 0.0
        if bbox is not None:
            bw, bh = self._bbox_size(tuple(float(v) for v in bbox))
            area = bw * bh
        area_term = min(1.0, area / 1600.0)
        support = 0.55 * min(1.0, conf / 0.25) + 0.30 * area_term + 0.15 * max(0.0, 1.0 - min(missed / 2.0, 1.0))
        if selected_track is not None and getattr(measurement_track, "track_id", None) == getattr(selected_track, "track_id", None):
            support += 0.08
        return max(0.0, min(1.0, support))

    def _update_desync(self, selected_track, measurement_track):
        c = self.context
        if selected_track is None or measurement_track is None:
            c.identity_desync_frames = 0
            c.identity_desync = False
            return
        if getattr(selected_track, "track_id", None) == getattr(measurement_track, "track_id", None):
            c.identity_desync_frames = 0
            c.identity_desync = False
        else:
            c.identity_desync_frames += 1
            c.identity_desync = c.identity_desync_frames >= self.desync_guard_frames

    def _transition_state(self, measurement_track):
        c = self.context
        prev = c.state
        if prev == STATE_IDLE:
            return STATE_WIDE_TRACKING if c.selected_id is not None else STATE_IDLE
        if prev == STATE_WIDE_TRACKING:
            if self.readiness.is_ready:
                return STATE_HANDOFF_READY
            return STATE_WIDE_TRACKING
        if prev == STATE_HANDOFF_READY:
            return STATE_ACQUIRE if measurement_track is not None else STATE_WIDE_TRACKING
        if prev == STATE_ACQUIRE:
            if measurement_track is None:
                return STATE_HOLD
            if c.local_lock_score >= 0.58:
                return STATE_REFINE
            return STATE_ACQUIRE
        if prev == STATE_REFINE:
            if measurement_track is None:
                return STATE_HOLD
            if c.local_lock_score >= 0.78:
                return STATE_LOCKED
            return STATE_REFINE
        if prev == STATE_LOCKED:
            if measurement_track is None:
                c.lock_loss_reason = "missing_measurement"
                return STATE_HOLD
            if c.active_track_missed >= self.active_missed_hard_frames:
                c.lock_loss_reason = "active_track_missed"
                return STATE_REACQUIRE
            if c.identity_desync:
                c.lock_loss_reason = "identity_desync"
                return STATE_REACQUIRE
            if c.jump_rejected:
                c.lock_loss_reason = "jump_rejected"
                return STATE_REACQUIRE
            if c.local_lock_score < 0.48:
                c.lock_loss_reason = "score_drop"
                return STATE_REACQUIRE
            return STATE_LOCKED
        if prev == STATE_HOLD:
            if measurement_track is not None:
                return STATE_REACQUIRE
            if c.hold_frames > self.hold_frames:
                c.lock_loss_reason = "hold_timeout"
                return STATE_LOST
            return STATE_HOLD
        if prev == STATE_REACQUIRE:
            if measurement_track is None and c.reacquire_frames > self.hold_frames:
                c.lock_loss_reason = "reacquire_timeout"
                return STATE_LOST
            if c.identity_desync:
                return STATE_REACQUIRE
            if c.active_track_missed >= self.active_missed_soft_frames:
                return STATE_REACQUIRE
            if c.local_lock_score >= 0.78 and c.measurement_support >= self.ui_truth_lock_support_min:
                return STATE_LOCKED
            return STATE_REACQUIRE
        if prev == STATE_LOST:
            return STATE_WIDE_TRACKING if c.selected_id is not None else STATE_IDLE
        return prev

    def update(self, frame_shape, selected_track, tracks, selected_id):
        c = self.context
        c.selected_id = selected_id
        c.active_track_missed = int(getattr(selected_track, "missed_frames", 999) or 999) if selected_track is not None else 999
        self._update_wide_readiness(selected_track, frame_shape)
        c.handoff_ready_score = self.readiness.readiness_score
        c.handoff_ready_streak = self.readiness.ready_streak

        measurement_track = self._choose_measurement_track(selected_track, tracks)
        if measurement_track is not None:
            c.steering_target_id = int(getattr(measurement_track, "track_id", -1))
            c.raw_center = tuple(float(v) for v in measurement_track.center_xy)
            c.raw_bbox = tuple(float(v) for v in measurement_track.bbox_xyxy)
        else:
            c.raw_center = None
            c.raw_bbox = None

        self._update_desync(selected_track, measurement_track)
        c.measurement_support = self._compute_measurement_support(measurement_track, selected_track)
        c.state = self._transition_state(measurement_track)

        anchor = self.anchor_filter.update(
            measured_center=c.raw_center,
            measured_bbox=c.raw_bbox,
            confidence=float(getattr(measurement_track, "confidence", 0.0) or 0.0) if measurement_track is not None else 0.0,
            state=c.state,
        )
        c.refined_center = anchor["refined_center"]
        c.refined_bbox = anchor["refined_bbox"]
        c.predicted_center = anchor["predicted_center"]
        c.last_jump_px = float(anchor["jump_px"])
        c.jump_rejected = bool(anchor["jump_rejected"])
        c.lock_confidence = float(anchor["anchor_confidence"])
        c.local_lock_score = self._compute_local_lock_score(measurement_track, c.lock_confidence, c.last_jump_px)
        if c.active_track_missed >= self.active_missed_soft_frames:
            c.local_lock_score *= 0.82
            c.lock_confidence *= 0.88
        if c.active_track_missed >= self.active_missed_hard_frames:
            c.local_lock_score *= 0.58
            c.lock_confidence *= 0.70
        if c.identity_desync_frames > 0:
            factor = max(0.45, 1.0 - 0.18 * float(c.identity_desync_frames))
            c.local_lock_score *= factor
            c.lock_confidence *= max(0.50, factor)
        c.ui_truthful_lock = bool(c.state == STATE_LOCKED and c.measurement_support >= self.ui_truth_lock_support_min and c.active_track_missed <= self.ui_truth_max_missed and not c.identity_desync)

        if c.state == STATE_ACQUIRE:
            c.acquire_frames += 1
            c.refine_frames = c.lock_frames = c.hold_frames = c.reacquire_frames = 0
        elif c.state == STATE_REFINE:
            c.refine_frames += 1
            c.acquire_frames = c.lock_frames = c.hold_frames = c.reacquire_frames = 0
        elif c.state == STATE_LOCKED:
            c.lock_frames += 1
            c.acquire_frames = c.refine_frames = c.hold_frames = c.reacquire_frames = 0
        elif c.state == STATE_HOLD:
            c.hold_frames += 1
            c.acquire_frames = c.refine_frames = c.lock_frames = c.reacquire_frames = 0
        elif c.state == STATE_REACQUIRE:
            c.reacquire_frames += 1
            c.acquire_frames = c.refine_frames = c.lock_frames = c.hold_frames = 0
        else:
            c.acquire_frames = c.refine_frames = c.lock_frames = c.hold_frames = c.reacquire_frames = 0

        if c.state in (STATE_REFINE, STATE_LOCKED) and c.local_lock_score >= 0.62 and c.refined_center is not None:
            c.last_good_center = c.refined_center
            c.last_good_bbox = c.refined_bbox

        return {
            "state": c.state,
            "soft_track": measurement_track,
            "steering_target_id": c.steering_target_id,
            "refined_center": c.refined_center,
            "refined_bbox": c.refined_bbox,
            "predicted_center": c.predicted_center,
            "lock_confidence": c.lock_confidence,
            "local_lock_score": c.local_lock_score,
            "jump_rejected": c.jump_rejected,
            "jump_px": c.last_jump_px,
            "handoff_ready_score": c.handoff_ready_score,
            "handoff_ready_streak": c.handoff_ready_streak,
            "lock_loss_reason": c.lock_loss_reason,
            "wide_center_jitter": self.readiness.center_jitter_px,
            "wide_bbox_jitter": self.readiness.bbox_size_jitter,
            "last_good_center": c.last_good_center,
            "last_good_bbox": c.last_good_bbox,
            "last_good_zoom": c.last_good_zoom,
            "active_track_missed": c.active_track_missed,
            "measurement_support": c.measurement_support,
            "identity_desync": c.identity_desync,
            "identity_desync_frames": c.identity_desync_frames,
            "ui_truthful_lock": c.ui_truthful_lock,
        }
