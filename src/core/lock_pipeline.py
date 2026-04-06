from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Optional, Sequence, Tuple
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


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _dist(a, b) -> float:
    if a is None or b is None:
        return 1e9
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.hypot(dx, dy)


def _bbox_wh(bbox):
    if bbox is None:
        return (0.0, 0.0)
    x1, y1, x2, y2 = bbox
    return (max(1.0, float(x2) - float(x1)), max(1.0, float(y2) - float(y1)))


def _bbox_similarity(a, b) -> float:
    if a is None or b is None:
        return 0.0
    aw, ah = _bbox_wh(a)
    bw, bh = _bbox_wh(b)
    dw = abs(aw - bw) / max(aw, bw, 1.0)
    dh = abs(ah - bh) / max(ah, bh, 1.0)
    return _clamp01(1.0 - 0.5 * (dw + dh))


def _velocity_similarity(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax, ay = a
    bx, by = b
    da = math.hypot(ax, ay)
    db = math.hypot(bx, by)
    if da < 1e-6 and db < 1e-6:
        return 1.0
    if da < 1e-6 or db < 1e-6:
        return 0.0
    cos = (ax * bx + ay * by) / max(da * db, 1e-6)
    mag = 1.0 - min(abs(da - db) / max(da, db, 1.0), 1.0)
    return _clamp01(0.7 * (0.5 * (cos + 1.0)) + 0.3 * mag)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values)) / max(1, len(values))


def _norm(v) -> float:
    if v is None:
        return 0.0
    return math.hypot(float(v[0]), float(v[1]))


def _direction_similarity(a, b) -> float:
    if a is None or b is None:
        return 0.0
    an = _norm(a)
    bn = _norm(b)
    if an < 1e-6 or bn < 1e-6:
        return 0.0
    cos = (float(a[0]) * float(b[0]) + float(a[1]) * float(b[1])) / max(an * bn, 1e-6)
    return _clamp01(0.5 * (cos + 1.0))


def _bbox_center(bbox):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5)


def _bbox_from_center_wh(center, wh):
    if center is None or wh is None:
        return None
    cx, cy = center
    w, h = wh
    hw = max(1.0, float(w)) * 0.5
    hh = max(1.0, float(h)) * 0.5
    return (float(cx) - hw, float(cy) - hh, float(cx) + hw, float(cy) + hh)


def _elliptical_distance(point, center, motion_vec, major, minor):
    if point is None or center is None:
        return 1e9
    dx = float(point[0]) - float(center[0])
    dy = float(point[1]) - float(center[1])
    mvx = float(motion_vec[0]) if motion_vec is not None else 0.0
    mvy = float(motion_vec[1]) if motion_vec is not None else 0.0
    norm = math.hypot(mvx, mvy)
    if norm < 1e-6:
        return math.hypot(dx / max(major, 1.0), dy / max(minor, 1.0))
    ux, uy = mvx / norm, mvy / norm
    vx, vy = -uy, ux
    along = dx * ux + dy * uy
    across = dx * vx + dy * vy
    return math.hypot(along / max(major, 1.0), across / max(minor, 1.0))


@dataclass
class ProxyTrack:
    track_id: int
    raw_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    center_xy: Tuple[float, float]
    confidence: float
    velocity_xy: Tuple[float, float] = (0.0, 0.0)
    hits: int = 0
    missed_frames: int = 0
    is_confirmed: bool = True
    is_valid_target: bool = True
    is_active_target: bool = True
    is_proxy: bool = True


def _bbox_diag(bbox) -> float:
    w, h = _bbox_wh(bbox)
    return math.hypot(w, h)


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return math.sqrt(_mean([(float(v) - m) ** 2 for v in values]))


@dataclass
class WideReadinessSnapshot:
    track_id: Optional[int] = None
    conf: float = 0.0
    hits: int = 0
    missed_frames: int = 0
    center: Optional[Tuple[float, float]] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    center_jitter_px: float = 0.0
    bbox_size_jitter: float = 0.0
    velocity_consistency: float = 0.0
    readiness_score: float = 0.0
    ready_streak: int = 0
    is_ready: bool = False


@dataclass
class OwnershipState:
    owner_selected_id: Optional[int] = None
    owner_track_id: Optional[int] = None
    owner_raw_id: Optional[int] = None
    owner_age_frames: int = 0
    owner_lock_frames: int = 0
    owner_reacquire_frames: int = 0
    owner_strength: float = 0.0
    ownership_confidence: float = 0.0
    last_owner_center: Optional[Tuple[float, float]] = None
    last_owner_bbox: Optional[Tuple[float, float, float, float]] = None
    last_owner_velocity: Optional[Tuple[float, float]] = None
    pending_track_id: Optional[int] = None
    pending_score: float = 0.0
    pending_frames: int = 0
    transfer_reason: Optional[str] = None
    last_reject_reason: Optional[str] = None


@dataclass
class LocalLockContext:
    selected_id: Optional[int] = None
    steering_target_id: Optional[int] = None
    state: str = STATE_IDLE
    raw_center: Optional[Tuple[float, float]] = None
    raw_bbox: Optional[Tuple[float, float, float, float]] = None
    refined_center: Optional[Tuple[float, float]] = None
    refined_bbox: Optional[Tuple[float, float, float, float]] = None
    predicted_center: Optional[Tuple[float, float]] = None
    motion_dir: Optional[Tuple[float, float]] = None
    last_measurement_center: Optional[Tuple[float, float]] = None
    lock_confidence: float = 0.0
    local_lock_score: float = 0.0
    handoff_ready_score: float = 0.0
    handoff_ready_streak: int = 0
    acquire_frames: int = 0
    refine_frames: int = 0
    lock_frames: int = 0
    hold_frames: int = 0
    reacquire_frames: int = 0
    bad_lock_frames: int = 0
    jump_rejected: bool = False
    last_jump_px: float = 0.0
    anchor_jump_px: float = 0.0
    last_good_center: Optional[Tuple[float, float]] = None
    last_good_bbox: Optional[Tuple[float, float, float, float]] = None
    last_good_zoom: float = 2.8
    lock_loss_reason: Optional[str] = None
    ownership: OwnershipState = field(default_factory=OwnershipState)
    identity_consistency_score: float = 0.0
    ownership_score: float = 0.0
    owner_score: float = 0.0
    owner_strength: float = 0.0
    transfer_margin: float = 0.0
    transfer_dwell_frames: int = 0
    jump_risk: float = 0.0
    ownership_reject_reason: Optional[str] = None
    motion_dir: Optional[Tuple[float, float]] = None
    last_measurement_center: Optional[Tuple[float, float]] = None
    projected_center: Optional[Tuple[float, float]] = None
    projected_bbox: Optional[Tuple[float, float, float, float]] = None


class RefinedAnchorFilter:
    def __init__(self, position_alpha: float = 0.42, bbox_alpha: float = 0.35):
        self.position_alpha = float(position_alpha)
        self.bbox_alpha = float(bbox_alpha)
        self.refined_center: Optional[Tuple[float, float]] = None
        self.refined_bbox: Optional[Tuple[float, float, float, float]] = None
        self.velocity: Tuple[float, float] = (0.0, 0.0)

    def reset(self) -> None:
        self.refined_center = None
        self.refined_bbox = None
        self.velocity = (0.0, 0.0)

    def predict(self) -> Optional[Tuple[float, float]]:
        if self.refined_center is None:
            return None
        return (self.refined_center[0] + self.velocity[0], self.refined_center[1] + self.velocity[1])

    def update(self, measured_center, measured_bbox, confidence: float, state: str, allow_update: bool = True):
        predicted = self.predict()
        if measured_center is None or not allow_update:
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

        if state == STATE_LOCKED:
            alpha = 0.28
        elif state == STATE_REFINE:
            alpha = 0.36
        elif state == STATE_REACQUIRE:
            alpha = 0.42
        elif state == STATE_ACQUIRE:
            alpha = 0.52
        else:
            alpha = self.position_alpha

        next_center = (base[0] + alpha * dx, base[1] + alpha * dy)
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

        return {
            "refined_center": self.refined_center,
            "refined_bbox": self.refined_bbox,
            "predicted_center": predicted,
            "jump_px": math.hypot(dx, dy),
            "jump_rejected": False,
            "anchor_confidence": float(confidence),
        }


class LockPipeline:
    def __init__(self, config: dict):
        self.cfg = config or {}
        hcfg = self.cfg.get("handoff") or {}
        self.ready_frames = int(hcfg.get("handoff_ready_frames", 5))
        self.hold_frames = int(hcfg.get("handoff_hold_frames", 10))
        self.reacquire_radius = float(hcfg.get("handoff_reacquire_radius", 165.0))
        self.jump_soft_threshold_px = float(hcfg.get("jump_soft_threshold_px", 24.0))
        self.jump_hard_threshold_px = float(hcfg.get("jump_hard_threshold_px", 48.0))
        self.owner_reacquire_bias_same_raw = float(hcfg.get("owner_reacquire_bias_same_raw", 0.18))
        self.owner_reacquire_penalty_other_raw = float(hcfg.get("owner_reacquire_penalty_other_raw", 0.14))
        self.identity_transfer_margin_locked = float(hcfg.get("ownership_transfer_margin_locked", 0.14))
        self.identity_transfer_margin_reacquire = float(hcfg.get("ownership_transfer_margin_reacquire", 0.10))
        self.identity_transfer_margin_refine = float(hcfg.get("ownership_transfer_margin_refine", 0.08))
        self.identity_transfer_dwell_locked = int(hcfg.get("ownership_transfer_dwell_locked", 4))
        self.identity_transfer_dwell_reacquire = int(hcfg.get("ownership_transfer_dwell_reacquire", 3))
        self.identity_transfer_dwell_refine = int(hcfg.get("ownership_transfer_dwell_refine", 2))
        self.identity_score_min_for_transfer = float(hcfg.get("identity_score_min_for_transfer", 0.42))
        self.identity_score_min_for_jump_accept = float(hcfg.get("identity_score_min_for_jump_accept", 0.58))
        self.initial_owner_score_min = float(hcfg.get("initial_owner_score_min", 0.26))
        self.initial_identity_score_min = float(hcfg.get("initial_identity_score_min", 0.32))
        self.lock_drop_frames_locked = int(hcfg.get("owner_drop_frames_locked", 5))
        self.lock_drop_frames_reacquire = int(hcfg.get("owner_drop_frames_reacquire", 3))
        self.reacquire_motion_major_scale = float(hcfg.get("reacquire_motion_major_scale", 2.2))
        self.reacquire_motion_minor_scale = float(hcfg.get("reacquire_motion_minor_scale", 1.2))
        self.proxy_max_frames = int(hcfg.get("proxy_max_frames", max(self.hold_frames, 12)))

        self.context = LocalLockContext()
        self.readiness = WideReadinessSnapshot()
        self.anchor_filter = RefinedAnchorFilter(
            position_alpha=float(hcfg.get("anchor_position_alpha", 0.42)),
            bbox_alpha=float(hcfg.get("anchor_bbox_alpha", 0.35)),
        )
        self._center_hist: Deque[Tuple[float, float]] = deque(maxlen=8)
        self._size_hist: Deque[Tuple[float, float]] = deque(maxlen=8)
        self._vel_hist: Deque[Tuple[float, float]] = deque(maxlen=8)
        self._prev_center: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self.context = LocalLockContext()
        self.readiness = WideReadinessSnapshot()
        self.anchor_filter.reset()
        self._center_hist.clear()
        self._size_hist.clear()
        self._vel_hist.clear()
        self._prev_center = None

    def _update_wide_readiness(self, selected_track, frame_shape):
        r = self.readiness
        c = self.context
        if selected_track is None:
            r.ready_streak = 0
            r.is_ready = False
            r.readiness_score = 0.0
            c.handoff_ready_score = 0.0
            c.handoff_ready_streak = 0
            return r

        center = tuple(float(v) for v in selected_track.center_xy)
        bbox = tuple(float(v) for v in selected_track.bbox_xyxy)
        size = _bbox_wh(bbox)

        self._center_hist.append(center)
        self._size_hist.append(size)
        if self._prev_center is not None:
            self._vel_hist.append((center[0] - self._prev_center[0], center[1] - self._prev_center[1]))
        self._prev_center = center

        center_jitter = 0.0
        if len(self._center_hist) >= 2:
            dists = [_dist(self._center_hist[i - 1], self._center_hist[i]) for i in range(1, len(self._center_hist))]
            center_jitter = _mean(dists)

        size_jitter = 0.0
        if len(self._size_hist) >= 2:
            diffs = []
            for i in range(1, len(self._size_hist)):
                pw, ph = self._size_hist[i - 1]
                cw, ch = self._size_hist[i]
                diffs.append(0.5 * (abs(cw - pw) / max(1.0, pw) + abs(ch - ph) / max(1.0, ph)))
            size_jitter = _mean(diffs)

        vel_consistency = 1.0
        if len(self._vel_hist) >= 2:
            mags = [math.hypot(vx, vy) for vx, vy in self._vel_hist]
            vel_consistency = 1.0 - min(_std(mags) / max(_mean(mags), 1.0), 1.0)

        conf = float(getattr(selected_track, "confidence", 0.0) or 0.0)
        hits = int(getattr(selected_track, "hits", 0) or 0)
        missed = int(getattr(selected_track, "missed_frames", 0) or 0)
        h, w = frame_shape[:2]
        edge_margin = min(center[0], center[1], w - center[0], h - center[1])

        score = 0.0
        score += min(1.0, conf / 0.25) * 0.30
        score += min(1.0, hits / 5.0) * 0.20
        score += max(0.0, 1.0 - center_jitter / 16.0) * 0.25
        score += max(0.0, 1.0 - size_jitter / 0.18) * 0.10
        score += max(0.0, min(1.0, edge_margin / 30.0)) * 0.05
        score += max(0.0, vel_consistency) * 0.10
        if missed > 0:
            score *= 0.35

        ready = score >= 0.72 and missed == 0 and (hits >= 3 or bool(getattr(selected_track, "is_confirmed", False)))

        r.track_id = int(getattr(selected_track, "track_id", -1))
        r.conf = conf
        r.hits = hits
        r.missed_frames = missed
        r.center = center
        r.bbox = bbox
        r.center_jitter_px = center_jitter
        r.bbox_size_jitter = size_jitter
        r.velocity_consistency = vel_consistency
        r.readiness_score = score
        r.ready_streak = r.ready_streak + 1 if ready else 0
        r.is_ready = ready and r.ready_streak >= self.ready_frames

        c.handoff_ready_score = r.readiness_score
        c.handoff_ready_streak = r.ready_streak
        return r

    def _collect_candidates(self, tracks: Iterable[object], selected_id: Optional[int]) -> List[object]:
        tracks = list(tracks or [])
        ordered: List[object] = []
        if selected_id is not None:
            for tr in tracks:
                if getattr(tr, "track_id", None) == selected_id:
                    ordered.append(tr)
        for tr in tracks:
            if tr not in ordered and bool(getattr(tr, "is_confirmed", False)):
                ordered.append(tr)
        for tr in tracks:
            if tr not in ordered:
                ordered.append(tr)
        return ordered

    def _predict_owner_center(self, ctx: LocalLockContext):
        """
        Stabilne przewidywanie pozycji ownera na podstawie ostatniego dobrego centrum
        i wektora ruchu. Bezpieczny fallback: jeśli brak danych, zwraca ostatni znany środek.
        """
        try:
            base = ctx.refined_center or ctx.last_good_center or ctx.ownership.last_owner_center
            if base is None:
                return None

            motion = ctx.motion_dir or ctx.ownership.last_owner_velocity
            if motion is None:
                return base

            mx = float(motion[0])
            my = float(motion[1])
            if abs(mx) < 1e-6 and abs(my) < 1e-6:
                return base

            return (float(base[0]) + mx, float(base[1]) + my)
        except Exception:
            return ctx.last_good_center or ctx.refined_center or ctx.ownership.last_owner_center

    def _identity_consistency_score(self, tr, ctx: LocalLockContext) -> float:
        if tr is None:
            return 0.0
        center = tuple(float(v) for v in tr.center_xy)
        bbox = tuple(float(v) for v in tr.bbox_xyxy)
        vel = getattr(tr, "velocity_xy", None)

        anchor_center = self._predict_owner_center(ctx) or ctx.refined_center or ctx.last_good_center or ctx.ownership.last_owner_center or self.readiness.center
        anchor_bbox = ctx.ownership.last_owner_bbox or ctx.last_good_bbox or self.readiness.bbox
        d_anchor = _dist(center, anchor_center)
        diag = max(20.0, _bbox_diag(anchor_bbox))
        spatial_gate = max(52.0, min(240.0, diag * 3.8))
        motion_vec = ctx.motion_dir or ctx.ownership.last_owner_velocity
        ell_major = max(spatial_gate, _norm(motion_vec) * self.reacquire_motion_major_scale + diag * 1.4)
        ell_minor = max(38.0, min(150.0, diag * self.reacquire_motion_minor_scale))
        ell_d = _elliptical_distance(center, anchor_center, motion_vec, ell_major, ell_minor)
        spatial = 1.0 - min(min(d_anchor / spatial_gate, ell_d), 1.0)

        bbox_sim = _bbox_similarity(bbox, anchor_bbox)
        vel_sim = _velocity_similarity(vel, ctx.ownership.last_owner_velocity)

        expected_dir = None
        if ctx.motion_dir is not None and _norm(ctx.motion_dir) > 1e-6:
            expected_dir = ctx.motion_dir
        elif ctx.ownership.last_owner_velocity is not None and _norm(ctx.ownership.last_owner_velocity) > 1e-6:
            expected_dir = ctx.ownership.last_owner_velocity
        elif anchor_center is not None and ctx.ownership.last_owner_center is not None:
            expected_dir = (
                float(anchor_center[0]) - float(ctx.ownership.last_owner_center[0]),
                float(anchor_center[1]) - float(ctx.ownership.last_owner_center[1]),
            )

        candidate_vec = None
        if anchor_center is not None:
            candidate_vec = (float(center[0]) - float(anchor_center[0]), float(center[1]) - float(anchor_center[1]))
        dir_sim = _direction_similarity(candidate_vec, expected_dir) if expected_dir is not None else 0.5

        owner_raw = ctx.ownership.owner_raw_id
        tr_raw = getattr(tr, "raw_id", None)
        hits_term = min(float(getattr(tr, "hits", 0) or 0.0) / 20.0, 1.0)
        missed_term = min(float(getattr(tr, "missed_frames", 0) or 0.0) / 5.0, 1.0)

        score = 0.44 * spatial + 0.16 * bbox_sim + 0.12 * vel_sim + 0.18 * dir_sim + 0.10 * hits_term
        if owner_raw is not None and tr_raw == owner_raw:
            score += 0.12
        score -= 0.08 * missed_term
        if spatial < 0.20:
            score -= 0.10
        if dir_sim < 0.25:
            score -= 0.08
        return _clamp01(score)

    def _ownership_score(self, tr, ctx: LocalLockContext, identity_score: float) -> float:
        if tr is None:
            return 0.0
        conf = float(getattr(tr, "confidence", 0.0) or 0.0)
        hits = float(getattr(tr, "hits", 0) or 0.0)
        missed = float(getattr(tr, "missed_frames", 0) or 0.0)
        bbox = getattr(tr, "bbox_xyxy", None)
        scale = max(36.0, _bbox_diag(ctx.last_good_bbox or ctx.ownership.last_owner_bbox or bbox))
        jump_px = _dist(getattr(tr, "center_xy", None), ctx.refined_center or ctx.last_good_center or ctx.ownership.last_owner_center)
        jump_risk = _clamp01(jump_px / max(scale * 2.8, 48.0))
        track_quality = _clamp01(0.55 * min(hits / 20.0, 1.0) + 0.25 * conf + 0.20 * (1.0 - min(missed / 3.0, 1.0)))
        same_raw_bonus = 0.08 if ctx.ownership.owner_raw_id is not None and getattr(tr, "raw_id", None) == ctx.ownership.owner_raw_id else 0.0
        score = (
            0.56 * identity_score
            + 0.10 * _clamp01(ctx.lock_confidence)
            + 0.08 * _clamp01(ctx.local_lock_score)
            + 0.06 * _clamp01(ctx.handoff_ready_score)
            + 0.18 * track_quality
            + same_raw_bonus
            - 0.10 * jump_risk
            - 0.05 * min(missed / 3.0, 1.0)
        )
        return _clamp01(score)

    def _owner_strength(self, ctx: LocalLockContext) -> float:
        own = ctx.ownership
        age_term = min(float(own.owner_age_frames) / 30.0, 1.0)
        lock_term = min(float(own.owner_lock_frames) / 20.0, 1.0)
        reacq_term = min(float(own.owner_reacquire_frames) / 10.0, 1.0)
        score = (
            0.35 * age_term
            + 0.35 * lock_term
            + 0.15 * _clamp01(ctx.lock_confidence)
            + 0.10 * _clamp01(ctx.local_lock_score)
            + 0.05 * (1.0 - min(reacq_term, 1.0))
        )
        return _clamp01(score)

    def _transfer_policy(self, ctx: LocalLockContext):
        owner_strength = self._owner_strength(ctx)
        ctx.owner_strength = owner_strength
        if ctx.state == STATE_LOCKED:
            margin = self.identity_transfer_margin_locked + 0.06 * owner_strength
            dwell = self.identity_transfer_dwell_locked
        elif ctx.state == STATE_REACQUIRE:
            margin = self.identity_transfer_margin_reacquire + 0.05 * owner_strength
            dwell = self.identity_transfer_dwell_reacquire
        elif ctx.state == STATE_REFINE:
            margin = self.identity_transfer_margin_refine + 0.04 * owner_strength
            dwell = self.identity_transfer_dwell_refine
        else:
            margin = 0.08
            dwell = 2
        ctx.transfer_margin = margin
        ctx.transfer_dwell_frames = dwell
        return margin, dwell

    def _candidate_jump_guard(self, tr, ctx: LocalLockContext, identity_score: float):
        center = getattr(tr, "center_xy", None)
        bbox = getattr(tr, "bbox_xyxy", None)

        reference_center = (
            ctx.refined_center
            or ctx.last_good_center
            or ctx.ownership.last_owner_center
            or self.readiness.center
        )
        reference_bbox = ctx.last_good_bbox or ctx.ownership.last_owner_bbox or self.readiness.bbox or bbox

        if center is None or reference_center is None:
            ctx.jump_risk = 0.0
            ctx.jump_rejected = False
            ctx.ownership_reject_reason = None
            return True, 0.0

        diag = max(24.0, _bbox_diag(reference_bbox))
        soft_thr = max(self.jump_soft_threshold_px, diag * 1.35)
        hard_thr = max(self.jump_hard_threshold_px, diag * 2.10)
        if ctx.state == STATE_LOCKED:
            soft_thr *= 0.85
            hard_thr *= 0.85
        elif ctx.state == STATE_REACQUIRE:
            soft_thr *= 1.10
            hard_thr *= 1.05

        jump_px = _dist(center, reference_center)
        ctx.jump_risk = _clamp01(jump_px / max(hard_thr, 1.0))
        bootstrap_mode = ctx.ownership.owner_track_id is None

        if bootstrap_mode:
            if jump_px > hard_thr and identity_score < max(0.22, self.initial_identity_score_min):
                ctx.jump_rejected = True
                ctx.ownership_reject_reason = "jump_identity_risk"
                return False, jump_px
            if jump_px > soft_thr and identity_score < 0.20:
                ctx.ownership_reject_reason = "jump_soft_risk"
                return False, jump_px
            ctx.jump_rejected = False
            ctx.ownership_reject_reason = None
            return True, jump_px

        strong_owner = _clamp01(ctx.ownership.ownership_confidence) > 0.55 or _clamp01(ctx.owner_strength) > 0.55
        if strong_owner and jump_px > soft_thr and identity_score < 0.72:
            ctx.ownership_reject_reason = "owner_anchor_gate"
            if jump_px > hard_thr * 0.85:
                ctx.jump_rejected = True
            return False, jump_px

        if jump_px > hard_thr and identity_score < self.identity_score_min_for_jump_accept:
            ctx.jump_rejected = True
            ctx.ownership_reject_reason = "jump_identity_risk"
            return False, jump_px

        if jump_px > soft_thr and identity_score < 0.52:
            ctx.ownership_reject_reason = "jump_soft_risk"
            return False, jump_px

        ctx.jump_rejected = False
        ctx.ownership_reject_reason = None
        return True, jump_px

    def _get_owner_proxy_score(self, scored, ctx: LocalLockContext) -> float:
        owner_id = ctx.ownership.owner_track_id
        if owner_id is None:
            return 0.0
        for item in scored:
            tr = item["track"]
            if tr is not None and getattr(tr, "track_id", None) == owner_id:
                return item["ownership_score"]
        return 0.0

    def _adopt_owner(self, tr, ctx: LocalLockContext, reason: str):
        own = ctx.ownership
        own.owner_selected_id = ctx.selected_id
        own.owner_track_id = getattr(tr, "track_id", None)
        own.owner_raw_id = getattr(tr, "raw_id", None)
        own.owner_age_frames = 1
        own.owner_lock_frames = 0
        own.owner_reacquire_frames = 0
        own.pending_track_id = None
        own.pending_score = 0.0
        own.pending_frames = 0
        own.transfer_reason = reason
        own.last_reject_reason = None
        own.last_owner_center = getattr(tr, "center_xy", None)
        own.last_owner_bbox = getattr(tr, "bbox_xyxy", None)
        own.last_owner_velocity = getattr(tr, "velocity_xy", None)

    def _update_owner_memory(self, tr, ctx: LocalLockContext):
        own = ctx.ownership
        if tr is None:
            return
        own.owner_age_frames += 1
        if ctx.state == STATE_LOCKED:
            own.owner_lock_frames += 1
        if ctx.state == STATE_REACQUIRE:
            own.owner_reacquire_frames += 1
        own.last_owner_center = getattr(tr, "center_xy", None)
        own.last_owner_bbox = getattr(tr, "bbox_xyxy", None)
        own.last_owner_velocity = getattr(tr, "velocity_xy", None)

    def _resolve_owner_candidate(self, candidates, ctx: LocalLockContext):
        scored = []
        for tr in candidates:
            id_score = self._identity_consistency_score(tr, ctx)
            own_score = self._ownership_score(tr, ctx, id_score)
            scored.append({"track": tr, "identity_score": id_score, "ownership_score": own_score})

        if not scored:
            return None, scored

        scored.sort(key=lambda x: x["ownership_score"], reverse=True)
        best = scored[0]
        owner_score = self._get_owner_proxy_score(scored, ctx)
        ctx.owner_score = owner_score
        margin, dwell = self._transfer_policy(ctx)

        best_tr = best["track"]
        best_score = best["ownership_score"]
        best_id_score = best["identity_score"]

        allowed, jump_px = self._candidate_jump_guard(best_tr, ctx, best_id_score)
        ctx.anchor_jump_px = jump_px
        ctx.last_jump_px = jump_px
        ctx.identity_consistency_score = best_id_score
        ctx.ownership_score = best_score

        if ctx.ownership.owner_track_id is None:
            # Initial bootstrap: pozwól wejść, jeśli kandydat jest wystarczająco sensowny
            # nawet gdy refined/last_good jeszcze nie są stabilnie zbudowane.
            if not allowed:
                ctx.ownership.last_reject_reason = ctx.ownership_reject_reason
                return None, scored

            bootstrap_score_ok = best_score >= self.initial_owner_score_min
            bootstrap_identity_ok = best_id_score >= self.initial_identity_score_min

            selected_match = (
                ctx.selected_id is not None
                and getattr(best_tr, "track_id", None) == ctx.selected_id
            )
            confirmed = bool(getattr(best_tr, "is_confirmed", False))
            hits = int(getattr(best_tr, "hits", 0) or 0)
            missed = int(getattr(best_tr, "missed_frames", 0) or 0)

            # Soft bootstrap path for stable selected target
            soft_bootstrap_ok = selected_match and confirmed and hits >= 3 and missed == 0 and best_id_score >= 0.20

            if (bootstrap_score_ok and bootstrap_identity_ok) or soft_bootstrap_ok:
                self._adopt_owner(best_tr, ctx, "initial_owner")
                ctx.ownership.ownership_confidence = max(ctx.ownership.ownership_confidence, 0.35)
                return best_tr, scored

            ctx.ownership.last_reject_reason = "initial_owner_threshold"
            return None, scored

        current_owner_id = ctx.ownership.owner_track_id
        if getattr(best_tr, "track_id", None) == current_owner_id:
            ctx.ownership.pending_track_id = None
            ctx.ownership.pending_frames = 0
            ctx.ownership.pending_score = 0.0
            self._update_owner_memory(best_tr, ctx)
            return best_tr, scored

        if not allowed:
            ctx.ownership.last_reject_reason = ctx.ownership_reject_reason
            return None, scored

        if best_id_score < self.identity_score_min_for_transfer:
            ctx.ownership.last_reject_reason = "identity_score_too_low"
            return None, scored

        if best_score <= owner_score + margin:
            ctx.ownership.pending_track_id = None
            ctx.ownership.pending_frames = 0
            ctx.ownership.pending_score = 0.0
            ctx.ownership.last_reject_reason = "owner_margin_not_met"
            return None, scored

        if ctx.ownership.pending_track_id != getattr(best_tr, "track_id", None):
            ctx.ownership.pending_track_id = getattr(best_tr, "track_id", None)
            ctx.ownership.pending_frames = 1
            ctx.ownership.pending_score = best_score
            ctx.ownership.last_reject_reason = "owner_transfer_dwell_started"
            return None, scored

        ctx.ownership.pending_frames += 1
        ctx.ownership.pending_score = max(ctx.ownership.pending_score, best_score)

        if ctx.ownership.pending_frames >= dwell:
            self._adopt_owner(best_tr, ctx, "ownership_transfer")
            return best_tr, scored

        ctx.ownership.last_reject_reason = "owner_transfer_waiting"
        return None, scored

    def _measurement_from_owner(self, candidates, ctx: LocalLockContext):
        owner_id = ctx.ownership.owner_track_id
        if owner_id is None:
            return None
        for tr in candidates:
            if getattr(tr, "track_id", None) == owner_id:
                return tr
        return None

    def _owner_consistent_proxy(self, candidates, ctx: LocalLockContext):
        best = None
        best_score = -1.0
        anchor_center = self._predict_owner_center(ctx) or ctx.refined_center or ctx.last_good_center or ctx.ownership.last_owner_center
        expected_dir = ctx.motion_dir or ctx.ownership.last_owner_velocity
        anchor_bbox = ctx.last_good_bbox or ctx.ownership.last_owner_bbox
        diag = _bbox_diag(anchor_bbox) if anchor_bbox is not None else 48.0
        max_proxy_dist = max(72.0, min(220.0, diag * 3.4))
        ell_major = max(max_proxy_dist, _norm(expected_dir) * self.reacquire_motion_major_scale + diag * 1.4)
        ell_minor = max(42.0, min(140.0, diag * self.reacquire_motion_minor_scale))
        for tr in candidates:
            id_score = self._identity_consistency_score(tr, ctx)
            if id_score < 0.52:
                continue
            center = getattr(tr, "center_xy", None)
            d = _dist(center, anchor_center)
            if d > ell_major * 1.05:
                continue
            ell_d = _elliptical_distance(center, anchor_center, expected_dir, ell_major, ell_minor)
            if ell_d > 1.15:
                continue
            dir_bonus = 0.0
            if expected_dir is not None and anchor_center is not None:
                candidate_vec = (
                    float(getattr(tr, "center_xy", (0.0, 0.0))[0]) - float(anchor_center[0]),
                    float(getattr(tr, "center_xy", (0.0, 0.0))[1]) - float(anchor_center[1]),
                )
                dir_bonus = 0.22 * _direction_similarity(candidate_vec, expected_dir)
            score = id_score + dir_bonus - 0.22 * min(d / max_proxy_dist, 1.0) - 0.18 * min(ell_d, 1.0)
            if ctx.ownership.owner_raw_id is not None and getattr(tr, "raw_id", None) == ctx.ownership.owner_raw_id:
                score += 0.15
            if bool(getattr(tr, "is_confirmed", False)):
                score += 0.04
            if score > best_score:
                best = tr
                best_score = score
        return best

    def _compute_lock_metrics(self, measurement_track, ctx: LocalLockContext):
        if measurement_track is None or ctx.refined_center is None:
            ctx.local_lock_score *= 0.95
            ctx.lock_confidence *= 0.96
            ctx.ownership.ownership_confidence *= 0.96
            return
        if bool(getattr(measurement_track, "is_proxy", False)):
            ctx.local_lock_score *= 0.985
            ctx.lock_confidence *= 0.988
            ctx.ownership.ownership_confidence *= 0.992
            return

        center = tuple(float(v) for v in measurement_track.center_xy)
        bbox = tuple(float(v) for v in measurement_track.bbox_xyxy)
        conf = float(getattr(measurement_track, "confidence", 0.0) or 0.0)
        hits = float(getattr(measurement_track, "hits", 0) or 0.0)
        missed = float(getattr(measurement_track, "missed_frames", 0) or 0.0)

        scale = max(40.0, _bbox_diag(ctx.last_good_bbox or ctx.ownership.last_owner_bbox or bbox))
        d_refined = _dist(center, ctx.refined_center or ctx.last_good_center)
        spatial = 1.0 - min(d_refined / max(scale * 1.8, 60.0), 1.0)
        bbox_sim = _bbox_similarity(bbox, ctx.last_good_bbox or ctx.ownership.last_owner_bbox)
        hit_term = min(hits / 20.0, 1.0)
        miss_term = 1.0 - min(missed / 3.0, 1.0)

        ctx.local_lock_score = _clamp01(
            0.42 * spatial + 0.22 * bbox_sim + 0.15 * hit_term + 0.11 * conf + 0.10 * miss_term
        )
        ctx.lock_confidence = _clamp01(
            0.48 * ctx.local_lock_score + 0.28 * ctx.identity_consistency_score + 0.12 * ctx.handoff_ready_score + 0.12 * conf
        )
        ctx.ownership.ownership_confidence = _clamp01(0.55 * ctx.local_lock_score + 0.45 * ctx.identity_consistency_score)

    def _update_last_good(self, ctx: LocalLockContext):
        if ctx.refined_center is None:
            return
        if ctx.state in (STATE_ACQUIRE, STATE_REFINE, STATE_LOCKED, STATE_REACQUIRE) and ctx.local_lock_score > 0.25:
            ctx.last_good_center = ctx.refined_center
            ctx.last_good_bbox = ctx.refined_bbox

    def _transition_state(self, measurement_track, ctx: LocalLockContext):
        current = ctx.state

        if ctx.ownership.owner_track_id is None and measurement_track is None:
            ctx.state = STATE_IDLE
            return

        if current in (STATE_IDLE, STATE_WIDE_TRACKING):
            if self.readiness.is_ready or ctx.ownership.owner_track_id is not None:
                ctx.state = STATE_HANDOFF_READY
            else:
                ctx.state = STATE_WIDE_TRACKING
            return

        if current == STATE_HANDOFF_READY:
            if measurement_track is not None and (self.readiness.is_ready or ctx.ownership.owner_track_id is not None):
                ctx.state = STATE_ACQUIRE
            else:
                ctx.state = STATE_WIDE_TRACKING
            return

        if current == STATE_ACQUIRE:
            ctx.acquire_frames += 1
            if measurement_track is None:
                ctx.state = STATE_HOLD
                ctx.lock_loss_reason = "measurement_missing"
                return
            if ctx.local_lock_score > 0.46 and ctx.identity_consistency_score > 0.42:
                ctx.state = STATE_REFINE
                ctx.refine_frames = 0
            return

        if current == STATE_REFINE:
            ctx.refine_frames += 1
            if measurement_track is None:
                ctx.state = STATE_HOLD
                ctx.lock_loss_reason = "measurement_missing"
                return
            if ctx.local_lock_score > 0.60 and ctx.lock_confidence > 0.30 and ctx.identity_consistency_score > 0.48:
                ctx.state = STATE_LOCKED
                ctx.lock_frames = 0
                ctx.bad_lock_frames = 0
                return
            if ctx.local_lock_score < 0.15:
                ctx.state = STATE_REACQUIRE
                ctx.lock_loss_reason = "score_drop"
            return

        if current == STATE_LOCKED:
            ctx.lock_frames += 1
            if measurement_track is None:
                ctx.bad_lock_frames += 1
            elif _clamp01(ctx.local_lock_score) < 0.14 and _clamp01(ctx.ownership.ownership_confidence) < 0.18:
                ctx.bad_lock_frames += 1
            else:
                ctx.bad_lock_frames = 0

            if ctx.bad_lock_frames >= self.lock_drop_frames_locked:
                ctx.state = STATE_REACQUIRE
                ctx.lock_loss_reason = "score_drop"
            return

        if current == STATE_HOLD:
            ctx.hold_frames += 1
            if measurement_track is not None and not bool(getattr(measurement_track, "is_proxy", False)):
                ctx.state = STATE_REACQUIRE
                return
            if ctx.hold_frames > max(self.hold_frames, self.proxy_max_frames):
                ctx.state = STATE_LOST
            return

        if current == STATE_REACQUIRE:
            ctx.reacquire_frames += 1
            if measurement_track is None:
                if ctx.reacquire_frames > self.lock_drop_frames_reacquire:
                    ctx.state = STATE_LOST
                return
            if ctx.local_lock_score > 0.46 and ctx.identity_consistency_score > 0.42:
                ctx.state = STATE_REFINE
                ctx.refine_frames = 0
            if ctx.local_lock_score > 0.66 and ctx.lock_confidence > 0.34 and ctx.identity_consistency_score > 0.50:
                ctx.state = STATE_LOCKED
                ctx.bad_lock_frames = 0
            return

        if current == STATE_LOST:
            if self.readiness.is_ready or ctx.ownership.owner_track_id is not None:
                ctx.state = STATE_HANDOFF_READY
            elif measurement_track is not None:
                ctx.state = STATE_WIDE_TRACKING

    def update(self, frame_shape, selected_track, tracks, selected_id):
        ctx = self.context
        requested_selected_id = selected_id
        ctx.selected_id = selected_id
        ctx.jump_rejected = False
        ctx.ownership_reject_reason = None
        ctx.lock_loss_reason = None
        ctx.projected_center = None
        ctx.projected_bbox = None

        self._update_wide_readiness(selected_track, frame_shape)
        candidates = self._collect_candidates(tracks, requested_selected_id)
        owner_candidate, _ = self._resolve_owner_candidate(candidates, ctx)

        owner_track_id = ctx.ownership.owner_track_id
        owner_alive = owner_track_id is not None and any(
            getattr(tr, "track_id", None) == owner_track_id for tr in candidates
        )
        owner_soft_hold = (
            owner_track_id is not None
            and not owner_alive
            and ctx.state in (STATE_LOCKED, STATE_REFINE, STATE_REACQUIRE, STATE_HOLD)
            and (ctx.ownership.owner_reacquire_frames <= max(4, self.lock_drop_frames_locked + 2))
        )
        owner_authoritative = owner_track_id is not None and (owner_alive or owner_soft_hold)

        if owner_authoritative:
            ctx.selected_id = owner_track_id
        else:
            ctx.selected_id = requested_selected_id

        measurement = self._measurement_from_owner(candidates, ctx)
        if measurement is None and owner_authoritative and ctx.state in (STATE_LOCKED, STATE_REACQUIRE, STATE_HOLD, STATE_REFINE):
            measurement = self._owner_consistent_proxy(candidates, ctx)
        if measurement is None and ctx.state == STATE_REACQUIRE:
            measurement = self._owner_consistent_proxy(candidates, ctx)
        if measurement is None and owner_candidate is not None:
            measurement = owner_candidate
        if measurement is None and selected_track is not None and (self.readiness.is_ready or ctx.ownership.owner_track_id is None):
            measurement = selected_track

        if owner_authoritative and measurement is not None:
            measurement_tid = getattr(measurement, "track_id", None)
            if measurement_tid is not None and owner_alive and measurement_tid != owner_track_id:
                owner_match = next((tr for tr in candidates if getattr(tr, "track_id", None) == owner_track_id), None)
                if owner_match is not None:
                    measurement = owner_match

        if owner_authoritative and measurement is None and owner_alive:
            owner_match = next((tr for tr in candidates if getattr(tr, "track_id", None) == owner_track_id), None)
            if owner_match is not None:
                measurement = owner_match

        ctx.steering_target_id = getattr(measurement, "track_id", None) if measurement is not None else None
        if owner_authoritative:
            ctx.steering_target_id = owner_track_id
        ctx.raw_center = getattr(measurement, "center_xy", None) if measurement is not None else None
        ctx.raw_bbox = getattr(measurement, "bbox_xyxy", None) if measurement is not None else None

        allow_update = not ctx.jump_rejected
        anchor = self.anchor_filter.update(
            measured_center=ctx.raw_center,
            measured_bbox=ctx.raw_bbox,
            confidence=float(getattr(measurement, "confidence", 0.0) or 0.0) if measurement is not None else 0.0,
            state=ctx.state,
            allow_update=allow_update,
        )
        ctx.refined_center = anchor["refined_center"]
        ctx.refined_bbox = anchor["refined_bbox"]
        ctx.predicted_center = anchor["predicted_center"]
        ctx.last_jump_px = float(anchor["jump_px"])
        if bool(anchor.get("jump_rejected", False)):
            ctx.jump_rejected = True

        current_center = getattr(measurement, "center_xy", None) if measurement is not None else None
        if current_center is not None and ctx.last_measurement_center is not None:
            inst_dir = (float(current_center[0]) - float(ctx.last_measurement_center[0]), float(current_center[1]) - float(ctx.last_measurement_center[1]))
            if _norm(inst_dir) > 1e-6:
                if ctx.motion_dir is None or _norm(ctx.motion_dir) < 1e-6:
                    ctx.motion_dir = inst_dir
                else:
                    ctx.motion_dir = (0.75 * float(ctx.motion_dir[0]) + 0.25 * float(inst_dir[0]), 0.75 * float(ctx.motion_dir[1]) + 0.25 * float(inst_dir[1]))
        if current_center is not None:
            ctx.last_measurement_center = tuple(float(v) for v in current_center)

        self._compute_lock_metrics(measurement, ctx)
        ctx.ownership.owner_strength = self._owner_strength(ctx)
        ctx.owner_strength = ctx.ownership.owner_strength
        self._transition_state(measurement, ctx)
        self._update_last_good(ctx)

        return {
            "state": ctx.state,
            "pipeline_state": ctx.state,
            "soft_track": measurement,
            "steering_target_id": ctx.steering_target_id,
            "refined_center": ctx.refined_center,
            "refined_bbox": ctx.refined_bbox,
            "predicted_center": ctx.predicted_center,
            "lock_confidence": ctx.lock_confidence,
            "local_lock_score": ctx.local_lock_score,
            "jump_rejected": ctx.jump_rejected,
            "jump_px": ctx.last_jump_px,
            "anchor_jump_px": ctx.anchor_jump_px,
            "handoff_ready_score": ctx.handoff_ready_score,
            "handoff_ready_streak": ctx.handoff_ready_streak,
            "wide_center_jitter": self.readiness.center_jitter_px,
            "wide_bbox_jitter": self.readiness.bbox_size_jitter,
            "lock_loss_reason": ctx.lock_loss_reason,
            "owner_track_id": ctx.ownership.owner_track_id,
            "owner_raw_id": ctx.ownership.owner_raw_id,
            "owner_strength": ctx.owner_strength,
            "ownership_confidence": ctx.ownership.ownership_confidence,
            "identity_consistency_score": ctx.identity_consistency_score,
            "ownership_score": ctx.ownership_score,
            "pending_track_id": ctx.ownership.pending_track_id,
            "pending_frames": ctx.ownership.pending_frames,
            "transfer_reason": ctx.ownership.transfer_reason,
            "ownership_reject_reason": ctx.ownership.last_reject_reason,
            "jump_risk": ctx.jump_risk,
            "projected_center": ctx.projected_center,
            "projected_bbox": ctx.projected_bbox,
            "last_good_center": ctx.last_good_center,
            "last_good_bbox": ctx.last_good_bbox,
            "last_good_zoom": ctx.last_good_zoom,
        }
