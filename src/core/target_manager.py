from __future__ import annotations

import math
from typing import Optional, Tuple


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


class TargetManager:
    def __init__(
        self,
        reacquire_radius_auto=135.0,
        reacquire_radius_manual=220.0,
        sticky_frames=22,
        switch_margin=0.40,
        switch_dwell=7,
        switch_cooldown=9,
        switch_persist=3,
        max_select_missed=2,
        min_start_conf=0.10,
        min_start_hits=2,
        min_confirmed_conf=0.10,
        min_hold_frames=5,
        predicted_dist_px=95.0,
        raw_id_bonus=1.8,
        current_target_bonus=4.8,
        selection_freeze_frames=8,
        hard_keep_missed=1,
        hard_keep_conf=0.18,
        hard_switch_min_gain=1.10,
        owner_switch_min_gap_px=22.0,
        degraded_switch_persist=2,
        healthy_switch_persist=5,
        fallback_recover_radius=185.0,
        fallback_bbox_min_similarity=0.42,
        same_owner_commit_persist=3,
        stale_owner_frames=10,
        stale_global_switch_persist=4,
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
        self.owner_switch_min_gap_px = float(owner_switch_min_gap_px)
        self.degraded_switch_persist = int(degraded_switch_persist)
        self.healthy_switch_persist = int(healthy_switch_persist)
        self.fallback_recover_radius = float(fallback_recover_radius)
        self.fallback_bbox_min_similarity = float(fallback_bbox_min_similarity)
        self.same_owner_commit_persist = int(same_owner_commit_persist)
        self.stale_owner_frames = int(stale_owner_frames)
        self.stale_global_switch_persist = int(stale_global_switch_persist)

        self.lock_age = 9999
        self.frame_id = 0
        self.last_switch_frame = -999999
        self.pending_id = None
        self.pending_count = 0

        self.last_selected_center = None
        self.last_selected_bbox = None
        self.last_selected_raw_id = None
        self.selection_freeze_id = None
        self.selection_freeze_left = 0
        self.selected_missing_frames = 0
        self.external_owner_hint_id = None
        self.external_owner_hint_ttl = 0
        self.external_owner_hint_center = None
        self.external_owner_hint_strength = 1.0
        self.identity_anchor = None
        self.identity_anchor_freshness = 0
        self._last_update_tracks = []

    def reset(self):
        self.selected_id = None
        self.manual_lock = False
        self.lock_age = 9999
        self.frame_id = 0
        self.last_switch_frame = -999999
        self.pending_id = None
        self.pending_count = 0
        self.last_selected_center = None
        self.last_selected_bbox = None
        self.last_selected_raw_id = None
        self.selection_freeze_id = None
        self.selection_freeze_left = 0
        self.selected_missing_frames = 0
        self.external_owner_hint_id = None
        self.external_owner_hint_ttl = 0
        self.external_owner_hint_center = None
        self.external_owner_hint_strength = 1.0
        self.identity_anchor = None
        self.identity_anchor_freshness = 0
        self._last_update_tracks = []

    def set_auto_mode(self):
        self.manual_lock = False
        self.selected_id = None
        self.lock_age = 9999
        self.pending_id = None
        self.pending_count = 0
        self.selection_freeze_id = None
        self.selection_freeze_left = 0
        self.selected_missing_frames = 0
        self.external_owner_hint_id = None
        self.external_owner_hint_ttl = 0
        self.external_owner_hint_center = None
        self.external_owner_hint_strength = 1.0
        self.identity_anchor = None
        self.identity_anchor_freshness = 0
        self._last_update_tracks = []

    def set_manual_target(self, tid):
        self.manual_lock = True
        self.selected_id = int(tid)
        self.lock_age = 0
        self.last_switch_frame = self.frame_id
        self.pending_id = None
        self.pending_count = 0
        self.selection_freeze_id = None
        self.selection_freeze_left = 0
        self.selected_missing_frames = 0

    def freeze_to(self, tid, frames=None):
        if tid is None:
            return
        self.selection_freeze_id = int(tid)
        self.selection_freeze_left = int(self.selection_freeze_frames if frames is None else frames)

    def clear_freeze(self):
        self.selection_freeze_id = None
        self.selection_freeze_left = 0

    def set_external_owner_hint(self, tid, ttl=3, center=None, strength=1.0):
        if tid is None:
            self.clear_external_owner_hint()
            return
        self.external_owner_hint_id = int(tid)
        self.external_owner_hint_ttl = max(0, int(ttl))
        self.external_owner_hint_center = tuple(float(v) for v in center) if center is not None else None
        self.external_owner_hint_strength = float(strength)

    def clear_external_owner_hint(self):
        self.external_owner_hint_id = None
        self.external_owner_hint_ttl = 0
        self.external_owner_hint_center = None
        self.external_owner_hint_strength = 1.0

    def _consume_external_owner_hint(self):
        if self.external_owner_hint_ttl > 0:
            self.external_owner_hint_ttl = max(0, int(self.external_owner_hint_ttl) - 1)
        if self.external_owner_hint_ttl <= 0 and self.external_owner_hint_id is not None:
            self.clear_external_owner_hint()

    def _find_track_by_id(self, tracks, tid):
        if tid is None:
            return None
        for tr in tracks or []:
            if int(getattr(tr, "track_id", -1)) == int(tid):
                return tr
        return None

    def _apply_external_owner_bias(self, tracks, predicted_center):
        hint_id = self.external_owner_hint_id
        if hint_id is None:
            return predicted_center
        hinted = self._find_track_by_id(tracks, hint_id)
        if hinted is None:
            return predicted_center
        hinted_missed = int(getattr(hinted, "missed_frames", 0) or 0)
        hinted_conf = float(getattr(hinted, "confidence", 0.0) or 0.0)
        if hinted_missed > max(self.hard_keep_missed, 1):
            return predicted_center
        if hinted_conf < max(0.08, self.hard_keep_conf * 0.8):
            return predicted_center
        self.selected_id = int(hint_id)
        self.lock_age = 0
        self._store_owner_reference(hinted)
        hint_center = self.external_owner_hint_center or self._score_center(hinted)
        if hint_center is not None:
            return hint_center
        return predicted_center

    def _distance(self, a: Optional[Point], b: Optional[Point]) -> float:
        if a is None or b is None:
            return float("inf")
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _score_center(self, tr):
        pred = getattr(tr, "predicted_center_xy", None)
        ctr = getattr(tr, "center_xy", None)
        ref = pred or ctr
        if ref is None:
            return None
        return (float(ref[0]), float(ref[1]))

    def _bbox_similarity(self, a: Optional[BBox], b: Optional[BBox]) -> float:
        if a is None or b is None:
            return 0.0
        aw = max(1.0, float(a[2]) - float(a[0]))
        ah = max(1.0, float(a[3]) - float(a[1]))
        bw = max(1.0, float(b[2]) - float(b[0]))
        bh = max(1.0, float(b[3]) - float(b[1]))
        dw = abs(aw - bw) / max(aw, bw)
        dh = abs(ah - bh) / max(ah, bh)
        return max(0.0, 1.0 - 0.5 * (dw + dh))

    def _store_owner_reference(self, tr) -> None:
        if tr is None:
            return
        center = self._score_center(tr)
        if center is not None:
            self.last_selected_center = center
        bbox = getattr(tr, "bbox_xyxy", None)
        if bbox is not None:
            self.last_selected_bbox = tuple(float(v) for v in bbox)
        rid = getattr(tr, "raw_id", None)
        if rid is not None:
            self.last_selected_raw_id = int(rid)

    def _recovery_radius(self) -> float:
        growth = 1.0 + min(0.80, 0.08 * float(self.selected_missing_frames))
        return max(self.reacquire_radius_auto, self.fallback_recover_radius) * growth

    def _fallback_same_owner_score(self, tr) -> float:
        center = self._score_center(tr)
        ref_center = self.last_selected_center
        dist = self._distance(center, ref_center)
        sim = self._bbox_similarity(getattr(tr, "bbox_xyxy", None), self.last_selected_bbox)
        conf = float(getattr(tr, "confidence", 0.0) or 0.0)
        raw_match = (
            self.last_selected_raw_id is not None
            and getattr(tr, "raw_id", None) is not None
            and int(getattr(tr, "raw_id")) == int(self.last_selected_raw_id)
        )
        raw_bonus = 1.8 if raw_match else 0.0
        radius = max(1.0, self._recovery_radius())
        return sim * 8.0 + conf * 5.0 + raw_bonus - (dist / radius) * 2.5

    def _same_owner_fallback_candidates(self, tracks):
        if self.last_selected_center is None and self.last_selected_bbox is None:
            return []

        radius = self._recovery_radius()
        out = []
        for tr in tracks:
            missed = int(getattr(tr, "missed_frames", 0) or 0)
            conf = float(getattr(tr, "confidence", 0.0) or 0.0)
            if missed > max(self.max_select_missed + 2, self.hard_keep_missed + 2):
                continue
            if conf < max(0.05, self.min_confirmed_conf * 0.60):
                continue

            center = self._score_center(tr)
            dist = self._distance(center, self.last_selected_center)
            sim = self._bbox_similarity(getattr(tr, "bbox_xyxy", None), self.last_selected_bbox)
            raw_match = (
                self.last_selected_raw_id is not None
                and getattr(tr, "raw_id", None) is not None
                and int(getattr(tr, "raw_id")) == int(self.last_selected_raw_id)
            )

            if raw_match:
                out.append(tr)
                continue

            strong_geom = sim >= max(self.fallback_bbox_min_similarity, 0.58) and dist <= radius * 1.15
            plausible_geom = sim >= self.fallback_bbox_min_similarity and dist <= radius
            if strong_geom or plausible_geom:
                out.append(tr)
        return out

    def find_active_track(self, tracks):
        if self.selected_id is None:
            return None

        tracks = list(tracks or [])
        if not tracks:
            return None

        for tr in tracks:
            if int(getattr(tr, "track_id", -1)) == int(self.selected_id):
                return tr

        fallback_candidates = []
        for tr in tracks:
            missed = int(getattr(tr, "missed_frames", 0) or 0)
            conf = float(getattr(tr, "confidence", 0.0) or 0.0)
            if missed > max(self.max_select_missed + 1, self.hard_keep_missed + 1):
                continue
            if conf < max(0.06, self.min_confirmed_conf * 0.8):
                continue
            fallback_candidates.append(tr)

        if not fallback_candidates:
            return None

        if self.last_selected_raw_id is not None:
            raw_matches = [
                tr for tr in fallback_candidates
                if getattr(tr, "raw_id", None) is not None
                and int(getattr(tr, "raw_id")) == int(self.last_selected_raw_id)
            ]
            if raw_matches:
                if self.last_selected_center is not None:
                    return min(raw_matches, key=lambda tr: self._distance(self._score_center(tr), self.last_selected_center))
                return raw_matches[0]

        same_owner = self._same_owner_fallback_candidates(fallback_candidates)
        if same_owner:
            return max(same_owner, key=self._fallback_same_owner_score)

        return None

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

            out.append(tr)
        return out

    def _score(self, tr, frame_shape, predicted_center=None, is_current=False):
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

        cx = x / max(1.0, w)
        cy = y / max(1.0, h)
        center_dist2 = (cx - 0.5) ** 2 + (cy - 0.5) ** 2

        score = 0.0
        score += conf * 10.0
        score += min(area_norm * 220.0, 2.0)
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

        if self.last_selected_bbox is not None:
            score += self._bbox_similarity(getattr(tr, "bbox_xyxy", None), self.last_selected_bbox) * 2.0

        return score

    def _compute_formation_signature(self, track, tracks):
        center = self._score_center(track)
        if center is None:
            return tuple()
        x1, y1, x2, y2 = track.bbox_xyxy
        area = max(1.0, (float(x2) - float(x1)) * (float(y2) - float(y1)))
        tid = int(getattr(track, "track_id", -1))
        neighbors = []
        for tr in tracks or []:
            if int(getattr(tr, "track_id", -1)) == tid:
                continue
            nc = self._score_center(tr)
            if nc is None:
                continue
            nx1, ny1, nx2, ny2 = tr.bbox_xyxy
            narea = max(1.0, (float(nx2) - float(nx1)) * (float(ny2) - float(ny1)))
            neighbors.append((nc[0] - center[0], nc[1] - center[1], narea / area))
        neighbors.sort(key=lambda v: v[0] * v[0] + v[1] * v[1])
        return tuple(neighbors[:3])

    def _signature_similarity(self, a, b):
        if not a or not b:
            return 0.5
        n = min(len(a), len(b))
        if n == 0:
            return 0.5
        sims = []
        for i in range(n):
            dx_a, dy_a, ar_a = a[i]
            dx_b, dy_b, ar_b = b[i]
            dist_diff = math.hypot(dx_a - dx_b, dy_a - dy_b)
            ar_diff = abs(math.log(max(0.01, ar_a) / max(0.01, ar_b)))
            sims.append(math.exp(-dist_diff / 50.0) * math.exp(-ar_diff))
        return sum(sims) / len(sims)

    def _continuity_score(self, track, tracks):
        if self.identity_anchor is None or track is None:
            return None
        center = self._score_center(track)
        if center is None:
            return 0.0
        avx, avy = self.identity_anchor["velocity"]
        velocity = getattr(track, "velocity_xy", (0.0, 0.0)) or (0.0, 0.0)
        dv = math.hypot(velocity[0] - avx, velocity[1] - avy)
        motion = math.exp(-(dv * dv) / (2.0 * 15.0 * 15.0))

        x1, y1, x2, y2 = track.bbox_xyxy
        area = max(1.0, (float(x2) - float(x1)) * (float(y2) - float(y1)))
        anchor_area = max(1.0, float(self.identity_anchor["area"]))
        log_ratio = math.log(area / anchor_area)
        scale = math.exp(-(log_ratio * log_ratio) / (2.0 * 0.5 * 0.5))

        current_sig = self._compute_formation_signature(track, tracks)
        formation = self._signature_similarity(current_sig, self.identity_anchor["formation_sig"])

        return 0.35 * motion + 0.20 * scale + 0.45 * formation

    def _set_identity_anchor(self, track, tracks):
        if track is None:
            return
        x1, y1, x2, y2 = track.bbox_xyxy
        area = max(1.0, (float(x2) - float(x1)) * (float(y2) - float(y1)))
        velocity = getattr(track, "velocity_xy", (0.0, 0.0)) or (0.0, 0.0)
        formation_sig = self._compute_formation_signature(track, tracks)
        self.identity_anchor = {
            "area": float(area),
            "velocity": (float(velocity[0]), float(velocity[1])),
            "formation_sig": formation_sig,
            "frame_idx": int(self.frame_id),
        }
        self.identity_anchor_freshness = 0

    def _advance_pending(self, candidate_id: int) -> int:
        if self.pending_id == int(candidate_id):
            self.pending_count += 1
        else:
            self.pending_id = int(candidate_id)
            self.pending_count = 1
        return self.pending_count

    def _commit_selected(self, tr, freeze_extra=0):
        self.selected_id = int(getattr(tr, "track_id"))
        self.lock_age = 0
        self.last_switch_frame = self.frame_id
        self.pending_id = None
        self.pending_count = 0
        self.selected_missing_frames = 0
        self._store_owner_reference(tr)
        self.freeze_to(self.selected_id, self.selection_freeze_frames + int(freeze_extra))
        self._set_identity_anchor(tr, self._last_update_tracks)
        return self.selected_id

    def update(self, tracks, predicted_center, frame_shape):
        self.frame_id += 1
        self._consume_external_owner_hint()
        if self.identity_anchor is not None:
            self.identity_anchor_freshness += 1
        tracks = list(tracks or [])
        self._last_update_tracks = tracks
        if not tracks:
            self.selected_missing_frames += 1
            self.lock_age += 1
            return self.selected_id

        active = self.find_active_track(tracks)
        if active is not None:
            self.selected_missing_frames = 0
            self._store_owner_reference(active)
        else:
            self.selected_missing_frames += 1

        if self.manual_lock:
            self.lock_age = 0 if active is not None else self.lock_age + 1
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
                self._store_owner_reference(frozen)
                self.pending_id = None
                self.pending_count = 0
                self.lock_age += 1
                self.selected_missing_frames = 0
                return self.selected_id
            if self.selection_freeze_left == 0:
                self.clear_freeze()

        candidates = self._eligible_tracks(tracks)
        if not candidates:
            self.lock_age += 1
            return self.selected_id

        predicted_center = self._apply_external_owner_bias(candidates, predicted_center)
        anchor = predicted_center or self.last_selected_center

        if self.selected_id is None:
            best = max(candidates, key=lambda tr: self._score(tr, frame_shape, predicted_center))
            pending = self._advance_pending(int(best.track_id))
            if pending >= 2:
                return self._commit_selected(best, freeze_extra=0)
            return self.selected_id

        # If exact ID vanished but a geometrically-consistent replacement exists,
        # adopt the new track ID quickly instead of staying latched to a ghost owner.
        if active is not None and int(getattr(active, "track_id", -1)) != int(self.selected_id):
            same_raw = (
                self.last_selected_raw_id is not None
                and getattr(active, "raw_id", None) is not None
                and int(getattr(active, "raw_id")) == int(self.last_selected_raw_id)
            )
            required = 1 if same_raw else max(2, self.same_owner_commit_persist)
            pending = self._advance_pending(int(getattr(active, "track_id")))
            if pending >= required:
                return self._commit_selected(active, freeze_extra=2)
            return self.selected_id

        if active is None and anchor is not None:
            close = []
            reacquire_radius = max(self.reacquire_radius_auto, self._recovery_radius())
            for tr in candidates:
                dist = self._distance(self._score_center(tr), anchor)
                if dist <= reacquire_radius:
                    close.append(tr)

            if close:
                best = max(close, key=lambda tr: self._score(tr, frame_shape, predicted_center))
                pending = self._advance_pending(int(best.track_id))
                if pending >= max(2, self.degraded_switch_persist):
                    return self._commit_selected(best, freeze_extra=2)
                return self.selected_id

            # Hard stale-owner escape hatch: if we have visible candidates for a while,
            # stop clinging to a dead owner and move to the strongest visible target.
            if self.selected_missing_frames >= self.stale_owner_frames:
                global_best = max(candidates, key=lambda tr: self._score(tr, frame_shape, predicted_center))
                pending = self._advance_pending(int(global_best.track_id))
                if pending >= max(2, self.stale_global_switch_persist):
                    return self._commit_selected(global_best, freeze_extra=1)

            self.lock_age += 1
            return self.selected_id

        if active is None:
            self.lock_age += 1
            return self.selected_id

        current_score = self._score(active, frame_shape, predicted_center, is_current=True)
        best = max(
            candidates,
            key=lambda tr: self._score(
                tr,
                frame_shape,
                predicted_center,
                is_current=(int(getattr(tr, "track_id", -1)) == int(self.selected_id)),
            ),
        )

        if int(best.track_id) == int(self.selected_id):
            self.pending_id = None
            self.pending_count = 0
            self.lock_age += 1
            return self.selected_id

        best_score = self._score(best, frame_shape, predicted_center, is_current=False)

        current_conf = float(getattr(active, "confidence", 0.0) or 0.0)
        current_missed = int(getattr(active, "missed_frames", 0) or 0)
        current_center = self._score_center(active)
        best_center = self._score_center(best)

        current_healthy = (
            bool(getattr(active, "is_confirmed", False))
            and current_missed <= self.hard_keep_missed
            and current_conf >= self.hard_keep_conf
        )

        score_gap = best_score - current_score
        center_gap = self._distance(best_center, current_center)
        dist_best_to_anchor = self._distance(best_center, anchor)
        dist_current_to_anchor = self._distance(current_center, anchor)
        same_raw = (
            self.last_selected_raw_id is not None
            and getattr(best, "raw_id", None) is not None
            and int(getattr(best, "raw_id")) == int(self.last_selected_raw_id)
        )

        if current_healthy:
            if (
                score_gap < self.hard_switch_min_gain
                and dist_best_to_anchor >= dist_current_to_anchor * 0.88
                and center_gap < max(self.owner_switch_min_gap_px, self.predicted_dist_px * 0.35)
                and not same_raw
            ):
                self.pending_id = None
                self.pending_count = 0
                self.lock_age += 1
                return self.selected_id

        current_degraded = (current_missed >= 3) or (current_conf < 0.16)
        near_anchor = self._distance(best_center, anchor) <= min(max(self.reacquire_radius_auto, self._recovery_radius()), self.predicted_dist_px * 1.8)

        margin = self.switch_margin * (0.35 if current_degraded else 1.35)
        ratio_ok = best_score > (current_score * (1.02 if current_degraded else 1.16))
        dwell_ok = self.lock_age >= (2 if current_degraded else max(self.min_hold_frames, self.switch_dwell))
        cooldown_ok = (self.frame_id - self.last_switch_frame) >= (3 if current_degraded else max(self.switch_cooldown, 10))
        geometry_separation_ok = center_gap >= (8.0 if current_degraded else self.owner_switch_min_gap_px)

        velocity_coherent = True
        if not current_degraded:
            avx, avy = getattr(active, "velocity_xy", (0.0, 0.0)) or (0.0, 0.0)
            bvx, bvy = getattr(best, "velocity_xy", (0.0, 0.0)) or (0.0, 0.0)
            owner_speed = math.sqrt(avx * avx + avy * avy)
            if owner_speed >= 4.0:
                dot = avx * bvx + avy * bvy
                velocity_coherent = dot >= 0.0

        # Identity-continuity guard. Jeśli anchor istnieje, kandydat musi mieć
        # porównywalny lub wyższy continuity_score niż obecny owner — chroni
        # przed przełączeniem na geometrycznie-zbliżonego sąsiada w formacji
        # identycznych obiektów (klasyczny scenariusz wrong-adjacent-object).
        continuity_ok = True
        if self.identity_anchor is not None:
            best_cont = self._continuity_score(best, tracks)
            active_cont = self._continuity_score(active, tracks)
            if best_cont is not None and active_cont is not None:
                if best_cont < active_cont * 0.9:
                    continuity_ok = False

        need_switch = (
            near_anchor
            and best_score > (current_score + margin)
            and ratio_ok
            and dwell_ok
            and cooldown_ok
            and geometry_separation_ok
            and velocity_coherent
            and continuity_ok
        )

        if need_switch:
            pending = self._advance_pending(int(best.track_id))
            required_persist = max(2, self.degraded_switch_persist) if current_degraded else max(self.healthy_switch_persist, self.switch_persist)
            if pending >= required_persist:
                return self._commit_selected(best, freeze_extra=4)
        else:
            self.pending_id = None
            self.pending_count = 0

        self.lock_age += 1
        return self.selected_id
