from __future__ import annotations

import math
from typing import Optional, Tuple


Point = Tuple[float, float]


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
        owner_switch_min_gap_px=22.0,
        degraded_switch_persist=2,
        healthy_switch_persist=4,
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

        self.lock_age = 9999
        self.frame_id = 0
        self.last_switch_frame = -999999
        self.pending_id = None
        self.pending_count = 0

        self.last_selected_center = None
        self.last_selected_raw_id = None
        self.selection_freeze_id = None
        self.selection_freeze_left = 0

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

    def set_auto_mode(self):
        self.manual_lock = False
        self.selected_id = None
        self.lock_age = 9999
        self.pending_id = None
        self.pending_count = 0

    def set_manual_target(self, tid):
        self.manual_lock = True
        self.selected_id = int(tid)
        self.lock_age = 0
        self.last_switch_frame = self.frame_id
        self.pending_id = None
        self.pending_count = 0
        self.selection_freeze_id = None
        self.selection_freeze_left = 0

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

        return score

    def update(self, tracks, predicted_center, frame_shape):
        self.frame_id += 1
        tracks = list(tracks or [])
        if not tracks:
            self.lock_age += 1
            return self.selected_id

        active = self.find_active_track(tracks)
        if active is not None:
            self.last_selected_center = self._score_center(active)
            rid = getattr(active, "raw_id", None)
            if rid is not None:
                self.last_selected_raw_id = int(rid)

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
                self.last_selected_center = self._score_center(frozen)
                rid = getattr(frozen, "raw_id", None)
                if rid is not None:
                    self.last_selected_raw_id = int(rid)
                self.pending_id = None
                self.pending_count = 0
                self.lock_age += 1
                return self.selected_id
            if self.selection_freeze_left == 0:
                self.clear_freeze()

        candidates = self._eligible_tracks(tracks)
        if not candidates:
            self.lock_age += 1
            return self.selected_id

        anchor = predicted_center or self.last_selected_center

        if self.selected_id is None:
            best = max(candidates, key=lambda tr: self._score(tr, frame_shape, predicted_center))
            if self.pending_id == int(best.track_id):
                self.pending_count += 1
            else:
                self.pending_id = int(best.track_id)
                self.pending_count = 1

            if self.pending_count >= 2:
                self.selected_id = int(best.track_id)
                self.lock_age = 0
                self.last_switch_frame = self.frame_id
                self.pending_id = None
                self.pending_count = 0
                self.last_selected_center = self._score_center(best)
                rid = getattr(best, "raw_id", None)
                if rid is not None:
                    self.last_selected_raw_id = int(rid)
            return self.selected_id

        if active is None and anchor is not None:
            close = []
            for tr in candidates:
                dist = self._distance(self._score_center(tr), anchor)
                if dist <= self.reacquire_radius_auto:
                    close.append(tr)

            if close:
                best = max(close, key=lambda tr: self._score(tr, frame_shape, predicted_center))
                if self.pending_id == int(best.track_id):
                    self.pending_count += 1
                else:
                    self.pending_id = int(best.track_id)
                    self.pending_count = 1

                if self.pending_count >= max(2, self.degraded_switch_persist):
                    self.selected_id = int(best.track_id)
                    self.lock_age = 0
                    self.last_switch_frame = self.frame_id
                    self.pending_id = None
                    self.pending_count = 0
                    self.last_selected_center = self._score_center(best)
                    rid = getattr(best, "raw_id", None)
                    if rid is not None:
                        self.last_selected_raw_id = int(rid)
                return self.selected_id

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
                and dist_best_to_anchor >= dist_current_to_anchor * 0.92
                and center_gap < max(self.owner_switch_min_gap_px, self.predicted_dist_px * 0.35)
                and not same_raw
            ):
                self.pending_id = None
                self.pending_count = 0
                self.lock_age += 1
                return self.selected_id

        current_degraded = (current_missed >= 3) or (current_conf < 0.16)
        near_anchor = self._distance(best_center, anchor) <= min(self.reacquire_radius_auto, self.predicted_dist_px)

        margin = self.switch_margin * (0.30 if current_degraded else 1.20)
        ratio_ok = best_score > (current_score * (1.01 if current_degraded else 1.12))
        dwell_ok = self.lock_age >= (2 if current_degraded else max(self.min_hold_frames, self.switch_dwell))
        cooldown_ok = (self.frame_id - self.last_switch_frame) >= (2 if current_degraded else max(self.switch_cooldown, 8))
        geometry_separation_ok = center_gap >= (8.0 if current_degraded else self.owner_switch_min_gap_px)

        need_switch = (
            near_anchor
            and best_score > (current_score + margin)
            and ratio_ok
            and dwell_ok
            and cooldown_ok
            and geometry_separation_ok
        )

        if need_switch:
            if self.pending_id == int(best.track_id):
                self.pending_count += 1
            else:
                self.pending_id = int(best.track_id)
                self.pending_count = 1

            required_persist = self.degraded_switch_persist if current_degraded else max(self.healthy_switch_persist, self.switch_persist)
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
                self.freeze_to(self.selected_id, self.selection_freeze_frames + 4)
                return self.selected_id
        else:
            self.pending_id = None
            self.pending_count = 0

        self.lock_age += 1
        return self.selected_id