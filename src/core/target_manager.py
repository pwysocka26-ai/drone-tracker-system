from __future__ import annotations

import math
from typing import Optional, Tuple


Point = Tuple[float, float]


class TargetManager:
    """
    V3:
    - mniej agresywny niż v2
    - stabilny selected_id
    - łatwiejszy reacquire w AUTO
    - przełączenie tylko gdy kandydat jest lokalny i wyraźnie lepszy
    """

    def __init__(
        self,
        reacquire_radius_auto=145.0,
        reacquire_radius_manual=220.0,
        sticky_frames=22,
        switch_margin=0.38,
        switch_dwell=7,
        switch_cooldown=8,
        switch_persist=3,
        max_select_missed=1,
        min_start_conf=0.10,
        min_start_hits=2,
        min_confirmed_conf=0.10,
        min_hold_frames=6,
        predicted_dist_px=90.0,
        raw_id_bonus=2.0,
        current_target_bonus=2.8,
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

        self.lock_age = 9999
        self.frame_id = 0
        self.last_switch_frame = -999999
        self.pending_id = None
        self.pending_count = 0

        self.last_selected_center = None
        self.last_selected_raw_id = None

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
        score -= missed * 2.0

        if is_current:
            score += self.current_target_bonus

        anchor = predicted_center or self.last_selected_center
        if anchor is not None:
            dist = self._distance(tuple(tr.center_xy), anchor)
            score += max(-2.5, 1.25 - dist / max(1.0, self.predicted_dist_px))

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
            self.last_selected_center = tuple(active.center_xy)
            rid = getattr(active, "raw_id", None)
            if rid is not None:
                self.last_selected_raw_id = int(rid)

        if self.manual_lock:
            self.lock_age = 0 if active is not None else self.lock_age + 1
            return self.selected_id

        candidates = self._eligible_tracks(tracks)
        if not candidates:
            self.lock_age += 1
            return self.selected_id

        anchor = predicted_center or self.last_selected_center

        # initial acquire: small dwell, not instant
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
                self.last_selected_center = tuple(best.center_xy)
                rid = getattr(best, "raw_id", None)
                if rid is not None:
                    self.last_selected_raw_id = int(rid)
            return self.selected_id

        # reacquire locally
        if active is None and anchor is not None:
            close = []
            for tr in candidates:
                dist = self._distance(tuple(tr.center_xy), anchor)
                if dist <= self.reacquire_radius_auto:
                    close.append(tr)

            if close:
                best = max(close, key=lambda tr: self._score(tr, frame_shape, predicted_center))
                self.selected_id = int(best.track_id)
                self.lock_age = 0
                self.last_switch_frame = self.frame_id
                self.pending_id = None
                self.pending_count = 0
                self.last_selected_center = tuple(best.center_xy)
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
        best_score = self._score(best, frame_shape, predicted_center, is_current=False)

        if int(best.track_id) == int(self.selected_id):
            self.pending_id = None
            self.pending_count = 0
            self.lock_age += 1
            return self.selected_id

        near_anchor = self._distance(tuple(best.center_xy), anchor) <= min(self.reacquire_radius_auto, self.predicted_dist_px)

        need_switch = (
            near_anchor
            and best_score > (current_score + self.switch_margin)
            and best_score > (current_score * 1.12)
            and self.lock_age >= self.min_hold_frames
            and (self.frame_id - self.last_switch_frame) >= self.switch_cooldown
        )

        if need_switch:
            if self.pending_id == int(best.track_id):
                self.pending_count += 1
            else:
                self.pending_id = int(best.track_id)
                self.pending_count = 1

            if self.pending_count >= self.switch_persist:
                self.selected_id = int(best.track_id)
                self.lock_age = 0
                self.last_switch_frame = self.frame_id
                self.pending_id = None
                self.pending_count = 0
                self.last_selected_center = tuple(best.center_xy)
                rid = getattr(best, "raw_id", None)
                if rid is not None:
                    self.last_selected_raw_id = int(rid)
                return self.selected_id
        else:
            self.pending_id = None
            self.pending_count = 0

        self.lock_age += 1
        return self.selected_id
