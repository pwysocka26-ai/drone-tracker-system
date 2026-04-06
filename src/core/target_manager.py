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
        switch_margin=0.48,
        switch_dwell=7,
        switch_cooldown=10,
        switch_persist=2,
        max_select_missed=2,
        min_start_conf=0.10,
        min_start_hits=2,
        min_confirmed_conf=0.10,
        min_hold_frames=5,
        predicted_dist_px=82.0,
        raw_id_bonus=1.8,
        current_target_bonus=2.6,
        raw_id_lock_bonus=3.8,
        takeoff_switch_margin=0.95,
        takeoff_switch_ratio=1.28,
        raw_id_match_radius_px=72.0,
        reacquire_prefer_same_raw_id=True,
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
        self.raw_id_lock_bonus = float(raw_id_lock_bonus)
        self.takeoff_switch_margin = float(takeoff_switch_margin)
        self.takeoff_switch_ratio = float(takeoff_switch_ratio)
        self.raw_id_match_radius_px = float(raw_id_match_radius_px)
        self.reacquire_prefer_same_raw_id = bool(reacquire_prefer_same_raw_id)

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
        score -= missed * 1.8

        if is_current:
            score += self.current_target_bonus

        anchor = predicted_center or self.last_selected_center
        if anchor is not None:
            dist = self._distance(tuple(tr.center_xy), anchor)
            score += max(-3.2, 1.4 - dist / max(1.0, self.predicted_dist_px))
            if dist > (self.predicted_dist_px * 1.35):
                score -= 1.2

        if self.last_selected_raw_id is not None and raw_id is not None:
            if int(raw_id) == int(self.last_selected_raw_id):
                score += self.raw_id_bonus
                if anchor is not None and self._distance(tuple(tr.center_xy), anchor) <= self.raw_id_match_radius_px:
                    score += self.raw_id_lock_bonus

        return score


    def _same_raw_id(self, tr, raw_id) -> bool:
        rid = getattr(tr, "raw_id", None)
        return rid is not None and raw_id is not None and int(rid) == int(raw_id)

    def _same_raw_id_near_anchor(self, tr, anchor) -> bool:
        if not self._same_raw_id(tr, self.last_selected_raw_id):
            return False
        if anchor is None:
            return True
        return self._distance(tuple(tr.center_xy), anchor) <= self.raw_id_match_radius_px

    def _switch_guard(self, active, best, current_score, best_score, anchor) -> bool:
        if active is None or best is None:
            return False
        active_conf = float(getattr(active, "confidence", 0.0) or 0.0)
        best_conf = float(getattr(best, "confidence", 0.0) or 0.0)
        active_hits = int(getattr(active, "hits", 0) or 0)
        best_hits = int(getattr(best, "hits", 0) or 0)
        active_missed = int(getattr(active, "missed_frames", 0) or 0)
        best_missed = int(getattr(best, "missed_frames", 0) or 0)

        best_same_raw = self._same_raw_id_near_anchor(best, anchor)
        active_same_raw = self._same_raw_id(active, self.last_selected_raw_id)

        if best_same_raw and not active_same_raw:
            return True

        if active_same_raw and not best_same_raw and active_missed <= 1 and active_hits >= 3:
            margin = best_score - current_score
            ratio = best_score / max(0.01, current_score)
            if margin < self.takeoff_switch_margin and ratio < self.takeoff_switch_ratio and best_conf < (active_conf + 0.18):
                return False

        if best_missed > 0 and active_missed == 0 and best_conf < (active_conf + 0.22):
            return False

        if best_hits < active_hits and best_conf < (active_conf + 0.12):
            return False

        return True

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
        if anchor is not None and self.last_selected_center is not None:
            if self._distance(anchor, self.last_selected_center) > max(self.reacquire_radius_auto * 1.15, self.raw_id_match_radius_px * 2.0):
                self.last_selected_raw_id = None

        if self.selected_id is None:
            same_raw_candidates = []
            if self.reacquire_prefer_same_raw_id and self.last_selected_raw_id is not None:
                for tr in candidates:
                    if self._same_raw_id_near_anchor(tr, anchor):
                        same_raw_candidates.append(tr)
            pool = same_raw_candidates if same_raw_candidates else candidates
            best = max(pool, key=lambda tr: self._score(tr, frame_shape, predicted_center))
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

        if active is None and anchor is not None:
            close = []
            for tr in candidates:
                dist = self._distance(tuple(tr.center_xy), anchor)
                if dist <= self.reacquire_radius_auto:
                    close.append(tr)

            if close:
                if self.reacquire_prefer_same_raw_id and self.last_selected_raw_id is not None:
                    same_raw_close = [tr for tr in close if self._same_raw_id_near_anchor(tr, anchor)]
                    pool = same_raw_close if same_raw_close else close
                else:
                    pool = close
                best = max(pool, key=lambda tr: self._score(tr, frame_shape, predicted_center))
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
            and best_score > (current_score * 1.10)
            and self.lock_age >= self.min_hold_frames
            and (self.frame_id - self.last_switch_frame) >= self.switch_cooldown
            and self._switch_guard(active, best, current_score, best_score, anchor)
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
