import math


class TargetManager:
    def __init__(
        self,
        reacquire_radius_auto=160.0,
        reacquire_radius_manual=260.0,
        sticky_frames=22,
        switch_margin=0.18,
        switch_dwell=4,
        switch_cooldown=6,
        max_select_missed=1,
        min_start_conf=0.10,
    ):
        self.selected_id = None
        self.manual_lock = False
        self.reacquire_radius_auto = float(reacquire_radius_auto)
        self.reacquire_radius_manual = float(reacquire_radius_manual)
        self.sticky_frames = int(sticky_frames)
        self.switch_margin = float(switch_margin)
        self.switch_dwell = int(switch_dwell)
        self.switch_cooldown = int(switch_cooldown)
        self.max_select_missed = int(max_select_missed)
        self.min_start_conf = float(min_start_conf)

        self.lock_age = 9999
        self.frame_id = 0
        self.last_switch_frame = -999999
        self.pending_id = None
        self.pending_count = 0

    def reset(self):
        self.selected_id = None
        self.manual_lock = False
        self.lock_age = 9999
        self.frame_id = 0
        self.last_switch_frame = -999999
        self.pending_id = None
        self.pending_count = 0

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
            if int(getattr(tr, 'track_id', -1)) == int(self.selected_id):
                return tr
        return None

    def _eligible_tracks(self, tracks):
        out = []
        for tr in tracks or []:
            missed = int(getattr(tr, 'missed_frames', 0) or 0)
            if missed > self.max_select_missed:
                continue

            conf = float(getattr(tr, 'confidence', 0.0) or 0.0)
            confirmed = bool(getattr(tr, 'is_confirmed', False))
            is_current = self.selected_id is not None and int(getattr(tr, 'track_id', -1)) == int(self.selected_id)

            if self.selected_id is None:
                if (not confirmed) and (conf < self.min_start_conf):
                    continue
            else:
                if (not confirmed) and (not is_current):
                    continue

            out.append(tr)
        return out

    def _score(self, tr, frame_shape, predicted_center=None):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = tr.bbox_xyxy
        x, y = tr.center_xy
        conf = float(getattr(tr, 'confidence', 0.0) or 0.0)
        hits = int(getattr(tr, 'hits', 0) or 0)
        missed = int(getattr(tr, 'missed_frames', 0) or 0)
        confirmed = bool(getattr(tr, 'is_confirmed', False))

        area = max(1.0, (x2 - x1) * (y2 - y1))
        area_norm = area / max(1.0, float(w * h))

        cx = x / max(1.0, w)
        cy = y / max(1.0, h)
        center_dist2 = (cx - 0.5) ** 2 + (cy - 0.5) ** 2

        score = 0.0
        score += conf * 10.0
        score += min(area_norm * 220.0, 2.0)
        score += max(-1.2, 1.0 - center_dist2 * 3.0)
        score += min(1.5, hits * 0.25)
        if confirmed:
            score += 2.0
        score -= missed * 1.8

        if predicted_center is not None:
            px, py = predicted_center
            dist = math.hypot(x - px, y - py)
            score += max(-2.0, 1.2 - dist / 110.0)

        return score

    def update(self, tracks, predicted_center, frame_shape):
        self.frame_id += 1
        tracks = list(tracks or [])
        if not tracks:
            self.lock_age += 1
            return self.selected_id

        active = self.find_active_track(tracks)
        if self.manual_lock:
            self.lock_age = 0 if active is not None else self.lock_age + 1
            return self.selected_id

        candidates = self._eligible_tracks(tracks)
        if not candidates:
            self.lock_age += 1
            return self.selected_id

        if self.selected_id is not None and active is None and predicted_center is not None:
            close = []
            for tr in candidates:
                dist = math.hypot(tr.center_xy[0] - predicted_center[0], tr.center_xy[1] - predicted_center[1])
                if dist <= self.reacquire_radius_auto:
                    close.append(tr)
            if close:
                best = max(close, key=lambda tr: self._score(tr, frame_shape, predicted_center))
                self.selected_id = best.track_id
                self.lock_age = 0
                self.last_switch_frame = self.frame_id
                self.pending_id = None
                self.pending_count = 0
                return self.selected_id

        best = max(candidates, key=lambda tr: self._score(tr, frame_shape, predicted_center))
        best_score = self._score(best, frame_shape, predicted_center)

        if self.selected_id is None:
            self.selected_id = best.track_id
            self.lock_age = 0
            self.last_switch_frame = self.frame_id
            self.pending_id = None
            self.pending_count = 0
            return self.selected_id

        if active is None:
            self.selected_id = best.track_id
            self.lock_age = 0
            self.last_switch_frame = self.frame_id
            self.pending_id = None
            self.pending_count = 0
            return self.selected_id

        current_score = self._score(active, frame_shape, predicted_center)
        current_missed = int(getattr(active, 'missed_frames', 0) or 0)
        best_confirmed = bool(getattr(best, 'is_confirmed', False))

        if best.track_id != self.selected_id and current_missed > 0 and best_confirmed:
            self.selected_id = best.track_id
            self.lock_age = 0
            self.last_switch_frame = self.frame_id
            self.pending_id = None
            self.pending_count = 0
            return self.selected_id

        need_switch = (
            best.track_id != self.selected_id
            and best_score > (current_score + self.switch_margin)
            and (self.frame_id - self.last_switch_frame) >= self.switch_cooldown
        )

        if need_switch:
            if self.pending_id == best.track_id:
                self.pending_count += 1
            else:
                self.pending_id = best.track_id
                self.pending_count = 1

            if self.pending_count >= self.switch_dwell:
                self.selected_id = best.track_id
                self.lock_age = 0
                self.last_switch_frame = self.frame_id
                self.pending_id = None
                self.pending_count = 0
                return self.selected_id
        else:
            self.pending_id = None
            self.pending_count = 0

        self.lock_age = 0
        return self.selected_id
