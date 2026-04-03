class TargetManager:
    def __init__(self, reacquire_radius_auto=140.0, reacquire_radius_manual=260.0, sticky_frames=20, switch_margin=0.25, switch_dwell=6):
        self.selected_id = None
        self.manual_lock = False
        self.last_switch_frame = 0
        self.frame_id = 0
        self.sticky_frames = int(sticky_frames)
        self.switch_margin = float(switch_margin)
        self.switch_dwell = int(switch_dwell)
        self.reacquire_radius_auto = float(reacquire_radius_auto)
        self.reacquire_radius_manual = float(reacquire_radius_manual)
        self.lock_age = 0

    def set_auto_mode(self):
        self.manual_lock = False

    def set_manual_target(self, tid):
        self.manual_lock = True
        self.selected_id = tid
        self.lock_age = 0
        self.last_switch_frame = self.frame_id

    def update(self, tracks, predicted_center, frame_shape):
        self.frame_id += 1
        tracks = list(tracks or [])
        if not tracks:
            self.lock_age += 1
            return
        if self.manual_lock:
            keep = next((t for t in tracks if t.track_id == self.selected_id), None)
            self.lock_age = 0 if keep is not None else self.lock_age + 1
            return

        best = None
        best_score = -1e9
        for tr in tracks:
            score = self._score(tr, frame_shape, predicted_center)
            if tr.track_id == self.selected_id:
                score += 5.0
            if score > best_score:
                best_score = score
                best = tr

        if best is None:
            self.lock_age += 1
            return

        if self.selected_id is None:
            self.selected_id = best.track_id
            self.lock_age = 0
            self.last_switch_frame = self.frame_id
            return

        current = next((t for t in tracks if t.track_id == self.selected_id), None)
        if current is None:
            self.selected_id = best.track_id
            self.lock_age = 0
            self.last_switch_frame = self.frame_id
            return

        current_score = self._score(current, frame_shape, predicted_center)
        if best.track_id != self.selected_id and best_score > current_score * (1.0 + self.switch_margin) and (self.frame_id - self.last_switch_frame) > self.switch_dwell:
            self.selected_id = best.track_id
            self.last_switch_frame = self.frame_id
            self.lock_age = 0
        else:
            self.lock_age += 1

    def _score(self, tr, frame_shape, predicted_center=None):
        h, w = frame_shape[:2]
        x, y = tr.center_xy
        cx = x / max(1.0, w)
        cy = y / max(1.0, h)
        center_dist = (cx - 0.5) ** 2 + (cy - 0.5) ** 2
        center_score = 1.0 - center_dist * 3.0
        conf = float(getattr(tr, "confidence", 0.0))
        area = (tr.bbox_xyxy[2] - tr.bbox_xyxy[0]) * (tr.bbox_xyxy[3] - tr.bbox_xyxy[1])
        area_norm = area / max(1.0, (w * h))
        score = conf * 5.0 + center_score * 2.0 + min(area_norm * 200.0, 2.0)
        if getattr(tr, "is_confirmed", False):
            score += 2.0
        if predicted_center is not None:
            px, py = predicted_center
            dx = x - px
            dy = y - py
            dist2 = dx * dx + dy * dy
            score += max(-2.0, 1.2 - dist2 / 25000.0)
        return score
