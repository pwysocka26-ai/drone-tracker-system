import numpy as np


class TargetManager:
    def __init__(self, reacquire_radius_auto=85.0, reacquire_radius_manual=120.0, sticky_frames=60):
        self.selected_id = None
        self.manual_lock = False
        self.reacquire_radius_auto = float(reacquire_radius_auto)
        self.reacquire_radius_manual = float(reacquire_radius_manual)
        self.sticky_frames = int(sticky_frames)
        self.lock_age = 9999

    def reset(self):
        self.selected_id = None
        self.manual_lock = False
        self.lock_age = 9999

    def set_manual_target(self, track_id: int):
        self.selected_id = int(track_id)
        self.manual_lock = True
        self.lock_age = 0

    def set_auto_mode(self):
        self.manual_lock = False
        self.selected_id = None
        self.lock_age = 9999

    def find_active_track(self, tracks):
        if self.selected_id is None:
            return None
        for tr in tracks:
            if tr.track_id == self.selected_id:
                return tr
        return None

    def _choose_best_auto_target(self, tracks, frame_shape):
        if not tracks:
            return None

        h, w = frame_shape[:2]
        cx = w / 2.0
        cy = h / 2.0
        best = None
        best_score = None

        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox_xyxy
            tcx, tcy = tr.center_xy
            area = max(1.0, (x2 - x1) * (y2 - y1))
            dist = np.hypot(tcx - cx, tcy - cy)

            # prefer: większy, pewniejszy, bliżej środka, wyżej w kadrze
            score = (
                tr.confidence * 1000.0
                + area * 0.020
                - dist * 0.60
                - y1 * 0.15
            )

            if best_score is None or score > best_score:
                best_score = score
                best = tr

        return best

    def update(self, tracks, predicted_center, frame_shape):
        if not tracks:
            self.lock_age += 1
            return self.selected_id

        active = self.find_active_track(tracks)
        if active is not None:
            self.lock_age = 0
            return self.selected_id

        # Jeśli chwilowo zgubiono target, próbuj odzyskać najbliższy do predykcji.
        if predicted_center is not None:
            radius = self.reacquire_radius_manual if self.manual_lock else self.reacquire_radius_auto
            candidates = []
            for t in tracks:
                dist = np.hypot(
                    t.center_xy[0] - predicted_center[0],
                    t.center_xy[1] - predicted_center[1],
                )
                candidates.append((dist, -t.confidence, t))
            candidates.sort(key=lambda x: (x[0], x[1]))
            best_dist, _, best_track = candidates[0]
            if best_dist <= radius:
                self.selected_id = best_track.track_id
                self.lock_age = 0
                return self.selected_id

        # MANUAL: nie przełączaj samoczynnie na "jakiś inny" cel.
        if self.manual_lock:
            self.lock_age += 1
            return self.selected_id

        # AUTO: trzymaj sticky przez kilka klatek, zanim zmienisz target.
        if self.selected_id is not None and self.lock_age < self.sticky_frames:
            self.lock_age += 1
            return self.selected_id

        best = self._choose_best_auto_target(tracks, frame_shape)
        if best is not None:
            self.selected_id = best.track_id
            self.lock_age = 0

        return self.selected_id
