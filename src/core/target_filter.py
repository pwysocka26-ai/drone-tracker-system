import math


class TargetFilter:
    def __init__(
        self,
        min_age_frames=4,
        min_area=35.0,
        max_area_ratio=0.05,
        max_jump=180.0,
        min_stability_score=0.28,
    ):
        self.min_age_frames = int(min_age_frames)
        self.min_area = float(min_area)
        self.max_area_ratio = float(max_area_ratio)
        self.max_jump = float(max_jump)
        self.min_stability_score = float(min_stability_score)

        self.state = {}

    def reset(self):
        self.state = {}

    def _area(self, bbox):
        x1, y1, x2, y2 = bbox
        return max(1.0, (x2 - x1) * (y2 - y1))

    def _dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, tracks, frame_shape):
        h, w = frame_shape[:2]
        frame_area = float(w * h)
        seen = set()

        for tr in tracks:
            tid = int(tr.track_id)
            seen.add(tid)

            area = self._area(tr.bbox_xyxy)
            cy = tr.center_xy[1]

            prev = self.state.get(
                tid,
                {
                    "age": 0,
                    "last_center": tr.center_xy,
                    "last_area": area,
                    "score": 0.0,
                },
            )

            age = prev["age"] + 1
            jump = self._dist(tr.center_xy, prev["last_center"])
            area_ratio = min(area, prev["last_area"]) / max(area, prev["last_area"])

            # --- scoring ---
            score = 0.0

            # confidence
            score += min(1.0, float(tr.confidence)) * 0.25

            # lifetime
            score += min(1.0, age / max(1.0, self.min_age_frames)) * 0.25

            # motion stability
            motion_score = 1.0 - min(1.0, jump / max(1.0, self.max_jump))
            score += motion_score * 0.20

            # bbox stability
            score += area_ratio * 0.15

            # sky bonus / lower frame penalty
            sky_score = 1.0 - min(1.0, cy / max(1.0, h))
            score += sky_score * 0.15

            # hard rejections
            area_ratio_frame = area / max(1.0, frame_area)
            rejected = False

            if area < self.min_area:
                rejected = True

            if area_ratio_frame > self.max_area_ratio:
                rejected = True

            if jump > self.max_jump * 1.8:
                rejected = True

            valid = (not rejected) and age >= self.min_age_frames and score >= self.min_stability_score

            tr.target_score = score
            tr.is_valid_target = valid

            self.state[tid] = {
                "age": age,
                "last_center": tr.center_xy,
                "last_area": area,
                "score": score,
            }

        # cleanup starych tracków
        stale = [tid for tid in self.state.keys() if tid not in seen]
        for tid in stale:
            del self.state[tid]

        return tracks
