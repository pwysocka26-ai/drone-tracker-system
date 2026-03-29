import math


class TargetFilter:
    def __init__(
        self,
        min_age_frames=6,
        min_area=80.0,
        max_area=120000.0,
        max_jump=220.0,
        max_aspect_ratio=8.0,
    ):
        self.min_age_frames = int(min_age_frames)
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.max_jump = float(max_jump)
        self.max_aspect_ratio = float(max_aspect_ratio)

        self.history = {}

    def reset(self):
        self.history = {}

    def _bbox_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return max(1.0, (x2 - x1) * (y2 - y1))

    def _bbox_aspect(self, bbox):
        x1, y1, x2, y2 = bbox
        w = max(1.0, (x2 - x1))
        h = max(1.0, (y2 - y1))
        r = max(w / h, h / w)
        return r

    def _dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def update(self, tracks):
        valid_tracks = []
        seen_ids = set()

        for tr in tracks:
            tid = int(tr.track_id)
            seen_ids.add(tid)

            area = self._bbox_area(tr.bbox_xyxy)
            aspect = self._bbox_aspect(tr.bbox_xyxy)

            state = self.history.get(tid, {
                "age": 0,
                "last_center": tr.center_xy,
                "last_area": area,
                "valid": False,
            })

            jump = self._dist(tr.center_xy, state["last_center"])
            area_ratio = min(area, state["last_area"]) / max(area, state["last_area"])

            state["age"] += 1
            state["last_center"] = tr.center_xy
            state["last_area"] = area

            geometry_ok = (
                area >= self.min_area and
                area <= self.max_area and
                aspect <= self.max_aspect_ratio
            )

            motion_ok = (
                jump <= self.max_jump and
                area_ratio >= 0.20
            )

            if state["age"] >= self.min_age_frames and geometry_ok and motion_ok:
                state["valid"] = True

            self.history[tid] = state

            tr.is_valid_target = state["valid"]

            if state["valid"]:
                valid_tracks.append(tr)

        # usuń stare wpisy których już nie ma
        to_delete = [tid for tid in self.history.keys() if tid not in seen_ids]
        for tid in to_delete:
            del self.history[tid]

        return valid_tracks
