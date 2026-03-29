import math


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / max(1.0, union)


def _area(box):
    x1, y1, x2, y2 = box
    return max(1.0, (x2 - x1) * (y2 - y1))


class StableTargetRegistry:
    def __init__(
        self,
        max_missing=25,
        lost_ttl=180,
        match_distance=140.0,
        recover_distance=220.0,
        min_iou=0.01,
    ):
        self.max_missing = int(max_missing)
        self.lost_ttl = int(lost_ttl)
        self.match_distance = float(match_distance)
        self.recover_distance = float(recover_distance)
        self.min_iou = float(min_iou)

        self.next_id = 1
        self.active = {}
        self.lost = {}

    def reset(self):
        self.next_id = 1
        self.active = {}
        self.lost = {}

    def _new_id(self):
        v = self.next_id
        self.next_id += 1
        return v

    def _make_state(self, tr, missing=0):
        return {
            "bbox_xyxy": tr.bbox_xyxy,
            "center_xy": tr.center_xy,
            "confidence": tr.confidence,
            "raw_id": tr.raw_id,
            "missing": int(missing),
        }

    def _match_pool(self, pool, tracks, used, distance_limit):
        assignments = {}

        for stable_id, state in pool.items():
            best_idx = None
            best_score = None

            for i, tr in enumerate(tracks):
                if i in used:
                    continue

                d = _dist(state["center_xy"], tr.center_xy)
                iou = _iou(state["bbox_xyxy"], tr.bbox_xyxy)

                if d > distance_limit and iou < self.min_iou:
                    continue

                area_old = _area(state["bbox_xyxy"])
                area_new = _area(tr.bbox_xyxy)
                area_ratio = min(area_old, area_new) / max(area_old, area_new)

                score = d - 160.0 * iou - 60.0 * area_ratio - 30.0 * tr.confidence

                if best_score is None or score < best_score:
                    best_score = score
                    best_idx = i

            if best_idx is not None:
                assignments[stable_id] = best_idx
                used.add(best_idx)

        return assignments

    def update(self, tracks):
        used = set()
        updated_tracks = []

        # 1. Najpierw dopasuj do aktywnych
        active_assignments = self._match_pool(
            self.active, tracks, used, self.match_distance
        )

        new_active = {}
        still_lost = dict(self.lost)

        # trafione aktywne
        for stable_id, idx in active_assignments.items():
            tr = tracks[idx]
            tr.track_id = stable_id
            new_active[stable_id] = self._make_state(tr, missing=0)
            updated_tracks.append(tr)

        # nietrafione aktywne -> przechodzą do lost
        for stable_id, state in self.active.items():
            if stable_id not in active_assignments:
                moved = dict(state)
                moved["missing"] = moved.get("missing", 0) + 1
                still_lost[stable_id] = moved

        # 2. Potem próbuj odzyskać stare lost ID
        recoverable_lost = {
            sid: state
            for sid, state in still_lost.items()
            if state.get("missing", 0) <= self.lost_ttl
        }

        lost_assignments = self._match_pool(
            recoverable_lost, tracks, used, self.recover_distance
        )

        for stable_id, idx in lost_assignments.items():
            tr = tracks[idx]
            tr.track_id = stable_id
            new_active[stable_id] = self._make_state(tr, missing=0)
            updated_tracks.append(tr)
            if stable_id in still_lost:
                del still_lost[stable_id]

        # 3. Cała reszta dostaje nowe ID
        for i, tr in enumerate(tracks):
            if i in used:
                continue
            stable_id = self._new_id()
            tr.track_id = stable_id
            new_active[stable_id] = self._make_state(tr, missing=0)
            updated_tracks.append(tr)

        # 4. Starzenie lost
        cleaned_lost = {}
        for stable_id, state in still_lost.items():
            state = dict(state)
            state["missing"] = state.get("missing", 0) + 1
            if state["missing"] <= self.lost_ttl:
                cleaned_lost[stable_id] = state

        self.active = new_active
        self.lost = cleaned_lost

        updated_tracks.sort(key=lambda t: t.track_id)
        return updated_tracks
