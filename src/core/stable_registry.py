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
        max_missing=20,
        lost_ttl=240,
        match_distance=180.0,
        recover_distance=320.0,
        min_iou=0.01,
        area_ratio_min=0.35,
    ):
        self.max_missing = int(max_missing)
        self.lost_ttl = int(lost_ttl)
        self.match_distance = float(match_distance)
        self.recover_distance = float(recover_distance)
        self.min_iou = float(min_iou)
        self.area_ratio_min = float(area_ratio_min)

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
            "bbox_xyxy": tuple(tr.bbox_xyxy),
            "center_xy": tuple(tr.center_xy),
            "confidence": float(tr.confidence),
            "raw_id": int(getattr(tr, "raw_id", tr.track_id)),
            "missing": int(missing),
        }

    def _compatible(self, state, tr, distance_limit):
        d = _dist(state["center_xy"], tr.center_xy)
        iou = _iou(state["bbox_xyxy"], tr.bbox_xyxy)

        old_area = _area(state["bbox_xyxy"])
        new_area = _area(tr.bbox_xyxy)
        area_ratio = min(old_area, new_area) / max(old_area, new_area)

        ok = (d <= distance_limit or iou >= self.min_iou) and area_ratio >= self.area_ratio_min
        return ok, d, iou, area_ratio

    def _score_match(self, state, tr, d, iou, area_ratio):
        conf = float(tr.confidence)
        raw_bonus = 0.0
        if int(getattr(tr, "raw_id", -1)) == int(state.get("raw_id", -999999)):
            raw_bonus = 35.0

        # im mniejszy score tym lepiej
        return (
            d
            - 220.0 * iou
            - 90.0 * area_ratio
            - 45.0 * conf
            - raw_bonus
        )

    def _assign(self, pool, tracks, used, distance_limit):
        pairs = []

        for stable_id, state in pool.items():
            for idx, tr in enumerate(tracks):
                if idx in used:
                    continue

                ok, d, iou, area_ratio = self._compatible(state, tr, distance_limit)
                if not ok:
                    continue

                score = self._score_match(state, tr, d, iou, area_ratio)
                pairs.append((score, stable_id, idx))

        pairs.sort(key=lambda x: x[0])

        assigned_ids = set()
        assigned_idx = set()
        assignments = {}

        for score, stable_id, idx in pairs:
            if stable_id in assigned_ids or idx in assigned_idx:
                continue
            assignments[stable_id] = idx
            assigned_ids.add(stable_id)
            assigned_idx.add(idx)
            used.add(idx)

        return assignments

    def update(self, tracks):
        used = set()
        output_tracks = []

        # 1. Dopasowanie do aktywnych ID
        active_assignments = self._assign(self.active, tracks, used, self.match_distance)

        new_active = {}
        new_lost = dict(self.lost)

        for stable_id, idx in active_assignments.items():
            tr = tracks[idx]
            tr.track_id = stable_id
            new_active[stable_id] = self._make_state(tr, missing=0)
            output_tracks.append(tr)

        # 2. Nietrafione aktywne przechodza do lost
        for stable_id, state in self.active.items():
            if stable_id not in active_assignments:
                moved = dict(state)
                moved["missing"] = moved.get("missing", 0) + 1
                new_lost[stable_id] = moved

        # 3. Proba odzyskania ID z lost
        recoverable_lost = {
            sid: state
            for sid, state in new_lost.items()
            if state.get("missing", 0) <= self.lost_ttl
        }

        lost_assignments = self._assign(recoverable_lost, tracks, used, self.recover_distance)

        for stable_id, idx in lost_assignments.items():
            tr = tracks[idx]
            tr.track_id = stable_id
            new_active[stable_id] = self._make_state(tr, missing=0)
            output_tracks.append(tr)
            if stable_id in new_lost:
                del new_lost[stable_id]

        # 4. Reszta dostaje nowe stale ID
        for idx, tr in enumerate(tracks):
            if idx in used:
                continue
            stable_id = self._new_id()
            tr.track_id = stable_id
            new_active[stable_id] = self._make_state(tr, missing=0)
            output_tracks.append(tr)

        # 5. Starzenie lost
        cleaned_lost = {}
        for stable_id, state in new_lost.items():
            st = dict(state)
            st["missing"] = st.get("missing", 0) + 1
            if st["missing"] <= self.lost_ttl:
                cleaned_lost[stable_id] = st

        self.active = new_active
        self.lost = cleaned_lost

        output_tracks.sort(key=lambda t: t.track_id)
        return output_tracks
