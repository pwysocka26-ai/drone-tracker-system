import math


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


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


class StableTargetRegistry:
    def __init__(self, max_missing=25, match_distance=140.0, min_iou=0.01):
        self.max_missing = int(max_missing)
        self.match_distance = float(match_distance)
        self.min_iou = float(min_iou)
        self.next_id = 1
        self.targets = {}

    def reset(self):
        self.next_id = 1
        self.targets = {}

    def _new_id(self):
        v = self.next_id
        self.next_id += 1
        return v

    def update(self, tracks):
        # tracks: lista obiektów z bbox_xyxy, center_xy, confidence, raw_id
        updated = []
        det_used = set()

        live_ids = list(self.targets.keys())
        # Najpierw próbuj dopasować stare targety do nowych detekcji
        for stable_id in live_ids:
            tgt = self.targets.get(stable_id)
            if tgt is None:
                continue

            best_idx = None
            best_score = None

            for i, tr in enumerate(tracks):
                if i in det_used:
                    continue

                d = _dist(tgt["center_xy"], tr.center_xy)
                iou = _iou(tgt["bbox_xyxy"], tr.bbox_xyxy)

                if d > self.match_distance and iou < self.min_iou:
                    continue

                # preferuj bliskość + zgodność bbox
                score = d - 120.0 * iou - 40.0 * tr.confidence

                if best_score is None or score < best_score:
                    best_score = score
                    best_idx = i

            if best_idx is not None:
                tr = tracks[best_idx]
                det_used.add(best_idx)

                tr.track_id = stable_id
                self.targets[stable_id] = {
                    "bbox_xyxy": tr.bbox_xyxy,
                    "center_xy": tr.center_xy,
                    "confidence": tr.confidence,
                    "raw_id": tr.raw_id,
                    "missing": 0,
                }
                updated.append(tr)
            else:
                self.targets[stable_id]["missing"] += 1

        # Nowe targety
        for i, tr in enumerate(tracks):
            if i in det_used:
                continue
            stable_id = self._new_id()
            tr.track_id = stable_id
            self.targets[stable_id] = {
                "bbox_xyxy": tr.bbox_xyxy,
                "center_xy": tr.center_xy,
                "confidence": tr.confidence,
                "raw_id": tr.raw_id,
                "missing": 0,
            }
            updated.append(tr)

        # Sprzątaj bardzo stare
        to_delete = []
        for stable_id, tgt in self.targets.items():
            if tgt["missing"] > self.max_missing:
                to_delete.append(stable_id)
        for stable_id in to_delete:
            del self.targets[stable_id]

        updated.sort(key=lambda t: t.track_id)
        return updated
