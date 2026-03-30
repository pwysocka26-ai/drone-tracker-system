import cv2
import numpy as np


class NarrowTemplateFallback:
    def __init__(self, match_threshold=0.45, search_radius=180):
        self.match_threshold = float(match_threshold)
        self.search_radius = int(search_radius)
        self.template = None
        self.last_bbox = None

    def reset(self):
        self.template = None
        self.last_bbox = None

    def _clip_bbox(self, bbox, shape):
        h, w = shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w, int(x2)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 + 2 or y2 <= y1 + 2:
            return None
        return (x1, y1, x2, y2)

    def update_template(self, frame, bbox):
        bbox = self._clip_bbox(bbox, frame.shape)
        if bbox is None:
            return False

        x1, y1, x2, y2 = bbox
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return False

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        # minimalny sensowny rozmiar template
        if gray.shape[0] < 8 or gray.shape[1] < 8:
            return False

        self.template = gray.copy()
        self.last_bbox = bbox
        return True

    def search(self, frame, center):
        if self.template is None or self.last_bbox is None or center is None:
            return None, 0.0

        h, w = frame.shape[:2]
        th, tw = self.template.shape[:2]

        cx, cy = int(center[0]), int(center[1])

        x1 = max(0, cx - self.search_radius)
        y1 = max(0, cy - self.search_radius)
        x2 = min(w, cx + self.search_radius)
        y2 = min(h, cy + self.search_radius)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, 0.0

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        if gray_roi.shape[0] < th + 2 or gray_roi.shape[1] < tw + 2:
            return None, 0.0

        res = cv2.matchTemplate(gray_roi, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < self.match_threshold:
            return None, float(max_val)

        mx, my = max_loc
        bx1 = x1 + mx
        by1 = y1 + my
        bx2 = bx1 + tw
        by2 = by1 + th

        bbox = self._clip_bbox((bx1, by1, bx2, by2), frame.shape)
        return bbox, float(max_val)
