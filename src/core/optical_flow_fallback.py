import cv2
import numpy as np


class NarrowOpticalFlowFallback:
    def __init__(self, max_points=30):
        self.max_points = int(max_points)
        self.prev_gray = None
        self.prev_points = None
        self.active = False
        self.last_center = None
        self.last_bbox = None

        self.feature_params = dict(
            maxCorners=self.max_points,
            qualityLevel=0.03,
            minDistance=4,
            blockSize=5,
        )

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

    def reset(self):
        self.prev_gray = None
        self.prev_points = None
        self.active = False
        self.last_center = None
        self.last_bbox = None

    def _clip_bbox(self, bbox, shape):
        h, w = shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(w - 2, x1))
        y1 = max(0, min(h - 2, y1))
        x2 = max(x1 + 2, min(w, x2))
        y2 = max(y1 + 2, min(h, y2))
        return x1, y1, x2, y2

    def init_from_bbox(self, frame, bbox_xyxy):
        x1, y1, x2, y2 = self._clip_bbox(bbox_xyxy, frame.shape)

        # zawężamy ROI do środka bboxa, żeby nie łapać tła przy krawędziach
        bw = x2 - x1
        bh = y2 - y1

        mx = max(2, int(bw * 0.18))
        my = max(2, int(bh * 0.18))

        rx1 = x1 + mx
        ry1 = y1 + my
        rx2 = x2 - mx
        ry2 = y2 - my

        if rx2 <= rx1 + 4 or ry2 <= ry1 + 4:
            rx1, ry1, rx2, ry2 = x1, y1, x2, y2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            self.reset()
            return False

        pts = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
        if pts is None or len(pts) < 4:
            self.reset()
            return False

        pts[:, 0, 0] += rx1
        pts[:, 0, 1] += ry1

        self.prev_gray = gray
        self.prev_points = pts.reshape(-1, 1, 2)
        self.active = True
        self.last_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.last_bbox = (x1, y1, x2, y2)
        return True

    def update(self, frame):
        if not self.active or self.prev_gray is None or self.prev_points is None or self.last_center is None:
            return False, None, 0, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )

        if next_pts is None or status is None:
            self.reset()
            return False, None, 0, None

        prev_good = self.prev_points[status.flatten() == 1]
        good_new = next_pts[status.flatten() == 1]

        if prev_good is None or good_new is None or len(good_new) < 4:
            self.reset()
            return False, None, len(good_new) if good_new is not None else 0, None

        prev_good = prev_good.reshape(-1, 2)
        good_new = good_new.reshape(-1, 2)

        # liczymy przesunięcie punktów
        deltas = good_new - prev_good
        dxs = deltas[:, 0]
        dys = deltas[:, 1]

        # mediana jest dużo odporniejsza na złe punkty niż średnia
        mdx = float(np.median(dxs))
        mdy = float(np.median(dys))

        new_cx = float(self.last_center[0] + mdx)
        new_cy = float(self.last_center[1] + mdy)

        # odrzucamy punkty zbyt daleko od mediany ruchu
        err = np.sqrt((dxs - mdx) ** 2 + (dys - mdy) ** 2)
        inliers = err < 6.0

        filtered_pts = good_new[inliers]
        if filtered_pts is None or len(filtered_pts) < 4:
            self.reset()
            return False, None, 0, None

        # sprawdzamy, czy punkty nie rozlazły się za bardzo
        spread_x = float(np.std(filtered_pts[:, 0]))
        spread_y = float(np.std(filtered_pts[:, 1]))
        if spread_x > 80 or spread_y > 80:
            self.reset()
            return False, None, 0, None

        # aktualizacja stanu
        self.prev_gray = gray
        self.prev_points = filtered_pts.reshape(-1, 1, 2)
        self.last_center = (new_cx, new_cy)

        if self.last_bbox is not None:
            x1, y1, x2, y2 = self.last_bbox
            bw = x2 - x1
            bh = y2 - y1
            self.last_bbox = (
                int(new_cx - bw / 2),
                int(new_cy - bh / 2),
                int(new_cx + bw / 2),
                int(new_cy + bh / 2),
            )

        return True, (new_cx, new_cy), len(filtered_pts), filtered_pts
