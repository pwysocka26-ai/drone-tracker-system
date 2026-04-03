import cv2


class NarrowCVTracker:
    def __init__(self):
        self.tracker = None
        self.active = False
        self.last_bbox = None
        self.tracker_name = None

    def _create_tracker(self):
        # Preferuj MOSSE, fallback do MIL
        try:
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
                self.tracker_name = "MOSSE"
                return cv2.legacy.TrackerMOSSE_create()
        except Exception:
            pass

        try:
            if hasattr(cv2, "TrackerMIL_create"):
                self.tracker_name = "MIL"
                return cv2.TrackerMIL_create()
        except Exception:
            pass

        try:
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMIL_create"):
                self.tracker_name = "MIL"
                return cv2.legacy.TrackerMIL_create()
        except Exception:
            pass

        return None

    def reset(self):
        self.tracker = None
        self.active = False
        self.last_bbox = None
        self.tracker_name = None

    def init_from_bbox(self, frame, bbox_xyxy):
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        w = x2 - x1
        h = y2 - y1
        if w < 6 or h < 6:
            return False

        tr = self._create_tracker()
        if tr is None:
            return False

        ok = tr.init(frame, (x1, y1, w, h))
        if ok is None:
            ok = True

        self.tracker = tr
        self.active = bool(ok)
        self.last_bbox = (x1, y1, x2, y2) if self.active else None
        return self.active

    def update(self, frame):
        if not self.active or self.tracker is None:
            return False, None

        ok, box = self.tracker.update(frame)
        if not ok:
            self.active = False
            return False, None

        x, y, w, h = box
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

        if w < 4 or h < 4:
            self.active = False
            return False, None

        self.last_bbox = (x1, y1, x2, y2)
        return True, self.last_bbox
