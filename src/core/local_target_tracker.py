from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


BBox = Tuple[int, int, int, int]


@dataclass
class LocalTrackResult:
    ok: bool
    bbox: Optional[BBox]
    score: float
    source: str
    center: Optional[Tuple[float, float]]


class LocalTargetTracker:
    """
    Lightweight local identity tracker used as a second layer after WIDE selection.

    Goal:
    - keep a local visual lock on the already selected target
    - protect against switching to a neighbour in tight formation
    - provide a short-term local reacquire signal for NARROW

    This tracker is intentionally conservative: if confidence drops, it degrades fast
    and lets the main pipeline fall back to reacquire instead of pretending identity is stable.
    """

    def __init__(self, tracker_type: str = "csrt", max_lost_frames: int = 8) -> None:
        self.tracker_type = tracker_type
        self.max_lost_frames = int(max_lost_frames)
        self._tracker: Any = None
        self._active: bool = False
        self._lost_frames: int = 0
        self._last_bbox: Optional[BBox] = None
        self._last_center: Optional[Tuple[float, float]] = None
        self._score: float = 0.0

    def reset(self) -> None:
        self._tracker = None
        self._active = False
        self._lost_frames = 0
        self._last_bbox = None
        self._last_center = None
        self._score = 0.0

    def is_active(self) -> bool:
        return self._active

    def state(self) -> Dict[str, Any]:
        return {
            "active": self._active,
            "lost_frames": self._lost_frames,
            "bbox": self._last_bbox,
            "score": self._score,
            "center": self._last_center,
        }

    def _make_tracker(self) -> Any:
        if cv2 is None:
            return None
        creators = [
            "TrackerCSRT_create",
            "legacy_TrackerCSRT_create",
            "TrackerKCF_create",
            "legacy_TrackerKCF_create",
        ]
        for name in creators:
            fn = getattr(cv2, name, None)
            if callable(fn):
                return fn()
        return None

    def initialize(self, frame: Any, bbox: BBox) -> bool:
        self.reset()
        tracker = self._make_tracker()
        if tracker is None or frame is None or bbox is None:
            return False

        x1, y1, x2, y2 = [int(v) for v in bbox]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        ok = tracker.init(frame, (x1, y1, w, h))
        if ok is False:
            return False

        self._tracker = tracker
        self._active = True
        self._last_bbox = (x1, y1, x1 + w, y1 + h)
        self._last_center = (x1 + w / 2.0, y1 + h / 2.0)
        self._score = 1.0
        return True

    def update(self, frame: Any) -> LocalTrackResult:
        if not self._active or self._tracker is None or frame is None:
            return LocalTrackResult(False, self._last_bbox, 0.0, "inactive", self._last_center)

        ok, box = self._tracker.update(frame)
        if not ok:
            self._lost_frames += 1
            self._score = max(0.0, self._score - 0.20)
            if self._lost_frames > self.max_lost_frames:
                self.reset()
            return LocalTrackResult(False, self._last_bbox, self._score, "tracker_lost", self._last_center)

        x, y, w, h = [int(v) for v in box]
        bbox = (x, y, x + max(1, w), y + max(1, h))
        self._last_bbox = bbox
        self._last_center = (bbox[0] + (bbox[2] - bbox[0]) / 2.0, bbox[1] + (bbox[3] - bbox[1]) / 2.0)
        self._lost_frames = 0
        self._score = min(1.0, max(0.55, self._score * 0.85 + 0.20))
        return LocalTrackResult(True, bbox, self._score, "local_tracker", self._last_center)
