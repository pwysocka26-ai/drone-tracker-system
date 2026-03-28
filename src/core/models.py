from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class Detection:
    bbox_xyxy: Tuple[float, float, float, float]
    center_xy: Tuple[float, float]
    confidence: float


@dataclass
class TrackedObject:
    track_id: int
    bbox_xyxy: Tuple[float, float, float, float]
    center_xy: Tuple[float, float]
    confidence: float
    age: int = 0
    missed: int = 0
    is_selected: bool = False


@dataclass
class TargetMessage:
    target_id: int
    center_xy: Tuple[float, float]
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    timestamp: float


@dataclass
class HeadCommand:
    pan_error: float
    tilt_error: float
    target_id: Optional[int] = None


@dataclass
class WideTrackerState:
    tracks: List[TrackedObject] = field(default_factory=list)
    selected_target_id: Optional[int] = None
