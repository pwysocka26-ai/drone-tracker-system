from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class WideState:
    frame: Any
    frame_id: int
    tracks: list
    selection_tracks: list
    selected_track: Any
    target_manager: Any
    last_backend: str
    last_yolo_boxes: int
    last_det_tracks: int
    drop_streak: int


@dataclass
class NarrowSnapshot:
    owner_id: Optional[int]
    owner_track: Any
    smooth_center: Any
    smooth_zoom: float
    hold_count: int


@dataclass
class NarrowState:
    predicted_center: Any
    smooth_center: Any
    smooth_zoom: float
    hold_count: int
    pan_speed: float
    tilt_speed: float
    owner_track: Any
    effective_track: Any
    reused_last_good: bool
    edge_limit_active: bool


@dataclass
class OwnerDecision:
    owner_track: Any
    mode: str
    reason: str
    pending_id: Optional[int] = None
