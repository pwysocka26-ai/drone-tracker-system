from typing import Optional, Tuple

from core.models import HeadCommand, TargetMessage


class NarrowTracker:
    def __init__(self):
        self.last_target: Optional[TargetMessage] = None

    def update(self, target: Optional[TargetMessage]) -> Optional[TargetMessage]:
        if target is not None:
            self.last_target = target
        return self.last_target


class HeadController:
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def compute(self, target: Optional[TargetMessage]) -> HeadCommand:
        if target is None:
            return HeadCommand(0.0, 0.0, None)

        cx, cy = target.center_xy
        screen_cx = self.frame_width / 2.0
        screen_cy = self.frame_height / 2.0

        pan_error = cx - screen_cx
        tilt_error = cy - screen_cy

        return HeadCommand(
            pan_error=pan_error,
            tilt_error=tilt_error,
            target_id=target.target_id,
        )
