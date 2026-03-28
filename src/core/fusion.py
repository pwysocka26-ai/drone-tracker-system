import time
from typing import Optional

from core.models import TargetMessage, WideTrackerState


class FusionBridge:
    def build_target_message(self, state: WideTrackerState) -> Optional[TargetMessage]:
        if state.selected_target_id is None:
            return None

        for tr in state.tracks:
            if tr.track_id == state.selected_target_id:
                return TargetMessage(
                    target_id=tr.track_id,
                    center_xy=tr.center_xy,
                    bbox_xyxy=tr.bbox_xyxy,
                    confidence=tr.confidence,
                    timestamp=time.time(),
                )
        return None
