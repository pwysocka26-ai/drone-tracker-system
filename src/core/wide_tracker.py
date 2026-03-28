from typing import List, Optional
import numpy as np

from core.models import Detection, TrackedObject, WideTrackerState


class WideTracker:
    def __init__(self, match_dist: float = 120.0, max_missed: int = 15):
        self.match_dist = match_dist
        self.max_missed = max_missed
        self.next_id = 1
        self.state = WideTrackerState()

    def update(self, detections: List[Detection]) -> WideTrackerState:
        existing = self.state.tracks
        used_track_ids = set()
        new_tracks: List[TrackedObject] = []

        for det in detections:
            cx, cy = det.center_xy
            best_track: Optional[TrackedObject] = None
            best_dist = 1e9

            for tr in existing:
                if tr.track_id in used_track_ids:
                    continue

                px, py = tr.center_xy
                dist = float(np.hypot(cx - px, cy - py))
                if dist < best_dist and dist < self.match_dist:
                    best_dist = dist
                    best_track = tr

            if best_track is None:
                tr = TrackedObject(
                    track_id=self.next_id,
                    bbox_xyxy=det.bbox_xyxy,
                    center_xy=det.center_xy,
                    confidence=det.confidence,
                    age=1,
                    missed=0,
                )
                self.next_id += 1
            else:
                px, py = best_track.center_xy
                sx = 0.78 * px + 0.22 * cx
                sy = 0.78 * py + 0.22 * cy
                tr = TrackedObject(
                    track_id=best_track.track_id,
                    bbox_xyxy=det.bbox_xyxy,
                    center_xy=(sx, sy),
                    confidence=det.confidence,
                    age=best_track.age + 1,
                    missed=0,
                )
                used_track_ids.add(best_track.track_id)

            new_tracks.append(tr)

        matched_ids = {t.track_id for t in new_tracks}
        for tr in existing:
            if tr.track_id not in matched_ids and tr.missed < self.max_missed:
                new_tracks.append(
                    TrackedObject(
                        track_id=tr.track_id,
                        bbox_xyxy=tr.bbox_xyxy,
                        center_xy=tr.center_xy,
                        confidence=max(0.2, tr.confidence - 0.03),
                        age=tr.age,
                        missed=tr.missed + 1,
                    )
                )

        new_tracks.sort(key=lambda t: t.track_id)

        selected = self.state.selected_target_id
        for tr in new_tracks:
            tr.is_selected = tr.track_id == selected

        self.state = WideTrackerState(
            tracks=new_tracks,
            selected_target_id=selected,
        )
        return self.state

    def select_target(self, track_id: Optional[int]) -> WideTrackerState:
        self.state.selected_target_id = track_id
        for tr in self.state.tracks:
            tr.is_selected = tr.track_id == track_id
        return self.state
