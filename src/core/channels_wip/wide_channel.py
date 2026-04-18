from __future__ import annotations

from .models import WideState


class WideChannel:
    def __init__(
        self,
        model,
        multi_tracker,
        target_manager,
        *,
        backend,
        tracker_name,
        conf,
        imgsz,
        classes,
        inference_every,
        search_fallback,
        search_conf,
        search_imgsz,
        search_interval,
        parse_tracks,
    ):
        self.model = model
        self.multi_tracker = multi_tracker
        self.target_manager = target_manager
        self.backend = backend
        self.tracker_name = tracker_name
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.classes = classes
        self.inference_every = max(1, int(inference_every))
        self.search_fallback = bool(search_fallback)
        self.search_conf = float(search_conf)
        self.search_imgsz = int(search_imgsz)
        self.search_interval = max(1, int(search_interval))
        self.parse_tracks = parse_tracks

        self.tracks = []
        self.last_backend = '-'
        self.last_yolo_boxes = 0
        self.last_det_tracks = 0
        self.drop_streak = 0

    def reset(self):
        self.tracks = []
        self.last_backend = '-'
        self.last_yolo_boxes = 0
        self.last_det_tracks = 0
        self.drop_streak = 0
        try:
            self.multi_tracker.reset()
        except Exception:
            pass

    def _infer(self, frame):
        if self.backend == 'predict':
            results = self.model.predict(
                source=frame,
                conf=self.conf,
                imgsz=self.imgsz,
                classes=self.classes,
                verbose=False,
            )
            backend_name = 'predict'
        else:
            results = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_name,
                conf=self.conf,
                imgsz=self.imgsz,
                classes=self.classes,
                verbose=False,
            )
            backend_name = 'track'

        result = results[0]
        boxes = getattr(result, 'boxes', None)
        yolo_boxes = int(len(boxes)) if (boxes is not None and getattr(boxes, 'xyxy', None) is not None) else 0
        det_tracks = self.parse_tracks(result, frame.shape)

        if self.search_fallback and (not det_tracks):
            search_results = self.model.predict(
                source=frame,
                conf=self.search_conf,
                imgsz=self.search_imgsz,
                classes=self.classes,
                verbose=False,
            )
            search_result = search_results[0]
            search_boxes = getattr(search_result, 'boxes', None)
            search_box_count = int(len(search_boxes)) if (search_boxes is not None and getattr(search_boxes, 'xyxy', None) is not None) else 0
            search_tracks = self.parse_tracks(search_result, frame.shape)
            if search_tracks:
                det_tracks = search_tracks
                backend_name = 'predict-search'
                yolo_boxes = search_box_count
            else:
                yolo_boxes = max(yolo_boxes, search_box_count)

        return det_tracks, backend_name, yolo_boxes

    def step(self, frame, frame_id, predicted_center):
        should_infer = (frame_id % self.inference_every == 0) or (not self.tracks)
        if should_infer:
            det_tracks, self.last_backend, self.last_yolo_boxes = self._infer(frame)
            self.last_det_tracks = int(len(det_tracks))
            self.drop_streak = (self.drop_streak + 1) if self.last_det_tracks == 0 else 0
            self.tracks = self.multi_tracker.update(det_tracks, frame.shape)

        tracks = list(self.tracks or [])
        selection_tracks = [t for t in tracks if getattr(t, 'is_confirmed', False)] or tracks
        self.target_manager.update(selection_tracks, predicted_center, frame.shape)
        selected_track = self.target_manager.find_active_track(selection_tracks)

        for tr in tracks:
            tr.is_active_target = False
            tr.is_valid_target = bool(getattr(tr, 'is_confirmed', False) or getattr(tr, 'hits', 0) >= 2)
        if selected_track is not None:
            selected_track.is_active_target = True

        return WideState(
            frame=frame,
            frame_id=frame_id,
            tracks=tracks,
            selection_tracks=selection_tracks,
            selected_track=selected_track,
            target_manager=self.target_manager,
            last_backend=self.last_backend,
            last_yolo_boxes=self.last_yolo_boxes,
            last_det_tracks=self.last_det_tracks,
            drop_streak=self.drop_streak,
        )
