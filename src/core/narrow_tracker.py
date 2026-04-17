import cv2
import numpy as np


class SimpleKalman2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            np.float32,
        )
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.12
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def reset(self):
        self.initialized = False

    def init_state(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return None
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def correct(self, x, y):
        if not self.initialized:
            self.init_state(x, y)
            return (x, y)
        m = np.array([[x], [y]], np.float32)
        est = self.kf.correct(m)
        return float(est[0, 0]), float(est[1, 0])


class _SyntheticTrack:
    def __init__(self, track_id, raw_id, bbox_xyxy, center_xy, confidence):
        self.track_id = int(track_id)
        self.raw_id = int(raw_id if raw_id is not None else track_id)
        self.bbox_xyxy = tuple(float(v) for v in bbox_xyxy)
        self.center_xy = tuple(float(v) for v in center_xy)
        self.confidence = float(confidence)
        self.missed_frames = 0
        self.hits = 999
        self.is_confirmed = True
        self.is_valid_target = True
        self.is_active_target = True


class NarrowTracker:
    def __init__(self, hold_frames=120):
        self.kalman = SimpleKalman2D()
        self.smooth_center = None
        self.smooth_zoom = 2.2
        self.hold_count = 0
        self.hold_frames = int(hold_frames)
        self.last_pan_speed = 0.0
        self.last_tilt_speed = 0.0
        self.degraded_count = 0
        self.large_target_count = 0
        self.last_good_center = None
        self.last_good_zoom = 2.2
        self.last_good_bbox = None
        self.last_owner_conf = 0.0
        self.last_owner_raw_id = None
        self.reacquire_grace_frames = max(10, int(self.hold_frames * 0.22))
        self.reacquire_boost_frames = max(14, int(self.hold_frames * 0.30))
        self.terminal_hold_frames = max(18, int(self.hold_frames * 0.36))
        self.reset_after_frames = max(72, int(self.hold_frames * 1.75))

        self.owner_id = None
        self.owner_track = None
        self.zoom_mode = "SEARCH"
        self.lock_state = "REACQUIRE"

        self.pending_owner_id = None
        self.pending_owner_count = 0
        self.owner_switch_commit_frames = 3

    def reset(self):
        self.kalman.reset()
        self.smooth_center = None
        self.smooth_zoom = 2.2
        self.hold_count = 0
        self.last_pan_speed = 0.0
        self.last_tilt_speed = 0.0
        self.degraded_count = 0
        self.large_target_count = 0
        self.last_good_center = None
        self.last_good_zoom = 2.2
        self.last_good_bbox = None
        self.last_owner_conf = 0.0
        self.last_owner_raw_id = None
        self.owner_id = None
        self.owner_track = None
        self.zoom_mode = "SEARCH"
        self.lock_state = "REACQUIRE"
        self.pending_owner_id = None
        self.pending_owner_count = 0

    def desired_zoom(self, frame, track):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = track.bbox_xyxy
        tw = max(1.0, x2 - x1)
        th = max(1.0, y2 - y1)
        rel = max(tw / max(1.0, w), th / max(1.0, h))

        if rel > 0.26:
            z = 8.0
        elif rel > 0.18:
            z = 11.0
        elif rel > 0.12:
            z = 14.5
        elif rel > 0.08:
            z = 17.5
        elif rel > 0.05:
            z = 19.5
        else:
            z = 19.5

        return float(np.clip(z, 1.4, 19.5))

    def _step_towards(self, desired, active, degraded=False):
        if desired is None:
            self.last_pan_speed = 0.0
            self.last_tilt_speed = 0.0
            return self.smooth_center

        if self.smooth_center is None:
            self.smooth_center = desired
            self.last_pan_speed = 0.0
            self.last_tilt_speed = 0.0
            return self.smooth_center

        ex = desired[0] - self.smooth_center[0]
        ey = desired[1] - self.smooth_center[1]

        dead_zone = 6 if degraded else 4
        if abs(ex) < dead_zone:
            ex = 0.0
        if abs(ey) < dead_zone:
            ey = 0.0

        if active:
            kp = 0.18 if degraded else 0.24
            max_step = 32.0 if degraded else 46.0
        else:
            kp = 0.08
            max_step = 16.0

        pan_speed = float(np.clip(ex * kp, -max_step, max_step))
        tilt_speed = float(np.clip(ey * kp, -max_step, max_step))

        smooth_alpha = 0.84 if degraded else 0.74
        pan_speed = smooth_alpha * self.last_pan_speed + (1.0 - smooth_alpha) * pan_speed
        tilt_speed = smooth_alpha * self.last_tilt_speed + (1.0 - smooth_alpha) * tilt_speed

        self.last_pan_speed = pan_speed
        self.last_tilt_speed = tilt_speed

        self.smooth_center = (
            self.smooth_center[0] + pan_speed,
            self.smooth_center[1] + tilt_speed,
        )
        return self.smooth_center

    def _find_track_by_id(self, tracks, track_id):
        if tracks is None or track_id is None:
            return None
        for tr in tracks:
            try:
                if int(getattr(tr, "track_id", -1)) == int(track_id):
                    return tr
            except Exception:
                continue
        return None

    def _track_quality(self, tr):
        if tr is None:
            return 0.0
        conf = float(getattr(tr, "confidence", 0.0) or 0.0)
        hits = int(getattr(tr, "hits", 0) or 0)
        missed = int(getattr(tr, "missed_frames", 0) or 0)
        confirmed = bool(getattr(tr, "is_confirmed", False))
        return conf * 10.0 + min(2.0, hits * 0.25) + (1.5 if confirmed else 0.0) - missed * 2.0

    def _healthy(self, tr):
        if tr is None:
            return False
        missed = int(getattr(tr, "missed_frames", 0) or 0)
        conf = float(getattr(tr, "confidence", 0.0) or 0.0)
        return missed <= 1 and conf >= 0.16

    def _make_synthetic_owner(self, center_xy):
        if self.owner_id is None or self.last_good_bbox is None or center_xy is None:
            return None
        x1, y1, x2, y2 = self.last_good_bbox
        bw = max(8.0, float(x2) - float(x1))
        bh = max(8.0, float(y2) - float(y1))
        cx, cy = float(center_xy[0]), float(center_xy[1])
        bbox = (cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh)
        conf = max(0.02, min(0.35, self.last_owner_conf * 0.6 if self.last_owner_conf else 0.08))
        return _SyntheticTrack(self.owner_id, self.last_owner_raw_id, bbox, (cx, cy), conf)

    def _resolve_owner_track(self, requested_track, tracks=None, manual_switch=False):
        current_track = self._find_track_by_id(tracks, self.owner_id)

        if manual_switch and requested_track is not None:
            self.owner_id = int(getattr(requested_track, "track_id", self.owner_id))
            self.pending_owner_id = None
            self.pending_owner_count = 0
            return requested_track

        if requested_track is None:
            if current_track is not None and int(getattr(current_track, "missed_frames", 0) or 0) <= 2:
                self.pending_owner_id = None
                self.pending_owner_count = 0
                return current_track
            return None

        req_id = int(getattr(requested_track, "track_id", -1))
        if self.owner_id is None:
            self.owner_id = req_id
            self.pending_owner_id = None
            self.pending_owner_count = 0
            return requested_track

        if req_id == int(self.owner_id):
            self.pending_owner_id = None
            self.pending_owner_count = 0
            return requested_track

        current_healthy = self._healthy(current_track)
        requested_quality = self._track_quality(requested_track)
        current_quality = self._track_quality(current_track)
        required = self.owner_switch_commit_frames if current_healthy else max(1, self.owner_switch_commit_frames - 1)
        strong_gain = requested_quality > (current_quality + (1.25 if current_healthy else 0.35))

        if self.pending_owner_id == req_id:
            self.pending_owner_count += 1
        else:
            self.pending_owner_id = req_id
            self.pending_owner_count = 1

        if (not current_healthy and self.pending_owner_count >= 1 and strong_gain) or self.pending_owner_count >= required:
            self.owner_id = req_id
            self.pending_owner_id = None
            self.pending_owner_count = 0
            self.hold_count = 0
            return requested_track

        if current_track is not None:
            return current_track
        return requested_track if not current_healthy else None

    def update(self, frame, requested_track, tracks=None, manual_switch=False):
        predicted_center = self.kalman.predict()
        desired_center = None

        owner_track = self._resolve_owner_track(
            requested_track=requested_track,
            tracks=tracks,
            manual_switch=manual_switch,
        )
        self.owner_track = owner_track

        if owner_track is not None:
            x1, y1, x2, y2 = owner_track.bbox_xyxy
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            h, w = frame.shape[:2]
            rel = max(bw / max(1.0, w), bh / max(1.0, h))
            missed = int(getattr(owner_track, "missed_frames", 0) or 0)
            conf = float(getattr(owner_track, "confidence", 0.0) or 0.0)

            large_target = rel > 0.12
            reacquire_recent = 0 < self.hold_count <= self.reacquire_boost_frames
            degraded = (missed >= 1) or (conf < 0.18) or reacquire_recent

            corrected = self.kalman.correct(owner_track.center_xy[0], owner_track.center_xy[1])
            desired_center = corrected
            predicted_center = corrected

            self.hold_count = 0
            self.degraded_count = self.degraded_count + 1 if degraded else 0
            self.large_target_count = self.large_target_count + 1 if large_target else 0

            desired_zoom = self.desired_zoom(frame, owner_track)
            if reacquire_recent:
                desired_zoom = 0.60 * self.smooth_zoom + 0.40 * desired_zoom

            if desired_zoom >= self.smooth_zoom:
                zoom_alpha = 0.78 if not degraded else 0.86
            else:
                zoom_alpha = 0.90 if not degraded else 0.94

            self.smooth_zoom = zoom_alpha * self.smooth_zoom + (1.0 - zoom_alpha) * desired_zoom
            self.smooth_zoom = float(np.clip(self.smooth_zoom, 1.4, 19.5))

            smooth_center = self._step_towards(desired_center, True, degraded=degraded)
            if smooth_center is not None:
                self.last_good_center = smooth_center
                self.last_good_zoom = self.smooth_zoom
                self.last_good_bbox = tuple(float(v) for v in owner_track.bbox_xyxy)
                self.last_owner_conf = conf
                rid = getattr(owner_track, "raw_id", None)
                if rid is not None:
                    self.last_owner_raw_id = int(rid)

            self.zoom_mode = "TARGET_FILL"
            self.lock_state = "TRACKING"
            self.owner_track = owner_track

        else:
            self.hold_count += 1
            self.degraded_count += 1
            desired_center = predicted_center if predicted_center is not None else self.last_good_center

            if self.hold_count > self.reset_after_frames:
                self.reset()
                return None, None, self.smooth_zoom, self.hold_count, 0.0, 0.0, None

            if desired_center is not None:
                smooth_center = self._step_towards(desired_center, False, degraded=True)
                if self.hold_count <= self.reacquire_grace_frames and self.last_good_center is not None:
                    gx = 0.84 * self.last_good_center[0] + 0.16 * smooth_center[0]
                    gy = 0.84 * self.last_good_center[1] + 0.16 * smooth_center[1]
                    smooth_center = (gx, gy)
                    self.smooth_center = smooth_center
            else:
                smooth_center = self.smooth_center

            target_zoom = self.last_good_zoom if self.last_good_zoom is not None else self.smooth_zoom
            zoom_hold_alpha = 0.994 if self.hold_count <= self.reacquire_grace_frames else 0.988
            self.smooth_zoom = zoom_hold_alpha * self.smooth_zoom + (1.0 - zoom_hold_alpha) * target_zoom
            self.smooth_zoom = float(np.clip(self.smooth_zoom, 1.4, 19.5))
            self.zoom_mode = "SEARCH"

            if self.hold_count <= self.terminal_hold_frames and self.owner_id is not None and self.last_good_bbox is not None:
                synth_center = smooth_center if smooth_center is not None else self.last_good_center
                self.owner_track = self._make_synthetic_owner(synth_center)
                self.lock_state = "HOLD"
            else:
                self.owner_track = None
                self.lock_state = "REACQUIRE"

        return (
            predicted_center,
            self.smooth_center,
            self.smooth_zoom,
            self.hold_count,
            self.last_pan_speed,
            self.last_tilt_speed,
            self.owner_track,
        )
