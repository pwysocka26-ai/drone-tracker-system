import cv2
import numpy as np


class SimpleKalman2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 1 * 0]],
            np.float32,
        )
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
        self.reacquire_grace_frames = max(8, int(self.hold_frames * 0.18))
        self.reacquire_boost_frames = max(12, int(self.hold_frames * 0.25))
        self.terminal_hold_frames = max(6, int(self.hold_frames * 0.10))

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

    def desired_zoom(self, frame, track):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = track.bbox_xyxy
        tw = max(1.0, x2 - x1)
        th = max(1.0, y2 - y1)
        rel = max(tw / w, th / h)

        if rel > 0.23:
            z = 1.65
        elif rel > 0.16:
            z = 1.85
        elif rel > 0.10:
            z = 2.10
        else:
            z = 0.060 / max(rel, 0.010)

        return float(np.clip(z, 1.55, 4.0))

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
            kp = 0.22 if degraded else 0.28
            max_step = 42.0 if degraded else 60.0
        else:
            kp = 0.10
            max_step = 20.0

        pan_speed = float(np.clip(ex * kp, -max_step, max_step))
        tilt_speed = float(np.clip(ey * kp, -max_step, max_step))

        smooth_alpha = 0.82 if degraded else 0.70
        pan_speed = smooth_alpha * self.last_pan_speed + (1.0 - smooth_alpha) * pan_speed
        tilt_speed = smooth_alpha * self.last_tilt_speed + (1.0 - smooth_alpha) * tilt_speed

        self.last_pan_speed = pan_speed
        self.last_tilt_speed = tilt_speed

        self.smooth_center = (
            self.smooth_center[0] + pan_speed,
            self.smooth_center[1] + tilt_speed,
        )
        return self.smooth_center

    def update(self, frame, active_track):
        predicted_center = self.kalman.predict()
        desired_center = None

        if active_track is not None:
            x1, y1, x2, y2 = active_track.bbox_xyxy
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            h, w = frame.shape[:2]
            rel = max(bw / max(1.0, w), bh / max(1.0, h))
            missed = int(getattr(active_track, "missed_frames", 0) or 0)
            conf = float(getattr(active_track, "confidence", 0.0) or 0.0)

            large_target = rel > 0.12
            reacquire_recent = 0 < self.hold_count <= self.reacquire_boost_frames
            degraded = (missed >= 1) or (conf < 0.18)
            degraded = degraded or reacquire_recent

            corrected = self.kalman.correct(active_track.center_xy[0], active_track.center_xy[1])
            desired_center = corrected
            predicted_center = corrected

            self.hold_count = 0
            self.degraded_count = self.degraded_count + 1 if degraded else 0
            self.large_target_count = self.large_target_count + 1 if large_target else 0

            desired_zoom = self.desired_zoom(frame, active_track)
            if reacquire_recent:
                desired_zoom = 0.72 * self.smooth_zoom + 0.28 * desired_zoom
            zoom_alpha = 0.96 if reacquire_recent else (0.94 if large_target else (0.92 if degraded else 0.88))
            self.smooth_zoom = zoom_alpha * self.smooth_zoom + (1.0 - zoom_alpha) * desired_zoom

            smooth_center = self._step_towards(desired_center, True, degraded=degraded)
            if smooth_center is not None and (
                (not reacquire_recent) or (conf >= 0.14) or (missed <= 1)
            ):
                self.last_good_center = smooth_center
                self.last_good_zoom = self.smooth_zoom

        else:
            self.hold_count += 1
            self.degraded_count += 1
            desired_center = predicted_center if predicted_center is not None else self.last_good_center

            soft_hold_frames = int(self.hold_frames * 1.35)
            final_hold_frames = int(self.hold_frames * 1.75)
            if self.hold_count > final_hold_frames:
                self.reset()
                return None, None, self.smooth_zoom, self.hold_count, 0.0, 0.0

            terminal_hold_active = self.hold_count <= self.terminal_hold_frames and self.last_good_center is not None

            if desired_center is None and self.last_good_center is not None:
                desired_center = self.last_good_center

            if terminal_hold_active:
                smooth_center = self.last_good_center
                self.smooth_center = smooth_center
                self.last_pan_speed = 0.0
                self.last_tilt_speed = 0.0
            elif desired_center is not None:
                smooth_center = self._step_towards(desired_center, False, degraded=True)
                if self.hold_count <= self.reacquire_grace_frames and self.last_good_center is not None:
                    gx = 0.82 * self.last_good_center[0] + 0.18 * smooth_center[0]
                    gy = 0.82 * self.last_good_center[1] + 0.18 * smooth_center[1]
                    smooth_center = (gx, gy)
                    self.smooth_center = smooth_center
            else:
                smooth_center = self.smooth_center

            target_zoom = self.last_good_zoom if self.last_good_zoom is not None else self.smooth_zoom
            if terminal_hold_active:
                self.smooth_zoom = 0.998 * self.smooth_zoom + 0.002 * target_zoom
            else:
                zoom_hold_alpha = 0.992 if self.hold_count <= self.reacquire_grace_frames else 0.985
                self.smooth_zoom = zoom_hold_alpha * self.smooth_zoom + (1.0 - zoom_hold_alpha) * target_zoom

            if self.hold_count > soft_hold_frames and self.last_good_center is not None:
                smooth_center = self.last_good_center
                self.smooth_center = smooth_center
                self.smooth_zoom = 0.996 * self.smooth_zoom + 0.004 * target_zoom

        return (
            predicted_center,
            smooth_center,
            self.smooth_zoom,
            self.hold_count,
            self.last_pan_speed,
            self.last_tilt_speed,
        )