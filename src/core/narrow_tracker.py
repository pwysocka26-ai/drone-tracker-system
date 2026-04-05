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


class NarrowTracker:
    def __init__(self, hold_frames=120):
        self.kalman = SimpleKalman2D()
        self.smooth_center = None
        self.smooth_zoom = 2.2
        self.hold_count = 0
        self.hold_frames = int(hold_frames)
        self.last_pan_speed = 0.0
        self.last_tilt_speed = 0.0

    def reset(self):
        self.kalman.reset()
        self.smooth_center = None
        self.smooth_zoom = 2.2
        self.hold_count = 0
        self.last_pan_speed = 0.0
        self.last_tilt_speed = 0.0

    def desired_zoom(self, frame, track):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = track.bbox_xyxy
        tw = max(1.0, x2 - x1)
        th = max(1.0, y2 - y1)
        rel = max(tw / w, th / h)

        # mały obiekt -> zoom in, duży -> zoom out
        z = 0.082 / max(rel, 0.009)
        return float(np.clip(z, 2.2, 5.0))

    def _step_towards(self, desired, active):
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

        # dead zone: blisko środka -> nie szarp
        if abs(ex) < 3:
            ex = 0.0
        if abs(ey) < 3:
            ey = 0.0

        kp = 0.36 if active else 0.20
        max_step = 82.0 if active else 34.0

        pan_speed = float(np.clip(ex * kp, -max_step, max_step))
        tilt_speed = float(np.clip(ey * kp, -max_step, max_step))

        # lekkie wygładzenie prędkości
        pan_speed = 0.58 * self.last_pan_speed + 0.42 * pan_speed
        tilt_speed = 0.58 * self.last_tilt_speed + 0.42 * tilt_speed

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
            corrected = self.kalman.correct(active_track.center_xy[0], active_track.center_xy[1])
            desired_center = corrected
            predicted_center = corrected
            self.hold_count = 0

            desired_zoom = self.desired_zoom(frame, active_track)
            self.smooth_zoom = 0.90 * self.smooth_zoom + 0.10 * desired_zoom
        else:
            self.hold_count += 1
            desired_center = predicted_center

            if self.hold_count > self.hold_frames:
                self.reset()
                return None, None, self.smooth_zoom, self.hold_count, 0.0, 0.0

            # podczas HOLD zoom nie powinien skakać
            self.smooth_zoom = 0.97 * self.smooth_zoom + 0.03 * self.smooth_zoom

        smooth_center = self._step_towards(desired_center, active_track is not None)

        return (
            predicted_center,
            smooth_center,
            self.smooth_zoom,
            self.hold_count,
            self.last_pan_speed,
            self.last_tilt_speed,
        )
