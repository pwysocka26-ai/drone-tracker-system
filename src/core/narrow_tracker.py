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
    STATE_LOST = 'LOST'
    STATE_ACQUIRE = 'ACQUIRE'
    STATE_STABILIZE = 'STABILIZE'
    STATE_LOCKED = 'LOCKED'
    STATE_HOLD = 'HOLD'

    def __init__(self, hold_frames=120):
        self.kalman = SimpleKalman2D()
        self.smooth_center = None
        self.smooth_zoom = 2.8
        self.hold_count = 0
        self.hold_frames = int(hold_frames)
        self.last_pan_speed = 0.0
        self.last_tilt_speed = 0.0
        self.state = self.STATE_LOST
        self.acquire_count = 0
        self.stable_count = 0
        self.last_error_norm = None
        self.jump_limited = False
        self.last_good_bbox = None
        self.last_good_center = None
        self.bbox_smooth = None
        self.last_good_bbox = None
        self.last_good_center = None
        self.bbox_smooth = None

    def reset(self):
        self.kalman.reset()
        self.smooth_center = None
        self.smooth_zoom = 2.8
        self.hold_count = 0
        self.last_pan_speed = 0.0
        self.last_tilt_speed = 0.0
        self.state = self.STATE_LOST
        self.acquire_count = 0
        self.stable_count = 0
        self.last_error_norm = None
        self.jump_limited = False

    def desired_zoom(self, frame, track, state):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = track.bbox_xyxy
        tw = max(1.0, x2 - x1)
        th = max(1.0, y2 - y1)
        rel = max(tw / w, th / h)

        base = 0.125 / max(rel, 0.0075)
        if state == self.STATE_ACQUIRE:
            base *= 0.92
        elif state == self.STATE_STABILIZE:
            base *= 1.00
        elif state == self.STATE_LOCKED:
            base *= 1.08
        return float(np.clip(base, 2.4, 5.8))

    def _choose_state(self, active_track, error_norm):
        if active_track is None:
            if self.hold_count > 0:
                return self.STATE_HOLD
            return self.STATE_LOST

        if self.state in (self.STATE_LOST, self.STATE_HOLD):
            self.acquire_count += 1
            self.stable_count = 0
            if self.acquire_count < 4:
                return self.STATE_ACQUIRE
            return self.STATE_STABILIZE

        if error_norm is None:
            return self.STATE_STABILIZE

        if error_norm < 18.0:
            self.stable_count += 1
        else:
            self.stable_count = 0

        if self.stable_count >= 6:
            return self.STATE_LOCKED
        return self.STATE_STABILIZE

    def _step_params(self, state):
        if state == self.STATE_ACQUIRE:
            return 0.48, 108.0, 2.0, 0.42
        if state == self.STATE_STABILIZE:
            return 0.38, 84.0, 2.0, 0.52
        if state == self.STATE_LOCKED:
            return 0.28, 56.0, 1.5, 0.68
        if state == self.STATE_HOLD:
            return 0.10, 12.0, 4.0, 0.82
        return 0.0, 0.0, 3.0, 0.76

    def _step_towards(self, desired, active, state):
        self.jump_limited = False
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
        self.last_error_norm = float((ex * ex + ey * ey) ** 0.5)

        kp, max_step, dead_zone, inertia = self._step_params(state)

        if abs(ex) < dead_zone:
            ex = 0.0
        if abs(ey) < dead_zone:
            ey = 0.0

        raw_pan = float(np.clip(ex * kp, -max_step, max_step))
        raw_tilt = float(np.clip(ey * kp, -max_step, max_step))

        if abs(ex) > max_step * 2.0 or abs(ey) > max_step * 2.0:
            self.jump_limited = True

        pan_speed = inertia * self.last_pan_speed + (1.0 - inertia) * raw_pan
        tilt_speed = inertia * self.last_tilt_speed + (1.0 - inertia) * raw_tilt

        self.last_pan_speed = pan_speed
        self.last_tilt_speed = tilt_speed

        self.smooth_center = (
            self.smooth_center[0] + pan_speed,
            self.smooth_center[1] + tilt_speed,
        )
        return self.smooth_center

    def _smooth_bbox(self, bbox, state):
        if bbox is None:
            return self.bbox_smooth
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        if self.bbox_smooth is None:
            self.bbox_smooth = (cx, cy, w, h)
            return self.bbox_smooth

        pcx, pcy, pw, ph = self.bbox_smooth
        diag = max(8.0, float((pw * pw + ph * ph) ** 0.5))
        if state == self.STATE_ACQUIRE:
            a_c, a_s = 0.42, 0.36
            max_jump_c = max(34.0, diag * 0.55)
            max_jump_s = max(22.0, 0.28 * max(pw, ph))
        elif state == self.STATE_STABILIZE:
            a_c, a_s = 0.58, 0.48
            max_jump_c = max(20.0, diag * 0.34)
            max_jump_s = max(14.0, 0.18 * max(pw, ph))
        else:
            a_c, a_s = 0.74, 0.64
            max_jump_c = max(12.0, diag * 0.20)
            max_jump_s = max(8.0, 0.12 * max(pw, ph))

        dcx = max(-max_jump_c, min(max_jump_c, cx - pcx))
        dcy = max(-max_jump_c, min(max_jump_c, cy - pcy))
        dw = max(-max_jump_s, min(max_jump_s, w - pw))
        dh = max(-max_jump_s, min(max_jump_s, h - ph))

        scx = pcx * a_c + (pcx + dcx) * (1.0 - a_c)
        scy = pcy * a_c + (pcy + dcy) * (1.0 - a_c)
        sw = pw * a_s + (pw + dw) * (1.0 - a_s)
        sh = ph * a_s + (ph + dh) * (1.0 - a_s)

        self.bbox_smooth = (scx, scy, sw, sh)
        return self.bbox_smooth

    def get_display_bbox(self):
        if self.bbox_smooth is None:
            return None
        cx, cy, w, h = self.bbox_smooth
        return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)

    def update(self, frame, active_track):
        predicted_center = self.kalman.predict()
        desired_center = None

        if active_track is not None:
            corrected = self.kalman.correct(active_track.center_xy[0], active_track.center_xy[1])
            desired_center = corrected
            predicted_center = corrected
            self.hold_count = 0
            self.last_good_center = corrected
        else:
            self.hold_count += 1
            if predicted_center is not None and self.last_good_center is not None:
                desired_center = (
                    0.80 * float(self.last_good_center[0]) + 0.20 * float(predicted_center[0]),
                    0.80 * float(self.last_good_center[1]) + 0.20 * float(predicted_center[1]),
                )
            else:
                desired_center = predicted_center or self.last_good_center

        error_norm = None
        if desired_center is not None and self.smooth_center is not None:
            dx = desired_center[0] - self.smooth_center[0]
            dy = desired_center[1] - self.smooth_center[1]
            error_norm = float((dx * dx + dy * dy) ** 0.5)

        self.state = self._choose_state(active_track, error_norm)

        if active_track is None and self.hold_count > self.hold_frames:
            self.reset()
            return None, None, self.smooth_zoom, self.hold_count, 0.0, 0.0, self.state, self.jump_limited

        if active_track is not None:
            self._smooth_bbox(active_track.bbox_xyxy, self.state)
            self.last_good_bbox = self.get_display_bbox()
            desired_zoom = self.desired_zoom(frame, active_track, self.state)
            if desired_zoom > self.smooth_zoom:
                alpha = 0.34 if self.state == self.STATE_ACQUIRE else 0.24
            else:
                alpha = 0.08
            self.smooth_zoom = (1.0 - alpha) * self.smooth_zoom + alpha * desired_zoom
        else:
            # hold zoom almost fixed for short losses
            self.smooth_zoom = 0.998 * self.smooth_zoom + 0.002 * self.smooth_zoom

        smooth_center = self._step_towards(desired_center, active_track is not None, self.state)

        return (
            predicted_center,
            smooth_center,
            self.smooth_zoom,
            self.hold_count,
            self.last_pan_speed,
            self.last_tilt_speed,
            self.state,
            self.jump_limited,
        )
