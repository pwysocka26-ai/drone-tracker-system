class HeadController:
    def __init__(self, kp=0.006, kd=0.030, max_speed=1.2, dead_zone=20.0, brake_zone=55.0, lock_zone=35.0):
        self.kp = kp
        self.kd = kd
        self.max_speed = max_speed
        self.dead_zone = dead_zone
        self.brake_zone = brake_zone
        self.lock_zone = lock_zone

        self.prev_pan_err = 0.0
        self.prev_tilt_err = 0.0

    def reset(self):
        self.prev_pan_err = 0.0
        self.prev_tilt_err = 0.0

    def _axis_update(self, err, prev_err):
        locked = abs(err) < self.lock_zone

        if locked:
            return 0.0, 0.0, True

        if abs(err) < self.dead_zone:
            err = 0.0

        derr = err - prev_err
        speed = self.kp * err + self.kd * derr

        if abs(err) < self.brake_zone:
            speed *= 0.25

        if err != 0.0 and prev_err != 0.0 and (err > 0) != (prev_err > 0):
            speed *= 0.15

        speed = max(-self.max_speed, min(self.max_speed, speed))
        return speed, err, False

    def update(self, pan_err, tilt_err):
        pan_speed, pan_err_used, pan_locked = self._axis_update(pan_err, self.prev_pan_err)
        tilt_speed, tilt_err_used, tilt_locked = self._axis_update(tilt_err, self.prev_tilt_err)

        self.prev_pan_err = pan_err_used
        self.prev_tilt_err = tilt_err_used

        return pan_speed, tilt_speed, pan_locked, tilt_locked
