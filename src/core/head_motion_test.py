import math


class HeadMotionTestMode:
    def __init__(self):
        self.enabled = False
        self.frame_idx = 0

        # amplituda sztucznego ruchu "glowicy" w pikselach wide-space
        self.pan_amplitude = 90.0
        self.tilt_amplitude = 45.0

        # predkosc ruchu
        self.pan_speed = 0.050
        self.tilt_speed = 0.035

    def toggle(self):
        self.enabled = not self.enabled
        self.frame_idx = 0

    def reset(self):
        self.frame_idx = 0

    def update(self):
        if not self.enabled:
            return 0.0, 0.0

        self.frame_idx += 1

        dx = self.pan_amplitude * math.sin(self.frame_idx * self.pan_speed)
        dy = self.tilt_amplitude * math.sin(self.frame_idx * self.tilt_speed)

        return dx, dy
