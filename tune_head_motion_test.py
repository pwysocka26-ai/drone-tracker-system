from pathlib import Path

p = Path("src/core/head_motion_test.py")

code = """import math


class HeadMotionTestMode:
    def __init__(self):
        self.enabled = False
        self.frame_idx = 0

        # delikatniejszy, bardziej realistyczny ruch testowy
        self.pan_amplitude = 2.5
        self.tilt_amplitude = 1.5

        self.pan_speed = 0.08
        self.tilt_speed = 0.055

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
"""

p.write_text(code, encoding="utf-8")
print("OK: head_motion_test.py tuned")
