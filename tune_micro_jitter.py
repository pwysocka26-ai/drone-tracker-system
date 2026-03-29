from pathlib import Path

p = Path("src/core/head_controller.py")
s = p.read_text(encoding="utf-8")

old = """    def _axis_update(self, err, prev_err):
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
"""

new = """    def _axis_update(self, err, prev_err):
        locked = abs(err) < self.lock_zone

        if locked:
            return 0.0, 0.0, True

        if abs(err) < self.dead_zone:
            err = 0.0

        derr = err - prev_err
        speed = self.kp * err + self.kd * derr

        # bardzo blisko srodka - mocne wygaszanie mikrodrgan
        if abs(err) < max(6.0, self.lock_zone * 1.5):
            speed *= 0.10
        elif abs(err) < self.brake_zone * 0.5:
            speed *= 0.18
        elif abs(err) < self.brake_zone:
            speed *= 0.25

        # po przejsciu przez zero mocno wyhamuj
        if err != 0.0 and prev_err != 0.0 and (err > 0) != (prev_err > 0):
            speed *= 0.10

        # bardzo male predkosci wyzeruj
        if abs(speed) < 0.03:
            speed = 0.0

        speed = max(-self.max_speed, min(self.max_speed, speed))
        return speed, err, False
"""

if old not in s:
    print("UWAGA: nie znalazlem bloku _axis_update")
else:
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("OK: head_controller micro-jitter damping added")
