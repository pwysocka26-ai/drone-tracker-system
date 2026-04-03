from pathlib import Path
import sys

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8-sig")

old = """dx_test, dy_test = head_motion_test.update()
        pan_speed += dx_test
        tilt_speed += dy_test"""

new = """dx_test, dy_test = head_motion_test.update()
        dx_test = max(-3.0, min(3.0, dx_test))
        dy_test = max(-2.0, min(2.0, dy_test))
        pan_speed += dx_test
        tilt_speed += dy_test"""

if old not in s:
    print("BLAD: nie znaleziono bloku head motion apply")
    sys.exit(1)

s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8-sig")
print("OK: head motion clamp added")
