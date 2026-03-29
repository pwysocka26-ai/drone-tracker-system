from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

old = """            ctrl_pan_speed, ctrl_tilt_speed = head_controller.update(pan_err, tilt_err)

            cx, cy = smooth_center
            cx -= ctrl_pan_speed * 8.0
            cy -= ctrl_tilt_speed * 8.0
            smooth_center = (cx, cy)

            pan_speed = ctrl_pan_speed
            tilt_speed = ctrl_tilt_speed
"""

new = """            ctrl_pan_speed, ctrl_tilt_speed = head_controller.update(pan_err, tilt_err)

            cx, cy = smooth_center

            if abs(pan_err) < 10:
                ctrl_pan_speed = 0.0
            if abs(tilt_err) < 10:
                ctrl_tilt_speed = 0.0

            cx -= ctrl_pan_speed * 8.0
            cy -= ctrl_tilt_speed * 8.0
            smooth_center = (cx, cy)

            pan_speed = ctrl_pan_speed
            tilt_speed = ctrl_tilt_speed
"""

s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("OK: added center freeze zone")
