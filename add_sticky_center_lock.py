from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

old = """            ctrl_pan_speed, ctrl_tilt_speed = head_controller.update(pan_err, tilt_err)

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

new = """            ctrl_pan_speed, ctrl_tilt_speed, pan_locked, tilt_locked = head_controller.update(pan_err, tilt_err)

            cx, cy = smooth_center

            # jeśli jesteśmy bardzo blisko środka, przyklej widok do celu
            if active_track is not None:
                tx, ty = active_track.center_xy
                if pan_locked:
                    cx = tx
                    ctrl_pan_speed = 0.0
                else:
                    cx -= ctrl_pan_speed * 5.0

                if tilt_locked:
                    cy = ty
                    ctrl_tilt_speed = 0.0
                else:
                    cy -= ctrl_tilt_speed * 5.0
            else:
                cx -= ctrl_pan_speed * 5.0
                cy -= ctrl_tilt_speed * 5.0

            smooth_center = (cx, cy)

            pan_speed = ctrl_pan_speed
            tilt_speed = ctrl_tilt_speed
"""

if old in s:
    s = s.replace(old, new)
else:
    print("UWAGA: nie znalazlem starego bloku sterowania")

p.write_text(s, encoding="utf-8")
print("OK: sticky center lock added")
