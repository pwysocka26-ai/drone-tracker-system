from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

old = """                if pan_locked:
                    cx = tx
                    ctrl_pan_speed = 0.0
                else:
                    cx -= ctrl_pan_speed * 5.0

                if tilt_locked:
                    cy = ty
                    ctrl_tilt_speed = 0.0
                else:
                    cy -= ctrl_tilt_speed * 5.0
"""

new = """                # SNAP TO TARGET (najważniejsze!)
                if abs(pan_err) < 35:
                    cx = tx
                    ctrl_pan_speed = 0.0
                else:
                    cx -= ctrl_pan_speed * 4.0

                if abs(tilt_err) < 35:
                    cy = ty
                    ctrl_tilt_speed = 0.0
                else:
                    cy -= ctrl_tilt_speed * 4.0
"""

s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("OK: snap-to-target added")
