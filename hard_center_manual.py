from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

old = """        if active_track is not None:
            tx, ty = active_track.center_xy

            if smooth_center is None:
                smooth_center = (tx, ty)

            pan_err = tx - smooth_center[0]
            tilt_err = ty - smooth_center[1]

            # reczny wybor celu ma byc bardziej agresywny niz auto
            if target_manager.manual_lock:
                alpha = 0.55
                snap_px = 30
            else:
                alpha = 0.28
                snap_px = 18

            cx = smooth_center[0] + alpha * pan_err
            cy = smooth_center[1] + alpha * tilt_err

            if abs(pan_err) < snap_px and abs(tilt_err) < snap_px:
                cx = tx
                cy = ty

            smooth_center = (cx, cy)
            pan_speed = pan_err * alpha
            tilt_speed = tilt_err * alpha
        else:
            pan_speed = 0.0
            tilt_speed = 0.0
"""

new = """        if active_track is not None:
            tx, ty = active_track.center_xy

            if smooth_center is None:
                smooth_center = (tx, ty)

            pan_err = tx - smooth_center[0]
            tilt_err = ty - smooth_center[1]

            if target_manager.manual_lock:
                # reczny lock: twarde ustawienie narrow na wybrany cel
                cx = tx
                cy = ty
                smooth_center = (cx, cy)
                pan_speed = pan_err
                tilt_speed = tilt_err
            else:
                alpha = 0.28
                snap_px = 18

                cx = smooth_center[0] + alpha * pan_err
                cy = smooth_center[1] + alpha * tilt_err

                if abs(pan_err) < snap_px and abs(tilt_err) < snap_px:
                    cx = tx
                    cy = ty

                smooth_center = (cx, cy)
                pan_speed = pan_err * alpha
                tilt_speed = tilt_err * alpha
        else:
            pan_speed = 0.0
            tilt_speed = 0.0
"""

if old not in s:
    print("UWAGA: nie znalazlem bloku manual lock")
else:
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("OK: manual lock changed to hard-center")
