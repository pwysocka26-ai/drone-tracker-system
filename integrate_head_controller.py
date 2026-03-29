from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

# 1. import
if "from core.head_controller import HeadController" not in s:
    s = s.replace(
        "from core.narrow_tracker import NarrowTracker",
        "from core.narrow_tracker import NarrowTracker\nfrom core.head_controller import HeadController"
    )

# 2. inicjalizacja
old_init = "    narrow_tracker = NarrowTracker(hold_frames=140)"
new_init = "    narrow_tracker = NarrowTracker(hold_frames=140)\n    head_controller = HeadController()"
if old_init in s and "head_controller = HeadController()" not in s:
    s = s.replace(old_init, new_init)

# 3. update narrow_tracker - nowa sygnatura bez zmian, ale później nadpiszemy pan_speed/tilt_speed z kontrolera
old_update = "        predicted_center, smooth_center, smooth_zoom, hold_count, pan_speed, tilt_speed = narrow_tracker.update(frame, active_track)"
new_update = "        predicted_center, smooth_center, smooth_zoom, hold_count, pan_speed, tilt_speed = narrow_tracker.update(frame, active_track)"
s = s.replace(old_update, new_update)

# 4. wstrzyknięcie HeadController zaraz po obliczeniu pan_err / tilt_err
anchor = """            pan_err = smooth_center[0] - frame.shape[1] / 2.0
            tilt_err = smooth_center[1] - frame.shape[0] / 2.0
"""
inject = """            pan_err = smooth_center[0] - frame.shape[1] / 2.0
            tilt_err = smooth_center[1] - frame.shape[0] / 2.0

            ctrl_pan_speed, ctrl_tilt_speed = head_controller.update(pan_err, tilt_err)

            cx, cy = smooth_center
            cx -= ctrl_pan_speed * 15.0
            cy -= ctrl_tilt_speed * 15.0
            smooth_center = (cx, cy)

            pan_speed = ctrl_pan_speed
            tilt_speed = ctrl_tilt_speed
"""
if anchor in s:
    s = s.replace(anchor, inject)

# 5. reset kontrolera przy przejściu na AUTO
old_auto = """        elif key == ord("0"):
            target_manager.set_auto_mode()
            narrow_tracker.reset()
"""
new_auto = """        elif key == ord("0"):
            target_manager.set_auto_mode()
            narrow_tracker.reset()
            head_controller = HeadController()
"""
if old_auto in s:
    s = s.replace(old_auto, new_auto)

# 6. reset kontrolera przy ręcznym wyborze celu
old_manual = """                target_manager.set_manual_target(tr.track_id)
                narrow_tracker.reset()
                narrow_tracker.kalman.init_state(tr.center_xy[0], tr.center_xy[1])
                narrow_tracker.smooth_center = tr.center_xy
"""
new_manual = """                target_manager.set_manual_target(tr.track_id)
                narrow_tracker.reset()
                head_controller = HeadController()
                narrow_tracker.kalman.init_state(tr.center_xy[0], tr.center_xy[1])
                narrow_tracker.smooth_center = tr.center_xy
"""
if old_manual in s:
    s = s.replace(old_manual, new_manual)

p.write_text(s, encoding="utf-8")
print("OK: HeadController integrated into app.py")
