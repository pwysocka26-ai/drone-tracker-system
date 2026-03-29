from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

old = """        cv2.putText(wide_debug, f"SELECTED ID: {target_manager.selected_id}", (20, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"HOLD COUNT: {hold_count}", (20, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"LOCK AGE: {target_manager.lock_age}", (20, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"PAN SPD: {pan_speed:.1f}  TILT SPD: {tilt_speed:.1f}", (20, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
"""

new = """        cv2.putText(wide_debug, f"SELECTED ID: {target_manager.selected_id}", (20, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"HOLD COUNT: {hold_count}", (20, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"LOCK AGE: {target_manager.lock_age}", (20, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(wide_debug, f"PAN SPD: {pan_speed:.1f}  TILT SPD: {tilt_speed:.1f}", (20, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        auto_text = "AUTO PICK ENABLED" if not target_manager.manual_lock else "AUTO PICK DISABLED"
        cv2.putText(wide_debug, auto_text, (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
"""

if old in s:
    s = s.replace(old, new)
else:
    print("UWAGA: nie znalazlem bloku debug text")

p.write_text(s, encoding="utf-8")
print("OK: auto target debug text added")
