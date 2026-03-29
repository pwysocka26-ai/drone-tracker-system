from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

old = """            cv2.putText(narrow_output, f"HOLD {hold_count}", (20, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if active_track is not None:
"""

new = """            cv2.putText(narrow_output, f"HOLD {hold_count}", (20, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            center_lock_text = "CENTER LOCK ON" if (abs(pan_err) < 12 and abs(tilt_err) < 12 and active_track is not None) else "CENTER LOCK OFF"
            cv2.putText(narrow_output, center_lock_text, (20, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if active_track is not None:
"""

if old in s:
    s = s.replace(old, new)
else:
    print("UWAGA: nie znalazlem miejsca na CENTER LOCK text")

p.write_text(s, encoding="utf-8")
print("OK: center lock debug added")
