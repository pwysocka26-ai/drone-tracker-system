from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
    "            cx -= ctrl_pan_speed * 15.0",
    "            cx -= ctrl_pan_speed * 8.0"
)
s = s.replace(
    "            cy -= ctrl_tilt_speed * 15.0",
    "            cy -= ctrl_tilt_speed * 8.0"
)

p.write_text(s, encoding="utf-8")
print("OK: reduced camera step")
