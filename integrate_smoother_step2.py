from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

old = "            narrow_tracker.reset()"
new = "            narrow_tracker.reset()\n            narrow_box_smoother.reset()"

if old in s and "narrow_box_smoother.reset()" not in s:
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("OK: smoother reset connected")
