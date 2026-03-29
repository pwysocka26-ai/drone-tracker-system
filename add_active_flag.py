from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

if "tr.is_active_target" not in s:
    s = s.replace(
        "if active_track is not None:",
        "for tr in tracks:\n                tr.is_active_target = False\n\n            if active_track is not None:\n                active_track.is_active_target = True"
    )

p.write_text(s, encoding="utf-8")
print("OK: active target flag added")
