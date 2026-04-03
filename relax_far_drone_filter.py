from pathlib import Path
import sys

p = Path("src/core/target_filter.py")
s = p.read_text(encoding="utf-8")

repls = [
    ("min_age_frames=4", "min_age_frames=2"),
    ("min_area=35.0", "min_area=12.0"),
    ("max_area_ratio=0.05", "max_area_ratio=0.08"),
    ("max_jump=180.0", "max_jump=260.0"),
    ("min_stability_score=0.28", "min_stability_score=0.12"),
]

changed = False
for old, new in repls:
    if old in s:
        s = s.replace(old, new, 1)
        changed = True

if not changed:
    print("BLAD: nie znalazlem parametrow do poluznienia w target_filter.py")
    sys.exit(1)

p.write_text(s, encoding="utf-8")
print("OK: target_filter relaxed for far/small drone")
