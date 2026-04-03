from pathlib import Path
import sys

p = Path("src/core/target_filter.py")
s = p.read_text(encoding="utf-8")

old = """    def __init__(
        self,
        min_age_frames=6,
        min_area=60.0,
        max_area_ratio=0.03,
        max_jump=120.0,
        min_stability_score=0.45,
    ):
"""

new = """    def __init__(
        self,
        min_age_frames=4,
        min_area=35.0,
        max_area_ratio=0.05,
        max_jump=180.0,
        min_stability_score=0.28,
    ):
"""

if old not in s:
    print("BLAD: nie znaleziono bloku __init__ w target_filter.py")
    sys.exit(1)

s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("OK: target filter softened")
