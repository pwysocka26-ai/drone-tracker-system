from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
"target = None",
"""target = getattr(self, "target", None)

if tracks:
    if target:
        # znajdź najbliższy poprzedniego targetu
        target = min(tracks, key=lambda t: (
            (t.center_xy[0] - target.center_xy[0])**2 +
            (t.center_xy[1] - target.center_xy[1])**2
        ))
    else:
        target = max(tracks, key=lambda t: t.area)

self.target = target
"""
)

p.write_text(s, encoding="utf-8")
print("OK: tracking memory added")
