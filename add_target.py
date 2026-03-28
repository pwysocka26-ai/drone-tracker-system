from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

s = s.replace(
"tracks = filtered",
"""tracks = filtered

# 🔥 WYBÓR GŁÓWNEGO CELU
target = None

if tracks:
    # wybierz największy obiekt (najbliższy)
    target = max(tracks, key=lambda t: t.area)
"""
)

p.write_text(s, encoding="utf-8")
print("OK: target selection added")
