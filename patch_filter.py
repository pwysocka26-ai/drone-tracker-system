from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

# 🔥 FILTR: usuń duże obiekty (budynki)
old = "tracks = tracker.update(detections)"
new = """tracks = tracker.update(detections)

# ===== FILTR OBIEKTÓW (KLUCZOWE) =====
filtered = []
h, w = frame.shape[:2]

for t in tracks:
    if t.bbox_xyxy is None:
        continue

    x1, y1, x2, y2 = t.bbox_xyxy
    bw = x2 - x1
    bh = y2 - y1

    area = bw * bh

    # ❌ odrzuć duże rzeczy (budynki)
    if area > 0.01 * w * h:
        continue

    # ❌ odrzuć dół kadru (budynki)
    cy = (y1 + y2) / 2
    if cy > 0.65 * h:
        continue

    # ✅ zostaw tylko małe obiekty w górze
    filtered.append(t)

tracks = filtered
"""

s = s.replace(old, new)

# 🔥 BONUS: preferuj najwyższe obiekty (samoloty)
s = s.replace(
    "best = max(tracks, key=lambda t: t.confidence)",
    "best = min(tracks, key=lambda t: t.center_xy[1])"
)

p.write_text(s, encoding="utf-8")
print("OK: filtering fixed")
