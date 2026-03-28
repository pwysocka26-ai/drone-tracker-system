from pathlib import Path

file = Path("src/core/app.py")
code = file.read_text(encoding="utf-8")

code = code.replace(
"(cx, cy, 0)",
"(0.7*px + 0.3*cx, 0.7*py + 0.3*cy, 0) if best_id in self.tracks else (cx, cy, 0)"
)

file.write_text(code, encoding="utf-8")
print("OK: smoothing added")
