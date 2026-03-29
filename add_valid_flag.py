from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

if "is_valid_target" not in s:
    s = s.replace(
        "for tr in tracks:",
        "for tr in tracks:\n        tr.is_valid_target = True"
    )

p.write_text(s, encoding="utf-8")
print("OK: valid_target flag added")
