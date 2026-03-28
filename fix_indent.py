from pathlib import Path

p = Path("src/core/app.py")
lines = p.read_text(encoding="utf-8").splitlines()

fixed = []
for line in lines:
    # usuń przypadkowe podwójne wcięcia przy return
    if "return frame, tracks" in line:
        fixed.append("    return frame, tracks")
    else:
        fixed.append(line)

p.write_text("\n".join(fixed), encoding="utf-8")
print("OK: indentation fixed")
