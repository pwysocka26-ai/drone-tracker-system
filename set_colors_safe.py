from pathlib import Path

p = Path("src/core/app.py")
lines = p.read_text(encoding="utf-8").splitlines()

new_lines = []

for line in lines:
    if "color = (0, 255, 0)" in line:
        indent = line[:line.index("c")]

        new_lines.append(indent + "if getattr(tr, 'is_active_target', False):")
        new_lines.append(indent + "    color = (0, 255, 0)  # zielony")
        new_lines.append(indent + "elif getattr(tr, 'is_valid_target', False):")
        new_lines.append(indent + "    color = (0, 255, 255)  # żółty")
        new_lines.append(indent + "else:")
        new_lines.append(indent + "    color = (0, 0, 255)  # czerwony")
    else:
        new_lines.append(line)

p.write_text("\n".join(new_lines), encoding="utf-8")
print("OK: color system applied safely")
