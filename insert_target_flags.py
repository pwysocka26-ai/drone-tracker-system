from pathlib import Path

p = Path("src/core/app.py")
lines = p.read_text(encoding="utf-8").splitlines()

new_lines = []
inserted = False

for line in lines:
    new_lines.append(line)

    if "active_track = target_manager.find_active_track(tracks)" in line and not inserted:
        indent = line[:line.index("a")]
        new_lines.append(indent + "for tr in tracks:")
        new_lines.append(indent + "    tr.is_active_target = False")
        new_lines.append(indent + "    tr.is_valid_target = True")
        new_lines.append(indent + "if active_track is not None:")
        new_lines.append(indent + "    active_track.is_active_target = True")
        inserted = True

p.write_text("\n".join(new_lines), encoding="utf-8")
print("OK: target flags inserted =", inserted)
