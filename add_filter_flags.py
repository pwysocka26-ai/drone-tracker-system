from pathlib import Path

p = Path("src/core/target_filter.py")
s = p.read_text(encoding="utf-8")

if "tr.is_valid_target" not in s:
    s = s.replace(
        "if state[\"valid\"]:\n                valid_tracks.append(tr)",
        "tr.is_valid_target = state[\"valid\"]\n\n            if state[\"valid\"]:\n                valid_tracks.append(tr)"
    )

p.write_text(s, encoding="utf-8")
print("OK: target validity flag added")
