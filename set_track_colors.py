from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

old = '        color = (0, 255, 255) if tr.track_id == selected_id else (0, 255, 0)\n'

new = '''        if getattr(tr, "is_active_target", False):
            color = (0, 255, 0)      # zielony
        elif getattr(tr, "is_valid_target", False):
            color = (0, 255, 255)    # żółty
        else:
            color = (0, 0, 255)      # czerwony
'''

if old not in s:
    print("UWAGA: nie znalazlem linii koloru w draw_tracks")
else:
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("OK: draw_tracks colors updated")
