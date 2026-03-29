from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

old = "color = (0, 255, 0)"

new = """
if getattr(tr, "is_active_target", False):
    color = (0, 255, 0)      # zielony
elif getattr(tr, "is_valid_target", False):
    color = (0, 255, 255)    # żółty
else:
    color = (0, 0, 255)      # czerwony
"""

s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("OK: 3-color system enabled")
