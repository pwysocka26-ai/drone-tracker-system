from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

if "from core.narrow_box_smoother import NarrowBoxSmoother" not in s:
    s = s.replace(
        "from core.head_controller import HeadController",
        "from core.head_controller import HeadController\nfrom core.narrow_box_smoother import NarrowBoxSmoother"
    )

if "narrow_box_smoother = NarrowBoxSmoother()" not in s:
    s = s.replace(
        "    head_controller = HeadController()",
        "    head_controller = HeadController()\n    narrow_box_smoother = NarrowBoxSmoother()"
    )

p.write_text(s, encoding="utf-8")
print("OK: smoother imported and created")
