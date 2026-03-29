from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

if "from core.target_filter import TargetFilter" not in s:
    s = s.replace(
        "from core.stable_registry import StableTargetRegistry",
        "from core.stable_registry import StableTargetRegistry\nfrom core.target_filter import TargetFilter"
    )

if "target_filter = TargetFilter()" not in s:
    s = s.replace(
        "    stable_registry = StableTargetRegistry(max_missing=25, match_distance=140.0, min_iou=0.01)",
        "    stable_registry = StableTargetRegistry(max_missing=25, match_distance=140.0, min_iou=0.01)\n    target_filter = TargetFilter()"
    )

s = s.replace(
    "            tracks = stable_registry.update(det_tracks)",
    "            tracks = stable_registry.update(det_tracks)\n            tracks = target_filter.update(tracks)"
)

s = s.replace(
    "            stable_registry.reset()\n            target_manager.set_auto_mode()\n            narrow_tracker.reset()",
    "            stable_registry.reset()\n            target_filter.reset()\n            target_manager.set_auto_mode()\n            narrow_tracker.reset()"
)

p.write_text(s, encoding="utf-8")
print("OK: wide target filter connected")
