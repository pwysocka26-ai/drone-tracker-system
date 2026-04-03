from pathlib import Path
import sys

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8-sig")

changed = False

# import
old_import = "from core.stable_registry import StableTargetRegistry"
new_import = "from core.stable_registry import StableTargetRegistry\nfrom core.target_filter import TargetFilter"

if "from core.target_filter import TargetFilter" not in s:
    if old_import not in s:
        print("BLAD: nie znaleziono importu StableTargetRegistry")
        sys.exit(1)
    s = s.replace(old_import, new_import, 1)
    changed = True

# init
old_init = "    narrow_tracker = NarrowTracker(hold_frames=140)"
new_init = "    narrow_tracker = NarrowTracker(hold_frames=140)\n    target_filter = TargetFilter()"

if "target_filter = TargetFilter()" not in s:
    if old_init not in s:
        print("BLAD: nie znaleziono inicjalizacji NarrowTracker")
        sys.exit(1)
    s = s.replace(old_init, new_init, 1)
    changed = True

# reset
old_reset = "            narrow_tracker.reset()"
new_reset = "            narrow_tracker.reset()\n            target_filter.reset()"

if "target_filter.reset()" not in s:
    if old_reset not in s:
        print("BLAD: nie znaleziono narrow_tracker.reset()")
        sys.exit(1)
    s = s.replace(old_reset, new_reset, 1)
    changed = True

# apply filter after stable registry
old_apply = "            tracks = stable_registry.update(det_tracks)"
new_apply = "            tracks = stable_registry.update(det_tracks)\n            tracks = target_filter.update(tracks, frame.shape)"

if "target_filter.update(tracks, frame.shape)" not in s:
    if old_apply not in s:
        print("BLAD: nie znaleziono stable_registry.update(det_tracks)")
        sys.exit(1)
    s = s.replace(old_apply, new_apply, 1)
    changed = True

# do not overwrite filter result to True for all tracks
old_flags = """        for tr in tracks:
            tr.is_active_target = False
            tr.is_valid_target = True
"""
new_flags = """        for tr in tracks:
            tr.is_active_target = False
            if not hasattr(tr, "is_valid_target"):
                tr.is_valid_target = True
"""

if old_flags in s:
    s = s.replace(old_flags, new_flags, 1)
    changed = True
else:
    print("UWAGA: nie znalazlem bloku flag is_valid_target=True dla wszystkich")

p.write_text(s, encoding="utf-8-sig")
print("OK: target filter integrated =", changed)
