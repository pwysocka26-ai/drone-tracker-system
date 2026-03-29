from pathlib import Path
import re
import sys

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

changed = False

# import
old_import = "from core.head_controller import HeadController"
new_import = "from core.head_controller import HeadController\nfrom core.operator_slots import OperatorSlotManager"

if "from core.operator_slots import OperatorSlotManager" not in s:
    if old_import not in s:
        print("BLAD: nie znaleziono importu HeadController")
        sys.exit(1)
    s = s.replace(old_import, new_import, 1)
    changed = True

# init
old_init = "    head_controller = HeadController()"
new_init = "    head_controller = HeadController()\n    operator_slots = OperatorSlotManager()"

if "operator_slots = OperatorSlotManager()" not in s:
    if old_init not in s:
        print("BLAD: nie znaleziono inicjalizacji HeadController")
        sys.exit(1)
    s = s.replace(old_init, new_init, 1)
    changed = True

# reset
old_reset = "            narrow_tracker.reset()"
new_reset = "            narrow_tracker.reset()\n            operator_slots.reset()"

if "operator_slots.reset()" not in s:
    if old_reset not in s:
        print("BLAD: nie znaleziono narrow_tracker.reset()")
        sys.exit(1)
    s = s.replace(old_reset, new_reset, 1)
    changed = True

# update tracks po filtrze
old_tracks = "            tracks = target_filter.update(tracks)"
new_tracks = "            tracks = target_filter.update(tracks)\n            tracks = operator_slots.update(tracks)"

if "tracks = operator_slots.update(tracks)" not in s:
    if old_tracks not in s:
        print("BLAD: nie znaleziono target_filter.update(tracks)")
        sys.exit(1)
    s = s.replace(old_tracks, new_tracks, 1)
    changed = True

# manual selection: klawisze 1..9 wybieraja operator_id
pattern = re.compile(
    r'''elif key in \(ord\("1"\), ord\("2"\), ord\("3"\)\):\s*
\s*visible = sorted\(tracks, key=lambda t: t\.center_xy\[0\]\)\s*
\s*idx = int\(chr\(key\)\) - 1\s*
\s*if idx < len\(visible\):\s*
\s*manual_lock = True\s*
\s*selected_id = visible\[idx\]\.track_id\s*
\s*start_xy = visible\[idx\]\.center_xy\s*
\s*kalman\.reset\(\)\s*
\s*kalman\.init_state\(start_xy\[0\], start_xy\[1\]\)\s*
\s*predicted_center = start_xy\s*
\s*smooth_center = start_xy\s*
\s*hold_count = 0\s*
\s*lock_age = 0''',
    re.MULTILINE
)

replacement = """elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7"), ord("8"), ord("9")):
                slot = int(chr(key))
                chosen = next((t for t in tracks if getattr(t, "operator_id", None) == slot), None)
                if chosen is not None:
                    manual_lock = True
                    selected_id = chosen.track_id
                    start_xy = chosen.center_xy
                    kalman.reset()
                    kalman.init_state(start_xy[0], start_xy[1])
                    predicted_center = start_xy
                    smooth_center = start_xy
                    hold_count = 0
                    lock_age = 0"""

s2, n = pattern.subn(replacement, s, count=1)
if n == 0:
    print("BLAD: nie znaleziono bloku wyboru manualnego 1..3")
    sys.exit(1)
s = s2
changed = True

# narrow label ma pokazywac operator_id
old_narrow = "active_track.track_id)"
new_narrow = 'getattr(active_track, "operator_id", active_track.track_id))'

if new_narrow not in s:
    if old_narrow not in s:
        print("BLAD: nie znaleziono wywolania draw_target_on_narrow z active_track.track_id")
        sys.exit(1)
    s = s.replace(old_narrow, new_narrow, 1)
    changed = True

p.write_text(s, encoding="utf-8")
print("OK: operator slots integrated =", changed)
