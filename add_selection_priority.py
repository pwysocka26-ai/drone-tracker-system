from pathlib import Path
import sys

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8-sig")

if "selection_priority" in s:
    print("OK: selection priority already exists")
    sys.exit(0)

old = """        visible_sorted = sorted(tracks, key=lambda t: t.track_id)

        predicted_center = narrow_tracker.kalman.predict()
        target_manager.update(tracks, predicted_center, frame.shape)
"""

new = """        visible_sorted = sorted(tracks, key=lambda t: t.track_id)

        fh, fw = frame.shape[:2]
        frame_cx = fw / 2.0
        frame_cy = fh / 2.0

        for tr in tracks:
            tx, ty = tr.center_xy
            dist_to_center = ((tx - frame_cx) ** 2 + (ty - frame_cy) ** 2) ** 0.5
            center_bonus = max(0.0, 1.0 - dist_to_center / max(1.0, (fw * 0.5)))

            valid_score = float(getattr(tr, "target_score", 0.0))
            conf_score = float(getattr(tr, "confidence", 0.0))

            persistence_bonus = 0.0
            if target_manager.selected_id is not None and tr.track_id == target_manager.selected_id:
                persistence_bonus = 0.35

            tr.selection_priority = (
                valid_score * 0.50
                + conf_score * 0.20
                + center_bonus * 0.30
                + persistence_bonus
            )

        predicted_center = narrow_tracker.kalman.predict()

        candidate_tracks = [t for t in tracks if getattr(t, "is_valid_target", False)]
        if candidate_tracks:
            candidate_tracks = sorted(candidate_tracks, key=lambda t: getattr(t, "selection_priority", 0.0), reverse=True)
            ordered_tracks = candidate_tracks + [t for t in tracks if not getattr(t, "is_valid_target", False)]
        else:
            ordered_tracks = tracks

        target_manager.update(ordered_tracks, predicted_center, frame.shape)
"""

if old not in s:
    print("BLAD: nie znaleziono miejsca do dodania selection priority")
    sys.exit(1)

s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8-sig")
print("OK: selection priority integrated")
