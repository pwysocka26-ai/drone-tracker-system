from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

old = """        wide_debug = crop_group(draw_tracks(frame, tracks, target_manager.selected_id), tracks, (1560, 450))
"""

new = """        debug_frame = draw_tracks(frame, tracks, target_manager.selected_id)

        visible_sorted = sorted(tracks, key=lambda t: t.center_xy[0])[:3]
        for i, tr in enumerate(visible_sorted, start=1):
            x1, y1, x2, y2 = [int(v) for v in tr.bbox_xyxy]
            cv2.putText(
                debug_frame,
                f"[{i}]",
                (x1, max(30, y1 - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2,
            )

        wide_debug = crop_group(debug_frame, tracks, (1560, 450))
"""

s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("OK: added labels 1/2/3")
