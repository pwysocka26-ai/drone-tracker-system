from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

# 1) dokładniejszy tracker: mniejszy dystans dopasowania
s = s.replace("tracker = StableTracker(max_age=16, match_dist=180)", "tracker = StableTracker(max_age=20, match_dist=90)")

# 2) wybór celu: trzymaj poprzedni ID, potem najbliższy poprzedniemu centrum, a nie najwyższe confidence
old_select = '''def auto_select_target(tracks, previous_target_id, previous_center):
    if not tracks:
        return previous_target_id

    ids = {t.track_id for t in tracks}

    if previous_target_id in ids:
        return previous_target_id

    if previous_center is not None:
        best = min(tracks, key=lambda t: np.hypot(t.center_xy[0] - previous_center[0], t.center_xy[1] - previous_center[1]))
        return best.track_id

    best = max(tracks, key=lambda t: t.confidence)
    return best.track_id
'''
new_select = '''def auto_select_target(tracks, previous_target_id, previous_center):
    if not tracks:
        return previous_target_id

    # 1. trzymaj poprzedni cel jeśli nadal istnieje
    for t in tracks:
        if t.track_id == previous_target_id:
            return previous_target_id

    # 2. jeśli zgubiliśmy cel, wybierz najbliższy poprzedniemu położeniu
    if previous_center is not None:
        best = min(
            tracks,
            key=lambda t: np.hypot(t.center_xy[0] - previous_center[0], t.center_xy[1] - previous_center[1])
        )
        return best.track_id

    # 3. na starcie wybierz obiekt najwyżej w kadrze i o największym polu
    def score(t):
        x1, y1, x2, y2 = t.bbox_xyxy
        area = max(1.0, (x2 - x1) * (y2 - y1))
        return (y1, -area)

    best = min(tracks, key=score)
    return best.track_id
'''
s = s.replace(old_select, new_select)

# 3) cel nie może znikać od razu - dłuższy hold
s = s.replace("if smooth_center is not None and lost_count < 18:", "if smooth_center is not None and lost_count < 30:")

# 4) mniej agresywny zoom automatyczny
old_zoom = '''def desired_zoom_from_target(frame, target: Track):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = target.bbox_xyxy
    tw = max(1.0, x2 - x1)
    th = max(1.0, y2 - y1)

    rel = max(tw / w, th / h)

    # mały obiekt => większy zoom
    desired = 0.11 / max(rel, 0.01)
    return float(np.clip(desired, 2.5, 8.0))
'''
new_zoom = '''def desired_zoom_from_target(frame, target: Track):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = target.bbox_xyxy
    tw = max(1.0, x2 - x1)
    th = max(1.0, y2 - y1)

    rel = max(tw / w, th / h)

    # bardziej konserwatywny zoom, żeby nie odlatywać poza cel
    desired = 0.06 / max(rel, 0.012)
    return float(np.clip(desired, 2.0, 5.0))
'''
s = s.replace(old_zoom, new_zoom)

# 5) mocniejsze wygładzanie środka
s = s.replace("a_center = 0.82", "a_center = 0.90")

# 6) wolniejsze zmiany zoomu
s = s.replace("smooth_zoom = 0.88 * smooth_zoom + 0.12 * desired_zoom", "smooth_zoom = 0.94 * smooth_zoom + 0.06 * desired_zoom")

# 7) startowy target ustaw na 1 zamiast przypadkowego
s = s.replace("target_id = 1", "target_id = 1")

# 8) w debug dodaj marker środka celu
old_debug = '''    for t in tracks:
        if t.bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
        color = (0, 255, 255) if t.track_id == selected_id else (0, 255, 0)
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            debug,
            f"DRON {t.track_id} {t.confidence:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
'''
new_debug = '''    for t in tracks:
        if t.bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
        cx, cy = [int(v) for v in t.center_xy]
        color = (0, 255, 255) if t.track_id == selected_id else (0, 255, 0)
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        cv2.circle(debug, (cx, cy), 4, color, -1)
        cv2.putText(
            debug,
            f"DRON {t.track_id} {t.confidence:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
'''
s = s.replace(old_debug, new_debug)

p.write_text(s, encoding="utf-8")
print("OK: tracking patch applied")
