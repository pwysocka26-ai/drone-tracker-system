from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

old_start = s.index("def detect_bright_objects(")
old_end = s.index("def make_panel(", old_start)

new_func = '''
def detect_bright_objects(frame, max_targets=3):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30 or area > 2000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue

        cx = x + w / 2
        cy = y + h / 2
        candidates.append((area, x, y, w, h, cx, cy))

    candidates.sort(reverse=True, key=lambda x: x[0])

    tracks = []
    centers = []
    min_dist = 40
    idx = 1

    for _, x, y, w, h, cx, cy in candidates:
        too_close = False
        for px, py in centers:
            dist = np.hypot(cx - px, cy - py)
            if dist < min_dist:
                too_close = True
                break

        if too_close:
            continue

        centers.append((cx, cy))
        tracks.append(
            TargetMessage(
                timestamp=time.time(),
                selected_track_id=idx,
                bbox_xyxy=(x, y, x + w, y + h),
                center_xy=(cx, cy),
                confidence=0.9,
                mode=f"DRON {idx}",
            )
        )

        idx += 1
        if len(tracks) >= max_targets:
            break

    return tracks

'''
s = s[:old_start] + new_func + s[old_end:]
p.write_text(s, encoding="utf-8")
print("OK: detect_bright_objects updated")
