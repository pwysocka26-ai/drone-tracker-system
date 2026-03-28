def detect_bright_objects(frame, max_targets=3):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 🔥 wykrywanie ciemnych obiektów (samoloty/drony)
    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # usuwanie szumu
    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tracks = []
    idx = 1

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)

        # filtr rozmiaru → ignorujemy szum
        if area < 30 or area > 2000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # filtr proporcji (samolot/dron ≠ linia)
        if w == 0 or h == 0:
            continue
        ratio = w / float(h)
        if ratio < 0.3 or ratio > 5:
            continue

        cx = x + w / 2
        cy = y + h / 2

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
