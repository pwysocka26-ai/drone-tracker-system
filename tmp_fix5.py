def narrow_worker(args, target_q, frame_q, debug_q, stop_event):
    smooth_center = None
    TARGET_ID = 2

    while not stop_event.is_set():
        frame = drain_queue_latest(frame_q)
        targets = drain_queue_latest(target_q)

        if frame is None or not targets:
            continue

        selected = None
        for t in targets:
            if t.selected_track_id == TARGET_ID:
                selected = t
                break

        if selected is None:
            continue

        cx, cy = selected.center_xy
        h, w = frame.shape[:2]

        # ?? bardzo mocny zoom (sta³y rozmiar kadru)
        zoom_w = int(w * 0.25)
        zoom_h = int(h * 0.25)

        if smooth_center is None:
            smooth_center = (cx, cy)
        else:
            a = 0.92
            smooth_center = (
                a * smooth_center[0] + (1 - a) * cx,
                a * smooth_center[1] + (1 - a) * cy,
            )

        x1 = int(smooth_center[0] - zoom_w / 2)
        y1 = int(smooth_center[1] - zoom_h / 2)
        x2 = int(smooth_center[0] + zoom_w / 2)
        y2 = int(smooth_center[1] + zoom_h / 2)

        # clamp
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        zoom = cv2.resize(crop, (1280, 720))

        cv2.putText(
            zoom,
            f"DRON {TARGET_ID}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2,
        )

        clamp_queue_put(debug_q, {"name": "narrow_output", "frame": zoom})

        time.sleep(1 / args.fps)
