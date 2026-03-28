def narrow_worker(args, target_q, frame_q, debug_q, stop_event):
    smooth_box = None
    TARGET_ID = 2

    while not stop_event.is_set():
        frame = drain_queue_latest(frame_q)
        targets = drain_queue_latest(target_q)

        if frame is None:
            continue

        if not targets:
            continue  # ?? NIE pokazuj ca³ej sceny

        selected = None
        for t in targets:
            if t.selected_track_id == TARGET_ID:
                selected = t
                break

        if selected is None:
            continue

        x1, y1, x2, y2 = selected.bbox_xyxy
        h, w = frame.shape[:2]

        # ?? wiêkszy zoom
        raw_box = expand_box((x1, y1, x2, y2), w, h, 2.5)

        if smooth_box is None:
            smooth_box = raw_box
        else:
            a = 0.90
            smooth_box = tuple(a * s + (1 - a) * b for s, b in zip(smooth_box, raw_box))

        zoom = crop_frame(frame, smooth_box, (1280, 720))

        cv2.putText(
            zoom,
            f"TRACKING DRON {TARGET_ID}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2,
        )

        clamp_queue_put(debug_q, {"name": "narrow_output", "frame": zoom})

        time.sleep(1 / args.fps)
