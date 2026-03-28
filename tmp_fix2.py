def narrow_worker(args, target_q, debug_q, stop_event):
    last_frame = None
    smooth_box = None

    SMOOTH = 0.85

    while not stop_event.is_set():
        msg = drain_queue_latest(debug_q)

        if msg and msg["name"] == "wide_program":
            last_frame = msg["frame"]

        target = drain_queue_latest(target_q)

        if last_frame is None:
            continue

        frame = last_frame.copy()
        h, w = frame.shape[:2]

        if target and target.bbox_xyxy:
            x1, y1, x2, y2 = target.bbox_xyxy
            box = expand_box((x1, y1, x2, y2), w, h, 0.8)

            if smooth_box is None:
                smooth_box = box
            else:
                smooth_box = tuple(
                    SMOOTH * s + (1 - SMOOTH) * b
                    for s, b in zip(smooth_box, box)
                )
        else:
            box = (0, 0, w, h)
            smooth_box = box

        zoom = crop_frame(frame, smooth_box, (1280, 720))

        clamp_queue_put(debug_q, {"name": "narrow_output", "frame": zoom})

        time.sleep(1 / args.fps)
