import cv2

from core.detector import detect_dark_objects
from core.fusion import FusionBridge
from core.narrow_tracker import HeadController, NarrowTracker
from core.ui import add_title, crop_group, draw_narrow, draw_tracks
from core.wide_tracker import WideTracker


def run_app(config):
    mode = config.get("mode", "video")
    video_cfg = config.get("video") or {}
    sim_cfg = config.get("simulation") or {}
    fps = config.get("fps", 30)

    wide_tracker = WideTracker()
    fusion = FusionBridge()
    narrow_tracker = NarrowTracker()

    cap = None
    sim = None

    if mode == "video":
        source = video_cfg.get("source", "video.mp4")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Nie moge otworzyc pliku: {source}")
            return
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps and src_fps > 0:
            fps = int(src_fps)
    else:
        from sim.simulator import FormationSimulator, draw_sim_frame, synthesize_tracks
        sim = FormationSimulator(
            sim_cfg.get("width", 1920),
            sim_cfg.get("height", 1080),
            sim_cfg.get("drones", 3),
            fps,
        )

    window_name = "Drone Tracker Multiview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 900)

    selected_id = None
    head_controller = None

    while True:
        if mode == "video":
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            detections = detect_dark_objects(frame, max_targets=16)
        else:
            drones, phase, sim_t = sim.step()
            frame = draw_sim_frame(sim.width, sim.height, drones, phase, sim_t)
            sim_dets = synthesize_tracks(drones, 0.01, 5.0, (88, 68))
            detections = []
            for d in sim_dets:
                from core.models import Detection
                detections.append(
                    Detection(
                        bbox_xyxy=d.bbox_xyxy,
                        center_xy=d.center_xy,
                        confidence=d.conf,
                    )
                )

        state = wide_tracker.update(detections)

        if selected_id is not None:
            state = wide_tracker.select_target(selected_id)

        if head_controller is None:
            head_controller = HeadController(frame.shape[1], frame.shape[0])

        target_msg = fusion.build_target_message(state)
        active_target = narrow_tracker.update(target_msg)
        head_cmd = head_controller.compute(active_target)

        wide_program = crop_group(frame, state.tracks, (780, 360))
        wide_debug = crop_group(draw_tracks(frame, state.tracks, state.selected_target_id), state.tracks, (1560, 450))
        narrow_output = draw_narrow(frame, active_target, (780, 360))

        if active_target is not None:
            cv2.putText(
                narrow_output,
                f"PAN ERR {head_cmd.pan_error:.1f}  TILT ERR {head_cmd.tilt_error:.1f}",
                (20, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        wide_program = add_title(wide_program, "WIDE PROGRAM")
        narrow_output = add_title(narrow_output, "NARROW OUTPUT")
        wide_debug = add_title(wide_debug, "WIDE DEBUG")

        dashboard = cv2.vconcat([cv2.hconcat([wide_program, narrow_output]), wide_debug])
        cv2.imshow(window_name, dashboard)

        key = cv2.waitKey(max(1, int(1000 / max(1, fps)))) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("1"):
            selected_id = 1
        elif key == ord("2"):
            selected_id = 2
        elif key == ord("3"):
            selected_id = 3
        elif key == ord("0"):
            selected_id = None
            state = wide_tracker.select_target(None)

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
