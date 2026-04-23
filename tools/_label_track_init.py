"""Interactive CSRT-tracker labeller.

Shows frame 0 of the video, user drags a ROI around the drone, then CSRT
propagates the bbox through the entire clip. Every `step` frames it writes
a positive sample (JPG + YOLO-format label) into the dataset folders, plus a
preview crop into review/ for later audit.

Keys inside the ROI window:
    drag with mouse, ENTER/SPACE = confirm, C = cancel
Keys during propagation:
    q = quit early, p = pause / resume
"""
import sys
from pathlib import Path

import cv2


MAX_DISPLAY_W = 1600


ZOOM_CROP_W = 600
ZOOM_CROP_H = 450


def select_initial_bbox(frame):
    h, w = frame.shape[:2]
    if w > MAX_DISPLAY_W:
        scale = MAX_DISPLAY_W / w
        disp = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        scale = 1.0
        disp = frame

    # Stage 1: click-to-zoom. Mouse click picks location; ESC falls back to
    # direct ROI on the scaled preview (useful when drone is large enough).
    print("[stage 1] click near drone on scaled preview (ESC = skip zoom, use scaled ROI)",
          flush=True)
    click = {"pt": None}

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["pt"] = (x, y)

    disp_annot = disp.copy()
    cv2.putText(disp_annot, "STAGE 1/2: CLICK near drone", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(disp_annot, "ESC = skip zoom, drag ROI on scaled",
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    stage1_win = "stage 1: click drone location"
    cv2.namedWindow(stage1_win)
    cv2.setMouseCallback(stage1_win, on_click)
    cv2.imshow(stage1_win, disp_annot)
    while True:
        k = cv2.waitKey(30) & 0xFF
        if click["pt"] is not None:
            break
        if k == 27:  # ESC
            print("[stage 1] ESC -> fallback to scaled ROI", flush=True)
            break
    cv2.destroyAllWindows()

    if click["pt"] is None:
        print("[fallback] drag ROI on scaled preview (ENTER=confirm, C=cancel)",
              flush=True)
        bbox = cv2.selectROI("fallback: ROI on scaled (ENTER=confirm, C=cancel)",
                             disp, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        if bbox[2] == 0 or bbox[3] == 0:
            return None
        return tuple(int(round(b / scale)) for b in bbox)

    # Stage 2: 1:1 crop around the click in full resolution, ROI on that.
    cx_full = int(click["pt"][0] / scale)
    cy_full = int(click["pt"][1] / scale)
    print(f"[stage 1] click at scaled=({click['pt'][0]},{click['pt'][1]}) "
          f"full=({cx_full},{cy_full})", flush=True)
    x0 = max(0, cx_full - ZOOM_CROP_W // 2)
    y0 = max(0, cy_full - ZOOM_CROP_H // 2)
    x1 = min(w, x0 + ZOOM_CROP_W)
    y1 = min(h, y0 + ZOOM_CROP_H)
    # re-align if we hit a frame edge so the crop stays full size when possible
    x0 = max(0, x1 - ZOOM_CROP_W)
    y0 = max(0, y1 - ZOOM_CROP_H)
    crop = frame[y0:y1, x0:x1].copy()
    print(f"[stage 2] 1:1 crop region x=[{x0},{x1}] y=[{y0},{y1}]  size={crop.shape[1]}x{crop.shape[0]}",
          flush=True)
    cv2.putText(crop, "STAGE 2/2: DRAG tight ROI on drone",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(crop, "ENTER=confirm, C=cancel",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    print("[stage 2] drag tight ROI on drone (ENTER=confirm, C=cancel)", flush=True)
    bbox_crop = cv2.selectROI("stage 2: ROI on 1:1 crop (ENTER=confirm, C=cancel)",
                              crop, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    if bbox_crop[2] == 0 or bbox_crop[3] == 0:
        return None
    bx, by, bw, bh = bbox_crop
    return (int(bx + x0), int(by + y0), int(bw), int(bh))


def write_positive(frame, bbox, img_dir, lbl_dir, review_dir,
                   name, class_id, fw, fh, save_review=False):
    x, y, w, h = bbox
    if x < 0 or y < 0 or x + w > fw or y + h > fh or w < 5 or h < 5:
        return False

    img_path = Path(img_dir) / f"{name}.jpg"
    lbl_path = Path(lbl_dir) / f"{name}.txt"
    cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    cx_n = (x + w / 2) / fw
    cy_n = (y + h / 2) / fh
    w_n = w / fw
    h_n = h / fh
    lbl_path.write_text(
        f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n",
        encoding="utf-8",
    )

    if save_review:
        pad = max(w, h) // 2
        rx1 = max(0, x - pad)
        ry1 = max(0, y - pad)
        rx2 = min(fw, x + w + pad)
        ry2 = min(fh, y + h + pad)
        crop = frame[ry1:ry2, rx1:rx2].copy()
        cv2.rectangle(crop, (x - rx1, y - ry1),
                      (x - rx1 + w, y - ry1 + h), (0, 255, 0), 2)
        cv2.imwrite(str(Path(review_dir) / f"{name}.jpg"), crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return True


def run(video: str, video_tag: str,
        img_dir: str, lbl_dir: str, review_dir: str,
        step: int = 10, review_every: int = 100,
        class_id: int = 0, show_live: bool = True,
        start_frame: int = 0, end_frame: int = 0) -> None:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"cannot read frame {start_frame}")
    fh, fw = frame.shape[:2]
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{video}: {total} frames, {fps:.1f} fps, {fw}x{fh}  start_frame={start_frame}")

    bbox0 = select_initial_bbox(frame)
    if bbox0 is None:
        raise SystemExit("no bbox selected -- aborted")
    print(f"init bbox on frame {start_frame}: {bbox0}")

    for d in (img_dir, lbl_dir, review_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox0)

    n_saved = 0
    n_lost = 0
    frame_idx = start_frame

    write_positive(frame, bbox0, img_dir, lbl_dir, review_dir,
                   f"{video_tag}_f{frame_idx:05d}", class_id, fw, fh,
                   save_review=True)
    n_saved += 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if end_frame > 0 and frame_idx > end_frame:
            print(f"[end_frame] reached {end_frame} -- stopping", flush=True)
            break
        tr_ok, bb = tracker.update(frame)
        if not tr_ok:
            n_lost += 1
            if show_live:
                preview = cv2.resize(frame, (960, int(960 * fh / fw)))
                cv2.putText(preview, f"LOST frame {frame_idx}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow("csrt live", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        x, y, w, h = [int(round(v)) for v in bb]
        if frame_idx % step == 0:
            save_rev = (frame_idx % review_every == 0)
            ok_save = write_positive(frame, (x, y, w, h),
                                     img_dir, lbl_dir, review_dir,
                                     f"{video_tag}_f{frame_idx:05d}",
                                     class_id, fw, fh, save_review=save_rev)
            if ok_save:
                n_saved += 1

        if show_live:
            preview = frame.copy()
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(preview, f"f={frame_idx}  saved={n_saved}  lost={n_lost}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            preview = cv2.resize(preview, (960, int(960 * fh / fw)))
            cv2.imshow("csrt live", preview)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('p'):
                while True:
                    k2 = cv2.waitKey(100) & 0xFF
                    if k2 == ord('p') or k2 == ord('q'):
                        break

    cap.release()
    cv2.destroyAllWindows()
    print(f"done: {video_tag}  saved={n_saved}  lost={n_lost}  total={frame_idx+1}")


if __name__ == "__main__":
    video = sys.argv[1]
    tag = sys.argv[2]
    img_dir = sys.argv[3] if len(sys.argv) > 3 else "training/v2/images/train"
    lbl_dir = sys.argv[4] if len(sys.argv) > 4 else "training/v2/labels/train"
    review_dir = sys.argv[5] if len(sys.argv) > 5 else "training/v2/review"
    step = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    start_frame = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    end_frame = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    run(video, tag, img_dir, lbl_dir, review_dir, step=step,
        start_frame=start_frame, end_frame=end_frame)
