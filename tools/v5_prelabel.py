"""V5 pre-labeller dla datasetu v6 (multi-class).

Puszcza model v5 (single class `dron_maly`, ale generalizuje na hexakoptery)
na video, mapuje detekcje na klasy v6 po rozmiarze bboxa:

    max(bbox_w, bbox_h) > size_threshold  -> dron_duzy (id=1)
    max(bbox_w, bbox_h) <= size_threshold -> dron_maly (id=0)

Output kompatybilny z `tools/_label_track_init.py` (ten sam format
JPG + YOLO .txt w append mode + review JPG), tak ze pre-label maly+duzy
plus CSRT-doklejone pilka skladaja sie na komplet 3-klasowy per frame.

usage:
    python tools/v5_prelabel.py --video <path> --tag <tag> \\
        --start-frame 10000 --end-frame 13000 --step 10
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


# Single-class (decyzja 2026-05-10): wszystkie etykiety = "dron".
# Mapy zachowane dla kompatybilności starych runów review JPG.
CLASS_NAMES = {0: "dron", 1: "dron", 2: "dron"}
CLASS_COLORS = {0: (0, 255, 0), 1: (0, 255, 0), 2: (0, 255, 0)}


def map_to_v6_class(w: int, h: int, size_threshold: int,
                    default_class: int = -1) -> int:
    """Jesli default_class >= 0 — zwroc go zawsze (override).
    W przeciwnym razie — heurystyka po rozmiarze (>threshold = duzy=1).
    """
    if default_class >= 0:
        return default_class
    return 1 if max(w, h) > size_threshold else 0


def write_positive(frame, bbox, img_dir, lbl_dir, review_dir,
                   name, class_id, fw, fh, save_review, conf):
    x, y, w, h = bbox
    if x < 0 or y < 0 or x + w > fw or y + h > fh or w < 5 or h < 5:
        return False

    img_path = Path(img_dir) / f"{name}.jpg"
    lbl_path = Path(lbl_dir) / f"{name}.txt"
    if not img_path.exists():
        cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    cx_n = (x + w / 2) / fw
    cy_n = (y + h / 2) / fh
    w_n = w / fw
    h_n = h / fh
    with open(lbl_path, "a", encoding="utf-8") as f:
        f.write(f"{class_id} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n")

    if save_review:
        pad = max(w, h) // 2 + 30
        rx1 = max(0, x - pad)
        ry1 = max(0, y - pad)
        rx2 = min(fw, x + w + pad)
        ry2 = min(fh, y + h + pad)
        crop = frame[ry1:ry2, rx1:rx2].copy()
        color = CLASS_COLORS.get(class_id, (0, 255, 0))
        cv2.rectangle(crop, (x - rx1, y - ry1),
                      (x - rx1 + w, y - ry1 + h), color, 2)
        label = f"{CLASS_NAMES[class_id]} c={conf:.2f} {w}x{h}px"
        cv2.putText(crop, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)
        cls_suffix = CLASS_NAMES[class_id]
        cv2.imwrite(str(Path(review_dir) / f"{name}_{cls_suffix}.jpg"),
                    crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return True


def run(video, tag, weights, start_frame, end_frame, step,
        conf_threshold, size_threshold, imgsz,
        img_dir, lbl_dir, review_dir, review_every,
        default_class=-1):
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
    print(f"video: {video}")
    print(f"  total={total} frames  fps={fps:.1f}  size={fw}x{fh}")
    print(f"  range: start={start_frame}  end={end_frame or total}  step={step}")
    print(f"  conf>={conf_threshold}  imgsz={imgsz}")
    if default_class >= 0:
        print(f"  default_class={default_class} ({CLASS_NAMES.get(default_class, '?')}) — override")
    else:
        print(f"  size_threshold={size_threshold}px (>{size_threshold}=duzy, <={size_threshold}=maly)")

    for d in (img_dir, lbl_dir, review_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    print(f"loaded weights: {weights}")

    n_frames_seen = 0
    n_frames_with_det = 0
    n_det_maly = 0
    n_det_duzy = 0
    n_det_skipped = 0
    frame_idx = start_frame
    current = frame  # already read above

    while True:
        if frame_idx % step == 0:
            n_frames_seen += 1
            results = model.predict(current, conf=conf_threshold, imgsz=imgsz,
                                    verbose=False)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                save_review = (frame_idx % review_every == 0)
                had_det = False
                for b in boxes:
                    xyxy = b.xyxy[0].cpu().numpy()
                    conf = float(b.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                    w = x2 - x1
                    h = y2 - y1
                    if w < 5 or h < 5:
                        n_det_skipped += 1
                        continue
                    v6_class = map_to_v6_class(w, h, size_threshold, default_class)
                    ok_save = write_positive(
                        current, (x1, y1, w, h), img_dir, lbl_dir, review_dir,
                        f"{tag}_f{frame_idx:05d}", v6_class, fw, fh,
                        save_review, conf,
                    )
                    if not ok_save:
                        n_det_skipped += 1
                        continue
                    had_det = True
                    if v6_class == 0:
                        n_det_maly += 1
                    else:
                        n_det_duzy += 1
                if had_det:
                    n_frames_with_det += 1

            if n_frames_seen % 50 == 0:
                print(f"  ... f={frame_idx}  seen={n_frames_seen}  "
                      f"with_det={n_frames_with_det}  "
                      f"maly={n_det_maly} duzy={n_det_duzy}", flush=True)

        ok, current = cap.read()
        if not ok:
            break
        frame_idx += 1
        if end_frame > 0 and frame_idx > end_frame:
            break

    cap.release()
    print()
    print(f"done: tag={tag}")
    print(f"  frames_seen        = {n_frames_seen}")
    print(f"  frames_with_det    = {n_frames_with_det}")
    print(f"  detections dron_maly = {n_det_maly}")
    print(f"  detections dron_duzy = {n_det_duzy}")
    print(f"  detections skipped   = {n_det_skipped}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tag", required=True,
                    help="np. 'chrcynno1353' — prefix nazw plikow w datasecie")
    ap.add_argument("--weights", default="data/weights/v5_best.pt")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=0,
                    help="0 = do konca video")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--conf", type=float, default=0.50)
    ap.add_argument("--size-threshold", type=int, default=30,
                    help="max(w,h) > threshold -> dron_duzy, else dron_maly "
                         "(ignorowane gdy --default-class >= 0)")
    ap.add_argument("--default-class", type=int, default=0,
                    help="Single-class default: 0=dron (decyzja 2026-05-10). "
                         "Legacy multi-class: -1 = uzyj size_threshold heurystyki")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--review-every", type=int, default=100,
                    help="zapisuje review JPG co N klatek")
    ap.add_argument("--img-dir", default="training/v6/images/train")
    ap.add_argument("--lbl-dir", default="training/v6/labels/train")
    ap.add_argument("--review-dir", default="training/v6/review")
    args = ap.parse_args()

    run(args.video, args.tag, args.weights,
        args.start_frame, args.end_frame, args.step,
        args.conf, args.size_threshold, args.imgsz,
        args.img_dir, args.lbl_dir, args.review_dir, args.review_every,
        default_class=args.default_class)
