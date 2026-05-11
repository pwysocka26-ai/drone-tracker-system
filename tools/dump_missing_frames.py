"""Dump candidate frames bez pre-labelu (blind spot v7) jako .jpg + pusty .txt.

Po `v5_prelabel.py` z step=10, niektóre klatki nie mają detekcji bo
v7 ich nie widzi — to dokładnie BLIND SPOT samples (najcenniejsze
dla v8, bo uczą model gdzie aktualnie zawodzi).

Skrypt:
- Liczy candidate indices (step=N)
- Sprawdza które są w prelabel/<tag>/images/
- Eksportuje brakujące jako .jpg + pusty .txt
- User w v6_review.py manualnie dodaje bbox (klawisz `a`)

Usage:
    python tools/dump_missing_frames.py --video <path> --tag <tag> \\
        --img-dir <dir> --lbl-dir <dir> [--step 10]
"""
import argparse
from pathlib import Path
import cv2


def run(video, tag, img_dir, lbl_dir, step):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {video}  total={total}  step={step}")

    img_dir = Path(img_dir); lbl_dir = Path(lbl_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Candidate indices
    candidates = list(range(0, total, step))

    # Existing frames in prelabel/images
    existing = set()
    for f in img_dir.iterdir():
        if not f.is_file() or not f.suffix.lower() == ".jpg":
            continue
        # seadrone_<tag>_f00050.jpg → 50
        stem = f.stem
        marker = "_f"
        if marker in stem:
            try:
                idx = int(stem.rsplit(marker, 1)[1])
                existing.add(idx)
            except ValueError:
                pass

    missing = [i for i in candidates if i not in existing]
    print(f"candidates: {len(candidates)}, existing: {len(existing)}, missing: {len(missing)}")

    if not missing:
        print("nothing to dump")
        return

    n_dumped = 0
    for idx in missing:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"  skip frame {idx} (read failed)")
            continue
        name = f"seadrone_{tag}_f{idx:05d}"
        jpg_path = img_dir / f"{name}.jpg"
        txt_path = lbl_dir / f"{name}.txt"
        cv2.imwrite(str(jpg_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        # Empty txt — user uzupełnia w v6_review
        txt_path.write_text("", encoding="utf-8")
        n_dumped += 1

    cap.release()
    print(f"dumped {n_dumped} frames + empty .txt to:")
    print(f"  {img_dir}")
    print(f"  {lbl_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tag", required=True, help="np. MAX_0004")
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--lbl-dir", required=True)
    ap.add_argument("--step", type=int, default=10)
    args = ap.parse_args()
    run(args.video, args.tag, args.img_dir, args.lbl_dir, args.step)
