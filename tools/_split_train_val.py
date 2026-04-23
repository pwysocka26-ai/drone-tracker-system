"""Stratified chronological train/val split for v2 drone dataset.

For each drone clip prefix, sort positives by frame index ascending
and move the LAST ~20% (highest frame_idx) from train/ to val/.
This avoids the leak that a random split would cause: consecutive frames
are near-duplicates, so random 80/20 would put near-identical samples
in both sets, inflating val metrics.

Hard negatives (fp_*, pex*) are left in train/ untouched.
Run once. Idempotent-safe: skips files that already moved.
"""
import shutil
from pathlib import Path

TRAIN_IMG = Path("training/v2/images/train")
TRAIN_LBL = Path("training/v2/labels/train")
VAL_IMG = Path("training/v2/images/val")
VAL_LBL = Path("training/v2/labels/val")

VAL_FRACTION = 0.20
CLIP_PREFIXES = ("dji0002", "dji0003", "dji0005")


def main() -> None:
    VAL_IMG.mkdir(parents=True, exist_ok=True)
    VAL_LBL.mkdir(parents=True, exist_ok=True)

    total_moved = 0
    for prefix in CLIP_PREFIXES:
        files = sorted(TRAIN_IMG.glob(f"{prefix}_f*.jpg"))
        if not files:
            print(f"[{prefix}] no files found in train/ -- skipping")
            continue

        n_total = len(files)
        n_val = round(n_total * VAL_FRACTION)
        val_files = files[-n_val:]
        first_stem = val_files[0].stem
        last_stem = val_files[-1].stem
        print(f"[{prefix}] total={n_total}  -> val={n_val}  range=[{first_stem} .. {last_stem}]")

        for img in val_files:
            lbl = TRAIN_LBL / (img.stem + ".txt")
            if not lbl.exists():
                print(f"  WARN: missing label for {img.name}, skipping")
                continue
            shutil.move(str(img), str(VAL_IMG / img.name))
            shutil.move(str(lbl), str(VAL_LBL / lbl.name))
            total_moved += 1

    print(f"\nmoved {total_moved} positive samples to val/")


if __name__ == "__main__":
    main()
