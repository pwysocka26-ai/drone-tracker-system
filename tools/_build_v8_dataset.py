"""Build v8 dataset = v7 + sea_drone, single-class.

Steps:
1. Copy v7/{images,labels}/{train,val} -> v8/{images,labels}/{train,val}
2. Convert v7 labels to single-class (class_id 0/1/2 -> 0)
3. Add sea_drone reviewed clips (chronological 80/20 split per klip)
4. Generate v8/data.yaml (nc=1, names=['dron'])

Sea-drone split strategy:
- chronologiczny (last 20% klatek per klip → val), zachowuje
  niezależność między consecutive frames (zgodne z v3 split tools/_split_train_val.py)
- empty .txt files SĄ kopiowane (legitimate negatives)

Usage:
    python tools/_build_v8_dataset.py
"""
import shutil
import os
from pathlib import Path
from collections import defaultdict


ROOT = Path(__file__).resolve().parent.parent
V7 = ROOT / "training" / "v7"
V8 = ROOT / "training" / "v8"
SEA_DRONE = ROOT / "data" / "sea_drone" / "prelabel"
SEA_CLIPS = ["MAX_0004", "DJI_0002", "DJI_20251218_0001_V"]
VAL_RATIO = 0.20


def copy_v7():
    """Copy v7 to v8 + convert labels to single-class."""
    print(f"=== Copy v7 -> v8 ===")
    for split in ["train", "val"]:
        for kind in ["images", "labels"]:
            src = V7 / kind / split
            dst = V8 / kind / split
            dst.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in src.iterdir():
                if not f.is_file():
                    continue
                shutil.copy2(f, dst / f.name)
                n += 1
            print(f"  {kind}/{split}: {n} files")

    # Convert v7 labels to single-class
    print(f"=== Convert v7 labels to single-class ===")
    for split in ["train", "val"]:
        n_changed = 0; n_files = 0
        for f in (V8 / "labels" / split).iterdir():
            if not f.is_file() or f.suffix != ".txt":
                continue
            n_files += 1
            content = f.read_text(encoding="utf-8").strip()
            if not content:
                continue
            new_lines = []
            for line in content.split("\n"):
                parts = line.split()
                if len(parts) != 5:
                    new_lines.append(line); continue
                if parts[0] != "0":
                    n_changed += 1
                    parts[0] = "0"
                new_lines.append(" ".join(parts))
            f.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        print(f"  labels/{split}: files={n_files}, lines_changed={n_changed}")


def add_sea_drone():
    """Add sea_drone to v8 with chronological 80/20 split per klip."""
    print(f"=== Add sea_drone (chronological 80/20 per klip) ===")
    total_added = defaultdict(int)
    for clip in SEA_CLIPS:
        img_dir = SEA_DRONE / clip / "images"
        lbl_dir = SEA_DRONE / clip / "labels"
        if not img_dir.exists():
            print(f"  skip {clip} (missing)")
            continue
        # Sort by frame index (filename ends with _f00050.jpg)
        all_jpg = sorted(
            [f for f in img_dir.iterdir() if f.suffix == ".jpg"],
            key=lambda p: int(p.stem.rsplit("_f", 1)[-1])
        )
        n = len(all_jpg)
        split_idx = int(n * (1 - VAL_RATIO))  # last 20% to val
        train_jpg = all_jpg[:split_idx]
        val_jpg = all_jpg[split_idx:]

        for split, files in [("train", train_jpg), ("val", val_jpg)]:
            for jpg in files:
                txt = lbl_dir / f"{jpg.stem}.txt"
                # Copy jpg
                dst_img = V8 / "images" / split / jpg.name
                shutil.copy2(jpg, dst_img)
                # Copy txt (even if empty, valid negative)
                dst_lbl = V8 / "labels" / split / f"{jpg.stem}.txt"
                if txt.exists():
                    shutil.copy2(txt, dst_lbl)
                else:
                    dst_lbl.write_text("", encoding="utf-8")
                total_added[split] += 1
        print(f"  {clip}: train={len(train_jpg)}, val={len(val_jpg)}")
    print(f"  TOTAL added: train={total_added['train']}, val={total_added['val']}")


def write_data_yaml():
    """Generate v8/data.yaml — single-class."""
    yaml = f"""path: {V8.as_posix()}
train: images/train
val: images/val
nc: 1
names: ['dron']
"""
    out = V8 / "data.yaml"
    out.write_text(yaml, encoding="utf-8")
    print(f"=== Wrote {out} ===")
    print(yaml)


def summary():
    print(f"=== v8 final state ===")
    for split in ["train", "val"]:
        n_img = len(list((V8 / "images" / split).iterdir()))
        n_lbl = len(list((V8 / "labels" / split).iterdir()))
        # count non-empty labels
        n_nonempty = 0; n_bboxes = 0
        for f in (V8 / "labels" / split).iterdir():
            if not f.suffix == ".txt":
                continue
            content = f.read_text(encoding="utf-8").strip()
            if content:
                n_nonempty += 1
                n_bboxes += len([l for l in content.split("\n") if l.strip()])
        print(f"  {split}: images={n_img}, labels={n_lbl}, non-empty={n_nonempty}, bboxes={n_bboxes}")


if __name__ == "__main__":
    if V8.exists():
        print(f"WARNING: {V8} already exists. Remove it first or back up.")
        import sys
        sys.exit(1)
    copy_v7()
    add_sea_drone()
    write_data_yaml()
    summary()
