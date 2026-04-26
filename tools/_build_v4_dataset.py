"""Build v4 dataset z CVAT export + opcjonalnie v3 merge.

CVAT export YOLO 1.1 zawiera (gdy 'Save images' = yes):
  obj_train_data/frame_NNNNNN.jpg
  obj_train_data/frame_NNNNNN.txt    <- "<class_id> <cx> <cy> <w> <h>" per line
  obj.names
  obj.data
  train.txt

Layout po run:
  training/v4/
    images/train/  images/val/
    labels/train/  labels/val/
    data.yaml

Split: chronologiczny per-task (ostatnie val-fraction frames -> val) zeby
unikac leakage (sasiednie klatki bardzo podobne; random split powodowal
near-duplicate w train+val co inflate-uje val mAP).

Klasy: filtruje tylko klasa 0 (dron_maly z labelling_guide.md). CVAT moze
miec 4 klasy zdefiniowane (tarcza/pilka/dron_duzy/dron_maly), ale v4
trening uzywa nc=1 na razie. Inne klasy w CVAT to placeholder.

Uzycie:
    python tools/_build_v4_dataset.py <cvat_export_zip> [--include-v3] [--val-split 0.20]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


V4_DIR = Path("training/v4")
V3_DIR = Path("training/v3")
DRONE_CLASS_ID = 0  # CVAT class index dla 'dron_maly' (pierwszy w obj.names)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cvat_zip", type=Path, help="CVAT YOLO 1.1 export ZIP (z save_images=yes)")
    p.add_argument("--include-v3", action="store_true",
                   help="Merge training/v3/ jako dodatkowe positives")
    p.add_argument("--val-split", type=float, default=0.20,
                   help="Frakcja danych do val (default 0.20)")
    p.add_argument("--keep-other-classes", action="store_true",
                   help="Zachowaj labelki innych klas niz dron_maly. Domyslnie filtrujemy.")
    p.add_argument("--out-dir", type=Path, default=V4_DIR,
                   help=f"Output dir (default {V4_DIR})")
    return p.parse_args()


def filter_labels(label_path: Path, keep_other: bool) -> list[str]:
    """Czyta plik label, zwraca tylko linie dla DRONE_CLASS_ID (chyba ze keep_other)."""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls_id = int(parts[0])
        except (ValueError, IndexError):
            continue
        if keep_other or cls_id == DRONE_CLASS_ID:
            out.append(line)
    return out


def chronological_split(image_paths: list[Path], val_fraction: float) -> tuple[list[Path], list[Path]]:
    """Sortuj po nazwie pliku, ostatnie N% → val (chronologiczny per-task split)."""
    sorted_paths = sorted(image_paths, key=lambda p: p.name)
    n_val = int(round(len(sorted_paths) * val_fraction))
    if n_val < 1 and len(sorted_paths) >= 5:
        n_val = 1
    train = sorted_paths[: len(sorted_paths) - n_val]
    val = sorted_paths[len(sorted_paths) - n_val :]
    return train, val


def copy_pair(img_src: Path, lbl_src: Path, img_dst_dir: Path, lbl_dst_dir: Path,
              keep_other: bool) -> bool:
    """Kopiuj parę (img + przefiltrowany label)."""
    if not img_src.exists():
        return False
    label_lines = filter_labels(lbl_src, keep_other)
    # Jeśli label pusty po filter -> hard negative (zachowaj img + pusty label)
    shutil.copy2(img_src, img_dst_dir / img_src.name)
    (lbl_dst_dir / (img_src.stem + ".txt")).write_text("\n".join(label_lines), encoding="utf-8")
    return True


def process_cvat_zip(cvat_zip: Path, val_fraction: float, keep_other: bool, out_dir: Path) -> tuple[int, int]:
    """Rozpakuj CVAT ZIP, splituj train/val, kopiuj do v4/."""
    if not cvat_zip.exists():
        print(f"ERROR: {cvat_zip} nie istnieje", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print(f"[cvat] rozpakowuje {cvat_zip} ({cvat_zip.stat().st_size/1e6:.1f} MB)...")
        with zipfile.ZipFile(cvat_zip) as z:
            z.extractall(tmp_path)

        # CVAT YOLO 1.1 layout: obj_train_data/<frame>.{jpg,txt}
        obj_dir = tmp_path / "obj_train_data"
        if not obj_dir.exists():
            print(f"ERROR: brak obj_train_data/ w ZIP -- czy to jest CVAT YOLO 1.1 z save_images=yes?", file=sys.stderr)
            sys.exit(1)

        images = [p for p in obj_dir.iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not images:
            print(f"ERROR: brak obrazow w obj_train_data/", file=sys.stderr)
            sys.exit(1)

        train_imgs, val_imgs = chronological_split(images, val_fraction)
        print(f"[cvat] split: train={len(train_imgs)}  val={len(val_imgs)} (chronologiczny)")

        train_ok = val_ok = 0
        for img in train_imgs:
            lbl = obj_dir / (img.stem + ".txt")
            if copy_pair(img, lbl, out_dir / "images" / "train", out_dir / "labels" / "train", keep_other):
                train_ok += 1
        for img in val_imgs:
            lbl = obj_dir / (img.stem + ".txt")
            if copy_pair(img, lbl, out_dir / "images" / "val", out_dir / "labels" / "val", keep_other):
                val_ok += 1

        return train_ok, val_ok


def merge_v3(out_dir: Path) -> tuple[int, int]:
    """Skopiuj v3 train/val obrazy + labelki do v4 (jako dodatkowe positives)."""
    if not V3_DIR.exists():
        print(f"WARN: {V3_DIR} nie istnieje, skip --include-v3", file=sys.stderr)
        return 0, 0

    train_ok = val_ok = 0
    for split in ("train", "val"):
        v3_imgs = list((V3_DIR / "images" / split).glob("*.jpg")) + \
                  list((V3_DIR / "images" / split).glob("*.jpeg")) + \
                  list((V3_DIR / "images" / split).glob("*.png"))
        for img in v3_imgs:
            lbl = V3_DIR / "labels" / split / (img.stem + ".txt")
            if not lbl.exists():
                continue
            shutil.copy2(img, out_dir / "images" / split / img.name)
            shutil.copy2(lbl, out_dir / "labels" / split / (img.stem + ".txt"))
            if split == "train":
                train_ok += 1
            else:
                val_ok += 1
    return train_ok, val_ok


def write_data_yaml(out_dir: Path) -> None:
    yaml_text = (
        f"path: {out_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['dron_maly']\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> int:
    args = parse_args()

    # 1. Layout
    for split in ("train", "val"):
        (args.out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 2. CVAT zip
    cvat_train, cvat_val = process_cvat_zip(args.cvat_zip, args.val_split, args.keep_other_classes, args.out_dir)
    print(f"[cvat]    copied train={cvat_train}  val={cvat_val}")

    # 3. Optional v3 merge
    if args.include_v3:
        v3_train, v3_val = merge_v3(args.out_dir)
        print(f"[v3]      copied train={v3_train}  val={v3_val}")

    # 4. data.yaml
    write_data_yaml(args.out_dir)

    # 5. Summary
    total_train = sum(1 for p in (args.out_dir / "images" / "train").iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    total_val = sum(1 for p in (args.out_dir / "images" / "val").iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"\n[v4 TOTAL] train={total_train}  val={total_val}  grand={total_train + total_val}")
    print(f"[v4]       data.yaml: {args.out_dir / 'data.yaml'}")
    print(f"\nGotowe -- mozesz odpalic trening:")
    print(f"    python training/train_v4.py --imgsz 640")
    print(f"    python training/train_v4.py --imgsz 960")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
