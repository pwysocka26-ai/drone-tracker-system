"""Build v3 dataset: merge Roboflow drone-detection set with v2.

Layout after run:
  training/v3/
    images/train/  images/val/
    labels/train/  labels/val/
    data.yaml

Sources:
  - data/roboflow_drone_v1/train/{images,labels}  (1445 images, split 80/20 here)
  - training/v2/images/{train,val} + labels       (already split, copy as-is)

Deterministic (seed=42). Copies (does not move) so v2 stays untouched.
"""
import random
import shutil
from pathlib import Path

ROBOFLOW_DIR = Path("data/roboflow_drone_v1/train")
V2_DIR = Path("training/v2")
V3_DIR = Path("training/v3")

VAL_FRACTION = 0.20
SEED = 42


def copy_pair(img_src: Path, lbl_src: Path, img_dst: Path, lbl_dst: Path) -> bool:
    if not img_src.exists() or not lbl_src.exists():
        return False
    shutil.copy2(img_src, img_dst / img_src.name)
    shutil.copy2(lbl_src, lbl_dst / lbl_src.name)
    return True


def main() -> None:
    # 1. create v3 layout
    for split in ("train", "val"):
        (V3_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (V3_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 2. split Roboflow 80/20 (extensions mixed: jpg/JPG/png in this dataset)
    rf_images = sorted(
        p for p in (ROBOFLOW_DIR / "images").iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    random.Random(SEED).shuffle(rf_images)
    n_val = int(round(len(rf_images) * VAL_FRACTION))
    rf_val = rf_images[:n_val]
    rf_train = rf_images[n_val:]
    print(f"[roboflow] total={len(rf_images)}  train={len(rf_train)}  val={len(rf_val)}")

    rf_train_ok = 0
    rf_val_ok = 0
    for img in rf_train:
        lbl = (ROBOFLOW_DIR / "labels") / (img.stem + ".txt")
        if copy_pair(img, lbl, V3_DIR / "images" / "train", V3_DIR / "labels" / "train"):
            rf_train_ok += 1
    for img in rf_val:
        lbl = (ROBOFLOW_DIR / "labels") / (img.stem + ".txt")
        if copy_pair(img, lbl, V3_DIR / "images" / "val", V3_DIR / "labels" / "val"):
            rf_val_ok += 1
    print(f"[roboflow] copied train={rf_train_ok}  val={rf_val_ok}")

    # 3. copy v2 as-is
    v2_train_images = sorted((V2_DIR / "images" / "train").glob("*.jpg"))
    v2_val_images = sorted((V2_DIR / "images" / "val").glob("*.jpg"))
    v2_train_ok = 0
    v2_val_ok = 0
    for img in v2_train_images:
        lbl = V2_DIR / "labels" / "train" / (img.stem + ".txt")
        if copy_pair(img, lbl, V3_DIR / "images" / "train", V3_DIR / "labels" / "train"):
            v2_train_ok += 1
    for img in v2_val_images:
        lbl = V2_DIR / "labels" / "val" / (img.stem + ".txt")
        if copy_pair(img, lbl, V3_DIR / "images" / "val", V3_DIR / "labels" / "val"):
            v2_val_ok += 1
    print(f"[v2]       copied train={v2_train_ok}  val={v2_val_ok}")

    # 4. data.yaml
    yaml_text = (
        "path: training/v3\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['drone']\n"
    )
    (V3_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")

    # 5. summary
    total_train = sum(1 for p in (V3_DIR / "images" / "train").iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    total_val = sum(1 for p in (V3_DIR / "images" / "val").iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"\n[v3 TOTAL] train={total_train}  val={total_val}  grand={total_train + total_val}")


if __name__ == "__main__":
    main()
