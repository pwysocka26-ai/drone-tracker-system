"""Train v3 YOLO model on merged v2 + Roboflow drone dataset.

v3 dataset (2134 images total):
  - train: 1707 (326 dji + 225 hard-neg + 1156 Roboflow drones)
  - val:   427  (82 dji + 56 hard-neg + 289 Roboflow drones)

Changes vs v2 config (2026-04-23):
  - epochs 30 -> 40 (larger + more diverse dataset needs longer convergence)
  - base model: still yolov8m.pt (COCO) -- clean learning from diversity,
    not a fine-tune of v2 best.pt. v2 was specialized to dji-style dark drones
    on overcast sea; starting from COCO lets v3 build broader feature basis.
  - imgsz=960: unchanged, matches production narrow-channel resolution.

Run from project root (local or Colab after unzip):
    python training/train_v3.py
"""
import sys
from pathlib import Path

from ultralytics import YOLO


DATA_YAML = "training/v3/data.yaml"
BASE_MODEL = "yolov8m.pt"
RUN_NAME = "v3_drone_m_imgsz960"
RUN_PROJECT = "training/runs"


def main() -> None:
    if not Path(DATA_YAML).exists():
        print(f"ERROR: {DATA_YAML} not found -- run from project root", file=sys.stderr)
        sys.exit(1)

    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA_YAML,
        imgsz=960,
        epochs=40,
        batch=16,                # reduce to 8 if OOM on smaller GPU
        device=0,
        patience=12,             # slightly longer patience for larger dataset
        name=RUN_NAME,
        project=RUN_PROJECT,

        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,

        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        translate=0.1,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        close_mosaic=12,

        plots=True,
        save=True,
        verbose=True,
    )

    print(f"\n[train_v3] training complete.")
    print(f"[train_v3] results: {RUN_PROJECT}/{RUN_NAME}/")
    print(f"[train_v3] best weights: {RUN_PROJECT}/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
