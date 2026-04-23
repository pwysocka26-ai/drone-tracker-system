"""Train v2 YOLO model on drone + hard-negatives dataset.

Config selected on 2026-04-22 for this deployment:
  - yolov8m: backbone capacity for 12-300 px object range in narrow channel
  - imgsz=960: matches production pipeline config (config/config.yaml)
                and covers critical 12-40 px zone for Mavic/piłka on 1000-1500 m
  - epochs=30: sufficient for ~326 train positives + 225 hard-neg bg
  - close_mosaic=10: disable mosaic for last 10 epochs for clean fine-tune

Run from project root (local or Colab after unzip):
    python training/train_v2.py
"""
import sys
from pathlib import Path

from ultralytics import YOLO


DATA_YAML = "training/v2/data.yaml"
BASE_MODEL = "yolov8m.pt"
RUN_NAME = "v2_drone_m_imgsz960"
RUN_PROJECT = "training/runs"


def main() -> None:
    if not Path(DATA_YAML).exists():
        print(f"ERROR: {DATA_YAML} not found -- run from project root", file=sys.stderr)
        sys.exit(1)

    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA_YAML,
        imgsz=960,
        epochs=30,
        batch=16,                # reduce to 8 if OOM on smaller GPU
        device=0,                # first GPU; set "cpu" for local debug only
        patience=10,             # early stop after 10 epochs without val improvement
        name=RUN_NAME,
        project=RUN_PROJECT,

        # optimizer
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,             # cosine LR schedule

        # small-object / multi-scale augmentations
        mosaic=1.0,              # 4-image mosaic boosts small-object coverage
        mixup=0.1,                # mild regularization
        copy_paste=0.0,           # off: aliased at our smallest training sizes
        scale=0.5,                # random resize 0.5..1.5x for scale invariance
        fliplr=0.5,               # drone orientation is not L/R biased
        flipud=0.0,               # keep gravity: do NOT flip vertical
        degrees=0.0,              # no rotation (orientation matters for context)
        translate=0.1,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        close_mosaic=10,          # turn off mosaic for last 10 epochs

        plots=True,
        save=True,
        verbose=True,
    )

    print(f"\n[train_v2] training complete.")
    print(f"[train_v2] results: {RUN_PROJECT}/{RUN_NAME}/")
    print(f"[train_v2] best weights: {RUN_PROJECT}/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
