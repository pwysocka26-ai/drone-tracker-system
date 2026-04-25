"""Train v4 YOLOv8s drone model — parametryzowany imgsz dla benchmark obu rozdzielczosci.

v4 dataset (D2-D3 plan, do wype�nienia po labellowaniu CVAT):
  - source: training/v4/data.yaml
  - labels: D2-D3 CVAT labellowanie + v3 base (~2134) + nowe (~2000+)

Plan D4 (28.04): trenuj OBA imgsz dla decyzji empirycznej:
  - yolov8s @ imgsz=640: ~30 ms inference estimate, lepszy speed gain do 19 ms target
  - yolov8s @ imgsz=960: ~50-60 ms estimate, kontynuacja v3 podejscia (lepszy recall malych)

Uzycie (Colab albo lokalnie z GPU):
    python training/train_v4.py --imgsz 640
    python training/train_v4.py --imgsz 960
    # Albo z customowymi hyperparametrami:
    python training/train_v4.py --imgsz 640 --epochs 60 --batch 32

Zmiany vs train_v3:
  - base model: yolov8s.pt zamiast yolov8m.pt (~11M vs 26M params, 2x szybszy)
  - epochs 40 -> 50 (mniejszy model wymaga wiecej czasu na convergence)
  - imgsz parametryzowany (CLI flag)
  - auto-export do ONNX FP32 + FP16 po treningu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train v4 YOLOv8s drone model")
    p.add_argument("--imgsz", type=int, default=640, choices=[640, 960],
                   help="Input image size. 640 = speed-first, 960 = accuracy-first")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32,
                   help="Batch size; reduce to 16 if OOM on T4/3060")
    p.add_argument("--data", default="training/v4/data.yaml",
                   help="Dataset yaml (musi istniec, D2-D3 labellowanie wype�nia)")
    p.add_argument("--base", default="yolov8s.pt",
                   help="Base model checkpoint (yolov8s.pt z COCO domyslnie)")
    p.add_argument("--device", default="0",
                   help="CUDA device id albo 'cpu'")
    p.add_argument("--patience", type=int, default=15,
                   help="Early stopping patience (epok bez poprawy mAP)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last checkpoint w runs/v4_<imgsz>/")
    p.add_argument("--no-export", action="store_true",
                   help="Skip ONNX export po treningu")
    return p.parse_args()


def train(args: argparse.Namespace) -> Path:
    """Trenuj YOLOv8s. Zwraca Path do best.pt."""
    if not Path(args.data).exists():
        print(f"ERROR: {args.data} not found.", file=sys.stderr)
        print(f"  Najpierw zlabelluj dane przez CVAT i wyeksportuj YOLO 1.1 do training/v4/", file=sys.stderr)
        print(f"  Patrz docs/cvat_setup.md sekcja 'Eksport po labellowaniu'", file=sys.stderr)
        sys.exit(1)

    from ultralytics import YOLO

    run_name = f"v4_drone_s_imgsz{args.imgsz}"
    run_project = "training/runs"

    print(f"[train_v4] base={args.base} imgsz={args.imgsz} epochs={args.epochs} batch={args.batch}")
    print(f"[train_v4] device={args.device} data={args.data}")
    print(f"[train_v4] output: {run_project}/{run_name}/")

    model = YOLO(args.base)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        name=run_name,
        project=run_project,
        resume=args.resume,

        # Optimizer parity z v3 (sprawdzone, dziala)
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,

        # Augmentations: silniejsze dla imgsz=640 (mniejszy input, wymaga wiekszej diversity)
        # mosaic=1.0 + mixup=0.15 (vs v3 0.1) -- maly dataset+ maly model
        mosaic=1.0,
        mixup=0.15,
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
        close_mosaic=10,

        plots=True,
        save=True,
        verbose=True,
    )

    best_pt = Path(f"{run_project}/{run_name}/weights/best.pt")
    if not best_pt.exists():
        print(f"WARN: {best_pt} not found after training", file=sys.stderr)
    else:
        print(f"[train_v4] best weights: {best_pt}")
    return best_pt


def export_onnx(best_pt: Path, imgsz: int) -> None:
    """Eksportuj best.pt do ONNX FP32 + FP16 dla pipeline'u C++."""
    if not best_pt.exists():
        print(f"WARN: skipping ONNX export, {best_pt} missing")
        return

    from ultralytics import YOLO

    model = YOLO(str(best_pt))

    # FP32
    print(f"[train_v4] eksportuje FP32 ONNX (imgsz={imgsz}, opset=12, simplify=False)...")
    fp32_path = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=False, dynamic=False)
    fp32_target = best_pt.parent / f"v4_best_imgsz{imgsz}.onnx"
    if Path(fp32_path).exists() and Path(fp32_path) != fp32_target:
        Path(fp32_path).rename(fp32_target)
    print(f"[train_v4]   -> {fp32_target} ({fp32_target.stat().st_size/1e6:.1f} MB)")

    # FP16 (re-load bo export modyfikuje stan)
    model = YOLO(str(best_pt))
    print(f"[train_v4] eksportuje FP16 ONNX (imgsz={imgsz}, half=True)...")
    fp16_path = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=False, dynamic=False, half=True)
    fp16_target = best_pt.parent / f"v4_best_fp16_imgsz{imgsz}.onnx"
    if Path(fp16_path).exists() and Path(fp16_path) != fp16_target:
        Path(fp16_path).rename(fp16_target)
    print(f"[train_v4]   -> {fp16_target} ({fp16_target.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    args = parse_args()
    best_pt = train(args)
    if not args.no_export:
        export_onnx(best_pt, args.imgsz)
    print(f"\n[train_v4] DONE. Aby benchmark inference w C++:")
    print(f"  ./cpp/build/Release/poc_inference.exe <jpg> {best_pt.parent}/v4_best_fp16_imgsz{args.imgsz}.onnx {args.imgsz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
