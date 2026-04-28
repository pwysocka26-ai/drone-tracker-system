"""Train v5 YOLOv8m drone model — atak na YOLO blindness na male drony.

Empiryczna motywacja (memory: project_v4_dataset_size_distribution_2026_04_27,
project_lock_loss_root_cause_2026_04_27):

- v4 dataset ma 64.5% bboxow <30 px short_side (median 12 px native),
  ale runtime imgsz=640 + 1920x1080 video skaluje drona 12 px → 4 px,
  ponizej yolov8 stride 8.
- v4@960 testowane (tools/eval_imgsz_comparison.py): detection rate
  18.8% → 19.6% (margines), ale mean_conf 0.251 → 0.469 (+87%).
  Niewystarczajace, bo drone 12 px @ 960 = 6 px (wciaz <stride 8).
- yolov8s @ imgsz>=960 jest "over-budgeted" (commit 8cbe953):
  v4@960 yolov8s 84.3% LOCKED vs v4@640 yolov8s 87.1% — maly backbone +
  duzy input nie pomaga.

v5 zmiany:
- base: yolov8s.pt → yolov8m.pt (~28M params, capacity dla imgsz=1280)
- imgsz: 640 → 1280 (drone 12 px native @ 1920x1080 → 8 px po scale,
  na granicy stride 8 — wykrywalny po raz pierwszy)
- batch: 32 → 16 (yolov8m@1280 jest VRAM-ciezki, fits w 16 GB T4 z 16)
- epochs: 50 → 60 (wiekszy model, dluzsza konwergencja)
- patience: 15 → 20 (wiekszy buffer dla cos LR plateau)
- copy_paste: 0.0 → 0.3 (KLUCZOWE dla small object recall —
  syntetyczne kopiowanie dronow na hard-negs zwieksza diversity poz pix)
- mixup: 0.15 → 0.10 (mniej agresywne dla wiekszego modelu)
- scale: 0.5 → 0.5 (bez zmian — zachowaj zoom-in/out variance)

Plan v6 (jesli v5 niewystarczy):
- P2 head (stride=4) dodany do yolov8m architecture (custom yaml)
- imgsz=1920 (drone 12 px native → 12 px po scale 1.0×, idealnie wykrywalny)

Uzycie (Colab Pro / RunPod RTX 4090):
    python training/train_v5.py
    # Albo eksperymenty:
    python training/train_v5.py --imgsz 1536 --epochs 80
    python training/train_v5.py --base yolov8l.pt  # wieksza capacity

Output:
    training/runs/v5_drone_m_imgsz1280/weights/best.pt
    training/runs/v5_drone_m_imgsz1280/weights/v5_best_imgsz1280.onnx
    training/runs/v5_drone_m_imgsz1280/weights/v5_best_fp16_imgsz1280.onnx
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
    p = argparse.ArgumentParser(description="Train v5 YOLOv8m drone model")
    p.add_argument("--imgsz", type=int, default=1280,
                   choices=[640, 960, 1280, 1536, 1920],
                   help="Input image size. 1280 default (drone median 12 px → "
                        "8 px po scale, na granicy stride 8)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size; reduce to 8 if OOM (T4 16 GB / 3060 12 GB)")
    p.add_argument("--data", default="training/v5/data.yaml",
                   help="Dataset yaml. v5 layout = kopia v4 (4032 obrazki, "
                        "64.5% bboxow <30 px wystarczy dla v5). "
                        "Generowane przez tools/_build_v5_dataset.py")
    p.add_argument("--base", default="yolov8m.pt",
                   help="Base checkpoint (yolov8m=28M, yolov8l=44M, yolov8x=68M)")
    p.add_argument("--device", default="0",
                   help="CUDA device id albo 'cpu'")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience")
    p.add_argument("--resume", action="store_true",
                   help="Resume z last checkpoint w runs/v5_<base>_imgsz<N>/")
    p.add_argument("--no-export", action="store_true",
                   help="Skip ONNX export po treningu")
    p.add_argument("--copy-paste", type=float, default=0.3,
                   help="Copy-paste aug probability (kluczowe dla small obj recall)")
    return p.parse_args()


def base_short(base: str) -> str:
    """yolov8m.pt -> 'm', yolov8l.pt -> 'l' itd."""
    stem = Path(base).stem.lower().replace("yolov8", "")
    return stem.rstrip(".pt") or "m"


def train(args: argparse.Namespace) -> Path:
    """Trenuj YOLOv8m. Zwraca Path do best.pt."""
    if not Path(args.data).exists():
        print(f"ERROR: {args.data} not found.", file=sys.stderr)
        sys.exit(1)

    from ultralytics import YOLO

    bs = base_short(args.base)
    run_name = f"v5_drone_{bs}_imgsz{args.imgsz}"
    run_project = "training/runs"

    print(f"[train_v5] base={args.base} imgsz={args.imgsz} epochs={args.epochs} batch={args.batch}")
    print(f"[train_v5] copy_paste={args.copy_paste} (small-obj augmentation)")
    print(f"[train_v5] device={args.device} data={args.data}")
    print(f"[train_v5] output: {run_project}/{run_name}/")

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

        # Optimizer (parity z v4)
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,

        # Augmentations: small-object focused
        # copy_paste 0.3 kluczowe — kopiuje drone instances na hard negs,
        # zwieksza positive samples dla rzadkich (40.4%) <10 px bboxow
        mosaic=1.0,
        mixup=0.10,
        copy_paste=args.copy_paste,
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
        print(f"[train_v5] best weights: {best_pt}")
    return best_pt


def export_onnx(best_pt: Path, imgsz: int) -> None:
    """Eksportuj best.pt do ONNX FP32 + FP16 dla pipeline'u C++."""
    if not best_pt.exists():
        print(f"WARN: skipping ONNX export, {best_pt} missing")
        return

    from ultralytics import YOLO

    # FP32
    model = YOLO(str(best_pt))
    print(f"[train_v5] eksportuje FP32 ONNX (imgsz={imgsz}, opset=12)...")
    fp32_path = model.export(format="onnx", imgsz=imgsz, opset=12,
                              simplify=False, dynamic=False)
    fp32_target = best_pt.parent / f"v5_best_imgsz{imgsz}.onnx"
    if Path(fp32_path).exists() and Path(fp32_path) != fp32_target:
        Path(fp32_path).rename(fp32_target)
    print(f"[train_v5]   -> {fp32_target} ({fp32_target.stat().st_size/1e6:.1f} MB)")

    # FP16
    model = YOLO(str(best_pt))
    print(f"[train_v5] eksportuje FP16 ONNX (imgsz={imgsz}, half=True)...")
    fp16_path = model.export(format="onnx", imgsz=imgsz, opset=12,
                              simplify=False, dynamic=False, half=True)
    fp16_target = best_pt.parent / f"v5_best_fp16_imgsz{imgsz}.onnx"
    if Path(fp16_path).exists() and Path(fp16_path) != fp16_target:
        Path(fp16_path).rename(fp16_target)
    print(f"[train_v5]   -> {fp16_target} ({fp16_target.stat().st_size/1e6:.1f} MB)")


def main() -> int:
    args = parse_args()
    best_pt = train(args)
    if not args.no_export:
        export_onnx(best_pt, args.imgsz)
    print(f"\n[train_v5] DONE. Pobierz weights na lokal i:")
    print(f"  ./cpp/build/Release/dtracker_main.exe \\")
    print(f"      --model data/weights/v5_best_fp16_imgsz{args.imgsz}.onnx \\")
    print(f"      --imgsz {args.imgsz}")
    print(f"\n[train_v5] Walidacja vs v4:")
    print(f"  python tools/eval_imgsz_comparison.py  # update MODELS list dla v5")
    print(f"  python tools/analyze_ghost_tracks.py artifacts/runs/<v5_run>/telemetry.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
