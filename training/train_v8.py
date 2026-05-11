"""Train v8 single-class (dron) na ROCm. Init z v7_best.pt (transfer).

v8 dataset: v7 (6440+1545) + sea_drone (748+189) = 7188+1734, single class.
Cel: fix sea blind spot (test.mp4 first half detection 0% -> >50%).

ROCm specific (z MEMORY 2026-05-02):
- amp=False (gfx1151 ma quirks z mixed precision)
- workers=2 (DataLoader stability na ROCm)
- device='0' (cuda alias dla ROCm w torch 2.12)

Usage (z aktywnego conda env rocm-yolo):
    # Smoke (5 epok, fast feedback)
    python training/train_v8.py --base yolov8m.pt --imgsz 1280 --epochs 5

    # Production (~14h ROCm, oczekiwane early-stop ~30-50 epok)
    python training/train_v8.py --base data/weights/v7_best.pt \\
        --imgsz 1280 --epochs 80 --batch 8 --patience 15
"""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--data", default="training/v8/data.yaml")
    # Default: init z v7 (transfer learning) — szybsze convergence niż yolov8m.pt cold start
    p.add_argument("--base", default="data/weights/v7_best.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--name", default=None,
                   help="run name (default: v8_<base_stem>_<imgsz>)")
    p.add_argument("--copy-paste", type=float, default=0.3,
                   help="Copy-paste aug. 0.3 zalecane dla yolov8m+ small obj")
    p.add_argument("--mixup", type=float, default=0.10,
                   help="Mixup aug. 0.10 zalecane dla yolov8m")
    p.add_argument("--cache", default=False,
                   help="DataLoader cache: False/True/'ram'/'disk'. UWAGA RAM:"
                        " v8 ma ~7188 train * 1280^2 * 3 = ~35 GB cache='ram'."
                        " 'disk' bezpieczniejsze przy 32 GB OS RAM.")
    p.add_argument("--amp", action="store_true",
                   help="Mixed precision (FP16). NIE zalecane na ROCm gfx1151.")
    p.add_argument("--close-mosaic", type=int, default=10,
                   help="Wyłącz mosaic w ostatnich N epokach")
    p.add_argument("--resume", default=None,
                   help="Resume z konkretnego ckpt (np. runs/.../last.pt)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.base)
    name = args.name or f"v8_{Path(args.base).stem}_{args.imgsz}"

    print(f"=== train_v8.py ===")
    print(f"base:    {args.base}")
    print(f"data:    {args.data}")
    print(f"imgsz:   {args.imgsz}")
    print(f"epochs:  {args.epochs}")
    print(f"batch:   {args.batch}")
    print(f"device:  {args.device}")
    print(f"workers: {args.workers}")
    print(f"name:    {name}")
    print()

    cache_val = args.cache
    if isinstance(cache_val, str) and cache_val.lower() in ("true", "1"):
        cache_val = True
    elif isinstance(cache_val, str) and cache_val.lower() in ("false", "0"):
        cache_val = False

    train_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project="training/runs",
        name=name,
        amp=args.amp,
        workers=args.workers,
        patience=args.patience,
        copy_paste=args.copy_paste,
        mixup=args.mixup,
        cache=cache_val,
        close_mosaic=args.close_mosaic,
        verbose=True,
    )
    if args.resume:
        train_kwargs["resume"] = args.resume

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
