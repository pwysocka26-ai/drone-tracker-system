"""Eksportuje v3 best.pt do ONNX FP16 dla benchmark inference.

FP16 redukuje pamiec modelu ~2x i zwykle przyspiesza inference 1.3-1.7x na
GPU/iGPU z native FP16 (Radeon 8060S, NVIDIA Tensor Cores). Trade-off:
mozliwy minimalny spadek mAP (typowo <1% dla yolov8).

Uzycie:
    python tools/export_v3_to_onnx_fp16.py [imgsz]
        imgsz: opcjonalne, domyslnie 960. Mozna podac 640 dla wiekszego speedup.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL = Path("data/weights/best.pt")


def main() -> int:
    if not MODEL.exists():
        print(f"ERROR: brak {MODEL}")
        return 1

    imgsz = 960
    if len(sys.argv) > 1:
        imgsz = int(sys.argv[1])

    out = Path(f"data/weights/v3_best_fp16_imgsz{imgsz}.onnx")

    from ultralytics import YOLO

    print(f"Laduje {MODEL}")
    model = YOLO(str(MODEL))

    print(f"Eksportuje do ONNX FP16 (imgsz={imgsz}, opset=12, simplify=False, half=True)")
    path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=12,
        simplify=False,
        dynamic=False,
        half=True,
    )

    exported = Path(path)
    if exported.exists() and exported != out:
        exported.rename(out)
        print(f"Przeniesiono: {exported} -> {out}")

    if out.exists():
        size_mb = out.stat().st_size / 1e6
        print(f"Gotowe: {out} ({size_mb:.1f} MB)")
    else:
        print(f"ERROR: {out} nie utworzone")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
