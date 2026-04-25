"""Eksportuje v3 best.pt do ONNX dla C++ inference przez ONNX Runtime.

Uzycie:
    python tools/export_v3_to_onnx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL = Path("data/weights/best.pt")
OUT = Path("data/weights/v3_best.onnx")
IMGSZ = 960  # v3 trenowany na 960, przy 640 drastyczny spadek recall na malych dronach


def main() -> int:
    if not MODEL.exists():
        print(f"ERROR: brak {MODEL}")
        return 1

    from ultralytics import YOLO

    print(f"Laduje {MODEL}")
    model = YOLO(str(MODEL))

    print(f"Eksportuje do ONNX (imgsz={IMGSZ}, opset=17)")
    # opset 17 jest dobrze wspierany przez ONNX Runtime 1.17+
    # dynamic batch/size NIE -- stala rozdzielczosc dla max optimizacji
    path = model.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=False, dynamic=False)

    exported = Path(path)
    if exported.exists() and exported != OUT:
        exported.rename(OUT)
        print(f"Przeniesiono: {exported} -> {OUT}")

    if OUT.exists():
        size_mb = OUT.stat().st_size / 1e6
        print(f"Gotowe: {OUT} ({size_mb:.1f} MB)")
    else:
        print(f"ERROR: {OUT} nie utworzone")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
