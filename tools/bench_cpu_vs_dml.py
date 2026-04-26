"""Benchmark inference czasu CPU vs DirectML (Radeon 8060S iGPU) dla v3 ONNX.

Uzywa poc_inference.exe z argv:
  poc_inference.exe <jpg> <onnx> <imgsz> [cpu]

Trzeci pozycyjny "cpu" wylaczna DirectML (CPU-only). Inaczej DirectML domyslnie.

Wynik: tabela markdown z 4 wariantami (FP32/FP16 x CPU/DML) + speedup.

Cel: dane do raportu / decyzji o hardware production (Jetson NVIDIA TensorRT
vs AMD Ryzen AI NPU) -- wiedziec ile DirectML faktycznie ratuje na obecnym
test rig (GMKtec Radeon 8060S).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


REPO = Path(__file__).resolve().parent.parent
POC = REPO / "cpp" / "build" / "Release" / "poc_inference.exe"
IMG = REPO / "artifacts" / "cvat_import" / "obj_train_data" / "frame_000150.jpg"

VARIANTS = [
    ("FP32 imgsz=960 DirectML", "v3_best.onnx", 960, False),
    ("FP32 imgsz=960 CPU",      "v3_best.onnx", 960, True),
    ("FP16 imgsz=960 DirectML", "v3_best_fp16_imgsz960.onnx", 960, False),
    ("FP16 imgsz=960 CPU",      "v3_best_fp16_imgsz960.onnx", 960, True),
]


def run_one(model_name: str, imgsz: int, cpu_only: bool) -> tuple[float, int, float]:
    """Zwraca (avg_inference_ms, num_detections, top_conf)."""
    model_path = REPO / "data" / "weights" / model_name
    if not model_path.exists():
        print(f"  SKIP: brak {model_path}")
        return (-1.0, -1, 0.0)
    args = [str(POC), str(IMG), str(model_path), str(imgsz)]
    if cpu_only:
        args.append("cpu")
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=120, encoding="utf-8")
    except subprocess.TimeoutExpired:
        return (-1.0, -1, 0.0)
    if out.returncode != 0:
        print(f"  ERROR: {out.stderr[:200]}")
        return (-1.0, -1, 0.0)
    text = out.stdout

    inf_ms = -1.0
    m = re.search(r"Avg inference time:\s*([\d.]+)\s*ms", text)
    if m:
        inf_ms = float(m.group(1))

    dets = -1
    m = re.search(r"DETECTIONS\s*\((\d+)\)", text)
    if m:
        dets = int(m.group(1))

    top_conf = 0.0
    m = re.search(r"conf=([\d.]+)", text)
    if m:
        top_conf = float(m.group(1))

    return (inf_ms, dets, top_conf)


def main() -> int:
    if not POC.exists():
        print(f"ERROR: brak {POC}", file=sys.stderr)
        print("Build first: cmake --build cpp/build --config Release --target poc_inference", file=sys.stderr)
        return 1
    if not IMG.exists():
        print(f"ERROR: brak {IMG}", file=sys.stderr)
        return 1

    print(f"Image: {IMG.name}\n")
    results = []
    for label, model, imgsz, cpu in VARIANTS:
        print(f"  Running {label}...")
        ms, dets, conf = run_one(model, imgsz, cpu)
        results.append((label, ms, dets, conf))

    print()
    print("# CPU vs DirectML benchmark")
    print()
    print(f"- POC binary: `cpp/build/Release/poc_inference.exe`")
    print(f"- Test image: `{IMG.relative_to(REPO).as_posix()}`")
    print(f"- 3 warmup + 10 measured runs per variant\n")
    print("| variant | inference (ms) | fps | detections | top conf |")
    print("|---|---|---|---|---|")
    baseline_ms = None
    for label, ms, dets, conf in results:
        if ms < 0:
            print(f"| {label} | (failed/skipped) | — | — | — |")
            continue
        fps = 1000.0 / ms if ms > 0 else 0
        print(f"| {label} | {ms:.1f} | {fps:.1f} | {dets} | {conf:.3f} |")
        if "FP32 imgsz=960 DirectML" in label:
            baseline_ms = ms

    if baseline_ms:
        print()
        print(f"## Speedup vs FP32 imgsz=960 DirectML ({baseline_ms:.1f} ms)")
        print()
        print("| variant | speedup |")
        print("|---|---|")
        for label, ms, _, _ in results:
            if ms < 0 or "FP32 imgsz=960 DirectML" in label:
                continue
            speedup = baseline_ms / ms
            arrow = "faster" if speedup > 1 else "slower"
            print(f"| {label} | {speedup:.2f}x {arrow} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
