#!/usr/bin/env bash
# Trenuj OBA warianty v4 yolov8s na D4 (28.04). Sequencyjnie -- jeden GPU,
# kazdy run ~5-6h na T4 (Colab Pro). Razem ~10-12h.
#
# Output:
#   training/runs/v4_drone_s_imgsz640/weights/best.pt
#   training/runs/v4_drone_s_imgsz640/weights/v4_best_fp16_imgsz640.onnx
#   training/runs/v4_drone_s_imgsz960/weights/best.pt
#   training/runs/v4_drone_s_imgsz960/weights/v4_best_fp16_imgsz960.onnx
#
# Po ukonczeniu D5 benchmark obu na test.mp4 + frame_000150 dla decyzji
# ktory model na demo 30.04.
#
# Uzycie (Colab Pro / lokalnie z GPU):
#     bash training/train_v4_both.sh
# Albo przez Python (rownolegle, dwa GPU):
#     python training/train_v4.py --imgsz 640 --device 0 &
#     python training/train_v4.py --imgsz 960 --device 1 &

set -e

cd "$(dirname "$0")/.."

echo "=== train v4 yolov8s @ imgsz=640 (speed-first) ==="
python training/train_v4.py --imgsz 640 "$@"

echo ""
echo "=== train v4 yolov8s @ imgsz=960 (accuracy-first, kontynuacja v3) ==="
python training/train_v4.py --imgsz 960 "$@"

echo ""
echo "=== DONE -- D5 benchmark obu wariantow ==="
ls -lh training/runs/v4_drone_s_imgsz*/weights/v4_best_fp16_imgsz*.onnx 2>/dev/null
