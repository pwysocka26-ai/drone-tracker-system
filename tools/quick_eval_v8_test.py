"""Szybki sanity eval v8 best.pt na test.mp4 — first half + full.

Cel: sprawdzić czy v8 widzi drone w klatkach 1-204 (blind spot v7).
Nie używa pipeline C++ — bezpośrednio ultralytics.predict() na CPU.

Usage:
    python tools/quick_eval_v8_test.py [--model PATH] [--conf 0.20] [--device cpu]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/detect/training/runs/v8_smoke/weights/best.pt")
    ap.add_argument("--video", default="data/test.mp4")
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = YOLO(args.model)
    print(f"Model: {args.model}")
    print(f"Video: {args.video}")
    print(f"Device: {args.device}, conf={args.conf}, imgsz={args.imgsz}")

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total}")

    # Track detection rate per range
    first_half_with = 0
    first_half_total = 0
    second_half_with = 0
    second_half_total = 0

    confs_first = []
    confs_second = []
    first_detection_idx = None

    boundary = 205  # z analyzy v7

    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                device=args.device, verbose=False)
        boxes = results[0].boxes
        has_det = boxes is not None and len(boxes) > 0
        if has_det:
            best_conf = float(boxes.conf.max().cpu().numpy())
            if first_detection_idx is None:
                first_detection_idx = idx
        else:
            best_conf = 0.0

        if idx < boundary:
            first_half_total += 1
            if has_det:
                first_half_with += 1
                confs_first.append(best_conf)
        else:
            second_half_total += 1
            if has_det:
                second_half_with += 1
                confs_second.append(best_conf)

        if (idx + 1) % 50 == 0:
            print(f"  frame {idx+1}/{total}: first_half={first_half_with}/{first_half_total} "
                  f"second_half={second_half_with}/{second_half_total}",
                  flush=True)

    cap.release()

    print()
    print("=== RESULTS ===")
    print(f"First half (1-{boundary-1}): {first_half_with}/{first_half_total} "
          f"= {100*first_half_with/max(1,first_half_total):.2f}% detection rate")
    if confs_first:
        print(f"  Conf range: {min(confs_first):.3f} - {max(confs_first):.3f}, "
              f"mean: {sum(confs_first)/len(confs_first):.3f}")
    print(f"Second half ({boundary}+): {second_half_with}/{second_half_total} "
          f"= {100*second_half_with/max(1,second_half_total):.2f}% detection rate")
    if confs_second:
        print(f"  Conf range: {min(confs_second):.3f} - {max(confs_second):.3f}, "
              f"mean: {sum(confs_second)/len(confs_second):.3f}")
    print(f"First detection at frame: {first_detection_idx}")

    print()
    print("=== COMPARISON vs v7 ===")
    print("v7: first_half=0%, second_half=100%, first_det=205")
    if first_half_with > 0:
        print(f"v8 FIX SUCCESS: blind spot pokryty w {first_half_with}/{first_half_total} klatkach")
    else:
        print("v8 NO IMPROVEMENT: blind spot nadal 0% — może wymagać większej liczby epok")


if __name__ == "__main__":
    main()
