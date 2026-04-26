"""Symuluj ruch kamery (pan/tilt/jitter) na istniejacym video.

Bierze source video (np. video_test_wide_short.mp4 1920x1080 gdzie v3 wykrywa
drona), wycina mniejsze okno crop (1280x720) i przesuwa to okno przez klatki
roznymi wzorcami ruchu. Output: video gdzie tlo "obraca sie" jakby kamera
sie poruszala, ale dron zachowuje sie naturalnie.

Cel: testowac MTT/Lock motion handling bez problemu domain shift (v3 wciaz
wykrywa drona w croped frame, bo to nasz video).

3 warianty:
  - slow_pan: 10 px/frame horizontal pan (~5 deg/s @ 35 deg FOV / 1280 px)
  - fast_pan: 50 px/frame horizontal pan (~25 deg/s)
  - jitter: random +/- 5 px each frame (mechanical vibration emulation)

Uzycie:
    python tools/_synth_camera_motion.py
"""
from __future__ import annotations

import sys
import random
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2  # type: ignore


SRC = Path("artifacts/test_videos/video_test_wide_short.mp4")
OUT_DIR = Path("artifacts/test_videos")
CROP_W = 1280
CROP_H = 720


def gen_slow_pan(n: int, margin_w: int, margin_h: int) -> list[tuple[int, int]]:
    """Bouncing slow pan, 10 px/frame horizontal."""
    coords = []
    x, direction = 0, 1
    for _ in range(n):
        coords.append((x, margin_h // 2))
        x += direction * 10
        if x >= margin_w:
            x = margin_w; direction = -1
        if x <= 0:
            x = 0; direction = 1
    return coords


def gen_fast_pan(n: int, margin_w: int, margin_h: int) -> list[tuple[int, int]]:
    """Bouncing fast pan, 50 px/frame horizontal."""
    coords = []
    x, direction = 0, 1
    for _ in range(n):
        coords.append((x, margin_h // 2))
        x += direction * 50
        if x >= margin_w:
            x = margin_w; direction = -1
        if x <= 0:
            x = 0; direction = 1
    return coords


def gen_jitter(n: int, margin_w: int, margin_h: int, seed: int = 42) -> list[tuple[int, int]]:
    """Mechanical vibration: random +/- 5 px around center (small-amplitude)."""
    rng = random.Random(seed)
    coords = []
    cx, cy = margin_w // 2, margin_h // 2
    x, y = cx, cy
    for _ in range(n):
        x += rng.randint(-5, 5)
        y += rng.randint(-5, 5)
        x = max(0, min(margin_w, x))
        y = max(0, min(margin_h, y))
        coords.append((x, y))
    return coords


def render(src: Path, out: Path, coords: list[tuple[int, int]],
           crop_w: int, crop_h: int) -> int:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out), fourcc, fps, (crop_w, crop_h))
    n_written = 0
    for x, y in coords:
        ok, img = cap.read()
        if not ok:
            break
        crop = img[y:y + crop_h, x:x + crop_w]
        writer.write(crop)
        n_written += 1
    cap.release()
    writer.release()
    return n_written


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: brak {SRC}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(SRC))
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Source: {SRC.name}  {w_in}x{h_in}  fps={fps}  frames={n_src}")

    margin_w = w_in - CROP_W
    margin_h = h_in - CROP_H
    if margin_w < 0 or margin_h < 0:
        print(f"ERROR: source za male, potrzebne >={CROP_W}x{CROP_H}", file=sys.stderr)
        return 1
    print(f"Crop: {CROP_W}x{CROP_H}, margin: {margin_w}x{margin_h}")

    # Wygeneruj 3 warianty
    variants = [
        ("synth_slow_pan",  gen_slow_pan(n_src,  margin_w, margin_h)),
        ("synth_fast_pan",  gen_fast_pan(n_src,  margin_w, margin_h)),
        ("synth_jitter",    gen_jitter(n_src,    margin_w, margin_h)),
    ]
    for name, coords in variants:
        out = OUT_DIR / f"video_test_{name}.mp4"
        n = render(SRC, out, coords, CROP_W, CROP_H)
        size_mb = out.stat().st_size / 1e6
        print(f"  {name:20s} -> {out.name}  ({n} frames, {size_mb:.1f} MB)")

    print(f"\nGotowe. Test:")
    for name, _ in variants:
        print(f"  ./cpp/build/Release/dtracker_main.exe --video artifacts/test_videos/video_test_{name}.mp4 --no-gui --no-record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
