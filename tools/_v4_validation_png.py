"""Generate v4 vs v3 PNG side-by-side comparisons na sea_drone scenario.

Wczytuje telemetry z 2 runow (v4@640, v3@960) na data/test.mp4 i rysuje
bounding boxes z obu modeli na tych samych klatkach (4 representatywne).

Output: artifacts/v4_validation/sea_drone/sea_drone_frame_NNNN.png
"""
from __future__ import annotations

import cv2
import json
from pathlib import Path


def load_telemetry(run_dir: Path) -> dict:
    rows: dict[int, dict] = {}
    with open(run_dir / "telemetry.jsonl", "r") as f:
        for line in f:
            r = json.loads(line)
            rows[r["frame_idx"]] = r
    return rows


def get_bbox_and_conf(t: dict) -> tuple[list | None, float]:
    """Wyciagnij bbox+conf z telemetry record. Format flat: active_track_bbox + _conf."""
    bb = t.get("active_track_bbox")
    if not bb or len(bb) < 4:
        return None, 0.0
    conf = float(t.get("active_track_conf") or 0.0)
    return list(bb[:4]), conf


def draw_bbox(img, bbox, conf, color, label_prefix: str):
    if not bbox:
        cv2.putText(img, f"{label_prefix} NO DET", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.putText(img, f"{label_prefix} c={conf:.2f}", (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def annotate_header(img, side_label: str, lock_state: str, bg_color):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 42), bg_color, -1)
    cv2.putText(img, f"{side_label}  lock={lock_state}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main() -> int:
    base = Path("C:/Users/pwyso/drone-tracker-system/artifacts/runs")
    v4 = load_telemetry(base / "2026-04-26_200838")
    v3 = load_telemetry(base / "2026-04-26_200919")

    video = Path("C:/Users/pwyso/drone-tracker-system/data/test.mp4")
    out_dir = Path("C:/Users/pwyso/drone-tracker-system/artifacts/v4_validation/sea_drone")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample = [50, 150, 250, 350]

    for fidx in sample:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            print(f"  [skip] frame {fidx}: read failed")
            continue
        f4 = frame.copy()
        f3 = frame.copy()
        t4 = v4.get(fidx, {})
        t3 = v3.get(fidx, {})

        bb4, c4 = get_bbox_and_conf(t4)
        bb3, c3 = get_bbox_and_conf(t3)
        draw_bbox(f4, bb4, c4, (0, 255, 0), "v4")
        draw_bbox(f3, bb3, c3, (255, 200, 0), "v3")

        ls4 = t4.get("narrow_lock_state", "?")
        ls3 = t3.get("narrow_lock_state", "?")
        annotate_header(f4, "v4 yolov8s @640 FP16", ls4, (0, 100, 0))
        annotate_header(f3, "v3 yolov8m @960 FP16", ls3, (100, 80, 0))

        side = cv2.hconcat([f4, f3])
        cv2.putText(side, f"sea_drone  frame {fidx}/{total}",
                    (10, side.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out_path = out_dir / f"sea_drone_frame_{fidx:04d}.png"
        cv2.imwrite(str(out_path), side)
        print(f"  -> {out_path.name}  v4: lock={ls4} {'BBOX' if bb4 else 'no det'}  "
              f"|  v3: lock={ls3} {'BBOX' if bb3 else 'no det'}")

    cap.release()
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
