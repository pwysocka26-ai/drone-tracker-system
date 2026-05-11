"""
Analyzer dla MTT duplicate spawning.

Czyta telemetry.jsonl, znajduje spawn events (pierwsza klatka gdzie pojawia
się nowy track_id), dla każdego loguje kontekst: jakie tracki już istniały,
distance/IoU do najblizszego, ile trackow ma missed_frames=0 w tej klatce
(proxy dla "ile detekcji w klatce").

Output: stats + lista konkretnych spawn events do code review.

Usage:
    python tools/analyze_mtt_duplicates.py artifacts/runs/<run>/telemetry.jsonl
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    bb = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / (aa + bb - inter)


def main(path: str) -> int:
    p = Path(path)
    if not p.exists():
        print(f"missing: {path}", file=sys.stderr)
        return 1

    seen_track_ids: set[int] = set()
    spawn_events: list[dict] = []
    multi_track_frames = 0
    confirmed_multi_frames = 0
    total_frames = 0

    track_history: dict[int, dict] = {}  # track_id -> {first_frame, last_frame, max_hits, was_confirmed}

    coexistence_pairs: Counter = Counter()  # (older_id, newer_id) -> frame count both alive
    last_alive: set[int] = set()

    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_frames += 1
            tracks = rec.get("tracks") or []
            frame_idx = rec.get("frame_idx", total_frames - 1)

            if len(tracks) >= 2:
                multi_track_frames += 1
            confirmed = [t for t in tracks if t.get("is_confirmed")]
            if len(confirmed) >= 2:
                confirmed_multi_frames += 1

            current_ids = {t["track_id"] for t in tracks}

            # spawn detection: track_id seen first time
            for t in tracks:
                tid = t["track_id"]
                if tid in seen_track_ids:
                    continue
                seen_track_ids.add(tid)

                others = [o for o in tracks if o["track_id"] != tid]
                # nearest existing track by center distance
                nearest = None
                nearest_dist = float("inf")
                for o in others:
                    dx = t["cx"] - o["cx"]; dy = t["cy"] - o["cy"]
                    d = math.hypot(dx, dy)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest = o

                # detections-this-frame proxy: tracks with missed_frames == 0
                fresh = [o for o in tracks if o.get("missed_frames", 0) == 0]

                spawn_events.append({
                    "frame_idx": frame_idx,
                    "new_id": tid,
                    "new_cx": t["cx"], "new_cy": t["cy"],
                    "new_bbox": t["bbox"], "new_conf": t["conf"],
                    "n_existing": len(others),
                    "n_fresh_total": len(fresh),  # incl. new track itself
                    "nearest_id": nearest["track_id"] if nearest else None,
                    "nearest_dist": nearest_dist if nearest else None,
                    "nearest_iou": iou(t["bbox"], nearest["bbox"]) if nearest else None,
                    "nearest_missed": nearest.get("missed_frames") if nearest else None,
                    "nearest_hits": nearest.get("hits") if nearest else None,
                    "nearest_confirmed": nearest.get("is_confirmed") if nearest else None,
                    "nearest_cx": nearest["cx"] if nearest else None,
                    "nearest_cy": nearest["cy"] if nearest else None,
                    "nearest_bbox": nearest["bbox"] if nearest else None,
                })

            # update lifecycle stats
            for t in tracks:
                tid = t["track_id"]
                h = track_history.setdefault(tid, {
                    "first_frame": frame_idx, "last_frame": frame_idx,
                    "max_hits": 0, "was_confirmed": False,
                })
                h["last_frame"] = frame_idx
                h["max_hits"] = max(h["max_hits"], t.get("hits", 0))
                if t.get("is_confirmed"):
                    h["was_confirmed"] = True

            # coexistence
            confirmed_ids = sorted(t["track_id"] for t in confirmed)
            for i in range(len(confirmed_ids)):
                for j in range(i + 1, len(confirmed_ids)):
                    coexistence_pairs[(confirmed_ids[i], confirmed_ids[j])] += 1

    # report
    print("=" * 72)
    print(f"FILE: {path}")
    print(f"Total frames: {total_frames}")
    print(f"Frames with 2+ tracks (any): {multi_track_frames} ({100*multi_track_frames/max(1,total_frames):.1f}%)")
    print(f"Frames with 2+ confirmed tracks: {confirmed_multi_frames} ({100*confirmed_multi_frames/max(1,total_frames):.1f}%)")
    print(f"Total spawn events (unique track_ids): {len(spawn_events)}")
    print()

    # Classify spawn events
    near_existing_close = [s for s in spawn_events if s["nearest_dist"] is not None and s["nearest_dist"] < 50]
    near_existing_iou_high = [s for s in spawn_events if s["nearest_iou"] is not None and s["nearest_iou"] > 0.3]
    spawn_with_fresh_existing = [s for s in spawn_events if s["nearest_missed"] == 0]
    spawn_with_zombie_existing = [s for s in spawn_events if s["nearest_missed"] is not None and s["nearest_missed"] > 0]

    print("=" * 72)
    print("SPAWN EVENT CLASSIFICATION")
    print(f"  Spawned <50 px from existing track:        {len(near_existing_close)}/{len(spawn_events)}")
    print(f"  Spawned with IoU>0.3 to existing track:    {len(near_existing_iou_high)}/{len(spawn_events)}")
    print(f"  Spawn while nearest existing missed=0:     {len(spawn_with_fresh_existing)}  <- both have det. this frame = 2 detekcje na 1 dronie?")
    print(f"  Spawn while nearest existing missed>0:     {len(spawn_with_zombie_existing)}  <- zombie kalman ghost obok swiezego")
    print()

    # Top 15 longest coexistences
    print("=" * 72)
    print("TOP CONFIRMED-PAIR COEXISTENCES (frames both alive & confirmed)")
    for (a, b), n in coexistence_pairs.most_common(15):
        ah = track_history.get(a, {})
        bh = track_history.get(b, {})
        print(f"  ID {a:4d} <-> ID {b:4d}: {n:5d} frames  | id{a} life [{ah.get('first_frame')}..{ah.get('last_frame')}] hits<={ah.get('max_hits')} | id{b} life [{bh.get('first_frame')}..{bh.get('last_frame')}] hits<={bh.get('max_hits')}")
    print()

    # Sample spawn events near existing
    print("=" * 72)
    print("SAMPLE SUSPICIOUS SPAWNS (top 15 by smallest distance to existing)")
    suspicious = sorted(
        [s for s in spawn_events if s["nearest_dist"] is not None],
        key=lambda s: s["nearest_dist"],
    )[:15]
    for s in suspicious:
        print(
            f"  frame {s['frame_idx']:5d}  new ID {s['new_id']:4d} cx={s['new_cx']:.1f} cy={s['new_cy']:.1f} "
            f"conf={s['new_conf']:.2f} bbox_w={s['new_bbox'][2]-s['new_bbox'][0]:.1f}x{s['new_bbox'][3]-s['new_bbox'][1]:.1f}"
        )
        print(
            f"      nearest ID {s['nearest_id']:4d} (missed={s['nearest_missed']}, hits={s['nearest_hits']}, "
            f"confirmed={s['nearest_confirmed']}) dist={s['nearest_dist']:.1f}px iou={s['nearest_iou']:.3f}"
        )

    # Histogram of nearest_dist for all spawns
    print()
    print("=" * 72)
    print("NEAREST-DIST HISTOGRAM (spawn -> existing track)")
    bins = [(0, 10), (10, 30), (30, 50), (50, 100), (100, 220), (220, 500), (500, 9999)]
    for lo, hi in bins:
        n = sum(1 for s in spawn_events if s["nearest_dist"] is not None and lo <= s["nearest_dist"] < hi)
        print(f"  [{lo:4d}, {hi:4d}): {n}")
    n_no_nearest = sum(1 for s in spawn_events if s["nearest_dist"] is None)
    print(f"  no other track:  {n_no_nearest}")

    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("usage: analyze_mtt_duplicates.py <telemetry.jsonl>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(args[0]))
