"""
Mierzy empirycznie lag narrow ramki vs centrum drona.

Per klatka liczymy:
  - owner.center (z active_track.bbox lub tracks[track_id==active_track_id])
  - narrow_center (smooth_center z PID)
  - lag_px = |owner.center - narrow_center|
  - velocity = |delta center between frames|

Output: histogram lag, lag vs velocity correlation, top frames z najwyzszym lag.

Usage:
    python tools/analyze_narrow_lag.py artifacts/runs/<run>/telemetry.jsonl
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main(path: str) -> int:
    p = Path(path)
    if not p.exists():
        print(f"missing: {path}", file=sys.stderr)
        return 1

    samples = []  # (frame_idx, owner_cx, owner_cy, narrow_cx, narrow_cy, lag, vel)

    last_owner_xy = None
    n_total = 0
    n_with_owner = 0
    n_with_narrow = 0

    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1

            active_id = rec.get("active_track_id")
            if active_id is None:
                last_owner_xy = None
                continue

            tracks = rec.get("tracks") or []
            owner = next((t for t in tracks if t["track_id"] == active_id), None)
            if owner is None:
                last_owner_xy = None
                continue
            n_with_owner += 1

            narrow_center = rec.get("narrow_center")
            if narrow_center is None or len(narrow_center) < 2:
                last_owner_xy = None
                continue
            n_with_narrow += 1

            owner_cx, owner_cy = owner["cx"], owner["cy"]
            narrow_cx, narrow_cy = narrow_center[0], narrow_center[1]

            lag = math.hypot(owner_cx - narrow_cx, owner_cy - narrow_cy)

            vel = 0.0
            if last_owner_xy is not None:
                vel = math.hypot(owner_cx - last_owner_xy[0], owner_cy - last_owner_xy[1])
            last_owner_xy = (owner_cx, owner_cy)

            samples.append({
                "frame": rec.get("frame_idx"),
                "owner_cx": owner_cx,
                "owner_cy": owner_cy,
                "narrow_cx": narrow_cx,
                "narrow_cy": narrow_cy,
                "lag": lag,
                "vel": vel,
                "lock_state": rec.get("narrow_lock_state"),
            })

    print(f"=== {path} ===")
    print(f"Total frames: {n_total}")
    print(f"Frames with owner: {n_with_owner}")
    print(f"Frames with narrow + owner: {n_with_narrow}")
    print()

    if not samples:
        return 0

    lags = [s["lag"] for s in samples]
    vels = [s["vel"] for s in samples]
    lags_sorted = sorted(lags)
    n = len(lags_sorted)

    def pct(p):
        return lags_sorted[int(p * n)]

    print(f"=== LAG (px) — narrow_center vs owner_center ===")
    print(f"  mean: {sum(lags)/n:.2f}")
    print(f"  p50:  {pct(0.50):.2f}")
    print(f"  p90:  {pct(0.90):.2f}")
    print(f"  p99:  {pct(0.99):.2f}")
    print(f"  max:  {lags_sorted[-1]:.2f}")
    print()

    bins = [(0,2),(2,5),(5,10),(10,20),(20,50),(50,100),(100,9999)]
    print("LAG histogram:")
    for lo, hi in bins:
        cnt = sum(1 for l in lags if lo <= l < hi)
        print(f"  [{lo:4d}, {hi:4d}) px: {cnt:6d} ({100*cnt/n:5.2f}%)")
    print()

    # Lag vs velocity correlation
    print("=== LAG vs VELOCITY ===")
    vel_bins = [(0,1),(1,2),(2,5),(5,10),(10,20),(20,9999)]
    for lo, hi in vel_bins:
        in_bin = [s["lag"] for s in samples if lo <= s["vel"] < hi]
        if in_bin:
            avg_lag = sum(in_bin) / len(in_bin)
            print(f"  vel [{lo:3d}, {hi:4d}) px/frame, n={len(in_bin):6d}, avg_lag={avg_lag:.2f} px")
    print()

    # Top 10 highest-lag frames
    print("=== TOP 15 HIGH-LAG FRAMES ===")
    top = sorted(samples, key=lambda s: -s["lag"])[:15]
    for s in top:
        print(f"  frame {s['frame']:5d}  lag={s['lag']:6.2f} px  vel={s['vel']:6.2f}  owner=({s['owner_cx']:7.1f},{s['owner_cy']:7.1f})  narrow=({s['narrow_cx']:7.1f},{s['narrow_cy']:7.1f})  state={s['lock_state']}")

    # Lag for stationary drone (vel < 1 px/frame) — should be ~0 if PID converges
    stationary = [s["lag"] for s in samples if s["vel"] < 1.0]
    if stationary:
        print()
        print(f"=== STATIONARY DRONE (vel < 1 px/frame), n={len(stationary)} ===")
        ss = sorted(stationary)
        print(f"  mean lag: {sum(ss)/len(ss):.2f} px")
        print(f"  p50:      {ss[len(ss)//2]:.2f}")
        print(f"  p90:      {ss[int(0.9*len(ss))]:.2f}")
        print(f"  max:      {ss[-1]:.2f}")
        if sum(ss)/len(ss) > 4.0:
            print("  WARNING: stationary drone has avg lag > pid_dead_zone_active=4 px -> dead zone bug?")

    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("usage: analyze_narrow_lag.py <telemetry.jsonl>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(args[0]))
