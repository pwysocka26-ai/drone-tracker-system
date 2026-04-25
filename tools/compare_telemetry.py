"""Porownuje dwa run-y dtracker_main per telemetry.jsonl.

Uzycie:
    python tools/compare_telemetry.py <run_dir_a> <run_dir_b> [--per-frame]

Args:
    run_dir_a, run_dir_b: sciezki do artifacts/runs/<timestamp>/
    --per-frame: dodatkowa analiza per-klatka (gdzie sie roznia)

Wynik: tabela markdown z metrykami obu runow + delta. Plus opcjonalnie
lista klatek gdzie selected_id sie rozni.

Typowe zastosowanie:
    # FP16 vs FP32 baseline
    python tools/compare_telemetry.py \\
        artifacts/runs/2026-04-25_215921 \\
        artifacts/runs/2026-04-25_224131
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_run(run_dir: Path) -> dict[str, Any]:
    """Zwraca slownik z metrykami z telemetry.jsonl + run_summary.json."""
    telemetry = run_dir / "telemetry.jsonl"
    summary = run_dir / "run_summary.json"
    if not telemetry.exists():
        print(f"ERROR: brak {telemetry}", file=sys.stderr)
        sys.exit(1)

    states: Counter[str] = Counter()
    sel_ids: Counter[int] = Counter()
    persistent_ids: Counter[int] = Counter()
    synth = 0
    csrt_active_frames = 0
    csrt_updated_frames = 0
    csrt_synthetic_frames = 0
    inf_ms: list[float] = []
    trk_ms: list[float] = []
    multi_tracks_total = 0
    frames_with_owner = 0
    sel_per_frame: dict[int, int | None] = {}
    n = 0

    for line in telemetry.open():
        r = json.loads(line)
        n += 1
        states[r["narrow_lock_state"]] += 1
        sel = r.get("selected_id")
        sel_per_frame[r["frame_idx"]] = sel
        if sel is not None:
            sel_ids[sel] += 1
            frames_with_owner += 1
        p = r.get("persistent_owner_id", -1)
        if p >= 0:
            persistent_ids[p] += 1
        if r.get("narrow_synthetic_hold"):
            synth += 1
        if r.get("csrt_active"):
            csrt_active_frames += 1
        if r.get("csrt_updated"):
            csrt_updated_frames += 1
        if r.get("csrt_synthetic_used"):
            csrt_synthetic_frames += 1
        inf_ms.append(r["inference_ms"])
        trk_ms.append(r["tracker_ms"])
        multi_tracks_total += r.get("multi_tracks", 0)

    summary_data: dict[str, Any] = {}
    if summary.exists():
        summary_data = json.loads(summary.read_text())

    return {
        "n": n,
        "states": dict(states),
        "unique_raw_track_id": len(sel_ids),
        "top_track_id": sel_ids.most_common(3),
        "unique_persistent": len(persistent_ids),
        "top_persistent": persistent_ids.most_common(3),
        "synthetic_hold": synth,
        "synthetic_hold_pct": 100 * synth / n if n else 0,
        "csrt_active_pct": 100 * csrt_active_frames / n if n else 0,
        "csrt_updated_pct": 100 * csrt_updated_frames / n if n else 0,
        "csrt_synthetic_pct": 100 * csrt_synthetic_frames / n if n else 0,
        "avg_inference_ms": sum(inf_ms) / n if n else 0,
        "avg_tracker_ms": sum(trk_ms) / n if n else 0,
        "p50_inference_ms": sorted(inf_ms)[n // 2] if n else 0,
        "frames_with_owner": frames_with_owner,
        "frames_with_owner_pct": 100 * frames_with_owner / n if n else 0,
        "multi_tracks_total": multi_tracks_total,
        "multi_tracks_avg": multi_tracks_total / n if n else 0,
        "lock_loss_events": summary_data.get("total_lock_loss_events", "?"),
        "reacquire_starts": summary_data.get("total_reacquire_starts", "?"),
        "reacquire_successes": summary_data.get("total_reacquire_successes", "?"),
        "reacquire_success_rate": summary_data.get("reacquire_success_rate"),
        "frames_locked": summary_data.get("total_time_in_locked_frames", "?"),
        "sel_per_frame": sel_per_frame,
    }


def render_compare(a: dict, b: dict, label_a: str, label_b: str) -> str:
    """Renderuje tabele markdown porownujaca dwa runy."""
    lines: list[str] = []
    lines.append(f"# Porownanie runow\n")
    lines.append(f"- A: `{label_a}`")
    lines.append(f"- B: `{label_b}`\n")

    rows = [
        ("frames", a["n"], b["n"]),
        ("LOCKED frames", a["states"].get("LOCKED", 0), b["states"].get("LOCKED", 0)),
        ("HOLD frames", a["states"].get("HOLD", 0), b["states"].get("HOLD", 0)),
        ("REACQUIRE frames", a["states"].get("REACQUIRE", 0), b["states"].get("REACQUIRE", 0)),
        ("ACQUIRE frames", a["states"].get("ACQUIRE", 0), b["states"].get("ACQUIRE", 0)),
        ("UNLOCKED frames", a["states"].get("UNLOCKED", 0), b["states"].get("UNLOCKED", 0)),
        ("frames z ownerem", a["frames_with_owner"], b["frames_with_owner"]),
        ("unique raw track_id", a["unique_raw_track_id"], b["unique_raw_track_id"]),
        ("unique persistent_owner_id", a["unique_persistent"], b["unique_persistent"]),
        ("synthetic_hold %", round(a["synthetic_hold_pct"], 1), round(b["synthetic_hold_pct"], 1)),
        ("CSRT active %", round(a["csrt_active_pct"], 1), round(b["csrt_active_pct"], 1)),
        ("CSRT update fired %", round(a["csrt_updated_pct"], 1), round(b["csrt_updated_pct"], 1)),
        ("CSRT synthetic used %", round(a["csrt_synthetic_pct"], 1), round(b["csrt_synthetic_pct"], 1)),
        ("avg inference ms", round(a["avg_inference_ms"], 1), round(b["avg_inference_ms"], 1)),
        ("avg tracker ms", round(a["avg_tracker_ms"], 2), round(b["avg_tracker_ms"], 2)),
        ("p50 inference ms", round(a["p50_inference_ms"], 1), round(b["p50_inference_ms"], 1)),
        ("multi_tracks avg / frame", round(a["multi_tracks_avg"], 2), round(b["multi_tracks_avg"], 2)),
        ("lock_loss_events", a["lock_loss_events"], b["lock_loss_events"]),
        ("reacquire success", a["reacquire_successes"], b["reacquire_successes"]),
    ]

    lines.append("| metryka | A | B | delta (B-A) |")
    lines.append("|---|---|---|---|")
    for name, va, vb in rows:
        try:
            delta = vb - va
            delta_s = f"{delta:+}" if isinstance(delta, int) else f"{delta:+.2f}"
        except Exception:
            delta_s = "—"
        lines.append(f"| {name} | {va} | {vb} | {delta_s} |")

    lines.append("")
    lines.append(f"**Top 3 raw track_id A**: {a['top_track_id']}")
    lines.append(f"**Top 3 raw track_id B**: {b['top_track_id']}")
    lines.append(f"**Top 3 persistent A**: {a['top_persistent']}")
    lines.append(f"**Top 3 persistent B**: {b['top_persistent']}")
    return "\n".join(lines)


def per_frame_diff(a: dict, b: dict, max_show: int = 30) -> str:
    """Lista klatek gdzie selected_id sie rozni."""
    spa = a["sel_per_frame"]
    spb = b["sel_per_frame"]
    common = set(spa.keys()) & set(spb.keys())
    diffs = []
    for fr in sorted(common):
        if spa[fr] != spb[fr]:
            diffs.append((fr, spa[fr], spb[fr]))

    n_common = len(common)
    n_diff = len(diffs)
    out = [f"\n## Per-frame selected_id diff"]
    out.append(f"Common frames: {n_common}, mismatches: {n_diff} ({100*n_diff/n_common:.1f}% gdy n_common>0)")
    if n_diff > 0:
        out.append(f"\n| frame | A.selected_id | B.selected_id |")
        out.append(f"|---|---|---|")
        for fr, sa, sb in diffs[:max_show]:
            out.append(f"| {fr} | {sa} | {sb} |")
        if n_diff > max_show:
            out.append(f"| ... | ({n_diff - max_show} more) | |")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    p.add_argument("--per-frame", action="store_true",
                   help="Dodatkowa analiza per-klatka (gdzie selected_id sie rozni)")
    args = p.parse_args()

    a = load_run(args.run_a)
    b = load_run(args.run_b)

    out = render_compare(a, b, args.run_a.name, args.run_b.name)
    print(out)

    if args.per_frame:
        print(per_frame_diff(a, b))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
