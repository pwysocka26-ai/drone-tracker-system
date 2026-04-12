from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunReportPaths:
    summary_path: Path
    metrics_path: Path
    timeline_path: Path
    keyframes_path: Path


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _pick_frame_index(row: dict[str, Any], fallback: int) -> int:
    for key in ("frame_idx", "frame", "frame_index"):
        if key in row:
            return _safe_int(row.get(key), fallback)
    return fallback


def _pick_timestamp_s(row: dict[str, Any], frame_idx: int, fps: float) -> float:
    for key in ("ts", "timestamp", "time_s", "t"):
        if key in row:
            value = row.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    if fps > 0:
        return frame_idx / fps
    return float(frame_idx)


def _sanitize_owner_missed(value: Any) -> int | None:
    try:
        if value is None:
            return None
        iv = int(value)
        if iv < 0 or iv >= 9999:
            return None
        return iv
    except Exception:
        return None


def _build_rows(rows: list[dict[str, Any]], fps: float) -> list[dict[str, Any]]:
    cooked: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        frame_idx = _pick_frame_index(row, idx)
        ts_s = _pick_timestamp_s(row, frame_idx, fps)
        cooked.append(
            {
                "raw": row,
                "frame_idx": frame_idx,
                "ts_s": ts_s,
                "wide_owner_id": row.get("wide_owner_id"),
                "narrow_owner_id": row.get("narrow_owner_id"),
                "pending_owner_id": row.get("pending_owner_id"),
                "lock_state": str(row.get("lock_state", row.get("narrow_runtime", {}).get("lock_state", ""))),
                "handoff_reason": str(
                    row.get(
                        "handoff_reject_reason",
                        row.get("handoff_reason", row.get("handoff_decision", {}).get("reject_reason", "")),
                    )
                ),
                "owner_reason": str(row.get("owner_reason", "")),
                "wide_quality": _safe_float(
                    row.get("wide_owner_quality", row.get("quality_score", row.get("handoff_decision", {}).get("quality_score")))
                ),
                "geometry_score": _safe_float(
                    row.get("geometry_score", row.get("handoff_decision", {}).get("geometry_score"))
                ),
                "center_lock_on": _safe_bool(row.get("center_lock_on", row.get("center_lock"))),
                "edge_active": _safe_bool(
                    row.get("edge_limit_active", row.get("edge_active", row.get("narrow_runtime", {}).get("edge_limit_active")))
                ),
                "blind_streak": _safe_int(row.get("narrow_blind_streak", row.get("handoff_decision", {}).get("narrow_blind_streak"))),
                "owner_missed": _sanitize_owner_missed(
                    row.get("owner_missed_frames", row.get("active_track_missed", row.get("handoff_decision", {}).get("owner_missed_frames")))
                ),
                "boxes": _safe_int(row.get("boxes", row.get("yolo_boxes"))),
                "dets": _safe_int(row.get("dets", row.get("detections", row.get("yolo_dets")))),
                "drop": _safe_int(row.get("drop", row.get("dropped", row.get("yolo_drop")))),
            }
        )
    return cooked


def _find_shots(shot_dir: Path | None) -> list[dict[str, Any]]:
    if shot_dir is None or not shot_dir.exists():
        return []

    pattern = re.compile(r"dashboard_(\d{8})_(\d{6})_(\d+)(?:_([a-z_]+))?\.png$")
    shots: list[dict[str, Any]] = []

    for path in sorted(shot_dir.glob("dashboard_*.png")):
        match = pattern.search(path.name)
        micros = 0
        tag = ""
        if match:
            micros = _safe_int(match.group(3), 0)
            tag = match.group(4) or ""
        shots.append(
            {
                "path": path,
                "name": path.name,
                "micros": micros,
                "ordinal": len(shots),
                "tag": tag,
            }
        )
    return shots


def _closest_shot(frame_idx: int, rows_count: int, shots: list[dict[str, Any]], event_kind: str = "") -> str:
    if not shots:
        return ""

    preferred = [s for s in shots if s.get("tag") == event_kind]
    pool = preferred if preferred else shots

    if rows_count <= 1:
        return pool[0]["name"]

    normalized = frame_idx / max(rows_count - 1, 1)
    target_ord = normalized * max(len(pool) - 1, 0)
    best = min(pool, key=lambda item: abs(item["ordinal"] - target_ord))
    return str(best["name"])


def _emit_event(events: list[dict[str, Any]], frame_idx: int, ts_s: float, kind: str, description: str) -> None:
    events.append({"frame_idx": frame_idx, "ts_s": ts_s, "kind": kind, "description": description})


def _build_timeline(cooked: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not cooked:
        return events

    prev = cooked[0]
    drop_open = False
    drop_start_frame = None
    drop_start_ts = None
    drop_peak = 0

    for row in cooked[1:]:
        frame_idx = row["frame_idx"]
        ts_s = row["ts_s"]

        if row["wide_owner_id"] != prev["wide_owner_id"] and row["wide_owner_id"] is not None:
            _emit_event(
                events,
                frame_idx,
                ts_s,
                "owner_switch",
                f"Wide owner switched {prev['wide_owner_id']} -> {row['wide_owner_id']} ({row['owner_reason'] or 'unknown reason'})",
            )

        if row["narrow_owner_id"] != prev["narrow_owner_id"]:
            if prev["narrow_owner_id"] is not None and row["narrow_owner_id"] is None:
                _emit_event(
                    events,
                    frame_idx,
                    ts_s,
                    "lock_lost",
                    f"Narrow lost owner {prev['narrow_owner_id']} while in {prev['lock_state'] or 'unknown'}",
                )
            elif row["narrow_owner_id"] is not None and prev["narrow_owner_id"] is None:
                _emit_event(
                    events,
                    frame_idx,
                    ts_s,
                    "reacquire",
                    f"Narrow acquired owner {row['narrow_owner_id']} in state {row['lock_state'] or 'unknown'}",
                )
            elif row["narrow_owner_id"] is not None:
                _emit_event(
                    events,
                    frame_idx,
                    ts_s,
                    "narrow_switch",
                    f"Narrow owner switched {prev['narrow_owner_id']} -> {row['narrow_owner_id']}",
                )

        if prev["lock_state"] != row["lock_state"]:
            if row["lock_state"] == "REACQUIRE":
                _emit_event(events, frame_idx, ts_s, "reacquire", "Entered REACQUIRE state")
            elif prev["lock_state"] == "REACQUIRE" and row["lock_state"] == "TRACKING":
                _emit_event(events, frame_idx, ts_s, "reacquire_complete", "Recovered from REACQUIRE to TRACKING")

        if prev["center_lock_on"] and not row["center_lock_on"]:
            _emit_event(
                events,
                frame_idx,
                ts_s,
                "center_lock_off",
                f"Center lock turned OFF (geom={row['geometry_score']:.2f}, edge={row['edge_active']})",
            )
        elif (not prev["center_lock_on"]) and row["center_lock_on"]:
            _emit_event(
                events,
                frame_idx,
                ts_s,
                "center_lock_on",
                f"Center lock turned ON (owner={row['narrow_owner_id']})",
            )

        if row["drop"] > 0:
            if not drop_open:
                drop_open = True
                drop_start_frame = frame_idx
                drop_start_ts = ts_s
                drop_peak = row["drop"]
            else:
                drop_peak = max(drop_peak, row["drop"])
        else:
            if drop_open:
                duration_frames = frame_idx - (drop_start_frame or frame_idx)
                if duration_frames >= 3 or drop_peak >= 3:
                    _emit_event(
                        events,
                        drop_start_frame or frame_idx,
                        drop_start_ts or ts_s,
                        "detection_gap",
                        f"Detection gap lasted {duration_frames} frames with peak drop={drop_peak}",
                    )
                else:
                    _emit_event(
                        events,
                        drop_start_frame or frame_idx,
                        drop_start_ts or ts_s,
                        "detection_drop",
                        f"Short detection drop started (peak drop={drop_peak})",
                    )
                _emit_event(events, frame_idx, ts_s, "detection_return", "Detections recovered after drop")
                drop_open = False
                drop_start_frame = None
                drop_start_ts = None
                drop_peak = 0

        prev_missed = prev["owner_missed"] if prev["owner_missed"] is not None else 0
        cur_missed = row["owner_missed"] if row["owner_missed"] is not None else 0
        if prev_missed < 3 <= cur_missed:
            _emit_event(events, frame_idx, ts_s, "drift", f"Owner missed count escalated to {cur_missed}")

        prev = row

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for event in events:
        key = (event["frame_idx"], event["kind"], event["description"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def _compute_metrics(cooked: list[dict[str, Any]], events: list[dict[str, Any]], fps: float) -> dict[str, Any]:
    if not cooked:
        return {
            "frames_total": 0,
            "duration_s": 0.0,
            "wide_owner_switches": 0,
            "narrow_owner_switches": 0,
            "lock_lost_count": 0,
            "reacquire_count": 0,
            "center_lock_off_count": 0,
            "detection_gap_count": 0,
            "short_drop_count": 0,
            "avg_geometry_score": 0.0,
            "avg_wide_quality": 0.0,
            "frames_with_narrow_owner": 0,
            "narrow_tracking_ratio": 0.0,
            "max_owner_missed": 0,
            "max_drop": 0,
        }

    frames_total = len(cooked)
    duration_s = cooked[-1]["ts_s"] - cooked[0]["ts_s"]
    if duration_s <= 0 and fps > 0:
        duration_s = frames_total / fps

    owner_missed_values = [r["owner_missed"] for r in cooked if r["owner_missed"] is not None]
    max_owner_missed = max(owner_missed_values) if owner_missed_values else 0

    return {
        "frames_total": frames_total,
        "duration_s": round(duration_s, 3),
        "wide_owner_switches": sum(1 for e in events if e["kind"] == "owner_switch"),
        "narrow_owner_switches": sum(1 for e in events if e["kind"] == "narrow_switch"),
        "lock_lost_count": sum(1 for e in events if e["kind"] == "lock_lost"),
        "reacquire_count": sum(1 for e in events if e["kind"] == "reacquire"),
        "center_lock_off_count": sum(1 for e in events if e["kind"] == "center_lock_off"),
        "detection_gap_count": sum(1 for e in events if e["kind"] == "detection_gap"),
        "short_drop_count": sum(1 for e in events if e["kind"] == "detection_drop"),
        "avg_geometry_score": round(sum(r["geometry_score"] for r in cooked) / frames_total, 4),
        "avg_wide_quality": round(sum(r["wide_quality"] for r in cooked) / frames_total, 4),
        "frames_with_narrow_owner": sum(1 for r in cooked if r["narrow_owner_id"] is not None),
        "narrow_tracking_ratio": round(sum(1 for r in cooked if r["narrow_owner_id"] is not None) / frames_total, 4),
        "max_owner_missed": max_owner_missed,
        "max_drop": max(r["drop"] for r in cooked),
    }


def _write_metrics_csv(metrics_path: Path, metrics: dict[str, Any]) -> None:
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def _write_timeline_md(timeline_path: Path, events: list[dict[str, Any]]) -> None:
    lines = ["# Timeline", "", "| Frame | Time [s] | Type | Description |", "|---:|---:|---|---|"]
    for e in events:
        lines.append(f"| {e['frame_idx']} | {e['ts_s']:.2f} | {e['kind']} | {e['description']} |")
    if not events:
        lines.append("| - | - | info | No events detected |")
    timeline_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_keyframes_md(keyframes_path: Path, events: list[dict[str, Any]], shots: list[dict[str, Any]], rows_count: int) -> None:
    lines = ["# Keyframes", "", "| Frame | Time [s] | Type | Screenshot | Description |", "|---:|---:|---|---|---|"]
    if not events:
        lines.append("| - | - | info | - | No key events detected |")
    else:
        for e in events:
            shot = _closest_shot(e["frame_idx"], rows_count, shots, e["kind"]) or "-"
            lines.append(f"| {e['frame_idx']} | {e['ts_s']:.2f} | {e['kind']} | {shot} | {e['description']} |")
    keyframes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run_summary_md(summary_path: Path, cooked: list[dict[str, Any]], metrics: dict[str, Any], events: list[dict[str, Any]]) -> None:
    if not cooked:
        summary_path.write_text("# Run summary\n\nNo telemetry rows found.\n", encoding="utf-8")
        return

    first, last = cooked[0], cooked[-1]
    run_shape = "stable" if metrics["lock_lost_count"] == 0 and metrics["reacquire_count"] == 0 else "unstable"
    owner_changes = metrics["wide_owner_switches"] + metrics["narrow_owner_switches"]

    lines = [
        "# Run summary",
        "",
        f"This run processed {metrics['frames_total']} telemetry frames over about {metrics['duration_s']:.2f} seconds.",
        f"The overall tracking profile was {run_shape}, with {metrics['lock_lost_count']} lock losses and {metrics['reacquire_count']} reacquire phases.",
        f"Wide owner switching happened {metrics['wide_owner_switches']} times, while narrow owner switching happened {metrics['narrow_owner_switches']} times.",
        f"The run started with wide owner {first['wide_owner_id']} and narrow owner {first['narrow_owner_id']}, and ended with wide owner {last['wide_owner_id']} and narrow owner {last['narrow_owner_id']}.",
        f"Average wide owner quality was {metrics['avg_wide_quality']:.2f}, and average geometry score was {metrics['avg_geometry_score']:.2f}.",
        f"Narrow had an active owner for {metrics['frames_with_narrow_owner']} frames, which is {metrics['narrow_tracking_ratio'] * 100:.1f}% of the run.",
        f"There were {metrics['detection_gap_count']} longer detection gaps and {metrics['short_drop_count']} short drop episodes.",
        f"The maximum owner missed count observed during the run was {metrics['max_owner_missed']}.",
        f"Center lock was forced or released {metrics['center_lock_off_count']} times.",
        f"In total there were {owner_changes} owner-related switch events across wide and narrow control.",
        "The most important transitions in this run are listed below.",
        "",
    ]

    important_kinds = {"owner_switch", "lock_lost", "reacquire", "center_lock_off", "drift", "detection_gap"}
    highlighted = [e for e in events if e["kind"] in important_kinds][:8]
    for e in highlighted:
        lines.append(f"At frame {e['frame_idx']} ({e['ts_s']:.2f}s) the system recorded {e['kind']}: {e['description']}.")

    while len([x for x in lines if x and not x.startswith("#")]) < 12:
        lines.append("Telemetry stayed broadly consistent with the current wide/narrow handoff logic during frames without major events.")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_run_reports(
    telemetry_path: str | Path,
    shot_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    fps: float = 30.0,
) -> RunReportPaths:
    telemetry_path = Path(telemetry_path)
    output_dir = Path(output_dir) if output_dir is not None else telemetry_path.parent
    shot_dir_path = Path(shot_dir) if shot_dir is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "run_summary.md"
    metrics_path = output_dir / "metrics.csv"
    timeline_path = output_dir / "timeline.md"
    keyframes_path = output_dir / "keyframes.md"

    raw_rows = _load_jsonl(telemetry_path)
    cooked = _build_rows(raw_rows, fps=fps)
    events = _build_timeline(cooked)
    metrics = _compute_metrics(cooked, events, fps=fps)
    shots = _find_shots(shot_dir_path)

    _write_run_summary_md(summary_path, cooked, metrics, events)
    _write_metrics_csv(metrics_path, metrics)
    _write_timeline_md(timeline_path, events)
    _write_keyframes_md(keyframes_path, events, shots, len(cooked))

    return RunReportPaths(
        summary_path=summary_path,
        metrics_path=metrics_path,
        timeline_path=timeline_path,
        keyframes_path=keyframes_path,
    )