from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunReportPaths:
    summary_path: Path
    metrics_path: Path
    timeline_path: Path
    keyframes_path: Path
    manifest_md_path: Path
    manifest_json_path: Path


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


def _md_scalar(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


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
                "lock_state": str(
                    row.get(
                        "lock_state",
                        row.get("narrow_runtime", {}).get("lock_state", ""),
                    )
                ),
                "handoff_reason": str(
                    row.get(
                        "handoff_reject_reason",
                        row.get(
                            "handoff_reason",
                            row.get("handoff_decision", {}).get("reject_reason", ""),
                        ),
                    )
                ),
                "owner_reason": str(row.get("owner_reason", "")),
                "wide_quality": _safe_float(
                    row.get(
                        "wide_owner_quality",
                        row.get(
                            "quality_score",
                            row.get("handoff_decision", {}).get("quality_score"),
                        ),
                    )
                ),
                "geometry_score": _safe_float(
                    row.get(
                        "geometry_score",
                        row.get("handoff_decision", {}).get("geometry_score"),
                    )
                ),
                "center_lock_on": _safe_bool(
                    row.get("center_lock_on", row.get("center_lock"))
                ),
                "edge_active": _safe_bool(
                    row.get(
                        "edge_limit_active",
                        row.get(
                            "edge_active",
                            row.get("narrow_runtime", {}).get("edge_limit_active"),
                        ),
                    )
                ),
                "blind_streak": _safe_int(
                    row.get(
                        "narrow_blind_streak",
                        row.get("handoff_decision", {}).get("narrow_blind_streak"),
                    )
                ),
                "owner_missed": _sanitize_owner_missed(
                    row.get(
                        "owner_missed_frames",
                        row.get(
                            "active_track_missed",
                            row.get("handoff_decision", {}).get("owner_missed_frames"),
                        ),
                    )
                ),
                "boxes": _safe_int(row.get("boxes", row.get("yolo_boxes"))),
                "dets": _safe_int(
                    row.get("dets", row.get("detections", row.get("yolo_dets")))
                ),
                "drop": _safe_int(
                    row.get("drop", row.get("dropped", row.get("yolo_drop")))
                ),
                "manual_lock": _safe_bool(row.get("manual_lock")),
                "active_track_area": _safe_float(row.get("active_track_area")),
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


def _closest_shot(
    frame_idx: int,
    rows_count: int,
    shots: list[dict[str, Any]],
    event_kind: str = "",
) -> str:
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


def _emit_event(
    events: list[dict[str, Any]],
    frame_idx: int,
    ts_s: float,
    kind: str,
    description: str,
) -> None:
    events.append(
        {
            "frame_idx": frame_idx,
            "ts_s": ts_s,
            "kind": kind,
            "description": description,
        }
    )


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
            _emit_event(
                events,
                frame_idx,
                ts_s,
                "drift",
                f"Owner missed count escalated to {cur_missed}",
            )

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


def _compute_metrics(
    cooked: list[dict[str, Any]],
    events: list[dict[str, Any]],
    fps: float,
) -> dict[str, Any]:
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
            "auto_frames": 0,
            "manual_frames": 0,
            "final_wide_owner": None,
            "final_narrow_owner": None,
        }

    frames_total = len(cooked)
    duration_s = cooked[-1]["ts_s"] - cooked[0]["ts_s"]
    if duration_s <= 0 and fps > 0:
        duration_s = frames_total / fps

    owner_missed_values = [r["owner_missed"] for r in cooked if r["owner_missed"] is not None]
    max_owner_missed = max(owner_missed_values) if owner_missed_values else 0
    last = cooked[-1]

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
        "narrow_tracking_ratio": round(
            sum(1 for r in cooked if r["narrow_owner_id"] is not None) / frames_total,
            4,
        ),
        "max_owner_missed": max_owner_missed,
        "max_drop": max(r["drop"] for r in cooked),
        "auto_frames": sum(1 for r in cooked if not r["manual_lock"]),
        "manual_frames": sum(1 for r in cooked if r["manual_lock"]),
        "final_wide_owner": last["wide_owner_id"],
        "final_narrow_owner": last["narrow_owner_id"],
    }


def _write_metrics_csv(metrics_path: Path, metrics: dict[str, Any]) -> None:
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])


def _write_timeline_md(timeline_path: Path, events: list[dict[str, Any]]) -> None:
    lines = [
        "# Timeline",
        "",
        "| Frame | Time [s] | Type | Description |",
        "|---:|---:|---|---|",
    ]

    for e in events:
        lines.append(
            f"| {e['frame_idx']} | {e['ts_s']:.2f} | {e['kind']} | {e['description']} |"
        )

    if not events:
        lines.append("| - | - | info | No events detected |")

    timeline_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_keyframes_md(
    keyframes_path: Path,
    events: list[dict[str, Any]],
    shots: list[dict[str, Any]],
    rows_count: int,
) -> None:
    lines = [
        "# Keyframes",
        "",
        "| Frame | Time [s] | Type | Screenshot | Description |",
        "|---:|---:|---|---|---|",
    ]

    if not events:
        lines.append("| - | - | info | - | No key events detected |")
    else:
        for e in events:
            shot = _closest_shot(e["frame_idx"], rows_count, shots, e["kind"]) or "-"
            lines.append(
                f"| {e['frame_idx']} | {e['ts_s']:.2f} | {e['kind']} | {shot} | {e['description']} |"
            )

    keyframes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _guess_failure_mode(metrics: dict[str, Any], events: list[dict[str, Any]]) -> str:
    if metrics["lock_lost_count"] > 0 and metrics["reacquire_count"] > 0:
        return "reacquire_failure_with_owner_instability"
    if metrics["center_lock_off_count"] >= 3:
        return "edge_geometry_break"
    if metrics["detection_gap_count"] >= 3:
        return "detection_dropout_burst"
    if metrics["wide_owner_switches"] + metrics["narrow_owner_switches"] >= 8:
        return "selected_id_instability"
    return "general_tracking_instability"


def _analysis_priority(events: list[dict[str, Any]]) -> list[str]:
    kinds = {e["kind"] for e in events}
    priority = ["timeline", "keyframes", "metrics", "telemetry"]
    if "lock_lost" in kinds or "reacquire" in kinds:
        return ["timeline", "keyframes", "telemetry", "metrics"]
    return priority


def _preferred_windows(events: list[dict[str, Any]]) -> list[list[int]]:
    priority_order = [
        "lock_lost",
        "reacquire",
        "center_lock_off",
        "drift",
        "owner_switch",
        "narrow_switch",
        "detection_gap",
        "detection_drop",
    ]
    picked: list[list[int]] = []
    seen_ranges: set[tuple[int, int]] = set()

    for kind in priority_order:
        for e in events:
            if e["kind"] != kind:
                continue

            start = max(0, int(e["frame_idx"]) - 12)
            end = int(e["frame_idx"]) + 12

            merged = False
            for idx, existing in enumerate(picked):
                if not (end < existing[0] - 6 or start > existing[1] + 6):
                    picked[idx] = [min(start, existing[0]), max(end, existing[1])]
                    merged = True
                    break

            if not merged:
                key = (start, end)
                if key not in seen_ranges:
                    picked.append([start, end])
                    seen_ranges.add(key)

            if len(picked) >= 3:
                break

        if len(picked) >= 3:
            break

    picked.sort(key=lambda x: x[0])
    return picked


def _git_text(repo_root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _git_meta(repo_root: Path) -> dict[str, Any]:
    branch = _git_text(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    commit = _git_text(repo_root, ["rev-parse", "--short", "HEAD"]) or "unknown"
    porcelain = _git_text(repo_root, ["status", "--porcelain"])
    dirty = bool(porcelain.strip()) if porcelain is not None else False
    return {
        "branch": branch,
        "commit": commit,
        "dirty_worktree": dirty,
    }


def _final_state(cooked: list[dict[str, Any]]) -> dict[str, Any]:
    if not cooked:
        return {
            "wide_owner": None,
            "narrow_owner": None,
            "tracking_state": "UNKNOWN",
            "center_lock": False,
            "lock_active": False,
        }

    last = cooked[-1]
    narrow_owner = last["narrow_owner_id"]
    lock_state = last["lock_state"] or "UNKNOWN"

    return {
        "wide_owner": last["wide_owner_id"],
        "narrow_owner": narrow_owner,
        "tracking_state": lock_state,
        "center_lock": bool(last["center_lock_on"]),
        "lock_active": bool(
            narrow_owner is not None
            and lock_state in {"TRACKING", "CENTER_LOCK", "LOCKED"}
        ),
    }


def _artifact_status(
    output_dir: Path,
    telemetry_path: Path,
    shot_dir: Path | None,
    video_dir: Path | None,
) -> dict[str, bool]:
    return {
        "run_summary": (output_dir / "run_summary.md").exists(),
        "timeline": (output_dir / "timeline.md").exists(),
        "keyframes": (output_dir / "keyframes.md").exists(),
        "metrics": (output_dir / "metrics.csv").exists(),
        "telemetry": telemetry_path.exists(),
        "images_dir": shot_dir.exists() if shot_dir is not None else False,
        "video_dir": video_dir.exists() if video_dir is not None else False,
    }


def _symptoms(
    metrics: dict[str, Any],
    events: list[dict[str, Any]],
    final_state: dict[str, Any],
) -> list[str]:
    symptoms: list[str] = []

    if metrics["narrow_owner_switches"] >= 2:
        symptoms.append("frequent_narrow_switches")
    if metrics["wide_owner_switches"] >= 3:
        symptoms.append("wide_owner_instability")
    if metrics["short_drop_count"] > 0 or metrics["detection_gap_count"] > 0:
        symptoms.append("repeated_short_detection_drops")
    if metrics["lock_lost_count"] > 0:
        symptoms.append("lock_loss_after_owner_instability")
    if metrics["reacquire_count"] > 0:
        symptoms.append("reacquire_entry_after_drift")
    if metrics["center_lock_off_count"] > 0:
        symptoms.append("center_lock_breaks")
    if final_state["narrow_owner"] is None:
        symptoms.append("run_ended_without_narrow_owner")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in symptoms:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped


def _recommended_first_action(
    metrics: dict[str, Any],
    events: list[dict[str, Any]],
    final_state: dict[str, Any],
) -> dict[str, str]:
    if metrics["lock_lost_count"] > 0 or metrics["reacquire_count"] > 0:
        return {
            "type": "inspect_and_patch",
            "module": "src/core/narrow_tracker.py",
            "change_kind": "reacquire_gate_tightening",
            "reason": "narrow owner instability around lock_lost and reacquire transitions",
        }
    if metrics["center_lock_off_count"] >= 2:
        return {
            "type": "inspect_and_patch",
            "module": "src/core/target_manager.py",
            "change_kind": "edge_geometry_gate_tightening",
            "reason": "center lock breaks suggest edge and geometry instability in owner selection",
        }
    if metrics["wide_owner_switches"] >= 3:
        return {
            "type": "inspect_and_patch",
            "module": "src/core/target_manager.py",
            "change_kind": "owner_selection_stabilization",
            "reason": "wide owner selection is unstable and likely drives downstream narrow churn",
        }
    return {
        "type": "inspect",
        "module": "src/core/app.py",
        "change_kind": "integration_trace_validation",
        "reason": "start from the integration path and validate event generation against telemetry",
    }


def _run_classification(metrics: dict[str, Any], final_state: dict[str, Any]) -> str:
    if not final_state.get("lock_active", False):
        if metrics.get("lock_lost_count", 0) > 0 or metrics.get("reacquire_count", 0) > 0:
            return "failed_end_state"
        return "unstable"

    if metrics.get("wide_owner_switches", 0) + metrics.get("narrow_owner_switches", 0) >= 8:
        return "unstable"

    if metrics.get("detection_gap_count", 0) >= 3:
        return "degraded"

    return "stable"


def _links(output_dir: Path, git_meta: dict[str, Any]) -> dict[str, str]:
    repo_slug = "pwysocka26-ai/drone-tracker-system"
    commit = git_meta.get("commit", "unknown")
    return {
        "github_repo": f"https://github.com/{repo_slug}",
        "commit_url": (
            f"https://github.com/{repo_slug}/commit/{commit}"
            if commit and commit != "unknown"
            else ""
        ),
        "run_folder": str(output_dir),
    }


def _write_run_summary_md(
    summary_path: Path,
    cooked: list[dict[str, Any]],
    metrics: dict[str, Any],
    events: list[dict[str, Any]],
) -> None:
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

    important_kinds = {
        "owner_switch",
        "lock_lost",
        "reacquire",
        "center_lock_off",
        "drift",
        "detection_gap",
    }
    highlighted = [e for e in events if e["kind"] in important_kinds][:8]
    for e in highlighted:
        lines.append(
            f"At frame {e['frame_idx']} ({e['ts_s']:.2f}s) the system recorded {e['kind']}: {e['description']}."
        )

    while len([x for x in lines if x and not x.startswith("#")]) < 12:
        lines.append(
            "Telemetry stayed broadly consistent with the current wide/narrow handoff logic during frames without major events."
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest_md(
    path: Path,
    run_id: str,
    metrics: dict[str, Any],
    events: list[dict[str, Any]],
    output_dir: Path,
    telemetry_path: Path,
    shot_dir: Path | None,
    video_dir: Path | None,
    git_meta: dict[str, Any],
    final_state: dict[str, Any],
    artifacts_status: dict[str, bool],
    symptoms: list[str],
    recommended_first_action: dict[str, str],
) -> None:
    manifest_version = "1.0"
    failure_mode = _guess_failure_mode(metrics, events)
    run_classification = _run_classification(metrics, final_state)
    links = _links(output_dir, git_meta)

    verdict = (
        "This run looks unstable. The main suspected issue is narrow owner instability under repeated detection gaps and edge-triggered geometry breaks."
        if failure_mode != "general_tracking_instability"
        else "This run looks somewhat unstable. The main suspected issue is general owner and handoff instability."
    )

    key_events = events[:8]
    windows = _preferred_windows(events)

    lines = [
        "# Latest Run Manifest",
        "",
        "## Run identity",
        f"- manifest_version: {manifest_version}",
        f"- run_id: {run_id}",
        f"- created_at: {run_id}",
        "- project_repo: pwysocka26-ai/drone-tracker-system",
        f"- branch: {_md_scalar(git_meta['branch'])}",
        f"- commit: {_md_scalar(git_meta['commit'])}",
        f"- dirty_worktree: {_md_scalar(git_meta['dirty_worktree'])}",
        "- scenario: video tracking run",
        "- source: local telemetry export",
        "",
        "## Quick verdict",
        verdict,
        "",
        "## Run classification",
        f"- run_classification: {run_classification}",
        "",
        "## Key metrics",
        f"- frames: {metrics['frames_total']}",
        f"- duration_s: {metrics['duration_s']}",
        f"- wide_owner_switches: {metrics['wide_owner_switches']}",
        f"- narrow_owner_switches: {metrics['narrow_owner_switches']}",
        f"- lock_losses: {metrics['lock_lost_count']}",
        f"- reacquire_phases: {metrics['reacquire_count']}",
        f"- detection_gap_count: {metrics['detection_gap_count']}",
        f"- short_drop_count: {metrics['short_drop_count']}",
        f"- avg_wide_owner_quality: {metrics['avg_wide_quality']}",
        f"- avg_geometry_score: {metrics['avg_geometry_score']}",
        f"- final_wide_owner: {_md_scalar(metrics['final_wide_owner'])}",
        f"- final_narrow_owner: {_md_scalar(metrics['final_narrow_owner'])}",
        "",
        "## Final state",
        f"- wide_owner: {_md_scalar(final_state['wide_owner'])}",
        f"- narrow_owner: {_md_scalar(final_state['narrow_owner'])}",
        f"- tracking_state: {_md_scalar(final_state['tracking_state'])}",
        f"- center_lock: {'ON' if final_state['center_lock'] else 'OFF'}",
        f"- lock_active: {_md_scalar(final_state['lock_active'])}",
        "",
        "## Key events",
    ]

    for e in key_events:
        lines.append(f"- frame {e['frame_idx']}: {e['kind']} — {e['description']}")

    lines += [
        "",
        "## Symptoms",
    ]
    if symptoms:
        for item in symptoms:
            lines.append(f"- {item}")
    else:
        lines.append("- none")

    lines += [
        "",
        "## Suspected failure mode",
        failure_mode,
        "",
        "## Analysis priority",
    ]
    for item in _analysis_priority(events):
        lines.append(f"- {item}")

    lines += [
        "",
        "## Recommended first action",
        f"- type: {recommended_first_action['type']}",
        f"- module: {recommended_first_action['module']}",
        f"- change_kind: {recommended_first_action['change_kind']}",
        f"- reason: {recommended_first_action['reason']}",
        "",
        "## Quick links",
        f"- github_repo: {links['github_repo']}",
        f"- commit_url: {links['commit_url'] or 'none'}",
        f"- run_folder: {links['run_folder']}",
        "",
        "## Artifact paths",
        f"- run_summary: {output_dir / 'run_summary.md'}",
        f"- timeline: {output_dir / 'timeline.md'}",
        f"- keyframes: {output_dir / 'keyframes.md'}",
        f"- metrics: {output_dir / 'metrics.csv'}",
        f"- telemetry: {telemetry_path}",
        f"- images_dir: {shot_dir if shot_dir is not None else output_dir / 'images'}",
        f"- video_dir: {video_dir if video_dir is not None else output_dir / 'video'}",
        "",
        "## Artifact status",
        f"- run_summary: {_md_scalar(artifacts_status['run_summary'])}",
        f"- timeline: {_md_scalar(artifacts_status['timeline'])}",
        f"- keyframes: {_md_scalar(artifacts_status['keyframes'])}",
        f"- metrics: {_md_scalar(artifacts_status['metrics'])}",
        f"- telemetry: {_md_scalar(artifacts_status['telemetry'])}",
        f"- images_dir: {_md_scalar(artifacts_status['images_dir'])}",
        f"- video_dir: {_md_scalar(artifacts_status['video_dir'])}",
        "",
        "## Recommended analysis windows",
    ]

    if windows:
        for w in windows:
            lines.append(f"- frames {w[0]}-{w[1]}")
    else:
        lines.append("- full run")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest_json(
    path: Path,
    run_id: str,
    metrics: dict[str, Any],
    events: list[dict[str, Any]],
    output_dir: Path,
    telemetry_path: Path,
    shot_dir: Path | None,
    video_dir: Path | None,
    git_meta: dict[str, Any],
    final_state: dict[str, Any],
    artifacts_status: dict[str, bool],
    symptoms: list[str],
    recommended_first_action: dict[str, str],
) -> None:
    manifest_version = "1.0"
    failure_mode = _guess_failure_mode(metrics, events)
    run_classification = _run_classification(metrics, final_state)
    links = _links(output_dir, git_meta)

    payload = {
        "manifest_version": manifest_version,
        "run_classification": run_classification,
        "run": {
            "run_id": run_id,
            "created_at": run_id,
            "repo": "pwysocka26-ai/drone-tracker-system",
            "branch": git_meta["branch"],
            "commit": git_meta["commit"],
            "dirty_worktree": git_meta["dirty_worktree"],
            "scenario": "video tracking run",
            "source": "local telemetry export",
        },
        "summary": {
            "quick_verdict": (
                "Run unstable. Main suspected issue is narrow owner instability under repeated detection gaps and edge-triggered geometry breaks."
                if failure_mode != "general_tracking_instability"
                else "Run somewhat unstable. Main suspected issue is general owner and handoff instability."
            ),
            "suspected_failure_mode": failure_mode,
            "analysis_priority": _analysis_priority(events),
        },
        "metrics": {
            "frames": metrics["frames_total"],
            "duration_s": metrics["duration_s"],
            "wide_owner_switches": metrics["wide_owner_switches"],
            "narrow_owner_switches": metrics["narrow_owner_switches"],
            "lock_losses": metrics["lock_lost_count"],
            "reacquire_phases": metrics["reacquire_count"],
            "detection_gap_count": metrics["detection_gap_count"],
            "short_drop_count": metrics["short_drop_count"],
            "avg_wide_owner_quality": metrics["avg_wide_quality"],
            "avg_geometry_score": metrics["avg_geometry_score"],
            "max_owner_missed": metrics["max_owner_missed"],
            "max_drop": metrics["max_drop"],
            "auto_frames": metrics["auto_frames"],
            "manual_frames": metrics["manual_frames"],
            "final_wide_owner": metrics["final_wide_owner"],
            "final_narrow_owner": metrics["final_narrow_owner"],
        },
        "final_state": final_state,
        "events": [
            {
                "frame": e["frame_idx"],
                "time_s": round(e["ts_s"], 2),
                "type": e["kind"],
                "description": e["description"],
            }
            for e in events[:12]
        ],
        "links": links,
        "artifacts": {
            "run_summary": str(output_dir / "run_summary.md"),
            "timeline": str(output_dir / "timeline.md"),
            "keyframes": str(output_dir / "keyframes.md"),
            "metrics": str(output_dir / "metrics.csv"),
            "telemetry": str(telemetry_path),
            "images_dir": str(shot_dir if shot_dir is not None else output_dir / "images"),
            "video_dir": str(video_dir if video_dir is not None else output_dir / "video"),
        },
        "artifacts_status": artifacts_status,
        "baseline": {
            "baseline_run_id": None,
            "baseline_manifest": None,
            "compare_focus": [
                "wide_owner_switches",
                "narrow_owner_switches",
                "lock_losses",
                "reacquire_phases",
            ],
        },
        "symptoms": symptoms,
        "recommended_first_action": recommended_first_action,
        "agent_hints": {
            "primary_modules_to_check": [
                "src/core/target_manager.py",
                "src/core/narrow_tracker.py",
                "src/core/app.py",
            ],
            "primary_recipes_to_try": [
                "selected_id_instability",
                "edge_geometry_break",
                "reacquire_failure",
            ],
            "preferred_analysis_window_frames": _preferred_windows(events),
            "patch_strategy": "minimal_safe_patch",
            "regression_focus": [
                "narrow_owner_switches",
                "lock_losses",
                "final_narrow_owner",
            ],
        },
    }

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _publish_latest(manifest_md: Path, manifest_json: Path, latest_dir: Path) -> None:
    latest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_md, latest_dir / "latest_run_manifest.md")
    shutil.copy2(manifest_json, latest_dir / "latest_run_manifest.json")


def generate_run_reports(
    telemetry_path: str | Path,
    shot_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    fps: float = 30.0,
    run_id: str | None = None,
    latest_dir: str | Path | None = "artifacts/latest",
    video_dir: str | Path | None = None,
) -> RunReportPaths:
    telemetry_path = Path(telemetry_path)
    output_dir = Path(output_dir) if output_dir is not None else telemetry_path.parent
    shot_dir_path = Path(shot_dir) if shot_dir is not None else None
    video_dir_path = Path(video_dir) if video_dir is not None else None

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "run_summary.md"
    metrics_path = output_dir / "metrics.csv"
    timeline_path = output_dir / "timeline.md"
    keyframes_path = output_dir / "keyframes.md"
    manifest_md_path = output_dir / "latest_run_manifest.md"
    manifest_json_path = output_dir / "latest_run_manifest.json"

    raw_rows = _load_jsonl(telemetry_path)
    cooked = _build_rows(raw_rows, fps=fps)
    events = _build_timeline(cooked)
    metrics = _compute_metrics(cooked, events, fps=fps)
    shots = _find_shots(shot_dir_path)

    _write_run_summary_md(summary_path, cooked, metrics, events)
    _write_metrics_csv(metrics_path, metrics)
    _write_timeline_md(timeline_path, events)
    _write_keyframes_md(keyframes_path, events, shots, len(cooked))

    resolved_run_id = run_id or output_dir.name
    repo_root = Path(__file__).resolve().parents[2]
    git_meta = _git_meta(repo_root)
    final_state = _final_state(cooked)
    artifacts_status = _artifact_status(
        output_dir,
        telemetry_path,
        shot_dir_path,
        video_dir_path,
    )
    symptoms = _symptoms(metrics, events, final_state)
    recommended_first_action = _recommended_first_action(metrics, events, final_state)

    _write_manifest_md(
        manifest_md_path,
        resolved_run_id,
        metrics,
        events,
        output_dir,
        telemetry_path,
        shot_dir_path,
        video_dir_path,
        git_meta,
        final_state,
        artifacts_status,
        symptoms,
        recommended_first_action,
    )
    _write_manifest_json(
        manifest_json_path,
        resolved_run_id,
        metrics,
        events,
        output_dir,
        telemetry_path,
        shot_dir_path,
        video_dir_path,
        git_meta,
        final_state,
        artifacts_status,
        symptoms,
        recommended_first_action,
    )

    if latest_dir is not None:
        _publish_latest(manifest_md_path, manifest_json_path, Path(latest_dir))

    return RunReportPaths(
        summary_path=summary_path,
        metrics_path=metrics_path,
        timeline_path=timeline_path,
        keyframes_path=keyframes_path,
        manifest_md_path=manifest_md_path,
        manifest_json_path=manifest_json_path,
    )