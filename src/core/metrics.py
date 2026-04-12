from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MetricsConfig:
    fps: float = 30.0
    stable_lock_frames: int = 5
    bad_aspect_low: float = 0.22
    bad_aspect_high: float = 4.5
    min_lock_area: float = 30.0
    ghost_motion_px: float = 12.0
    ghost_min_frames: int = 6


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_track(frame: Dict[str, Any], track_id: Any) -> Optional[Dict[str, Any]]:
    if track_id is None:
        return None
    for tr in frame.get("tracks", []):
        if tr.get("track_id") == track_id:
            return tr
    return None


def compute_metrics(rows: List[Dict[str, Any]], cfg: MetricsConfig | None = None) -> Dict[str, Any]:
    cfg = cfg or MetricsConfig()
    fps = cfg.fps

    ttft_frame = None
    for r in rows:
        if int(r.get("multi_tracks", 0)) > 0:
            ttft_frame = int(r["frame_idx"])
            break

    ttfl_frame = None
    streak = 0
    for r in rows:
        ok = (r.get("selected_id") is not None) and (r.get("active_track_id") is not None)
        streak = streak + 1 if ok else 0
        if streak >= cfg.stable_lock_frames:
            ttfl_frame = int(r["frame_idx"]) - cfg.stable_lock_frames + 1
            break

    auto_rows = [r for r in rows if str(r.get("mode", "AUTO")).upper() == "AUTO"]
    duration_s = max((rows[-1]["time_s"] - rows[0]["time_s"]) if len(rows) >= 2 else 0.0, 1e-9)

    unique_track_ids = sorted(
        {tr.get("track_id") for r in auto_rows for tr in r.get("tracks", []) if tr.get("track_id") is not None}
    )
    id_churn_per_min = len(unique_track_ids) / max(duration_s / 60.0, 1e-9)

    switches = 0
    prev_selected = None
    prev_valid = False
    for r in auto_rows:
        cur = r.get("selected_id")
        valid = cur is not None and r.get("active_track_id") is not None
        if prev_valid and valid and cur != prev_selected:
            switches += 1
        if valid:
            prev_selected = cur
        prev_valid = valid

    false_positive_locks = 0
    all_locks = 0
    for r in rows:
        active_id = r.get("active_track_id")
        if active_id is None:
            continue
        all_locks += 1
        tr = _find_track(r, active_id)
        if tr is None:
            continue
        area = float(tr.get("area", 0.0) or 0.0)
        aspect = float(tr.get("aspect", 1.0) or 1.0)
        if area < cfg.min_lock_area or aspect < cfg.bad_aspect_low or aspect > cfg.bad_aspect_high:
            false_positive_locks += 1

    ghost_episodes = 0
    ghost_streak = 0
    prev_center = None
    for r in rows:
        center = r.get("narrow_center")
        active_id = r.get("active_track_id")
        moving = False
        if center is not None and prev_center is not None:
            dx = float(center[0]) - float(prev_center[0])
            dy = float(center[1]) - float(prev_center[1])
            moving = (dx * dx + dy * dy) ** 0.5 >= cfg.ghost_motion_px
        prev_center = center if center is not None else prev_center

        bad = active_id is None and moving
        if bad:
            ghost_streak += 1
            if ghost_streak == cfg.ghost_min_frames:
                ghost_episodes += 1
        else:
            ghost_streak = 0

    return {
        "frames": len(rows),
        "duration_s": duration_s,
        "ttft_frames": ttft_frame,
        "ttft_s": (ttft_frame / fps) if ttft_frame is not None else None,
        "ttfl_frames": ttfl_frame,
        "ttfl_s": (ttfl_frame / fps) if ttfl_frame is not None else None,
        "unique_track_ids_auto": unique_track_ids,
        "id_churn_per_min": id_churn_per_min,
        "id_switch_rate_auto": switches / max(duration_s / 60.0, 1e-9),
        "id_switches_auto_total": switches,
        "false_positive_locks": false_positive_locks,
        "all_locks": all_locks,
        "false_positive_lock_rate": (false_positive_locks / all_locks) if all_locks else 0.0,
        "ghost_drift_episodes": ghost_episodes,
    }
