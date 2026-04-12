from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


def _safe_number(x: Any) -> Any:
    if isinstance(x, (int, float, bool)) or x is None:
        return x
    try:
        return float(x)
    except Exception:
        return None


def _bbox_metrics(bbox: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    aspect = w / max(1.0, h)
    return area, aspect


class TelemetryLogger:
    def __init__(self, run_name: str, fps: float = 30.0, root: str = "artifacts/telemetry") -> None:
        self.run_name = run_name
        self.fps = float(fps)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / f"{run_name}.jsonl"
        self._fh = self.path.open("w", encoding="utf-8")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def log_frame(
        self,
        frame_idx: int,
        mode: str,
        selected_id: Optional[int],
        active_track: object | None,
        tracks: Iterable[object],
        narrow_center: Optional[Sequence[float]],
        center_lock: bool,
        drift_gate_open: bool,
        **extra,
    ) -> None:
        tracks = list(tracks or [])
        active_track_id = getattr(active_track, "track_id", None) if active_track is not None else None

        items: list[Dict[str, Any]] = []
        for tr in tracks:
            bbox = tuple(float(v) for v in getattr(tr, "bbox_xyxy", (0, 0, 0, 0)))
            area, aspect = _bbox_metrics(bbox)
            cx, cy = getattr(tr, "center_xy", (None, None))
            items.append(
                {
                    "track_id": getattr(tr, "track_id", None),
                    "raw_id": getattr(tr, "raw_id", None),
                    "conf": _safe_number(getattr(tr, "confidence", None)),
                    "hits": getattr(tr, "hits", None),
                    "missed_frames": getattr(tr, "missed_frames", None),
                    "is_confirmed": bool(getattr(tr, "is_confirmed", False)),
                    "bbox": list(bbox),
                    "area": area,
                    "aspect": aspect,
                    "cx": _safe_number(cx),
                    "cy": _safe_number(cy),
                }
            )

        rec: Dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "time_s": float(frame_idx) / max(1.0, self.fps),
            "mode": mode,
            "multi_tracks": len(tracks),
            "selected_id": selected_id,
            "active_track_id": active_track_id,
            "active_track_missed": getattr(active_track, "missed_frames", None) if active_track is not None else None,
            "active_track_conf": _safe_number(getattr(active_track, "confidence", None)) if active_track is not None else None,
            "narrow_center": list(narrow_center) if narrow_center is not None else None,
            "center_lock": bool(center_lock),
            "drift_gate_open": bool(drift_gate_open),
            "tracks": items,
        }
        for key, value in (extra or {}).items():
            if hasattr(value, "__dict__"):
                try:
                    value = dict(value.__dict__)
                except Exception:
                    pass
            if isinstance(value, (list, dict, str, bool, int, float)) or value is None:
                rec[key] = value
            else:
                rec[key] = _safe_number(value)
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()


# === V6 PIVOT telemetry helpers ===
def telemetry_local_identity_state(ctx):
    tracker = getattr(ctx, "local_target_tracker", None)
    if tracker is None:
        return {
            "local_identity_active": False,
            "local_identity_score": 0.0,
            "local_identity_lost_frames": 0,
            "local_identity_bbox": None,
        }
    try:
        st = tracker.state()
        return {
            "local_identity_active": bool(st.get("active", False)),
            "local_identity_score": float(st.get("score", 0.0) or 0.0),
            "local_identity_lost_frames": int(st.get("lost_frames", 0) or 0),
            "local_identity_bbox": st.get("bbox"),
        }
    except Exception:
        return {
            "local_identity_active": False,
            "local_identity_score": 0.0,
            "local_identity_lost_frames": 0,
            "local_identity_bbox": None,
        }
