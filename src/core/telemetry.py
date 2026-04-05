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
        self._prev_center: Optional[tuple[float, float]] = None
        self._prev_zoom: Optional[float] = None

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
        steering_target_id: Optional[int] = None,
        lock_state: Optional[str] = None,
        pan_error_px: Optional[float] = None,
        tilt_error_px: Optional[float] = None,
        radial_error_px: Optional[float] = None,
        zoom: Optional[float] = None,
        jump_limited: bool = False,
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

        center_delta_px = None
        if narrow_center is not None and self._prev_center is not None:
            dx = float(narrow_center[0]) - float(self._prev_center[0])
            dy = float(narrow_center[1]) - float(self._prev_center[1])
            center_delta_px = float((dx * dx + dy * dy) ** 0.5)
        if narrow_center is not None:
            self._prev_center = (float(narrow_center[0]), float(narrow_center[1]))

        zoom_delta = None
        if zoom is not None and self._prev_zoom is not None:
            zoom_delta = float(zoom) - float(self._prev_zoom)
        if zoom is not None:
            self._prev_zoom = float(zoom)

        rec: Dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "time_s": float(frame_idx) / max(1.0, self.fps),
            "mode": mode,
            "multi_tracks": len(tracks),
            "selected_id": selected_id,
            "active_track_id": active_track_id,
            "steering_target_id": steering_target_id,
            "active_track_missed": getattr(active_track, "missed_frames", None) if active_track is not None else None,
            "active_track_conf": _safe_number(getattr(active_track, "confidence", None)) if active_track is not None else None,
            "narrow_center": list(narrow_center) if narrow_center is not None else None,
            "center_lock": bool(center_lock),
            "lock_state": lock_state,
            "pan_error_px": _safe_number(pan_error_px),
            "tilt_error_px": _safe_number(tilt_error_px),
            "radial_error_px": _safe_number(radial_error_px),
            "center_delta_px": _safe_number(center_delta_px),
            "zoom": _safe_number(zoom),
            "zoom_delta": _safe_number(zoom_delta),
            "jump_limited": bool(jump_limited),
            "drift_gate_open": bool(drift_gate_open),
            "tracks": items,
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
