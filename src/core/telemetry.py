from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

# Target-correctness validation thresholds. A reference target is established
# only after the tracker holds a stable, isolated, well-centered identity for
# this many frames — never immediately from first-acquire.
_REFERENCE_WINDOW_FRAMES = 20
_REFERENCE_GEOMETRY_MIN = 0.65
_REFERENCE_TELEPORT_MAX_PX = 60.0
_REFERENCE_NEIGHBOR_MIN_DIST_PX = 60.0
_REFERENCE_RESET_LOSS_FRAMES = 60
_WRONG_NEIGHBOR_RADIUS_PX = 100.0


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


def _sanitize_owner_missed(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        iv = int(value)
        if iv < 0:
            return None
        if iv >= 9999:
            return None
        return iv
    except Exception:
        return None


class TelemetryLogger:
    def __init__(
        self,
        run_name: str,
        fps: float = 30.0,
        root: str = "artifacts/runs",
        run_dir: str | Path | None = None,
    ) -> None:
        self.run_name = run_name
        self.fps = float(fps)

        if run_dir is not None:
            self.run_dir = Path(run_dir)
        else:
            self.run_dir = Path(root) / run_name

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "telemetry.jsonl"
        self._fh = self.path.open("w", encoding="utf-8")
        # Target-correctness state (reference only locked in after validation window).
        self._reference: Dict[str, Any] = {"raw_id": None, "area": None, "frame_idx": None, "established": False}
        self._prev_owner_center: Optional[tuple] = None
        self._prev_selected_id: Optional[int] = None
        self._prev_active_raw_id: Optional[int] = None
        self._candidate_count: int = 0
        self._off_reference_streak: int = 0
        self._lost_frames_count: int = 0
        self._prev_on_reference: Optional[bool] = None
        self._prev_nearest_distance: Optional[float] = None

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def log_crash(self, frame_idx: int, exc: BaseException) -> None:
        import traceback
        rec = {
            "event": "crash",
            "frame_idx": int(frame_idx),
            "time_s": float(frame_idx) / max(1.0, self.fps),
            "exc_type": type(exc).__name__,
            "exc_str": str(exc),
            "traceback": traceback.format_exc(),
        }
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

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
            vx, vy = getattr(tr, "velocity_xy", (None, None))
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
                    "vx": _safe_number(vx),
                    "vy": _safe_number(vy),
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
            "active_track_vx": _safe_number(getattr(active_track, "velocity_xy", (None, None))[0]) if active_track is not None else None,
            "active_track_vy": _safe_number(getattr(active_track, "velocity_xy", (None, None))[1]) if active_track is not None else None,
            "narrow_center": list(narrow_center) if narrow_center is not None else None,
            "center_lock": bool(center_lock),
            "drift_gate_open": bool(drift_gate_open),
            "tracks": items,
        }

        # --- Target-correctness instrumentation (read-only, telemetry-only) ---
        active_raw_id: Optional[int] = None
        active_center: Optional[tuple] = None
        active_area: Optional[float] = None
        if active_track is not None:
            try:
                x1, y1, x2, y2 = (float(v) for v in active_track.bbox_xyxy)
                active_area = max(1.0, (x2 - x1) * (y2 - y1))
                cx, cy = active_track.center_xy
                active_center = (float(cx), float(cy))
                rid = getattr(active_track, "raw_id", None)
                active_raw_id = int(rid) if rid is not None else None
            except Exception:
                active_raw_id = None
                active_center = None
                active_area = None

        owner_teleport_px: Optional[float] = None
        if active_center is not None and self._prev_owner_center is not None:
            dx = active_center[0] - self._prev_owner_center[0]
            dy = active_center[1] - self._prev_owner_center[1]
            owner_teleport_px = float((dx * dx + dy * dy) ** 0.5)

        neighbors: list[Dict[str, Any]] = []
        if active_center is not None and active_area:
            active_tid = getattr(active_track, "track_id", None)
            for tr in tracks:
                tid = getattr(tr, "track_id", None)
                if tid is None or (active_tid is not None and int(tid) == int(active_tid)):
                    continue
                ncx, ncy = getattr(tr, "center_xy", (None, None))
                if ncx is None or ncy is None:
                    continue
                nx1, ny1, nx2, ny2 = (float(v) for v in getattr(tr, "bbox_xyxy", (0, 0, 0, 0)))
                narea = max(1.0, (nx2 - nx1) * (ny2 - ny1))
                dx = float(ncx) - active_center[0]
                dy = float(ncy) - active_center[1]
                rid = getattr(tr, "raw_id", None)
                neighbors.append({
                    "track_id": int(tid),
                    "raw_id": int(rid) if rid is not None else None,
                    "conf": float(getattr(tr, "confidence", 0.0) or 0.0),
                    "distance_px": float((dx * dx + dy * dy) ** 0.5),
                    "area_ratio_to_owner": narea / active_area,
                })
        neighbors.sort(key=lambda n: n["distance_px"])
        nearest_neighbor = neighbors[0] if neighbors else None

        geom_score = float((extra or {}).get("geometry_score", 0.0) or 0.0)

        identity_stable = (
            selected_id is not None
            and active_raw_id is not None
            and self._prev_selected_id == selected_id
            and self._prev_active_raw_id == active_raw_id
        )
        tracking_clean = (
            bool(center_lock)
            and geom_score >= _REFERENCE_GEOMETRY_MIN
            and (owner_teleport_px is None or owner_teleport_px < _REFERENCE_TELEPORT_MAX_PX)
        )
        neighbor_clean = (
            not neighbors
            or neighbors[0]["distance_px"] > _REFERENCE_NEIGHBOR_MIN_DIST_PX
        )

        already_established = bool(self._reference.get("established"))
        if identity_stable and tracking_clean and neighbor_clean:
            self._candidate_count += 1
            if self._candidate_count >= _REFERENCE_WINDOW_FRAMES and not already_established:
                self._reference = {
                    "raw_id": active_raw_id,
                    "area": active_area,
                    "frame_idx": int(frame_idx),
                    "established": True,
                }
        else:
            self._candidate_count = 0

        if selected_id is None:
            self._lost_frames_count += 1
            if self._lost_frames_count >= _REFERENCE_RESET_LOSS_FRAMES:
                self._reference = {"raw_id": None, "area": None, "frame_idx": None, "established": False}
                self._candidate_count = 0
                self._off_reference_streak = 0
        else:
            self._lost_frames_count = 0

        reference_established = bool(self._reference.get("established"))
        reference_raw_id = self._reference.get("raw_id")
        reference_area = self._reference.get("area")
        if reference_established and active_raw_id is not None:
            on_reference_target: Optional[bool] = (active_raw_id == reference_raw_id)
        else:
            on_reference_target = None

        if on_reference_target is True:
            self._off_reference_streak = 0
        elif on_reference_target is False:
            self._off_reference_streak += 1

        wrong_neighbor_event = bool(
            reference_established
            and self._prev_on_reference is True
            and on_reference_target is False
            and self._prev_nearest_distance is not None
            and self._prev_nearest_distance < _WRONG_NEIGHBOR_RADIUS_PX
        )

        area_ratio_to_reference = None
        if reference_area and active_area:
            area_ratio_to_reference = active_area / reference_area

        rec["active_raw_id"] = active_raw_id
        rec["owner_teleport_px"] = owner_teleport_px
        rec["neighbor_count"] = len(neighbors)
        rec["nearest_neighbor"] = nearest_neighbor
        rec["candidate_window_frames"] = self._candidate_count
        rec["reference_established"] = reference_established
        rec["reference_raw_id"] = reference_raw_id
        rec["reference_frame_idx"] = self._reference.get("frame_idx")
        rec["reference_bbox_area"] = reference_area
        rec["on_reference_target"] = on_reference_target
        rec["off_reference_streak"] = self._off_reference_streak
        rec["wrong_neighbor_event"] = wrong_neighbor_event
        rec["area_ratio_to_reference"] = area_ratio_to_reference

        self._prev_owner_center = active_center
        self._prev_selected_id = selected_id
        self._prev_active_raw_id = active_raw_id
        self._prev_on_reference = on_reference_target
        self._prev_nearest_distance = (nearest_neighbor["distance_px"] if nearest_neighbor else None)

        for key, value in (extra or {}).items():
            if key == "owner_missed_frames":
                rec[key] = _sanitize_owner_missed(value)
                continue

            if isinstance(value, (list, dict, str, bool, int, float)) or value is None:
                rec[key] = value
            else:
                rec[key] = _safe_number(value)

        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()


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