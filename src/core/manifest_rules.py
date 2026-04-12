from __future__ import annotations

from typing import Any


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _event_count(events: list[dict[str, Any]], kind: str) -> int:
    return sum(1 for e in events if str(e.get("kind")) == kind)


def _has_edge_center_lock_break(events: list[dict[str, Any]]) -> bool:
    for e in events:
        if str(e.get("kind")) != "center_lock_off":
            continue
        desc = str(e.get("description", "")).lower()
        compact = desc.replace(" ", "")
        if "edge=true" in compact:
            return True
    return False


def _late_run_failure(events: list[dict[str, Any]], metrics: dict[str, Any]) -> bool:
    frames_total = max(1, _safe_int(metrics.get("frames_total", metrics.get("frames"))))
    late_threshold = int(frames_total * 0.80)
    for e in events:
        frame = _safe_int(e.get("frame_idx", e.get("frame")), -1)
        if frame >= late_threshold:
            if str(e.get("kind", e.get("type"))) in {"lock_lost", "reacquire", "drift", "center_lock_off"}:
                return True
    return False


def classify_run(
    metrics: dict[str, Any],
    final_state: dict[str, Any],
    timeline_events: list[dict[str, Any]],
) -> dict[str, str]:
    lock_losses = _safe_int(metrics.get("lock_lost_count", metrics.get("lock_losses")))
    reacquire_phases = _safe_int(metrics.get("reacquire_count", metrics.get("reacquire_phases")))
    wide_switches = _safe_int(metrics.get("wide_owner_switches"))
    narrow_switches = _safe_int(metrics.get("narrow_owner_switches"))
    short_drop_count = _safe_int(metrics.get("short_drop_count"))
    detection_gap_count = _safe_int(metrics.get("detection_gap_count"))
    max_drop = _safe_int(metrics.get("max_drop"))

    final_narrow_owner = final_state.get("narrow_owner", metrics.get("final_narrow_owner"))
    tracking_state = str(final_state.get("tracking_state", "") or "")
    lock_active = bool(final_state.get("lock_active", False))

    failed_end_state = (
        final_narrow_owner is None
        or tracking_state == "REACQUIRE"
        or not lock_active
    )

    edge_breaks = _event_count(timeline_events, "center_lock_off")
    edge_triggered = _has_edge_center_lock_break(timeline_events)

    if failed_end_state:
        if lock_losses >= 1 and reacquire_phases >= 1:
            return {
                "run_classification": "failed_end_state",
                "secondary_classification": "reacquire_failure",
            }
        return {
            "run_classification": "failed_end_state",
            "secondary_classification": "handoff_instability" if (wide_switches >= 6 or narrow_switches >= 6) else "mixed_failure",
        }

    if edge_breaks >= 2 and edge_triggered:
        return {
            "run_classification": "edge_geometry_failure",
            "secondary_classification": "handoff_instability" if (wide_switches >= 6 or narrow_switches >= 6) else "mixed_failure",
        }

    if (short_drop_count >= 10 and max_drop >= 30) or detection_gap_count >= 2:
        return {
            "run_classification": "detection_dropout_dominated",
            "secondary_classification": "handoff_instability" if (wide_switches >= 6 or narrow_switches >= 6) else "mixed_failure",
        }

    if wide_switches >= 6 or narrow_switches >= 6:
        return {
            "run_classification": "handoff_instability",
            "secondary_classification": "unstable_but_recovered" if lock_active else "mixed_failure",
        }

    if lock_losses == 0 and reacquire_phases == 0 and narrow_switches <= 2 and final_narrow_owner is not None and lock_active:
        return {
            "run_classification": "stable",
            "secondary_classification": "stable",
        }

    if final_narrow_owner is not None and lock_active:
        return {
            "run_classification": "unstable_but_recovered",
            "secondary_classification": "mixed_failure",
        }

    return {
        "run_classification": "mixed_failure",
        "secondary_classification": "mixed_failure",
    }


def detect_symptoms(
    metrics: dict[str, Any],
    final_state: dict[str, Any],
    timeline_events: list[dict[str, Any]],
) -> dict[str, Any]:
    wide_switches = _safe_int(metrics.get("wide_owner_switches"))
    narrow_switches = _safe_int(metrics.get("narrow_owner_switches"))
    short_drop_count = _safe_int(metrics.get("short_drop_count"))
    detection_gap_count = _safe_int(metrics.get("detection_gap_count"))
    max_drop = _safe_int(metrics.get("max_drop"))
    center_lock_off_count = _safe_int(metrics.get("center_lock_off_count"))
    lock_losses = _safe_int(metrics.get("lock_lost_count", metrics.get("lock_losses")))
    reacquire_phases = _safe_int(metrics.get("reacquire_count", metrics.get("reacquire_phases")))
    avg_geometry_score = _safe_float(metrics.get("avg_geometry_score"))
    avg_wide_owner_quality = _safe_float(metrics.get("avg_wide_quality", metrics.get("avg_wide_owner_quality")))

    drift_count = _event_count(timeline_events, "drift")
    final_narrow_owner = final_state.get("narrow_owner", metrics.get("final_narrow_owner"))

    symptoms: list[str] = []

    if wide_switches >= 6:
        symptoms.append("frequent_wide_switches")
        symptoms.append("wide_owner_instability")

    if narrow_switches >= 6:
        symptoms.append("frequent_narrow_switches")
        symptoms.append("narrow_owner_instability")

    if short_drop_count >= 10:
        symptoms.append("repeated_short_detection_drops")

    if detection_gap_count >= 1 or max_drop >= 30:
        symptoms.append("long_detection_gap")

    if center_lock_off_count >= 2:
        symptoms.append("center_lock_breaks")

    if _has_edge_center_lock_break(timeline_events):
        symptoms.append("edge_triggered_center_lock_breaks")

    if lock_losses >= 1 and narrow_switches >= 6:
        symptoms.append("lock_loss_after_owner_instability")

    if drift_count >= 1 and reacquire_phases >= 1:
        symptoms.append("reacquire_entry_after_drift")

    if final_narrow_owner is None:
        symptoms.append("run_ended_without_narrow_owner")

    if avg_geometry_score < 0.70:
        symptoms.append("geometry_score_low")

    if avg_wide_owner_quality < 0.50:
        symptoms.append("owner_quality_low")

    if _late_run_failure(timeline_events, metrics):
        symptoms.append("late_run_failure")

    manual_frames = _safe_int(metrics.get("manual_frames"))
    if manual_frames == 0:
        symptoms.append("manual_mode_unused")

    deduped: list[str] = []
    seen: set[str] = set()
    for s in symptoms:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    primary_symptom = "none"
    priority_order = [
        "run_ended_without_narrow_owner",
        "lock_loss_after_owner_instability",
        "reacquire_entry_after_drift",
        "narrow_owner_instability",
        "wide_owner_instability",
        "edge_triggered_center_lock_breaks",
        "center_lock_breaks",
        "long_detection_gap",
        "repeated_short_detection_drops",
        "geometry_score_low",
        "owner_quality_low",
        "late_run_failure",
        "manual_mode_unused",
    ]
    for item in priority_order:
        if item in deduped:
            primary_symptom = item
            break

    return {
        "primary_symptom": primary_symptom,
        "symptoms": deduped,
    }


def recommend_first_action(
    classification: dict[str, str],
    symptom_result: dict[str, Any],
    hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_classification = str(classification.get("run_classification", "mixed_failure"))
    symptoms = list(symptom_result.get("symptoms", []))
    primary_symptom = str(symptom_result.get("primary_symptom", "none"))

    if run_classification == "failed_end_state" or run_classification == "reacquire_failure":
        return {
            "type": "inspect_and_patch",
            "module": "src/core/narrow_tracker.py",
            "change_kind": "reacquire_gate_tightening",
            "reason": "run ends without active narrow owner and requires stronger reacquire and hold logic",
            "confidence": 0.90,
            "priority": "high",
        }

    if primary_symptom in {"frequent_narrow_switches", "narrow_owner_instability"} or "narrow_owner_instability" in symptoms:
        return {
            "type": "inspect_and_patch",
            "module": "src/core/target_manager.py",
            "change_kind": "switch_hysteresis_increase",
            "reason": "narrow-side instability suggests owner switching thresholds and persistence are too weak",
            "confidence": 0.84,
            "priority": "high",
        }

    if "edge_triggered_center_lock_breaks" in symptoms or run_classification == "edge_geometry_failure":
        return {
            "type": "inspect_and_patch",
            "module": "src/core/app.py",
            "change_kind": "center_lock_edge_guard",
            "reason": "center lock repeatedly breaks under edge geometry and should degrade more gracefully",
            "confidence": 0.82,
            "priority": "high",
        }

    if "wide_owner_instability" in symptoms:
        return {
            "type": "inspect_and_patch",
            "module": "src/core/target_manager.py",
            "change_kind": "switch_cooldown_increase",
            "reason": "wide owner selection is too unstable and likely drives downstream churn",
            "confidence": 0.78,
            "priority": "medium",
        }

    if "repeated_short_detection_drops" in symptoms or run_classification == "detection_dropout_dominated":
        return {
            "type": "inspect_and_patch",
            "module": "config/config.yaml",
            "change_kind": "config_first_tuning",
            "reason": "detection dropout dominates the failure signature and should be validated at configuration level first",
            "confidence": 0.72,
            "priority": "medium",
        }

    return {
        "type": "inspect",
        "module": "src/core/app.py",
        "change_kind": "fallback_alignment",
        "reason": "start from integration flow and verify runtime transitions against generated telemetry",
        "confidence": 0.60,
        "priority": "medium",
    }


def build_quick_verdict(
    metrics: dict[str, Any],
    final_state: dict[str, Any],
    classification: dict[str, str],
    symptom_result: dict[str, Any],
) -> str:
    run_classification = str(classification.get("run_classification", "mixed_failure"))
    secondary_classification = str(classification.get("secondary_classification", "mixed_failure"))
    final_narrow_owner = final_state.get("narrow_owner", metrics.get("final_narrow_owner"))
    avg_geometry_score = _safe_float(metrics.get("avg_geometry_score"))
    avg_wide_owner_quality = _safe_float(metrics.get("avg_wide_quality", metrics.get("avg_wide_owner_quality")))
    lock_losses = _safe_int(metrics.get("lock_lost_count", metrics.get("lock_losses")))
    reacquire_count = _safe_int(metrics.get("reacquire_count", metrics.get("reacquire_phases")))
    primary_symptom = str(symptom_result.get("primary_symptom", "none"))

    if run_classification == "stable":
        return "Run stable. No major terminal failure pattern was detected."

    if (
        final_narrow_owner is None
        and avg_geometry_score < 0.60
        and avg_wide_owner_quality < 0.35
        and lock_losses >= 1
    ):
        return "Run failed in late-stage tracking after quality degradation, edge-triggered center lock break, and reacquire loss."

    if run_classification == "failed_end_state" and secondary_classification == "reacquire_failure":
        return "Run ended in failed end state after lock loss and unsuccessful reacquire recovery."

    if primary_symptom == "narrow_owner_instability":
        return "Run unstable due to repeated narrow owner instability and weak recovery behavior."

    if reacquire_count >= 1 and lock_losses >= 1:
        return "Run unstable with lock-loss and reacquire transitions concentrated in the failing phase."

    return "Run unstable. Main suspected issue is narrow owner instability under repeated detection gaps and edge-triggered geometry breaks."


def build_primary_modules_to_check(
    recommended_first_action: dict[str, Any],
    classification: dict[str, str],
    symptom_result: dict[str, Any],
) -> list[str]:
    first_module = str(recommended_first_action.get("module", "") or "")
    run_classification = str(classification.get("run_classification", "mixed_failure"))
    symptoms = list(symptom_result.get("symptoms", []))

    ordered: list[str] = []

    def add(module: str) -> None:
        if module and module not in ordered:
            ordered.append(module)

    add(first_module)

    if first_module == "src/core/narrow_tracker.py":
        add("src/core/target_manager.py")
        add("src/core/app.py")
    elif first_module == "src/core/target_manager.py":
        add("src/core/narrow_tracker.py")
        add("src/core/app.py")
    elif first_module == "src/core/app.py":
        add("src/core/narrow_tracker.py")
        add("src/core/target_manager.py")
    else:
        add("src/core/narrow_tracker.py")
        add("src/core/target_manager.py")
        add("src/core/app.py")

    if run_classification == "detection_dropout_dominated" or "repeated_short_detection_drops" in symptoms:
        add("config/config.yaml")

    return ordered


def _window_around(frame: int, pre_margin: int, post_margin: int) -> list[int]:
    return [max(0, int(frame) - int(pre_margin)), int(frame) + int(post_margin)]


def _merge_windows(windows: list[list[int]], merge_gap: int = 6) -> list[list[int]]:
    if not windows:
        return []

    windows = sorted(windows, key=lambda x: x[0])
    merged = [windows[0]]

    for start, end in windows[1:]:
        last = merged[-1]
        if start <= last[1] + merge_gap:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])

    return merged


def select_analysis_windows(timeline_events: list[dict[str, Any]]) -> list[list[int]]:
    windows: list[list[int]] = []

    lock_lost_events = [e for e in timeline_events if str(e.get("kind")) == "lock_lost"]
    reacquire_events = [e for e in timeline_events if str(e.get("kind")) == "reacquire"]
    drift_events = [e for e in timeline_events if str(e.get("kind")) == "drift"]
    center_lock_off_events = [e for e in timeline_events if str(e.get("kind")) == "center_lock_off"]
    owner_switch_events = [e for e in timeline_events if str(e.get("kind")) in {"owner_switch", "narrow_switch"}]
    detection_drop_events = [e for e in timeline_events if str(e.get("kind")) in {"detection_gap", "detection_drop"}]

    if lock_lost_events:
        e = lock_lost_events[-1]
        windows.append(_window_around(_safe_int(e.get("frame_idx")), 12, 12))

    if reacquire_events:
        e = reacquire_events[-1]
        windows.append(_window_around(_safe_int(e.get("frame_idx")), 12, 15))

    if drift_events:
        e = drift_events[-1]
        windows.append(_window_around(_safe_int(e.get("frame_idx")), 10, 12))

    if center_lock_off_events:
        e = center_lock_off_events[-1]
        windows.append(_window_around(_safe_int(e.get("frame_idx")), 10, 10))

    if owner_switch_events:
        burst = sorted(_safe_int(e.get("frame_idx")) for e in owner_switch_events)
        if burst:
            burst_start = burst[0]
            burst_end = burst[min(len(burst) - 1, 2)]
            windows.append([max(0, burst_start - 5), burst_end + 5])

    if detection_drop_events:
        e = detection_drop_events[-1]
        windows.append(_window_around(_safe_int(e.get("frame_idx")), 8, 10))

    merged = _merge_windows(windows, merge_gap=6)
    return merged[:4]


def top_failure_frames(timeline_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    severity_rank = {
        "lock_lost": 100,
        "reacquire": 95,
        "center_lock_off": 90,
        "drift": 85,
        "detection_gap": 70,
        "detection_drop": 55,
        "owner_switch": 30,
        "narrow_switch": 25,
    }

    def score(event: dict[str, Any]) -> tuple[int, int]:
        kind = str(event.get("kind", event.get("type")))
        frame = _safe_int(event.get("frame_idx", event.get("frame")))
        base = severity_rank.get(kind, 0)

        if frame < 15 and kind not in {"lock_lost", "reacquire"}:
            base -= 40

        return (base, frame)

    ranked = sorted(timeline_events, key=score, reverse=True)

    picked: list[dict[str, Any]] = []
    used_frames: set[int] = set()

    for e in ranked:
        kind = str(e.get("kind", e.get("type")))
        if kind not in {"lock_lost", "reacquire", "center_lock_off", "drift", "detection_gap", "detection_drop", "owner_switch", "narrow_switch"}:
            continue

        frame = _safe_int(e.get("frame_idx", e.get("frame")))
        base = severity_rank.get(kind, 0)
        if frame < 15 and kind not in {"lock_lost", "reacquire"}:
            base -= 40
        if base <= 0:
            continue

        if any(abs(frame - uf) <= 6 for uf in used_frames):
            continue

        picked.append(
            {
                "frame": frame,
                "time_s": round(_safe_float(e.get("ts_s", e.get("time_s"))), 2),
                "type": kind,
                "description": str(e.get("description", "")),
            }
        )
        used_frames.add(frame)
        if len(picked) >= 3:
            break

    picked.sort(key=lambda x: x["frame"])
    return picked


def select_manifest_events(timeline_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not timeline_events:
        return []

    severity_order = {
        "lock_lost": 100,
        "reacquire": 90,
        "center_lock_off": 80,
        "drift": 70,
        "owner_switch": 60,
        "narrow_switch": 55,
        "detection_gap": 50,
        "detection_drop": 40,
        "reacquire_complete": 35,
        "center_lock_on": 20,
        "detection_return": 10,
    }

    tail_events = sorted(
        timeline_events[-3:],
        key=lambda e: _safe_int(e.get("frame_idx", e.get("frame"))),
    )
    important = sorted(
        timeline_events,
        key=lambda e: (
            severity_order.get(str(e.get("kind")), 0),
            _safe_int(e.get("frame_idx", e.get("frame"))),
        ),
        reverse=True,
    )

    merged: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    for e in important[:10] + tail_events:
        frame = _safe_int(e.get("frame_idx", e.get("frame")))
        kind = str(e.get("kind", e.get("type")))
        key = (frame, kind)
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "frame_idx": frame,
                "ts_s": _safe_float(e.get("ts_s", e.get("time_s"))),
                "kind": kind,
                "description": str(e.get("description", "")),
            }
        )

    merged.sort(key=lambda e: e["frame_idx"])
    return merged[:12]