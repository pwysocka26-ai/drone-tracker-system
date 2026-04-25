// NarrowTracker — smooth center + zoom dla wizualizacji narrow kanalu.
// D3 upgrade: PID _step_towards + adaptive velocity feedforward (parity
// z src/core/narrow_tracker.py:165-210, 385-402).
//
// Adaptive feedforward: scale 0.2-1.0 zaleznie od |velocity|:
//   |v| < 2 px/frame: scale=0.2 (drone stacjonarny, maly feedforward)
//   |v| >= 5 px/frame: scale=1.0 (szybki manewr, pelny feedforward)
//   pomiedzy: liniowa interpolacja
// desired_center = bbox_center + ff_scale * velocity
//
// PID _step_towards: proportional gain z deadzone + max_step clamp +
// EMA smoothing pan/tilt speed (anti-jerk).
#pragma once

#include <optional>

#include "dtracker/kalman.hpp"
#include "dtracker/track.hpp"

namespace dtracker {

struct NarrowConfig {
    float display_center_alpha = 0.78f;
    float display_size_alpha = 0.50f;
    float display_max_center_step = 42.0f;
    float display_max_size_step = 50.0f;
    float screen_fill = 0.18f;         // docelowy rozmiar drona w kadrze narrow
    float zoom_min = 1.0f;
    float zoom_max = 10.0f;

    // D3 PID + adaptive feedforward
    float pid_kp_active = 0.24f;          // proportional gain (active path)
    float pid_kp_active_degraded = 0.18f; // gain dla degraded ownera
    float pid_kp_hold = 0.05f;            // gain bez ownera (hold path)
    float pid_max_step_active = 46.0f;    // clamp speed (active)
    float pid_max_step_active_degraded = 32.0f;
    float pid_max_step_hold = 6.0f;
    float pid_dead_zone_active = 4.0f;    // ignore errors below dead_zone
    float pid_dead_zone_degraded = 6.0f;
    float pid_smooth_alpha = 0.74f;       // EMA pan/tilt speed (active)
    float pid_smooth_alpha_degraded = 0.84f;

    float ff_v_low = 2.0f;                // poniej ff_scale=0.2
    float ff_v_high = 5.0f;               // powyzej ff_scale=1.0
    float ff_scale_low = 0.2f;
    float ff_scale_high = 1.0f;

    // Fix 1: last_good persistence -- narrow trzyma synthetic crop podczas
    // chwilowej utraty ownera (HOLD/REACQUIRE), zamiast gasic widok.
    // Fix 4: 100 -> 300 (6 s @ 50fps). Visual review klatka 24s pokazala
    // czarny narrow gdy hold_count > 100 mimo ze TM trzymal persistent #1.
    // 6 s pokrywa wszystkie obserwowane przerwy detekcji w teście.
    int   max_hold_frames = 300;
    float hold_zoom_decay = 0.992f;       // EMA powolne zoom out podczas hold
};

struct NarrowState {
    std::optional<Point2> smooth_center;
    float smooth_size = 200.0f;        // aktualny rozmiar narrow crop (px)
    float zoom = 1.0f;
    bool has_owner = false;            // realny owner LUB synthetic-hold (Fix 1)

    // D3 PID state (across frames)
    double last_pan_speed = 0.0;
    double last_tilt_speed = 0.0;

    // Diag
    double last_ff_scale = 0.0;
    double last_velocity_magnitude = 0.0;
    bool last_degraded = false;

    // Fix 1: last_good persistence -- gdy real owner zniknal, trzymamy
    // ostatni dobry crop przez max_hold_frames klatek, zeby narrow nie czernial.
    std::optional<Point2> last_good_center;
    std::optional<BBox>   last_good_bbox;
    float last_good_size = 200.0f;
    float last_good_zoom = 1.0f;
    int   hold_count = 0;          // ile klatek bez real ownera
    bool  is_synthetic = false;    // true gdy aktualnie renderujemy z last_good
};

class NarrowTracker {
public:
    explicit NarrowTracker(NarrowConfig cfg = {}, int frame_w = 1920, int frame_h = 1080);

    // Update step: owner_track opcjonalny (moze byc nullptr, wtedy smooth drift).
    void update(const Track* owner, bool is_locked);

    const NarrowState& state() const { return s_; }

    // Narrow crop bbox (x1, y1, x2, y2) w klatkach zrodlowej.
    BBox narrow_crop() const;

private:
    // D3: PID step do desired_center z deadzone + clamp + EMA smoothing
    void step_towards_(const Point2& desired, bool active, bool degraded);

    // Adaptive velocity feedforward: zwraca (desired_center, ff_scale, |v|)
    Point2 compute_desired_center_(const Track& owner, double& out_ff_scale,
                                    double& out_v_mag) const;

    NarrowConfig cfg_;
    NarrowState s_;
    int frame_w_, frame_h_;
};

}  // namespace dtracker
