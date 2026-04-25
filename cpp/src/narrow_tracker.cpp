#include "dtracker/narrow_tracker.hpp"

#include <algorithm>
#include <cmath>

namespace dtracker {

NarrowTracker::NarrowTracker(NarrowConfig cfg, int frame_w, int frame_h)
    : cfg_(cfg), frame_w_(frame_w), frame_h_(frame_h) {}

// D3: Adaptive velocity feedforward
//   |v| < ff_v_low (2.0): ff_scale = 0.2
//   |v| >= ff_v_high (5.0): ff_scale = 1.0
//   pomiedzy: liniowa interpolacja
// desired_center = bbox_center + ff_scale * velocity
Point2 NarrowTracker::compute_desired_center_(const Track& owner,
                                               double& out_ff_scale,
                                               double& out_v_mag) const {
    double vx = owner.velocity.x;
    double vy = owner.velocity.y;
    double v_mag = std::hypot(vx, vy);
    double ff_scale;
    if (v_mag < cfg_.ff_v_low) {
        ff_scale = cfg_.ff_scale_low;
    } else if (v_mag >= cfg_.ff_v_high) {
        ff_scale = cfg_.ff_scale_high;
    } else {
        double t = (v_mag - cfg_.ff_v_low) / (cfg_.ff_v_high - cfg_.ff_v_low);
        ff_scale = cfg_.ff_scale_low + t * (cfg_.ff_scale_high - cfg_.ff_scale_low);
    }
    out_ff_scale = ff_scale;
    out_v_mag = v_mag;
    return Point2{owner.center.x + ff_scale * vx,
                   owner.center.y + ff_scale * vy};
}

// D3: PID _step_towards z deadzone + max_step clamp + EMA smoothing pan/tilt
void NarrowTracker::step_towards_(const Point2& desired, bool active, bool degraded) {
    if (!s_.smooth_center) {
        s_.smooth_center = desired;
        s_.last_pan_speed = 0.0;
        s_.last_tilt_speed = 0.0;
        return;
    }

    double ex = desired.x - s_.smooth_center->x;
    double ey = desired.y - s_.smooth_center->y;

    double dead_zone = degraded ? cfg_.pid_dead_zone_degraded : cfg_.pid_dead_zone_active;
    if (std::abs(ex) < dead_zone) ex = 0.0;
    if (std::abs(ey) < dead_zone) ey = 0.0;

    double kp;
    double max_step;
    double smooth_alpha;
    if (active) {
        kp = degraded ? cfg_.pid_kp_active_degraded : cfg_.pid_kp_active;
        max_step = degraded ? cfg_.pid_max_step_active_degraded : cfg_.pid_max_step_active;
        smooth_alpha = degraded ? cfg_.pid_smooth_alpha_degraded : cfg_.pid_smooth_alpha;
    } else {
        // Hold path (no owner): keep nearly still during owner loss.
        kp = cfg_.pid_kp_hold;
        max_step = cfg_.pid_max_step_hold;
        smooth_alpha = cfg_.pid_smooth_alpha_degraded;
    }

    double pan_speed = std::clamp(ex * kp, -max_step, max_step);
    double tilt_speed = std::clamp(ey * kp, -max_step, max_step);

    pan_speed = smooth_alpha * s_.last_pan_speed + (1.0 - smooth_alpha) * pan_speed;
    tilt_speed = smooth_alpha * s_.last_tilt_speed + (1.0 - smooth_alpha) * tilt_speed;

    s_.last_pan_speed = pan_speed;
    s_.last_tilt_speed = tilt_speed;

    s_.smooth_center = Point2{s_.smooth_center->x + pan_speed,
                                s_.smooth_center->y + tilt_speed};
}

void NarrowTracker::update(const Track* owner, bool is_locked) {
    (void)is_locked;

    // Fix 1: gdy nie ma real ownera, sprawdz czy trzymac synthetic z last_good.
    if (!owner) {
        ++s_.hold_count;

        // Hold path: jest last_good_center i nie przekroczylismy budzetu.
        if (s_.last_good_center && s_.hold_count <= cfg_.max_hold_frames) {
            s_.is_synthetic = true;
            s_.has_owner = true;  // narrow renderuje sie dalej

            // Desired = last_good_center (zalozenie: dron zatrzymal sie / mozliwa
            // ekstrapolacja z last velocity). Tu trzymamy stale, bezpieczniej.
            Point2 desired = *s_.last_good_center;
            // Hold path: low gain, EMA wytlumia speed do 0 (anti-jerk powrot)
            step_towards_(desired, false, true);

            // Zoom drift powolnie out (na wypadek gdyby dron zaraz wrocil dalej):
            float target_zoom = s_.last_good_zoom;
            s_.zoom = cfg_.hold_zoom_decay * s_.zoom + (1.0f - cfg_.hold_zoom_decay) * target_zoom;
            // smooth_size pozostaje stable
            return;
        }

        // Budzet hold przekroczony albo nie ma last_good -- czerni narrow.
        s_.has_owner = false;
        s_.is_synthetic = false;
        // Wytlumiamy speed do 0
        if (s_.smooth_center) {
            double smooth_alpha = cfg_.pid_smooth_alpha_degraded;
            s_.last_pan_speed = smooth_alpha * s_.last_pan_speed;
            s_.last_tilt_speed = smooth_alpha * s_.last_tilt_speed;
            s_.smooth_center = Point2{s_.smooth_center->x + s_.last_pan_speed,
                                       s_.smooth_center->y + s_.last_tilt_speed};
        }
        return;
    }

    // Real owner is back -- reset hold counter, zapisz last_good na koniec.
    s_.has_owner = true;
    s_.is_synthetic = false;
    s_.hold_count = 0;

    // Adaptive velocity feedforward
    double ff_scale, v_mag;
    Point2 desired = compute_desired_center_(*owner, ff_scale, v_mag);
    s_.last_ff_scale = ff_scale;
    s_.last_velocity_magnitude = v_mag;

    // Degraded jesli owner missed > 0 lub conf < 0.18
    bool degraded = (owner->missed_frames >= 1) || (owner->confidence < 0.18f);
    s_.last_degraded = degraded;

    // PID step do desired_center
    step_towards_(desired, true, degraded);

    // Smooth size (EMA z clamped max step) -- bez zmian z poprzedniej wersji
    float target_size = std::max(std::max(owner->bbox.x2 - owner->bbox.x1,
                                          owner->bbox.y2 - owner->bbox.y1),
                                 20.0f);
    target_size = target_size / std::max(0.01f, cfg_.screen_fill);
    float size_diff = target_size - s_.smooth_size;
    size_diff = std::clamp(size_diff, -cfg_.display_max_size_step, cfg_.display_max_size_step);
    s_.smooth_size += (1.0f - cfg_.display_size_alpha) * size_diff;
    s_.smooth_size = std::max(50.0f, s_.smooth_size);

    // Zoom = frame_size / crop_size (proxy)
    s_.zoom = std::clamp(static_cast<float>(std::min(frame_w_, frame_h_)) / s_.smooth_size,
                          cfg_.zoom_min, cfg_.zoom_max);

    // Fix 1: zapisz last_good po obliczeniu smooth state (na wypadek przyszlej utraty)
    if (s_.smooth_center) {
        s_.last_good_center = s_.smooth_center;
        s_.last_good_zoom = s_.zoom;
        s_.last_good_size = s_.smooth_size;
        s_.last_good_bbox = owner->bbox;
    }
}

BBox NarrowTracker::narrow_crop() const {
    if (!s_.smooth_center) {
        return BBox{0, 0, static_cast<float>(frame_w_), static_cast<float>(frame_h_)};
    }
    float half = s_.smooth_size * 0.5f;
    float cx = static_cast<float>(s_.smooth_center->x);
    float cy = static_cast<float>(s_.smooth_center->y);
    float x1 = std::max(0.0f, cx - half);
    float y1 = std::max(0.0f, cy - half);
    float x2 = std::min(static_cast<float>(frame_w_), cx + half);
    float y2 = std::min(static_cast<float>(frame_h_), cy + half);
    return BBox{x1, y1, x2, y2};
}

}  // namespace dtracker
