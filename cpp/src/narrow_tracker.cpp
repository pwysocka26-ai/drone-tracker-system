#include "dtracker/narrow_tracker.hpp"

#include <algorithm>
#include <cmath>

namespace dtracker {

NarrowTracker::NarrowTracker(NarrowConfig cfg, int frame_w, int frame_h)
    : cfg_(cfg), frame_w_(frame_w), frame_h_(frame_h) {}

void NarrowTracker::update(const Track* owner, bool is_locked) {
    if (!owner) {
        s_.has_owner = false;
        // drift smooth_center z velocity (jesli kiedys bylo)
        return;
    }

    s_.has_owner = true;
    Point2 target_center = owner->center;
    float target_size = std::max(std::max(owner->bbox.x2 - owner->bbox.x1,
                                          owner->bbox.y2 - owner->bbox.y1),
                                 20.0f);
    // Rozmiar narrow crop tak aby bbox zajmowal screen_fill frakcje kadru
    target_size = target_size / std::max(0.01f, cfg_.screen_fill);

    // Smooth center (EMA z clamped max step)
    if (!s_.smooth_center) {
        s_.smooth_center = target_center;
    } else {
        double dx = target_center.x - s_.smooth_center->x;
        double dy = target_center.y - s_.smooth_center->y;
        double len = std::hypot(dx, dy);
        if (len > cfg_.display_max_center_step) {
            dx *= cfg_.display_max_center_step / len;
            dy *= cfg_.display_max_center_step / len;
        }
        s_.smooth_center->x += (1.0 - cfg_.display_center_alpha) * dx;
        s_.smooth_center->y += (1.0 - cfg_.display_center_alpha) * dy;
    }

    // Smooth size (EMA z clamped max step)
    float size_diff = target_size - s_.smooth_size;
    size_diff = std::clamp(size_diff, -cfg_.display_max_size_step, cfg_.display_max_size_step);
    s_.smooth_size += (1.0f - cfg_.display_size_alpha) * size_diff;
    s_.smooth_size = std::max(50.0f, s_.smooth_size);

    // Zoom = frame_size / crop_size (proxy)
    s_.zoom = std::clamp(static_cast<float>(std::min(frame_w_, frame_h_)) / s_.smooth_size,
                          cfg_.zoom_min, cfg_.zoom_max);
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
