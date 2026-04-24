// NarrowTracker — smooth center + zoom dla wizualizacji narrow kanalu.
// Uproszczony: brak pelnego PID z adaptive velocity feedforward (dodajemy post-demo).
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
};

struct NarrowState {
    std::optional<Point2> smooth_center;
    float smooth_size = 200.0f;        // aktualny rozmiar narrow crop (px)
    float zoom = 1.0f;
    bool has_owner = false;
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
    NarrowConfig cfg_;
    NarrowState s_;
    int frame_w_, frame_h_;
};

}  // namespace dtracker
