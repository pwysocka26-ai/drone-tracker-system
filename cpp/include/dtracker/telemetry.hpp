// Telemetry — zapis JSON per-klatka zgodny z Python telemetry.jsonl format.
// Kluczowe pola (parity z Pythonem do weryfikacji parity-testow):
//   frame_idx, time_s, selected_id, active_track_id, active_track_bbox,
//   active_track_conf, narrow_lock_state, narrow_lock_phase,
//   multi_tracks (len), narrow_center, center_lock.
#pragma once

#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "dtracker/lock_pipeline.hpp"
#include "dtracker/track.hpp"

namespace dtracker {

struct FrameTelemetry {
    int frame_idx = 0;
    double time_s = 0.0;
    std::optional<int> selected_id;
    int persistent_owner_id = -1;        // Fix 2: logical owner (independent of track_id)
    std::optional<Track> active_track;
    LockState lock_state = LockState::UNLOCKED;
    std::vector<Track> multi_tracks;
    std::optional<Point2> narrow_center;
    bool center_lock = false;
    bool narrow_synthetic_hold = false;  // Fix 1: narrow renderuje z last_good
    int narrow_hold_count = 0;
    // Fix 5 diagnostic: dlaczego narrow czerni mimo has_owner=true
    bool narrow_has_owner = false;
    float narrow_smooth_size = 0.0f;
    float narrow_crop_x1 = 0.0f;
    float narrow_crop_y1 = 0.0f;
    float narrow_crop_x2 = 0.0f;
    float narrow_crop_y2 = 0.0f;
    bool narrow_rendered = false;        // czy narrow_vis nie-empty po renderingu
    double inference_ms = 0.0;
    double tracker_ms = 0.0;
};

class TelemetryWriter {
public:
    explicit TelemetryWriter(const std::string& path);
    ~TelemetryWriter();

    void write(const FrameTelemetry& rec);
    void close();

private:
    std::ofstream ofs_;
};

}  // namespace dtracker
