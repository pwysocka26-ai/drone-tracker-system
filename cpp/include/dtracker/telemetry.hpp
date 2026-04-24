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
    std::optional<Track> active_track;
    LockState lock_state = LockState::UNLOCKED;
    std::vector<Track> multi_tracks;
    std::optional<Point2> narrow_center;
    bool center_lock = false;
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
