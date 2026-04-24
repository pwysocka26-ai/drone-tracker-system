// Track struct (equivalent MultiTrackState z Pythona).
#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "dtracker/kalman.hpp"
#include "dtracker/types.hpp"

namespace dtracker {

struct Track {
    int track_id = -1;
    int raw_id = -1;                // ID z ByteTrack / detektora (jeśli dostępne)
    BBox bbox;                      // aktualny bbox (po smoothingu)
    Point2 center{0, 0};
    float confidence = 0.0f;
    Point2 velocity{0, 0};
    int age = 1;
    int hits = 1;
    int missed_frames = 0;
    bool is_confirmed = false;

    bool is_valid_target = true;
    bool is_active_target = false;

    std::vector<Point2> history;    // ostatnie N centrów
    std::optional<Point2> predicted_center;
    bool updated_this_frame = false;

    std::unique_ptr<SimpleKalman2D> kalman;

    // Deep copy (unique_ptr nie kopiuje się trywialnie).
    Track clone() const {
        Track t;
        t.track_id = track_id;
        t.raw_id = raw_id;
        t.bbox = bbox;
        t.center = center;
        t.confidence = confidence;
        t.velocity = velocity;
        t.age = age;
        t.hits = hits;
        t.missed_frames = missed_frames;
        t.is_confirmed = is_confirmed;
        t.is_valid_target = is_valid_target;
        t.is_active_target = is_active_target;
        t.history = history;
        t.predicted_center = predicted_center;
        t.updated_this_frame = updated_this_frame;
        if (kalman) t.kalman = std::make_unique<SimpleKalman2D>(kalman->clone());
        return t;
    }
};

}  // namespace dtracker
