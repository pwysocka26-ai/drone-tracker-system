// MultiTargetTracker — agreguje detekcje w stabilne tracki z własnym Kalmanem.
// Port 1:1 z Python src/core/multi_target_tracker.py (zasadniczo).
#pragma once

#include <cstdint>
#include <vector>

#include "dtracker/track.hpp"
#include "dtracker/types.hpp"

namespace dtracker {

struct MTTConfig {
    int max_missed_frames = 36;
    int confirm_hits = 2;
    float max_center_distance = 220.0f;
    float min_iou_for_match = 0.01f;
    float velocity_alpha = 0.65f;
    int history_size = 12;
    bool use_kalman = true;
    double kalman_process_noise = 0.03;
    double kalman_measurement_noise = 0.20;
};

class MultiTargetTracker {
public:
    explicit MultiTargetTracker(MTTConfig cfg = {});

    void reset();

    // Update step: dostarcz aktualne detekcje, dostaniesz listę trackow po update.
    std::vector<Track> update(const Detections& dets);

    // Read-only view (kopia dla output).
    std::vector<Track> tracks() const;

private:
    struct Match { int track_idx; int det_idx; };

    Point2 predict_center_(Track& tr) const;
    float iou_(const BBox& a, const BBox& b) const;
    BBox smooth_bbox_(const BBox& old, const BBox& fresh) const;
    float size_penalty_(const BBox& a, const BBox& b) const;
    float motion_penalty_(const Track& tr, const Detection& det) const;
    // <0 = no match
    float match_score_(const Track& tr, const Point2& predicted, const Detection& det) const;

    std::vector<Match> greedy_match_(const std::vector<Track>& tracks,
                                     const std::vector<Point2>& predicted_centers,
                                     const Detections& dets,
                                     std::vector<int>& unmatched_tracks,
                                     std::vector<int>& unmatched_dets) const;

    void apply_detection_(Track& tr, const Detection& det);
    void mark_missed_(Track& tr);
    Track spawn_(const Detection& det);

    MTTConfig cfg_;
    std::vector<Track> tracks_;
    int next_id_ = 1;
};

}  // namespace dtracker
