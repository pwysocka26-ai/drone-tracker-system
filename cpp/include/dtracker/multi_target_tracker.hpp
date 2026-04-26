// MultiTargetTracker — agreguje detekcje w stabilne tracki z własnym Kalmanem.
// Port 1:1 z Python src/core/multi_target_tracker.py (zasadniczo).
//
// Camera Motion Compensation (CMC, 2026-04-26): MTT moze opcjonalnie estymowac
// globalny ruch kamery miedzy klatkami (Lucas-Kanade na background features +
// estimateAffinePartial2D) i korygowac predicted_center o ten wektor. Bez CMC
// tracker traci track przy szybkim PTZ pan (>25 deg/s -> LOCKED 50%->16%).
// Aktywacja: passuj `frame` do update(); brak frame = CMC disabled (legacy).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "dtracker/track.hpp"
#include "dtracker/types.hpp"

namespace cv { class Mat; }

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

    // CMC config -- DISABLED by default (2026-04-26 measurement: bit-perfect
    // identyczne wyniki z/bez CMC dla naszych test scenarios bo MTT max_center_distance=220
    // juz absorbuje motion ≤200 px/frame naturally. CMC kosztuje ~20 ms/klatka.
    // Wlacz cmc_enabled=true gdy realny scenariusz ma sudden snap >400 deg/s
    // (typowo: gimbal vibration na helikopterze, rapid acquisition mode).
    bool   cmc_enabled = false;
    int    cmc_max_features = 200;
    double cmc_quality_level = 0.01;
    double cmc_min_distance = 8.0;
    int    cmc_grid_step = 64;     // pomijaj features w pobliżu detected tracks
};

class MultiTargetTracker {
public:
    explicit MultiTargetTracker(MTTConfig cfg = {});
    ~MultiTargetTracker();

    void reset();

    // Update step bez CMC (legacy): tylko detekcje.
    std::vector<Track> update(const Detections& dets);

    // Update step z CMC: passuj `frame` (BGR), MTT estymuje globalny ruch
    // kamery miedzy klatkami i koryguje predicted_center kazdego trackera.
    std::vector<Track> update(const Detections& dets, const cv::Mat& frame_bgr);

    // Read-only view (kopia dla output).
    std::vector<Track> tracks() const;

    // Diag: ostatni globalny camera motion vector (pixel space).
    Point2 last_camera_motion() const { return camera_motion_; }
    int    last_camera_motion_inliers() const { return camera_motion_inliers_; }

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

    // CMC helpers
    void estimate_camera_motion_(const cv::Mat& frame_bgr, const Detections& dets);
    void apply_camera_motion_to_predictions_();

    MTTConfig cfg_;
    std::vector<Track> tracks_;
    int next_id_ = 1;

    // CMC state
    struct CmcState;
    std::unique_ptr<CmcState> cmc_;
    Point2 camera_motion_{0.0, 0.0};       // ostatni global dx, dy
    int    camera_motion_inliers_ = 0;     // ile features inliers w affine
};

}  // namespace dtracker
