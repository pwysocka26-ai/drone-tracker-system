// LocalTargetTracker — visual CSRT fallback dla recovery.
// Port src/core/local_target_tracker.py.
//
// Wymaga OpenCV z contrib (opencv_tracking module). Buildowane lokalnie
// w third_party/opencv_contrib_build/. Gdy contrib niedostepny, klasa
// kompiluje sie ale init() zawsze zwraca false (graceful degradation).
//
// Cel:
// - lokalny visual lock na wybranym celu, niezalezny od YOLO detekcji
// - mostek dla narrow podczas dropoutow YOLO (np. dron za chmurą)
// - chroni przed switch na geometrycznie-zblizonego sasiada
#pragma once

#include <memory>
#include <optional>

#include <opencv2/core/mat.hpp>

#include "dtracker/kalman.hpp"   // Point2
#include "dtracker/types.hpp"    // BBox

namespace dtracker {

struct LocalTrackResult {
    bool ok = false;
    std::optional<BBox> bbox;
    float score = 0.0f;
    const char* source = "none";
    std::optional<Point2> center;
};

class LocalTargetTracker {
public:
    LocalTargetTracker(int max_lost_frames = 8);
    ~LocalTargetTracker();

    // Inicjalizuje CSRT na danej klatce + ROI.
    // Zwraca true gdy CSRT zainicjalizowal sie pomyslnie.
    bool init(const cv::Mat& frame, const BBox& roi);

    // Update: tracker sledzi cel w nowej klatce.
    // Zwraca LocalTrackResult z bbox (jesli ok=true) i score.
    LocalTrackResult update(const cv::Mat& frame);

    void reset();
    bool is_active() const { return active_; }
    int lost_frames() const { return lost_frames_; }
    float score() const { return score_; }
    std::optional<BBox> last_bbox() const { return last_bbox_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    bool active_ = false;
    int lost_frames_ = 0;
    int max_lost_frames_;
    std::optional<BBox> last_bbox_;
    std::optional<Point2> last_center_;
    float score_ = 0.0f;
};

}  // namespace dtracker
