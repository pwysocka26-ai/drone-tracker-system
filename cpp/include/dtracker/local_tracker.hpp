// LocalTargetTracker — visual CSRT fallback dla recovery.
// UWAGA (30.04 MVP): wymaga OpenCV contrib (opencv_tracking) — prebuilt Windows OpenCV
// nie zawiera contrib. Na demo ten modul jest DISABLED, Python pipeline mial go jako fallback.
// Post-demo: zbudowac OpenCV z contrib (CSRT) albo zaimplementowac template-matching proxy.
#pragma once

#include <opencv2/core/mat.hpp>
#include <optional>

#include "dtracker/types.hpp"

namespace dtracker {

struct LocalTrackResult {
    bool ok = false;
    std::optional<BBox> bbox;
    float score = 0.0f;
    const char* source = "none";
};

class LocalTargetTracker {
public:
    LocalTargetTracker() = default;

    // Stub: obecnie zwraca zawsze ok=false.
    // TODO: w OpenCV z contrib (opencv_tracking) uzyj cv::legacy::TrackerCSRT.
    bool init(const cv::Mat& frame, const BBox& roi) {
        (void)frame;
        (void)roi;
        return false;
    }

    LocalTrackResult update(const cv::Mat& frame) {
        (void)frame;
        return {};
    }

    void reset() {}
    bool is_active() const { return false; }
};

}  // namespace dtracker
