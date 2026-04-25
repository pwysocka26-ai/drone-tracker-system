#include "dtracker/local_tracker.hpp"

#include <opencv2/core.hpp>

// Wymaga OpenCV contrib (opencv_tracking module). Header tracking.hpp jest
// dostepny po rebuild OpenCV z BUILD_LIST=...,tracking,...
// Gdy contrib niedostepny (DTRACKER_NO_OPENCV_CONTRIB defined), klasa
// kompiluje sie z pustym Impl -- wszystkie metody return false / no-op.
#if !defined(DTRACKER_NO_OPENCV_CONTRIB)
#  include <opencv2/tracking.hpp>
#endif

namespace dtracker {

struct LocalTargetTracker::Impl {
#if !defined(DTRACKER_NO_OPENCV_CONTRIB)
    cv::Ptr<cv::Tracker> tracker;
#endif
};

LocalTargetTracker::LocalTargetTracker(int max_lost_frames)
    : impl_(std::make_unique<Impl>()), max_lost_frames_(max_lost_frames) {}

LocalTargetTracker::~LocalTargetTracker() = default;

void LocalTargetTracker::reset() {
#if !defined(DTRACKER_NO_OPENCV_CONTRIB)
    impl_->tracker.release();
#endif
    active_ = false;
    lost_frames_ = 0;
    last_bbox_.reset();
    last_center_.reset();
    score_ = 0.0f;
}

bool LocalTargetTracker::init(const cv::Mat& frame, const BBox& roi) {
    reset();
    if (frame.empty()) return false;

#if defined(DTRACKER_NO_OPENCV_CONTRIB)
    (void)roi;
    return false;
#else
    // BBox -> cv::Rect (clamped do klatki)
    int x1 = std::max(0, static_cast<int>(roi.x1));
    int y1 = std::max(0, static_cast<int>(roi.y1));
    int x2 = std::min(frame.cols, static_cast<int>(roi.x2));
    int y2 = std::min(frame.rows, static_cast<int>(roi.y2));
    if (x2 <= x1 || y2 <= y1) return false;
    int w = x2 - x1;
    int h = y2 - y1;
    if (w < 4 || h < 4) return false;  // CSRT wymaga min ~8x8, dajemy bufor

    try {
        impl_->tracker = cv::TrackerCSRT::create();
        impl_->tracker->init(frame, cv::Rect(x1, y1, w, h));
    } catch (const cv::Exception&) {
        impl_->tracker.release();
        return false;
    }

    active_ = true;
    last_bbox_ = BBox{static_cast<float>(x1), static_cast<float>(y1),
                      static_cast<float>(x1 + w), static_cast<float>(y1 + h)};
    last_center_ = Point2{static_cast<double>(x1 + w / 2.0),
                           static_cast<double>(y1 + h / 2.0)};
    score_ = 1.0f;
    return true;
#endif
}

LocalTrackResult LocalTargetTracker::update(const cv::Mat& frame) {
    LocalTrackResult res;
    res.bbox = last_bbox_;
    res.center = last_center_;
    res.score = score_;

#if defined(DTRACKER_NO_OPENCV_CONTRIB)
    res.source = "no_contrib";
    return res;
#else
    if (!active_ || !impl_->tracker || frame.empty()) {
        res.source = "inactive";
        return res;
    }

    cv::Rect updated;
    bool ok = false;
    try {
        ok = impl_->tracker->update(frame, updated);
    } catch (const cv::Exception&) {
        ok = false;
    }

    if (!ok) {
        ++lost_frames_;
        score_ = std::max(0.0f, score_ - 0.20f);
        if (lost_frames_ > max_lost_frames_) {
            reset();
            res.source = "tracker_lost_reset";
        } else {
            res.source = "tracker_lost";
        }
        res.score = score_;
        return res;
    }

    int x = updated.x;
    int y = updated.y;
    int w = std::max(1, updated.width);
    int h = std::max(1, updated.height);
    last_bbox_ = BBox{static_cast<float>(x), static_cast<float>(y),
                      static_cast<float>(x + w), static_cast<float>(y + h)};
    last_center_ = Point2{x + w / 2.0, y + h / 2.0};
    lost_frames_ = 0;
    // Score: EMA z floorem 0.55, ceil 1.0 (parity z Python)
    score_ = std::min(1.0f, std::max(0.55f, score_ * 0.85f + 0.20f));

    res.ok = true;
    res.bbox = last_bbox_;
    res.center = last_center_;
    res.score = score_;
    res.source = "csrt";
    return res;
#endif
}

}  // namespace dtracker
