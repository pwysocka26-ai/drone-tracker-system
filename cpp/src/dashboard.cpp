#include "dtracker/dashboard.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace dtracker {

Dashboard::Dashboard(DashboardConfig cfg) : cfg_(cfg) {
    if (cfg_.show_gui) {
        cv::namedWindow(cfg_.wide_title, cv::WINDOW_NORMAL);
        cv::resizeWindow(cfg_.wide_title, cfg_.wide_w, cfg_.wide_h);
        cv::namedWindow(cfg_.narrow_title, cv::WINDOW_NORMAL);
        cv::resizeWindow(cfg_.narrow_title, cfg_.narrow_w, cfg_.narrow_h);
    }
}

Dashboard::~Dashboard() {
    if (cfg_.show_gui) {
        try {
            cv::destroyAllWindows();
        } catch (...) {}
    }
}

static cv::Scalar color_for_state(LockState s) {
    switch (s) {
        case LockState::LOCKED:    return cv::Scalar(0, 255, 0);
        case LockState::ACQUIRE:   return cv::Scalar(0, 200, 255);
        case LockState::HOLD:      return cv::Scalar(0, 255, 255);
        case LockState::REACQUIRE: return cv::Scalar(0, 100, 255);
        default:                   return cv::Scalar(120, 120, 120);
    }
}

int Dashboard::render(const cv::Mat& frame_bgr,
                       const std::vector<Track>& tracks,
                       int selected_id,
                       LockState lock_state,
                       const NarrowState& narrow_state,
                       const BBox& narrow_crop) {
    if (!cfg_.show_gui) return -1;
    if (frame_bgr.empty()) return -1;

    cv::Mat wide = frame_bgr.clone();

    // Wszystkie tracki — szare bboxy
    for (const auto& t : tracks) {
        cv::Scalar col(120, 120, 120);
        if (t.track_id == selected_id) col = color_for_state(lock_state);
        cv::rectangle(wide,
                      cv::Point(static_cast<int>(t.bbox.x1), static_cast<int>(t.bbox.y1)),
                      cv::Point(static_cast<int>(t.bbox.x2), static_cast<int>(t.bbox.y2)),
                      col, 2);
        std::ostringstream ss;
        ss << "id=" << t.track_id << " c=" << std::setprecision(2) << t.confidence;
        cv::putText(wide, ss.str(),
                    cv::Point(static_cast<int>(t.bbox.x1), static_cast<int>(t.bbox.y1) - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1);
    }

    // Narrow crop rectangle na wide
    cv::rectangle(wide,
                  cv::Point(static_cast<int>(narrow_crop.x1), static_cast<int>(narrow_crop.y1)),
                  cv::Point(static_cast<int>(narrow_crop.x2), static_cast<int>(narrow_crop.y2)),
                  cv::Scalar(255, 255, 255), 1);

    // Status text
    std::ostringstream status;
    status << "lock=" << to_string(lock_state) << "  owner=" << selected_id
           << "  tracks=" << tracks.size();
    cv::putText(wide, status.str(), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, color_for_state(lock_state), 2);

    cv::imshow(cfg_.wide_title, wide);

    // Narrow crop z oryginalnej klatki
    if (narrow_state.has_owner) {
        int x1 = std::max(0, static_cast<int>(narrow_crop.x1));
        int y1 = std::max(0, static_cast<int>(narrow_crop.y1));
        int x2 = std::min(frame_bgr.cols, static_cast<int>(narrow_crop.x2));
        int y2 = std::min(frame_bgr.rows, static_cast<int>(narrow_crop.y2));
        if (x2 > x1 && y2 > y1) {
            cv::Mat narrow = frame_bgr(cv::Rect(x1, y1, x2 - x1, y2 - y1));
            cv::Mat narrow_resized;
            cv::resize(narrow, narrow_resized, cv::Size(cfg_.narrow_w, cfg_.narrow_h), 0, 0, cv::INTER_LINEAR);
            cv::imshow(cfg_.narrow_title, narrow_resized);
        }
    }

    int key = cv::waitKey(1);
    return key;
}

}  // namespace dtracker
