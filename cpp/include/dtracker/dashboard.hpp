// Dashboard — minimalne rendering wide + narrow + overlays.
// Wykorzystuje cv::imshow. W production (Jetson bez GUI) wylaczone flagą.
#pragma once

#include <opencv2/core/mat.hpp>

#include "dtracker/lock_pipeline.hpp"
#include "dtracker/narrow_tracker.hpp"
#include "dtracker/track.hpp"

namespace dtracker {

struct DashboardConfig {
    bool show_gui = true;
    int wide_w = 960;
    int wide_h = 540;
    int narrow_w = 640;
    int narrow_h = 640;
    std::string wide_title = "wide";
    std::string narrow_title = "narrow";
};

class Dashboard {
public:
    explicit Dashboard(DashboardConfig cfg = {});
    ~Dashboard();

    // Render klatke + overlays. Zwraca klawisz jesli nacisniety (albo -1).
    int render(const cv::Mat& frame_bgr,
               const std::vector<Track>& tracks,
               int selected_id,
               LockState lock_state,
               const NarrowState& narrow_state,
               const BBox& narrow_crop);

private:
    DashboardConfig cfg_;
};

}  // namespace dtracker
