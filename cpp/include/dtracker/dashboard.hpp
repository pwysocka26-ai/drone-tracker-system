// Dashboard — minimalne rendering wide + narrow + overlays.
// Wykorzystuje cv::imshow. W production (Jetson bez GUI) wylaczone flagą.
#pragma once

#include <optional>
#include <unordered_map>

#include <opencv2/core/mat.hpp>

#include "dtracker/angular.hpp"
#include "dtracker/lock_pipeline.hpp"
#include "dtracker/narrow_tracker.hpp"
#include "dtracker/track.hpp"
#include "dtracker/types.hpp"

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
               const BBox& narrow_crop,
               const std::optional<AngularOffset>& angular = std::nullopt);

private:
    DashboardConfig cfg_;
    // Last frame Kalman velocity per track_id — używane do wykrywania sign flip
    // (direction change) → wtedy NIE predykujemy bbox bo Kalman vel jest stale.
    std::unordered_map<int, Point2> last_velocities_;
    // Last frame bbox center per track_id — używane do liczenia INSTANTANEOUS
    // velocity (1-frame delta) która jest mniej laggy niż Kalman vel przy
    // szybkim ruchu. Dla prediction używamy inst_vel (bardziej fresh), a sign-flip
    // detection nadal Kalman vel (mniej noise).
    std::unordered_map<int, Point2> last_centers_;
};

}  // namespace dtracker
