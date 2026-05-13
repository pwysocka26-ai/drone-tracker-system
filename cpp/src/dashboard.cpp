#include "dtracker/dashboard.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace dtracker {

// Helper: kropkowany prostokat dla wizualizacji Kalman-drift trackow.
// Sygnalizuje operatorowi "MTT propaguje pozycje, nie ma swiezej detekcji".
// Empiryka: project_ghost_tracks_legit_signal_2026_04_27 — agresywny filter
// ukrylby legit small-drone targets (id=42 case), wiec rysujemy z roznym
// stylem zamiast ukrywac.
static void draw_dashed_rect(cv::Mat& img, cv::Point p1, cv::Point p2,
                              const cv::Scalar& color, int thickness,
                              int dash_len = 6, int gap_len = 4) {
    auto draw_dashed_line = [&](cv::Point a, cv::Point b) {
        const double dx_total = static_cast<double>(b.x - a.x);
        const double dy_total = static_cast<double>(b.y - a.y);
        const double len = std::sqrt(dx_total * dx_total + dy_total * dy_total);
        if (len < 1.0) return;
        const double dx = dx_total / len;
        const double dy = dy_total / len;
        double pos = 0.0;
        bool draw = true;
        while (pos < len) {
            const double seg = draw ? dash_len : gap_len;
            const double end = std::min(pos + seg, len);
            if (draw) {
                cv::Point sa(a.x + static_cast<int>(pos * dx),
                             a.y + static_cast<int>(pos * dy));
                cv::Point sb(a.x + static_cast<int>(end * dx),
                             a.y + static_cast<int>(end * dy));
                cv::line(img, sa, sb, color, thickness);
            }
            pos = end;
            draw = !draw;
        }
    };
    cv::Point tr(p2.x, p1.y);
    cv::Point bl(p1.x, p2.y);
    draw_dashed_line(p1, tr);
    draw_dashed_line(tr, p2);
    draw_dashed_line(p2, bl);
    draw_dashed_line(bl, p1);
}

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
                       const BBox& narrow_crop,
                       const std::optional<AngularOffset>& angular,
                       int persistent_owner_id) {
    if (!cfg_.show_gui) return -1;
    if (frame_bgr.empty()) return -1;

    cv::Mat wide = frame_bgr.clone();

    // Wszystkie tracki — wizualne rozroznienie wg stanu (anti-ghost UX):
    //   owner             -> kolor lock_state, solid grubsza
    //   confirmed swieza  -> szara solid (klasyczny "candidate")
    //   confirmed missed  -> szara KROPKOWANA (Kalman propaguje, brak detekcji)
    //   unconfirmed       -> ciemnoszara cienka (niepewny single-hit)
    // Empiryka: project_ghost_tracks_legit_signal_2026_04_27 — NIE filtrujemy
    // unconfirmed bo czesto sa to legit small-drone targets z YOLO blindness.
    for (const auto& t : tracks) {
        const bool is_owner = (t.track_id == selected_id);
        const bool is_kalman_drift = (!is_owner && t.is_confirmed && t.missed_frames > 0);
        const bool is_unconfirmed = (!is_owner && !t.is_confirmed);

        cv::Scalar col;
        int thickness;
        if (is_owner) {
            col = color_for_state(lock_state);
            thickness = 2;
        } else if (is_unconfirmed) {
            col = cv::Scalar(90, 90, 90);
            thickness = 1;
        } else {
            col = cv::Scalar(140, 140, 140);
            thickness = is_kalman_drift ? 1 : 2;
        }

        // Predictive bbox: kompensuje inference latency (~26 ms) przez extrapolację
        // pozycji. Używamy INSTANTANEOUS velocity (1-frame delta z last_centers_)
        // zamiast Kalman vel (która laguje przy szybkim ruchu/acceleracji).
        // Sign-flip detection na Kalman vel (mniej noise) — gdy direction change
        // → lookahead=0.
        // 2026-05-13: lookahead 1.0 -> 2.5. Cykl ~100 ms (3 klatki @ 30 fps),
        // bbox spozniony 2-3 klatki za dronem w fazie ACQUIRE/HOLD. User: "drone
        // wyprzedza bbox". Sign-flip clamp dalej chroni przed overshoot na manewry.
        double lookahead = (is_owner || (!is_kalman_drift && !is_unconfirmed)) ? 2.5 : 0.0;
        // Centrum bbox (potrzebne do inst_vel + zachowania w mapach)
        const double cx_now = (t.bbox.x1 + t.bbox.x2) * 0.5;
        const double cy_now = (t.bbox.y1 + t.bbox.y2) * 0.5;
        // Sign flip clamp (na Kalman vel)
        if (lookahead > 0.0) {
            auto it = last_velocities_.find(t.track_id);
            if (it != last_velocities_.end()) {
                const auto& prev = it->second;
                const bool flip_x = (prev.x * t.velocity.x < 0.0) &&
                                    (std::abs(prev.x) > 1.0) && (std::abs(t.velocity.x) > 1.0);
                const bool flip_y = (prev.y * t.velocity.y < 0.0) &&
                                    (std::abs(prev.y) > 1.0) && (std::abs(t.velocity.y) > 1.0);
                if (flip_x || flip_y) {
                    lookahead = 0.0;  // Direction change — nie predykuj
                }
            }
        }
        // Instantaneous velocity (fresh, mniej laggy niż Kalman vel).
        // Fallback: Kalman vel jeśli brak last_center.
        double vel_x = t.velocity.x;
        double vel_y = t.velocity.y;
        {
            auto it = last_centers_.find(t.track_id);
            if (it != last_centers_.end()) {
                const double inst_vx = cx_now - it->second.x;
                const double inst_vy = cy_now - it->second.y;
                // Clip outliers (noisy detection bbox jitter) — clamp do +/- 30 px/frame
                vel_x = std::max(-30.0, std::min(30.0, inst_vx));
                vel_y = std::max(-30.0, std::min(30.0, inst_vy));
            }
        }
        const double dx = vel_x * lookahead;
        const double dy = vel_y * lookahead;
        // 2026-05-13: drugi padding (po filter_and_pad) -> 0%. Z v8 ciasne bboxy
        // wystarczaja, dwukrotny padding 15%/20% dawal bbox 1.7x wiekszy niz drone.
        const double pad_w = 0.0;
        const double pad_h = 0.0;
        cv::Point p1(static_cast<int>(t.bbox.x1 - pad_w + dx),
                     static_cast<int>(t.bbox.y1 - pad_h + dy));
        cv::Point p2(static_cast<int>(t.bbox.x2 + pad_w + dx),
                     static_cast<int>(t.bbox.y2 + pad_h + dy));
        if (is_kalman_drift) {
            draw_dashed_rect(wide, p1, p2, col, thickness);
        } else {
            cv::rectangle(wide, p1, p2, col, thickness);
        }

        std::ostringstream ss;
        // Owner: persistent_owner_id (stable identity z TM Fix 2). Inni: raw track_id.
        int label_id = (is_owner && persistent_owner_id >= 0) ? persistent_owner_id : t.track_id;
        ss << "id=" << label_id << " c=" << std::setprecision(2) << t.confidence;
        if (is_kalman_drift) ss << " K" << t.missed_frames;
        else if (is_unconfirmed) ss << " ?";
        cv::putText(wide, ss.str(),
                    cv::Point(p1.x, p1.y - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1);
    }

    // Zachowaj velocity (Kalman, do sign-flip) i center (do inst_vel) dla
    // następnej klatki. Czyścimy entries dla zaginionych track_id.
    {
        std::unordered_map<int, Point2> next_vels;
        std::unordered_map<int, Point2> next_centers;
        next_vels.reserve(tracks.size());
        next_centers.reserve(tracks.size());
        for (const auto& t : tracks) {
            next_vels[t.track_id] = t.velocity;
            next_centers[t.track_id] = Point2{(t.bbox.x1 + t.bbox.x2) * 0.5,
                                              (t.bbox.y1 + t.bbox.y2) * 0.5};
        }
        last_velocities_ = std::move(next_vels);
        last_centers_ = std::move(next_centers);
    }

    // Narrow crop bracketts na wide — 4 corner L-shapes (military targeting style)
    // zamiast pelnego prostokata. User feedback 2026-05-13: "biala ramka zle wyglada".
    // Brackets sa wyrazniejsze (thickness 2), mniej dominujace w widoku, i nie traca
    // krawedzi po resize 4K -> display.
    {
        cv::Point tl(static_cast<int>(narrow_crop.x1), static_cast<int>(narrow_crop.y1));
        cv::Point br(static_cast<int>(narrow_crop.x2), static_cast<int>(narrow_crop.y2));
        const int box_w = br.x - tl.x;
        const int box_h = br.y - tl.y;
        const int len = std::max(20, std::min(box_w, box_h) / 6);  // 1/6 box size, min 20 px
        const cv::Scalar bracket_col(255, 255, 255);
        const int bracket_thick = 2;
        cv::Point tr(br.x, tl.y);
        cv::Point bl(tl.x, br.y);
        // top-left
        cv::line(wide, tl, cv::Point(tl.x + len, tl.y), bracket_col, bracket_thick);
        cv::line(wide, tl, cv::Point(tl.x, tl.y + len), bracket_col, bracket_thick);
        // top-right
        cv::line(wide, tr, cv::Point(tr.x - len, tr.y), bracket_col, bracket_thick);
        cv::line(wide, tr, cv::Point(tr.x, tr.y + len), bracket_col, bracket_thick);
        // bottom-left
        cv::line(wide, bl, cv::Point(bl.x + len, bl.y), bracket_col, bracket_thick);
        cv::line(wide, bl, cv::Point(bl.x, bl.y - len), bracket_col, bracket_thick);
        // bottom-right
        cv::line(wide, br, cv::Point(br.x - len, br.y), bracket_col, bracket_thick);
        cv::line(wide, br, cv::Point(br.x, br.y - len), bracket_col, bracket_thick);
    }

    // Status text — owner pokazuje persistent_owner_id (stable mimo MTT raw swap)
    std::ostringstream status;
    int status_owner_id = (persistent_owner_id >= 0) ? persistent_owner_id : selected_id;
    status << "lock=" << to_string(lock_state) << "  owner=" << status_owner_id
           << "  tracks=" << tracks.size();
    cv::putText(wide, status.str(), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, color_for_state(lock_state), 2);

    // Angular target position (gimbal opt-in) — prawy gorny rog.
    if (angular) {
        const int x0 = wide.cols - 290;
        int y = 30;
        const auto col = cv::Scalar(0, 255, 255);  // yellow
        const auto fmt = [](float v) {
            std::ostringstream os;
            os << std::fixed << std::setprecision(2) << v;
            return os.str();
        };
        cv::putText(wide, "AZ: " + fmt(angular->target_az_mrad) + " mrad  (d " + fmt(angular->delta_az_mrad) + ")",
                    cv::Point(x0, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1);
        y += 18;
        cv::putText(wide, "EL: " + fmt(angular->target_el_mrad) + " mrad  (d " + fmt(angular->delta_el_mrad) + ")",
                    cv::Point(x0, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1);
        y += 18;
        cv::putText(wide, "theta: " + fmt(angular->theta_mrad) + " mrad",
                    cv::Point(x0, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1);
    }

    cv::imshow(cfg_.wide_title, wide);

    // Narrow crop z oryginalnej klatki
    // 2026-05-13 bugfix: namedWindow w init tworzylo puste biale okno przy starcie.
    // Bez owner'a imshow nigdy nie byl wywolany -> bialy placeholder OpenCV widoczny
    // jako "gigantyczne biale okno" w pierwszych klatkach i przy zgubieniu celu.
    // Fix: czarny placeholder z "NO LOCK" tekstem zamiast pustego okna.
    bool drew_narrow = false;
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
            drew_narrow = true;
        }
    }
    if (!drew_narrow) {
        cv::Mat placeholder = cv::Mat::zeros(cfg_.narrow_h, cfg_.narrow_w, CV_8UC3);
        cv::putText(placeholder, "NO LOCK",
                    cv::Point(cfg_.narrow_w / 2 - 80, cfg_.narrow_h / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(80, 80, 80), 2);
        cv::imshow(cfg_.narrow_title, placeholder);
    }

    int key = cv::waitKey(1);
    return key;
}

}  // namespace dtracker
