#include "dtracker/multi_target_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>          // calcOpticalFlowPyrLK
#include <opencv2/calib3d.hpp>                  // estimateAffinePartial2D

namespace dtracker {

static constexpr float RAW_ID_BONUS = 0.60f;
static constexpr float BBOX_ALPHA = 0.72f;
static constexpr float SIZE_JUMP_LIMIT = 1.35f;

// ---------- CMC (Camera Motion Compensation) ----------
struct MultiTargetTracker::CmcState {
    cv::Mat prev_gray;
    std::vector<cv::Point2f> prev_features;
};

static Point2 bbox_center(const BBox& b) {
    return {0.5 * (b.x1 + b.x2), 0.5 * (b.y1 + b.y2)};
}

MultiTargetTracker::MultiTargetTracker(MTTConfig cfg)
    : cfg_(cfg), cmc_(std::make_unique<CmcState>()) {}

MultiTargetTracker::~MultiTargetTracker() = default;

void MultiTargetTracker::reset() {
    tracks_.clear();
    next_id_ = 1;
    if (cmc_) {
        cmc_->prev_gray.release();
        cmc_->prev_features.clear();
    }
    camera_motion_ = Point2{0.0, 0.0};
    camera_motion_inliers_ = 0;
}

std::vector<Track> MultiTargetTracker::tracks() const {
    std::vector<Track> out;
    out.reserve(tracks_.size());
    for (const auto& t : tracks_) out.push_back(t.clone());
    return out;
}

Point2 MultiTargetTracker::predict_center_(Track& tr) const {
    Point2 p;
    if (cfg_.use_kalman && tr.kalman) {
        p = tr.kalman->predict();
    } else {
        p = {tr.center.x + tr.velocity.x, tr.center.y + tr.velocity.y};
    }
    // CMC soft correction: dodaj globalny camera motion vector (jesli zmierzony).
    // NIE modyfikuje track.center/bbox/kalman -- tylko predicted_center
    // dla potrzeb matching. Po matching, normalny correct Kalman z measured
    // center (absolutne pixel coords) -- velocity over time absorbuje srednia
    // ego-motion, ale CMC daje per-frame correction dla zmian kierunku PTZ.
    if (camera_motion_inliers_ >= 15) {
        p.x += camera_motion_.x;
        p.y += camera_motion_.y;
    }
    tr.predicted_center = p;
    return p;
}

float MultiTargetTracker::iou_(const BBox& a, const BBox& b) const {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    if (inter <= 0.0f) return 0.0f;
    float aa = std::max(1.0f, (a.x2 - a.x1) * (a.y2 - a.y1));
    float bb = std::max(1.0f, (b.x2 - b.x1) * (b.y2 - b.y1));
    float uni = aa + bb - inter;
    return inter / std::max(1.0f, uni);
}

BBox MultiTargetTracker::smooth_bbox_(const BBox& old, const BBox& fresh) const {
    float ow = std::max(1.0f, old.x2 - old.x1);
    float oh = std::max(1.0f, old.y2 - old.y1);
    float nw = std::max(1.0f, fresh.x2 - fresh.x1);
    float nh = std::max(1.0f, fresh.y2 - fresh.y1);
    float alpha = BBOX_ALPHA;
    if (nw > ow * SIZE_JUMP_LIMIT || nh > oh * SIZE_JUMP_LIMIT) {
        alpha = std::max(BBOX_ALPHA, 0.84f);
    }
    return BBox{
        alpha * old.x1 + (1.0f - alpha) * fresh.x1,
        alpha * old.y1 + (1.0f - alpha) * fresh.y1,
        alpha * old.x2 + (1.0f - alpha) * fresh.x2,
        alpha * old.y2 + (1.0f - alpha) * fresh.y2,
    };
}

float MultiTargetTracker::size_penalty_(const BBox& a, const BBox& b) const {
    float wa = std::max(1.0f, a.x2 - a.x1);
    float ha = std::max(1.0f, a.y2 - a.y1);
    float wb = std::max(1.0f, b.x2 - b.x1);
    float hb = std::max(1.0f, b.y2 - b.y1);
    float dw = std::fabs(wa - wb) / std::max(wa, wb);
    float dh = std::fabs(ha - hb) / std::max(ha, hb);
    return 0.5f * (dw + dh);
}

float MultiTargetTracker::motion_penalty_(const Track& tr, const Detection& det) const {
    Point2 ref = tr.predicted_center.value_or(tr.center);
    double det_cx = 0.5 * (det.bbox.x1 + det.bbox.x2);
    double det_cy = 0.5 * (det.bbox.y1 + det.bbox.y2);
    double implied_vx = det_cx - ref.x;
    double implied_vy = det_cy - ref.y;
    double dvx = implied_vx - tr.velocity.x;
    double dvy = implied_vy - tr.velocity.y;
    double mag = std::hypot(dvx, dvy) / 45.0;
    return static_cast<float>(std::min(2.0, mag));
}

// Sentinel for "no match" -- valid scores can be negative (lower=better).
// Greedy_match filtruje to przez `s < NO_MATCH_SCORE`.
static constexpr float NO_MATCH_SCORE = std::numeric_limits<float>::max();

float MultiTargetTracker::match_score_(const Track& tr, const Point2& predicted, const Detection& det) const {
    double det_cx = 0.5 * (det.bbox.x1 + det.bbox.x2);
    double det_cy = 0.5 * (det.bbox.y1 + det.bbox.y2);
    float dist = static_cast<float>(std::hypot(predicted.x - det_cx, predicted.y - det_cy));
    if (dist > cfg_.max_center_distance) return NO_MATCH_SCORE;
    float iou = iou_(tr.bbox, det.bbox);
    float size_pen = size_penalty_(tr.bbox, det.bbox);
    float motion_pen = motion_penalty_(tr, det);
    if (iou < cfg_.min_iou_for_match && size_pen > 0.75f) return NO_MATCH_SCORE;
    float score = dist + 60.0f * size_pen + 30.0f * motion_pen - 25.0f * iou - 10.0f * det.conf;
    // raw_id bonus -- tutaj Detection nie ma raw_id z ByteTrack; pomijam w MVP
    return score;
}

std::vector<MultiTargetTracker::Match>
MultiTargetTracker::greedy_match_(const std::vector<Track>& tracks,
                                  const std::vector<Point2>& predicted_centers,
                                  const Detections& dets,
                                  std::vector<int>& unmatched_tracks,
                                  std::vector<int>& unmatched_dets) const {
    struct Pair { float score; int ti; int di; };
    std::vector<Pair> pairs;
    for (size_t ti = 0; ti < tracks.size(); ++ti) {
        for (size_t di = 0; di < dets.size(); ++di) {
            float s = match_score_(tracks[ti], predicted_centers[ti], dets[di]);
            // Bug fix 2026-04-25: score moze byc ujemny (lower=better, IoU bonus
            // moze przewazyc dist). NO_MATCH_SCORE jest jedynym sentinelem.
            if (s < NO_MATCH_SCORE) pairs.push_back({s, static_cast<int>(ti), static_cast<int>(di)});
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.score < b.score; });

    std::vector<bool> track_used(tracks.size(), false);
    std::vector<bool> det_used(dets.size(), false);
    std::vector<Match> out;
    for (const auto& p : pairs) {
        if (track_used[p.ti] || det_used[p.di]) continue;
        track_used[p.ti] = true;
        det_used[p.di] = true;
        out.push_back({p.ti, p.di});
    }
    for (size_t i = 0; i < tracks.size(); ++i)
        if (!track_used[i]) unmatched_tracks.push_back(static_cast<int>(i));
    for (size_t i = 0; i < dets.size(); ++i)
        if (!det_used[i]) unmatched_dets.push_back(static_cast<int>(i));
    return out;
}

void MultiTargetTracker::apply_detection_(Track& tr, const Detection& det) {
    Point2 old_center = tr.center;
    BBox smoothed = smooth_bbox_(tr.bbox, det.bbox);
    Point2 measured = bbox_center(smoothed);

    Point2 new_center;
    double vx = 0.0, vy = 0.0;
    if (cfg_.use_kalman) {
        if (!tr.kalman) {
            tr.kalman = std::make_unique<SimpleKalman2D>(cfg_.kalman_process_noise, cfg_.kalman_measurement_noise);
            tr.kalman->init_state(old_center.x, old_center.y, tr.velocity.x, tr.velocity.y);
        }
        Point2 corrected = tr.kalman->correct(measured.x, measured.y);
        new_center = corrected;
        Point2 v = tr.kalman->velocity();
        vx = v.x;
        vy = v.y;
    } else {
        new_center = measured;
        double measured_vx = new_center.x - old_center.x;
        double measured_vy = new_center.y - old_center.y;
        vx = cfg_.velocity_alpha * tr.velocity.x + (1.0 - cfg_.velocity_alpha) * measured_vx;
        vy = cfg_.velocity_alpha * tr.velocity.y + (1.0 - cfg_.velocity_alpha) * measured_vy;
    }
    tr.predicted_center = Point2{new_center.x + vx, new_center.y + vy};
    tr.bbox = smoothed;
    tr.center = new_center;
    tr.confidence = det.conf;
    tr.raw_id = det.cls;   // brak raw_id ByteTrack tutaj; cls jako proxy
    tr.velocity = {vx, vy};
    ++tr.age;
    ++tr.hits;
    tr.missed_frames = 0;
    tr.is_confirmed = tr.hits >= cfg_.confirm_hits;
    tr.updated_this_frame = true;

    tr.history.push_back(tr.center);
    if (static_cast<int>(tr.history.size()) > cfg_.history_size) {
        tr.history.erase(tr.history.begin(), tr.history.begin() + (tr.history.size() - cfg_.history_size));
    }
}

void MultiTargetTracker::mark_missed_(Track& tr) {
    ++tr.age;
    ++tr.missed_frames;
    Point2 old_center = tr.center;
    double dx, dy;
    if (cfg_.use_kalman && tr.kalman) {
        Point2 pred = tr.kalman->predict();
        dx = pred.x - old_center.x;
        dy = pred.y - old_center.y;
        tr.center = pred;
        tr.predicted_center = pred;
        tr.velocity = tr.kalman->velocity();
    } else {
        dx = tr.velocity.x;
        dy = tr.velocity.y;
        tr.center = {tr.center.x + dx, tr.center.y + dy};
        tr.predicted_center = tr.center;
        tr.velocity = {tr.velocity.x * 0.92, tr.velocity.y * 0.92};
    }
    tr.bbox.x1 += static_cast<float>(dx);
    tr.bbox.y1 += static_cast<float>(dy);
    tr.bbox.x2 += static_cast<float>(dx);
    tr.bbox.y2 += static_cast<float>(dy);

    tr.history.push_back(tr.center);
    if (static_cast<int>(tr.history.size()) > cfg_.history_size) {
        tr.history.erase(tr.history.begin(), tr.history.begin() + (tr.history.size() - cfg_.history_size));
    }
}

Track MultiTargetTracker::spawn_(const Detection& det) {
    Track t;
    t.track_id = next_id_++;
    t.raw_id = det.cls;
    t.bbox = det.bbox;
    t.center = bbox_center(det.bbox);
    t.confidence = det.conf;
    t.velocity = {0, 0};
    t.age = 1;
    t.hits = 1;
    t.missed_frames = 0;
    t.is_confirmed = cfg_.confirm_hits <= 1;
    t.updated_this_frame = true;
    t.history.push_back(t.center);
    t.predicted_center = t.center;
    if (cfg_.use_kalman) {
        t.kalman = std::make_unique<SimpleKalman2D>(cfg_.kalman_process_noise, cfg_.kalman_measurement_noise);
        t.kalman->init_state(t.center.x, t.center.y, 0.0, 0.0);
    }
    return t;
}

// CMC: szacuj globalny ruch kamery z optical flow background features
// (LK na features wyciagnietych poza pobliscym detected tracks). Wstrzykuj
// w predicted_center kazdego tracka aby match score nie 'pełzł' za scena.
void MultiTargetTracker::estimate_camera_motion_(const cv::Mat& frame_bgr,
                                                  const Detections& dets) {
    camera_motion_ = Point2{0.0, 0.0};
    camera_motion_inliers_ = 0;
    if (!cfg_.cmc_enabled || !cmc_ || frame_bgr.empty()) return;

    cv::Mat gray;
    cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);

    // Pierwsza klatka -- tylko zapisz, bez motion estimate
    if (cmc_->prev_gray.empty()) {
        cmc_->prev_gray = gray;
        cmc_->prev_features.clear();
        // Wyciagnij background features z pierwszej klatki -- pomijaj te w pobliżu detekcji
        cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1) * 255;
        for (const auto& d : dets) {
            int x1 = std::max(0, static_cast<int>(d.bbox.x1) - cfg_.cmc_grid_step);
            int y1 = std::max(0, static_cast<int>(d.bbox.y1) - cfg_.cmc_grid_step);
            int x2 = std::min(gray.cols, static_cast<int>(d.bbox.x2) + cfg_.cmc_grid_step);
            int y2 = std::min(gray.rows, static_cast<int>(d.bbox.y2) + cfg_.cmc_grid_step);
            if (x2 > x1 && y2 > y1) {
                mask(cv::Rect(x1, y1, x2 - x1, y2 - y1)).setTo(0);
            }
        }
        cv::goodFeaturesToTrack(gray, cmc_->prev_features,
                                 cfg_.cmc_max_features, cfg_.cmc_quality_level,
                                 cfg_.cmc_min_distance, mask);
        return;
    }

    // Mam prev_gray + prev_features. Track LK.
    if (cmc_->prev_features.size() < 8) {
        // Za malo features w prev klatce -- zlap nowe
        cmc_->prev_gray = gray;
        cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1) * 255;
        for (const auto& d : dets) {
            int x1 = std::max(0, static_cast<int>(d.bbox.x1) - cfg_.cmc_grid_step);
            int y1 = std::max(0, static_cast<int>(d.bbox.y1) - cfg_.cmc_grid_step);
            int x2 = std::min(gray.cols, static_cast<int>(d.bbox.x2) + cfg_.cmc_grid_step);
            int y2 = std::min(gray.rows, static_cast<int>(d.bbox.y2) + cfg_.cmc_grid_step);
            if (x2 > x1 && y2 > y1) mask(cv::Rect(x1, y1, x2 - x1, y2 - y1)).setTo(0);
        }
        cv::goodFeaturesToTrack(gray, cmc_->prev_features,
                                 cfg_.cmc_max_features, cfg_.cmc_quality_level,
                                 cfg_.cmc_min_distance, mask);
        return;
    }

    std::vector<cv::Point2f> next_features;
    std::vector<unsigned char> status;
    std::vector<float> err;
    try {
        cv::calcOpticalFlowPyrLK(cmc_->prev_gray, gray, cmc_->prev_features,
                                  next_features, status, err,
                                  cv::Size(21, 21), 3);
    } catch (const cv::Exception&) {
        cmc_->prev_gray = gray;
        cmc_->prev_features.clear();
        return;
    }

    std::vector<cv::Point2f> good_prev, good_next;
    good_prev.reserve(cmc_->prev_features.size());
    good_next.reserve(next_features.size());
    for (size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            good_prev.push_back(cmc_->prev_features[i]);
            good_next.push_back(next_features[i]);
        }
    }

    if (good_prev.size() >= 8) {
        std::vector<unsigned char> inliers;
        cv::Mat aff = cv::estimateAffinePartial2D(good_prev, good_next,
                                                   inliers, cv::RANSAC, 3.0);
        if (!aff.empty()) {
            // Affine 2x3 prev->next: [a, b, tx; -b, a, ty].
            // Empirycznie tx, ty = jak prev_features przesunęły sie do next.
            // To jest DOKLADNIE motion ze obiekt stat. zrobil w window coords:
            // gdy kamera pan w prawo, obiekty przesuwaja sie w LEWO w window (-tx).
            // Track predicted_center powinien byc PRZESUNIETY zgodnie z tx
            // (bo track sie tez przesunal w window coords). NIE negate.
            // Plus: clamp outliers (estimateAffine sometimes daje numeric escape)
            double tx = aff.at<double>(0, 2);
            double ty = aff.at<double>(1, 2);
            int n_inliers = 0;
            for (auto v : inliers) if (v) ++n_inliers;
            // Sanity clamp: realistyczny max camera motion ~200 px/frame
            const double CLAMP = 200.0;
            if (std::abs(tx) > CLAMP || std::abs(ty) > CLAMP || n_inliers < 15) {
                // Outlier estimate -- ignore tej klatki
                camera_motion_ = Point2{0.0, 0.0};
                camera_motion_inliers_ = 0;
            } else {
                camera_motion_ = Point2{tx, ty};
                camera_motion_inliers_ = n_inliers;
            }
        }
    }

    // Setup dla nastepnej klatki -- nowe features (re-detect kazda klatka)
    cv::Mat mask = cv::Mat::ones(gray.size(), CV_8UC1) * 255;
    for (const auto& d : dets) {
        int x1 = std::max(0, static_cast<int>(d.bbox.x1) - cfg_.cmc_grid_step);
        int y1 = std::max(0, static_cast<int>(d.bbox.y1) - cfg_.cmc_grid_step);
        int x2 = std::min(gray.cols, static_cast<int>(d.bbox.x2) + cfg_.cmc_grid_step);
        int y2 = std::min(gray.rows, static_cast<int>(d.bbox.y2) + cfg_.cmc_grid_step);
        if (x2 > x1 && y2 > y1) mask(cv::Rect(x1, y1, x2 - x1, y2 - y1)).setTo(0);
    }
    cmc_->prev_gray = gray;
    cv::goodFeaturesToTrack(gray, cmc_->prev_features,
                             cfg_.cmc_max_features, cfg_.cmc_quality_level,
                             cfg_.cmc_min_distance, mask);
}

// Stub kept for ABI -- noop (CMC stosujemy w predict_center_ zamiast).
void MultiTargetTracker::apply_camera_motion_to_predictions_() {}

std::vector<Track> MultiTargetTracker::update(const Detections& dets,
                                                const cv::Mat& frame_bgr) {
    estimate_camera_motion_(frame_bgr, dets);
    return update(dets);
}

std::vector<Track> MultiTargetTracker::update(const Detections& dets) {
    for (auto& tr : tracks_) tr.updated_this_frame = false;

    std::vector<Point2> predicted;
    predicted.reserve(tracks_.size());
    // predict_center_ aplikuje CMC correction soft (tylko predicted, nie state)
    for (auto& tr : tracks_) predicted.push_back(predict_center_(tr));

    std::vector<int> unmatched_tracks, unmatched_dets;
    auto matches = greedy_match_(tracks_, predicted, dets, unmatched_tracks, unmatched_dets);

    for (const auto& m : matches) {
        apply_detection_(tracks_[m.track_idx], dets[m.det_idx]);
    }
    for (int ti : unmatched_tracks) mark_missed_(tracks_[ti]);
    for (int di : unmatched_dets) tracks_.push_back(spawn_(dets[di]));

    // Usuń martwe
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [this](const Track& t) { return t.missed_frames > cfg_.max_missed_frames; }),
        tracks_.end());

    std::sort(tracks_.begin(), tracks_.end(),
              [](const Track& a, const Track& b) { return a.track_id < b.track_id; });

    return this->tracks();
}

}  // namespace dtracker
