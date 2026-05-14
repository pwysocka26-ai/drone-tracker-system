// dtracker_main: pelny pipeline end-to-end.
// video -> YOLO ONNX detect -> filter+pad -> MTT -> TM -> Lock -> Narrow ->
// dashboard + telemetry + recording.
// Parity z Python src/main.py + src/core/app.py (D6 plan).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "dtracker/angular.hpp"
#include "dtracker/gimbal_source.hpp"
#include "dtracker/dashboard.hpp"
#include "dtracker/inference.hpp"
#include "dtracker/io/file_frame_source.hpp"
#include "dtracker/io/frame_source.hpp"
#include "dtracker/local_tracker.hpp"
#include "dtracker/lock_pipeline.hpp"
#include "dtracker/multi_target_tracker.hpp"
#include "dtracker/narrow_tracker.hpp"
#include "dtracker/target_manager.hpp"
#include "dtracker/telemetry.hpp"
#include "dtracker/types.hpp"

namespace fs = std::filesystem;
using namespace dtracker;

// ====================== CLI parsing ======================

struct CliArgs {
    std::string video = "../../../artifacts/test_videos/video_test_wide_short.mp4";
    // Phase 3: dual-camera support. Jesli --video-wide + --video-narrow podane,
    // pipeline pracuje z 2 osobnymi streamami (wide=detection, narrow=refinement).
    // Backward-compat: tylko --video -> narrow generowany jako virtual crop wide.
    std::string video_wide;     // empty = use --video
    std::string video_narrow;   // empty = use virtual crop z wide
    // Default: v5 yolov8m @ imgsz=1280 FP16 (milestone 2026-04-29 small-drone).
    // A/B vs v4@640 (full 25000 klatek video_test_wide.mp4):
    //   LOCKED   24.8% -> 95.1% (+70.3 pp)
    //   lock_loss   406 -> 2    (200x lepiej)
    //   reacq succ  5.4% -> 50%
    //   inference   -    -> 26.1 ms (37 fps na DirectML iGPU Radeon 8060S)
    // v4 fallback: --model data/weights/v4_best_fp16_imgsz640.onnx --imgsz 640
    //              --min-area 200 --min-side 8.
    std::string model = "../../../data/weights/v5_best_fp16_imgsz1280.onnx";
    std::string out_dir = "../../../artifacts/runs";
    bool gui = true;
    bool record = true;
    int imgsz = 1280;
    // 2026-05-14 sesja stabilizacja: 0.20 -> 0.10. v8 na test.mp4 dawal det
    // coverage 218/428 (51%) przy 0.20, 357/428 (83%) przy 0.10. LOCKED count
    // taki sam (218), FP companions handled przez TM (id=3 ghost spawning ale
    // sel zostaje na id=2 owner). Cel: szybsze ACQUIRE i mniej track death.
    float conf = 0.10f;
    // Min bbox area (px^2) / side (px) post-NMS, pre-MTT. Defaults 25 / 4
    // dopuszczaja drony 5-15 px (area 25-225 px^2) -- niezbedne dla v5@1280.
    // Dla v4@640: 200 / 8 (legacy threshold).
    float min_area = 25.0f;
    float min_side = 4.0f;
    int max_frames = -1;
    bool use_directml = true;
    // Gimbal data (faza B: angular target position w telemetry).
    // fov_h_deg=0 (default) -> angular calc OFF. fov_h_deg>0 -> ON, wymaga
    // axis_az_mrad + axis_el_mrad (z encoderow glowicy). fov_v wyliczany
    // automatycznie z aspect ratio (square pixels assumption).
    float fov_h_deg = 0.0f;
    float axis_az_mrad = 0.0f;
    float axis_el_mrad = 0.0f;
    // Per-frame gimbal source (CSV). Priorytet: --gimbal-csv > --fov-h-deg/--axis-* > OFF.
    std::string gimbal_csv;
    // Async preprocess pipeline (Fala 1a). 1-frame lag w detection ale visual
    // alignment zachowane (display frame N-1 z detections N-1).
    // Default ON od 2026-05-13 — 20-23% szybszy cycle, sync output bit-exact.
    bool async = true;
    // 2026-05-13: Async display thread (Opcja A). Wide window renderuje source
    // klatki w real-time, tracker w osobnym watku async. Bbox moze byc N-2/N-3
    // klatek stary, ale display nie czeka na tracker = zero wizualnego lagu.
    bool display_thread = false;
};

static CliArgs parse_args(int argc, char** argv) {
    CliArgs a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto take = [&](const char* flag, std::string& dst) {
            if (s == flag && i + 1 < argc) { dst = argv[++i]; return true; }
            return false;
        };
        auto take_int = [&](const char* flag, int& dst) {
            if (s == flag && i + 1 < argc) { dst = std::atoi(argv[++i]); return true; }
            return false;
        };
        auto take_float = [&](const char* flag, float& dst) {
            if (s == flag && i + 1 < argc) { dst = static_cast<float>(std::atof(argv[++i])); return true; }
            return false;
        };
        if (take("--video", a.video)) continue;
        if (take("--video-wide", a.video_wide)) continue;
        if (take("--video-narrow", a.video_narrow)) continue;
        if (take("--model", a.model)) continue;
        if (take("--out-dir", a.out_dir)) continue;
        if (take_int("--imgsz", a.imgsz)) continue;
        if (take_float("--conf", a.conf)) continue;
        if (take_float("--min-area", a.min_area)) continue;
        if (take_float("--min-side", a.min_side)) continue;
        if (take_float("--fov-h-deg", a.fov_h_deg)) continue;
        if (take_float("--axis-az-mrad", a.axis_az_mrad)) continue;
        if (take_float("--axis-el-mrad", a.axis_el_mrad)) continue;
        if (take("--gimbal-csv", a.gimbal_csv)) continue;
        if (take_int("--max-frames", a.max_frames)) continue;
        if (s == "--no-gui") { a.gui = false; continue; }
        if (s == "--no-record") { a.record = false; continue; }
        if (s == "--cpu") { a.use_directml = false; continue; }
        if (s == "--async") { a.async = true; continue; }
        if (s == "--no-async") { a.async = false; continue; }
        if (s == "--display-thread") { a.display_thread = true; continue; }
        if (s == "-h" || s == "--help") {
            std::cout << "Usage: dtracker_main [--video PATH] [--model PATH] [--out-dir PATH]"
                      << " [--imgsz N] [--conf F] [--min-area F] [--min-side F]"
                      << " [--fov-h-deg F] [--axis-az-mrad F] [--axis-el-mrad F] [--gimbal-csv PATH]"
                      << " [--max-frames N] [--no-gui] [--no-record] [--cpu] [--no-async] [--display-thread]\n";
            std::exit(0);
        }
    }
    return a;
}

// ====================== parse_tracks port ======================

// Filter z app.py:parse_tracks. Drops noise + maly drone padding.
// area>200, aspect 0.10..10, drop bottom 18% (samolot/dron rzadko leci nisko),
// padding 15% horizontal + 20% vertical (propellery YOLO odcina).
static Detections filter_and_pad(const Detections& raw, int frame_w, int frame_h,
                                  float min_area, float min_side) {
    Detections out;
    out.reserve(raw.size());
    const float bottom_y = static_cast<float>(frame_h) * 0.82f;
    const float max_area = static_cast<float>(frame_w) * static_cast<float>(frame_h) * 0.035f;
    for (const auto& d : raw) {
        float bw = d.bbox.width();
        float bh = d.bbox.height();
        float area = bw * bh;
        float cy = d.bbox.cy();
        float aspect = bw / std::max(1.0f, bh);
        if (d.conf < 0.08f) continue;
        if (cy > bottom_y) continue;
        if (bw < min_side || bh < min_side) continue;
        if (area < min_area) continue;
        if (area > max_area) continue;
        if (aspect < 0.10f || aspect > 10.0f) continue;

        // 2026-05-13: padding 15%/20% -> 8%/10%. Kompromis: pelne 15/20 dawalo
        // (z dashboard render padding) bbox 1.7x wiekszy niz drone, pelne 0/0
        // bylo za ciasne (CSRT init bez marginu = utrata celu, narrow zoom
        // wycina propellery). 8%/10% to wystarczajacy margin bez "ramka ogromna".
        float pad_w = bw * 0.08f;
        float pad_h = bh * 0.10f;
        Detection p = d;
        p.bbox.x1 = std::max(0.0f, d.bbox.x1 - pad_w);
        p.bbox.y1 = std::max(0.0f, d.bbox.y1 - pad_h);
        p.bbox.x2 = std::min(static_cast<float>(frame_w), d.bbox.x2 + pad_w);
        p.bbox.y2 = std::min(static_cast<float>(frame_h), d.bbox.y2 + pad_h);
        out.push_back(p);
    }
    return out;
}

// ====================== ROI search (reacquire fallback) ======================
//
// Port src/core/app.py:_build_reacquire_roi + _predict_tracks_in_roi +
// _merge_track_lists. Cel: gdy YOLO traci ownera, robimy drugi inference
// na ROI 5x wokol last_good_center z nizszym conf -- znacznie wiekszy szans
// zlapania drona w trakcie HOLD/REACQUIRE.

struct RoiRect {
    int x1, y1, x2, y2;
    bool valid;
};

static RoiRect build_reacquire_roi(int frame_w, int frame_h,
                                    const Point2& ref_center,
                                    const std::optional<BBox>& ref_bbox,
                                    float expand,
                                    int min_size, int max_size) {
    float bw = 0.0f, bh = 0.0f;
    if (ref_bbox) {
        bw = ref_bbox->x2 - ref_bbox->x1;
        bh = ref_bbox->y2 - ref_bbox->y1;
    } else {
        bw = bh = std::max(40.0f, static_cast<float>(min_size) * 0.20f);
    }
    float roi_w = std::max(static_cast<float>(min_size), bw * expand);
    float roi_h = std::max(static_cast<float>(min_size), bh * expand);
    const float aspect = 16.0f / 9.0f;
    if (roi_w / std::max(1.0f, roi_h) < aspect) {
        roi_w = roi_h * aspect;
    } else {
        roi_h = roi_w / aspect;
    }
    roi_w = std::min(static_cast<float>(max_size), std::max(120.0f, roi_w));
    roi_h = std::min(static_cast<float>(max_size), std::max(120.0f, roi_h));

    int x1 = static_cast<int>(ref_center.x - roi_w * 0.5f);
    int y1 = static_cast<int>(ref_center.y - roi_h * 0.5f);
    int x2 = static_cast<int>(ref_center.x + roi_w * 0.5f);
    int y2 = static_cast<int>(ref_center.y + roi_h * 0.5f);
    x1 = std::max(0, std::min(x1, frame_w - 1));
    y1 = std::max(0, std::min(y1, frame_h - 1));
    x2 = std::max(0, std::min(x2, frame_w));
    y2 = std::max(0, std::min(y2, frame_h));
    if (x2 - x1 < 32 || y2 - y1 < 32) return {0, 0, 0, 0, false};
    return {x1, y1, x2, y2, true};
}

// Detect na crop, potem mapuj bboxy z powrotem do globalnych wspolrzednych.
static Detections detect_in_roi(YoloOnnxDetector& detector, const cv::Mat& frame,
                                 const RoiRect& roi, float conf_override) {
    if (!roi.valid) return {};
    cv::Mat crop = frame(cv::Rect(roi.x1, roi.y1, roi.x2 - roi.x1, roi.y2 - roi.y1));
    Detections crop_dets = detector.detect_with_conf(crop, conf_override);
    Detections mapped;
    mapped.reserve(crop_dets.size());
    for (auto& d : crop_dets) {
        Detection m = d;
        m.bbox.x1 += roi.x1;
        m.bbox.y1 += roi.y1;
        m.bbox.x2 += roi.x1;
        m.bbox.y2 += roi.y1;
        mapped.push_back(m);
    }
    return mapped;
}

// Merge: dodaj kandydata gdy nie duplikuje istniejacej detekcji (IoU < thresh
// AND center distance > thresh_px). Gdy duplikuje, zachowaj wyzszy conf.
static float bbox_iou_local(const BBox& a, const BBox& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float uni = a.area() + b.area() - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// Cross-class dedup detekcji w pojedynczej klatce. YOLO NMS jest per-class
// (inference.cpp:nms), wiec dwie detekcje na tym samym dronie z roznymi klasami
// nie sa scalane. Plus: per-class IoU 0.45 jest za rygorystyczne dla malych
// dronow (~14x13 px), gdzie sub-pixel jitter daje IoU 0.30-0.40 mimo ze to ten
// sam target. Empirycznie (artifacts/runs/2026-05-09_234124, analyzer):
// 12/16 spawn events to duplikat <10 px od istniejacego confirmed track.
//
// Strategia: po sortowaniu desc. po conf, suppress kazdego kandydata ktory ma
// (IoU > iou_thresh) LUB (center distance <= center_thresh_px) z ktoremkolwiek
// zachowanym. Cross-class (klasy ignorowane). Center distance jako fallback
// dla malych obiektow gdzie IoU jest noisy.
static Detections nms_dedup(const Detections& dets, float iou_thresh, float center_thresh_px) {
    if (dets.size() <= 1) return dets;
    std::vector<int> idx(dets.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return dets[a].conf > dets[b].conf; });
    std::vector<bool> keep(dets.size(), true);
    for (size_t i = 0; i < idx.size(); ++i) {
        int ii = idx[i];
        if (!keep[ii]) continue;
        for (size_t j = i + 1; j < idx.size(); ++j) {
            int jj = idx[j];
            if (!keep[jj]) continue;
            float iou_val = bbox_iou_local(dets[ii].bbox, dets[jj].bbox);
            if (iou_val > iou_thresh) { keep[jj] = false; continue; }
            float dx = dets[ii].bbox.cx() - dets[jj].bbox.cx();
            float dy = dets[ii].bbox.cy() - dets[jj].bbox.cy();
            float center_dist = std::sqrt(dx * dx + dy * dy);
            if (center_dist <= center_thresh_px) { keep[jj] = false; }
        }
    }
    Detections out;
    out.reserve(dets.size());
    for (size_t i = 0; i < dets.size(); ++i) if (keep[i]) out.push_back(dets[i]);
    return out;
}

static Detections merge_detection_lists(const Detections& primary, const Detections& secondary,
                                         float iou_thresh, float center_thresh_px) {
    Detections merged = primary;
    for (const auto& cand : secondary) {
        int dup_idx = -1;
        for (size_t i = 0; i < merged.size(); ++i) {
            if (bbox_iou_local(merged[i].bbox, cand.bbox) >= iou_thresh) { dup_idx = static_cast<int>(i); break; }
            float dx = merged[i].bbox.cx() - cand.bbox.cx();
            float dy = merged[i].bbox.cy() - cand.bbox.cy();
            if (std::sqrt(dx * dx + dy * dy) <= center_thresh_px) { dup_idx = static_cast<int>(i); break; }
        }
        if (dup_idx < 0) {
            merged.push_back(cand);
        } else if (cand.conf > merged[dup_idx].conf) {
            merged[dup_idx] = cand;
        }
    }
    return merged;
}

// ====================== composite for video ======================

// Wide left + narrow crop right, hconcat do 1920x1080.
static cv::Mat make_composite(const cv::Mat& wide, const cv::Mat& narrow,
                               int target_w, int target_h) {
    int half_w = target_w / 2;
    cv::Mat wide_resized, narrow_resized;
    cv::resize(wide, wide_resized, cv::Size(half_w, target_h));
    if (!narrow.empty()) {
        cv::resize(narrow, narrow_resized, cv::Size(target_w - half_w, target_h));
    } else {
        narrow_resized = cv::Mat::zeros(target_h, target_w - half_w, CV_8UC3);
    }
    cv::Mat out;
    cv::hconcat(wide_resized, narrow_resized, out);
    return out;
}

// ====================== utils ======================

static std::string ts_now() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H%M%S", &tm_buf);
    return buf;
}

static const char* end_state_verdict(LockState s) {
    switch (s) {
        case LockState::LOCKED:    return "LOCKED";
        case LockState::HOLD:      return "HOLD";
        case LockState::REACQUIRE: return "REACQUIRE";
        case LockState::ACQUIRE:   return "ACQUIRE";
        default:                   return "NO_OWNER";
    }
}

static void write_run_summary(const fs::path& path, int frames,
                               std::optional<int> final_owner,
                               LockState final_phase,
                               const LockPipelineState& ls) {
    std::ofstream o(path);
    o << std::fixed << std::setprecision(4);
    o << "{\n";
    o << "  \"session_duration_frames\": " << frames << ",\n";
    o << "  \"final_narrow_owner_id\": ";
    if (final_owner) o << *final_owner; else o << "null";
    o << ",\n";
    o << "  \"final_lock_phase\": \"" << to_string(final_phase) << "\",\n";
    o << "  \"end_state_verdict\": \"" << end_state_verdict(final_phase) << "\",\n";
    o << "  \"total_lock_loss_events\": " << ls.total_lock_loss_events << ",\n";
    o << "  \"total_reacquire_starts\": " << ls.total_reacquire_starts << ",\n";
    o << "  \"total_reacquire_successes\": " << ls.total_reacquire_successes << ",\n";
    o << "  \"reacquire_success_rate\": ";
    if (ls.total_reacquire_starts > 0) {
        o << (double)ls.total_reacquire_successes / (double)ls.total_reacquire_starts;
    } else {
        o << "null";
    }
    o << ",\n";
    o << "  \"total_time_in_locked_frames\": " << ls.total_frames_locked << ",\n";
    o << "  \"total_time_in_recovering_frames\": " << ls.total_frames_reacquire << ",\n";
    o << "  \"total_time_in_hold_frames\": " << ls.total_frames_hold << "\n";
    o << "}\n";
}

// Helper: dashed rectangle dla wizualizacji Kalman-drift trackow w recording
// (parity z dashboard.cpp draw_dashed_rect — anti-ghost UX).
static void draw_dashed_rect_local(cv::Mat& img, cv::Point p1, cv::Point p2,
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

// Wide frame z overlays (per-track bbox, narrow crop rect, status banner) -- na
// recording. Dashboard::render robi wlasne imshow, my potrzebujemy "to-Mat" wersji.
//
// Anti-ghost wizualne rozroznienie (parity z dashboard.cpp, ref. memory
// project_ghost_tracks_legit_signal_2026_04_27):
//   owner             -> kolor lock_state, solid grubsza
//   confirmed swieza  -> szara solid (klasyczny "candidate")
//   confirmed missed  -> szara KROPKOWANA (Kalman propaguje, brak detekcji)
//   unconfirmed       -> ciemnoszara cienka (niepewny single-hit)
static cv::Mat draw_wide_overlays(const cv::Mat& frame, const std::vector<Track>& tracks,
                                   int sel_id, int persistent_id,
                                   LockState lock_state, const BBox& crop,
                                   const NarrowState& nstate,
                                   const std::optional<AngularOffset>& angular = std::nullopt) {
    cv::Mat vis = frame.clone();
    cv::Scalar lock_color(0, 255, 0);
    if (lock_state == LockState::ACQUIRE)        lock_color = cv::Scalar(0, 200, 255);
    else if (lock_state == LockState::HOLD)      lock_color = cv::Scalar(0, 255, 255);
    else if (lock_state == LockState::REACQUIRE) lock_color = cv::Scalar(0, 100, 255);
    else if (lock_state == LockState::LOCKED)    lock_color = cv::Scalar(0, 255, 0);
    else                                          lock_color = cv::Scalar(120, 120, 120);

    for (const auto& t : tracks) {
        const bool is_owner = (t.track_id == sel_id);
        const bool is_kalman_drift = (!is_owner && t.is_confirmed && t.missed_frames > 0);
        const bool is_unconfirmed = (!is_owner && !t.is_confirmed);

        cv::Scalar col;
        int thickness;
        if (is_owner) {
            col = lock_color;
            thickness = 2;
        } else if (is_unconfirmed) {
            col = cv::Scalar(90, 90, 90);
            thickness = 1;
        } else {
            col = cv::Scalar(140, 140, 140);
            thickness = is_kalman_drift ? 1 : 2;
        }

        cv::Point p1(static_cast<int>(t.bbox.x1), static_cast<int>(t.bbox.y1));
        cv::Point p2(static_cast<int>(t.bbox.x2), static_cast<int>(t.bbox.y2));
        if (is_kalman_drift) {
            draw_dashed_rect_local(vis, p1, p2, col, thickness);
        } else {
            cv::rectangle(vis, p1, p2, col, thickness);
        }

        std::ostringstream ss;
        // Fix 4c: "tid=" zamiast "id=" -- disambiguation z persistent owner
        // ID w bannerze (#X). Per-track label = raw track_id z MTT.
        ss << "tid=" << t.track_id << " c=" << std::fixed << std::setprecision(2) << t.confidence;
        if (is_kalman_drift) ss << " K" << t.missed_frames;
        else if (is_unconfirmed) ss << " ?";
        cv::putText(vis, ss.str(),
                    cv::Point(p1.x, p1.y - 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1);
    }
    // Narrow crop rectangle: bialy gdy real owner, bialy przerywany gdy synthetic hold
    cv::Scalar crop_color = nstate.is_synthetic ? cv::Scalar(0, 255, 255) : cv::Scalar(255, 255, 255);
    cv::rectangle(vis,
                  cv::Point(static_cast<int>(crop.x1), static_cast<int>(crop.y1)),
                  cv::Point(static_cast<int>(crop.x2), static_cast<int>(crop.y2)),
                  crop_color, 1);

    std::ostringstream banner;
    banner << "lock=" << to_string(lock_state)
           << "  owner=#" << persistent_id   // Fix 2: persistent ID, nie raw track_id
           << " (tid=" << sel_id << ")"
           << "  tracks=" << tracks.size();
    if (nstate.is_synthetic) {
        banner << "  HOLD " << nstate.hold_count;
    }
    cv::putText(vis, banner.str(), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, lock_color, 2);

    // Angular target position (parity z dashboard.cpp render)
    if (angular) {
        const int x0 = vis.cols - 290;
        int y = 30;
        const auto col = cv::Scalar(0, 255, 255);
        const auto fmt = [](float v) {
            std::ostringstream os;
            os << std::fixed << std::setprecision(2) << v;
            return os.str();
        };
        cv::putText(vis, "AZ: " + fmt(angular->target_az_mrad) + " mrad  (d " + fmt(angular->delta_az_mrad) + ")",
                    cv::Point(x0, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1);
        y += 18;
        cv::putText(vis, "EL: " + fmt(angular->target_el_mrad) + " mrad  (d " + fmt(angular->delta_el_mrad) + ")",
                    cv::Point(x0, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1);
        y += 18;
        cv::putText(vis, "theta: " + fmt(angular->theta_mrad) + " mrad",
                    cv::Point(x0, y), cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1);
    }
    return vis;
}

// ====================== main ======================

int main(int argc, char** argv) {
    CliArgs a = parse_args(argc, argv);

    // Phase 2 HAL: IFrameSource zamiast cv::VideoCapture (vendor-pluggable).
    // Phase 3 dual-camera: source_wide + source_narrow (opcjonalny). Sync per
    // frame index. Backward-compat: --video (single) --> source_narrow=null,
    // narrow generowany przez virtual crop wide (legacy behavior).
    std::string wide_uri = a.video_wide.empty() ? a.video : a.video_wide;
    auto source = std::make_shared<dtracker::io::FileFrameSource>();  // wide
    if (!source->open(wide_uri)) {
        std::cerr << "FATAL: cannot open wide video: " << wide_uri << "\n";
        return 1;
    }
    const auto& src_info = source->info();
    int frame_w = src_info.width;
    int frame_h = src_info.height;
    double fps = src_info.fps;
    if (fps <= 0.0 || fps > 240.0) fps = 30.0;
    long total = src_info.total_frames;
    std::cout << "Wide:   " << frame_w << "x" << frame_h << " @ " << fps
              << " fps, " << total << " frames (codec=" << src_info.codec << ")\n";

    // Phase 3: opcjonalny narrow stream (osobna kamera vs virtual crop)
    std::shared_ptr<dtracker::io::IFrameSource> narrow_source;
    bool dual_camera_mode = !a.video_narrow.empty();
    if (dual_camera_mode) {
        narrow_source = std::make_shared<dtracker::io::FileFrameSource>();
        if (!narrow_source->open(a.video_narrow)) {
            std::cerr << "FATAL: cannot open narrow video: " << a.video_narrow << "\n";
            return 1;
        }
        const auto& ninfo = narrow_source->info();
        std::cout << "Narrow: " << ninfo.width << "x" << ninfo.height << " @ " << ninfo.fps
                  << " fps, " << ninfo.total_frames << " frames (codec=" << ninfo.codec << ")\n";
        std::cout << "MODE:   dual-camera (wide+narrow physical streams)\n";
    } else {
        std::cout << "MODE:   single-camera (narrow = virtual crop wide)\n";
    }

    std::string run_id = ts_now();
    fs::path run_dir = fs::path(a.out_dir) / run_id;
    fs::path images_dir = run_dir / "images";
    fs::path video_dir = run_dir / "video";
    fs::create_directories(images_dir);
    fs::create_directories(video_dir);
    std::cout << "Run: " << run_dir.string() << "\n";

    YoloConfig ycfg;
    ycfg.model_path = a.model;
    ycfg.imgsz = a.imgsz;
    ycfg.conf_threshold = a.conf;
    ycfg.nms_iou_threshold = 0.45f;
    ycfg.use_directml = a.use_directml;
    std::cout << "Init detector (DirectML=" << (a.use_directml ? "yes" : "no") << ")..." << std::flush;
    YoloOnnxDetector detector(ycfg);
    std::cout << " OK\n";

    // Gimbal source: CSV per-frame (priorytet) lub statyczne args.
    std::optional<GimbalCsvSource> gimbal_csv_source;
    if (!a.gimbal_csv.empty()) {
        try {
            gimbal_csv_source.emplace(a.gimbal_csv);
            std::cout << "Gimbal CSV: " << a.gimbal_csv << " ("
                      << gimbal_csv_source->size() << " entries)\n";
        } catch (const std::exception& e) {
            std::cerr << "FATAL: gimbal CSV load failed: " << e.what() << "\n";
            return 1;
        }
    } else if (a.fov_h_deg > 0.0f) {
        std::cout << "Gimbal STATIC: fov_h=" << a.fov_h_deg << " deg, axis_az="
                  << a.axis_az_mrad << " mrad, axis_el=" << a.axis_el_mrad << " mrad\n";
    }

    MTTConfig mtt_cfg;
    // 2026-05-14: probowano 80 (przeziycie sea blind spot 38 klatek), ale ghost
    // track id=1 dryfowal i blokowal spawn track id=2 -> LOCKED frame 155 vs 145
    // przy 36. Wniosek: lepiej szybka smierc + clean respawn niz dlugie ghost.
    mtt_cfg.max_missed_frames = 36;
    mtt_cfg.confirm_hits = 2;
    // 2026-05-13: 220 -> 300. Memory: drone manewr 445 px > 220. 400 bylo za duzo
    // (matching kradl detekcje sasiadow). 250 nie wystarczalo (user: dalej problem
    // ze sledzeniem). 300 to kompromis - obsluguje wiekszosc manewrow.
    mtt_cfg.max_center_distance = 300.0f;
    mtt_cfg.velocity_alpha = 0.65f;
    MultiTargetTracker mtt(mtt_cfg);

    // TM tuning (hipoteza 2, 2026-05-09): zminimalizowac ID swap przy duplikatach MTT.
    // Empirycznie: 12% klatek ma 2+ confirmed tracki, MTT spawnuje duplikaty na ten
    // sam fizyczny drone. switch_margin 0.55 -> 3.0 wymaga kandydata 4x lepszego.
    TMConfig tm_cfg;
    tm_cfg.switch_margin = 3.0f;
    TargetManager tm(tm_cfg);
    LockPipeline lock;

    // LocalTargetTracker (CSRT) jako wizualny fallback gdy YOLO traci ownera.
    // Init przy zdrowej detekcji ownera, update co klatka, fallback przy gapie.
    LocalTargetTracker local_tracker(/*max_lost_frames=*/20);
    int local_tracker_owner_id = -1;  // raw track_id na ktorym CSRT byl init'owany
    int local_tracker_init_frame = -10000;  // klatka ostatniego init (anti-drift refresh)
    const float local_tracker_min_score = 0.55f;
    // Anti-drift: forced re-init co N klatek na zdrowym ownerze. Empirycznie
    // (memory: project_csrt_drift_diagnosis_2026_04_27) gap >= 60 klatek bez
    // update'u podnosi fail rate first-update z 4% (baseline) do 11%. Refresh
    // co ~3s przy stabilnym ownerze odswieza appearance model bez kosztu lazy.
    constexpr int CSRT_REFRESH_INTERVAL = 90;
    constexpr float CSRT_REFRESH_MIN_CONF = 0.50f;

    NarrowConfig narrow_cfg;
    narrow_cfg.display_center_alpha = 0.78f;  // UWAGA: dead config, nie czytany w narrow_tracker.cpp
    narrow_cfg.display_size_alpha = 0.50f;
    narrow_cfg.display_max_size_step = 50.0f;
    // Narrow PID tuning. Empirical lag (artifacts/runs/2026-05-10_102204,
    // analyze_narrow_lag.py): stationary 0.93 px ok, ale fast manewry vel>=20
    // px/frame daje lag 8-14 px (frame 12500-12510 case). Dominujacy dlawik:
    // EMA na speed (smooth_alpha) -- kazda klatka tylko 60% nowej predkosci.
    // A1 (2026-05-10): wylaczyc EMA na speed -- speed = pelne new kazdej klatki.
    // Dla error 28 px / kp 0.5 / max_step 46: step = 14 px/frame, lag steady-
    // state spada z ~12 px do ~6-8 px dla vel=20. Stationary tracking nie
    // pogarsza sie (bbox jest stabilny dzieki MTT smooth_bbox + Kalman).
    narrow_cfg.pid_kp_active = 0.5f;
    narrow_cfg.pid_dead_zone_active = 1.0f;
    narrow_cfg.pid_smooth_alpha = 0.0f;
    NarrowTracker narrow(narrow_cfg, frame_w, frame_h);

    DashboardConfig dcfg;
    dcfg.show_gui = a.gui;
    Dashboard dashboard(dcfg);

    TelemetryWriter telemetry((run_dir / "telemetry.jsonl").string());

    cv::VideoWriter video_writer;
    const int composite_w = 1920;
    const int composite_h = 1080;
    if (a.record) {
        fs::path out_video = video_dir / "tracker_analysis.mp4";
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        video_writer.open(out_video.string(), fourcc, fps,
                          cv::Size(composite_w, composite_h));
        if (!video_writer.isOpened()) {
            std::cerr << "WARN: cannot open VideoWriter for " << out_video.string() << "\n";
        } else {
            std::cout << "REC: " << out_video.string() << "\n";
        }
    }

    int frame_idx = 0;
    cv::Mat frame;
    cv::Mat narrow_frame;
    dtracker::io::Frame io_frame;
    dtracker::io::Frame io_frame_narrow;
    dtracker::io::Frame io_frame_next;
    dtracker::io::Frame io_frame_narrow_next;
    auto t_start = std::chrono::steady_clock::now();
    double total_inf_ms = 0.0;
    double total_track_ms = 0.0;
    double total_cycle_ms = 0.0;
    std::vector<double> all_inf_ms;
    std::vector<double> all_cycle_ms;
    std::vector<double> all_read_ms;
    std::vector<double> all_post_inf_ms;
    std::vector<double> all_post_trk_ms;
    // Breakdown timers (2026-05-13 diag): split post_inf na podetapy
    std::vector<double> all_filter_roi_ms;       // t_inf1 -> t_trk0 (filter + dedup + ROI search)
    std::vector<double> all_mtt_lock_ms;         // t_trk0 -> t_mtt_tm_lock1 (MTT + TM + lock)
    std::vector<double> all_csrt_ms;             // t_mtt_tm_lock1 -> t_csrt1 (CSRT lifecycle + update)
    std::vector<double> all_narrow_ms;           // t_csrt1 -> t_trk1 (narrow.update)
    all_inf_ms.reserve(static_cast<size_t>(std::max(0, a.max_frames > 0 ? a.max_frames : 1024)));
    all_cycle_ms.reserve(all_inf_ms.capacity());
    all_read_ms.reserve(all_inf_ms.capacity());
    all_post_inf_ms.reserve(all_inf_ms.capacity());
    all_post_trk_ms.reserve(all_inf_ms.capacity());
    all_filter_roi_ms.reserve(all_inf_ms.capacity());
    all_mtt_lock_ms.reserve(all_inf_ms.capacity());
    all_csrt_ms.reserve(all_inf_ms.capacity());
    all_narrow_ms.reserve(all_inf_ms.capacity());
    bool quit = false;

    // Async pipeline pre-load (Fala 1a): pre-read frame 0, enqueue dla worker'a.
    // I/O thread odrzucony (testowany 2026-05-09): nie daje poprawy na Strix
    // Halo z 3+ threadami zamiast 2 (CPU/iGPU power budget shared).
    bool async_eof = false;
    if (a.async) {
        if (!source->read(io_frame) || io_frame.image.empty()) {
            std::cerr << "ERROR: empty source on async preload\n";
            return 1;
        }
        if (dual_camera_mode) {
            if (!narrow_source->read(io_frame_narrow) || io_frame_narrow.image.empty()) {
                std::cerr << "ERROR: narrow EOF on async preload\n";
                return 1;
            }
        }
        detector.enqueue(io_frame.image);
    }

    // ROI search (port src/core/app.py:roi_search): sekundarny YOLO inference
    // na ROI 4.8x wokol last_good_center gdy primary detection traci ownera.
    // Conf=0.06 (vs primary 0.20) -- znacznie szersza siec dla reacquire.
    int drop_streak = 0;
    const int   roi_required_drop = 1;
    const float roi_expand = 4.8f;
    const int   roi_min_size = 280;
    const int   roi_max_size = 1200;
    const float roi_conf = 0.06f;
    const float roi_merge_iou = 0.16f;
    const float roi_merge_center_px = 48.0f;

    // Pre-MTT detection dedup. Cross-class, IoU 0.45 + center 8px fallback dla
    // malych dronow. Cel: kasacja MTT duplicate spawn (12/16 spawn events
    // <10px od istniejacego confirmed track na empirical run 2026-05-09_234124).
    const float dedup_iou = 0.45f;
    const float dedup_center_px = 8.0f;

    // Runtime toggles (R = recording, T = telemetry). Domyslny stan z CLI flag.
    bool recording_active = a.record;
    bool telemetry_active = true;

    // ====================================================================
    // === DISPLAY THREAD MODE (Opcja A, 2026-05-13) ===
    // Wide window renderuje source w real-time, tracker pracuje async.
    // Bbox moze byc 1-3 klatki stary, ale display nie czeka na tracker.
    // MVP: bez ROI search, CSRT, telemetry, recording (do dolozenia po
    // weryfikacji konceptu).
    // ====================================================================
    if (a.display_thread) {
        struct DisplaySnapshot {
            int frame_idx = -1;
            std::vector<Track> tracks;
            int sel_id = -1;
            LockState lock_state = LockState::UNLOCKED;
            NarrowState narrow_state;
            BBox narrow_crop = BBox{0, 0, 0, 0};
            std::optional<AngularOffset> angular;
            int persistent_owner_id = -1;
        };
        struct QueuedFrame { int frame_idx; cv::Mat frame; };
        std::queue<QueuedFrame> work_queue;
        std::mutex queue_mtx;
        std::condition_variable queue_cv;
        std::atomic<bool> stop_worker{false};
        // MSVC nie wspiera std::atomic<std::shared_ptr<T>> trivially copyable
        // — uzywamy mutex-protected shared_ptr (read/write rzadkie, locking taniejszy
        // od kopii structu z vector<Track>).
        std::shared_ptr<DisplaySnapshot> latest_snap = std::make_shared<DisplaySnapshot>();
        latest_snap->narrow_crop = BBox{0, 0, static_cast<float>(frame_w), static_cast<float>(frame_h)};
        std::mutex snap_mtx;

        std::thread worker_thread([&]() {
            int worker_drop_streak = 0;
            while (!stop_worker.load()) {
                QueuedFrame qf;
                {
                    std::unique_lock<std::mutex> lk(queue_mtx);
                    queue_cv.wait(lk, [&]{ return !work_queue.empty() || stop_worker.load(); });
                    if (stop_worker.load() && work_queue.empty()) break;
                    qf = std::move(work_queue.front());
                    work_queue.pop();
                }

                // MVP tracker pipeline (synchronous detect, no ROI/CSRT)
                Detections raw = detector.detect(qf.frame);
                Detections filtered = filter_and_pad(raw, frame_w, frame_h, a.min_area, a.min_side);
                filtered = nms_dedup(filtered, dedup_iou, dedup_center_px);

                std::vector<Track> tracks = mtt.update(filtered, qf.frame);
                worker_drop_streak = filtered.empty() ? worker_drop_streak + 1 : 0;
                std::optional<int> sel = tm.select(tracks);
                LockState lock_state = lock.step(sel, tracks);

                const Track* owner = nullptr;
                if (sel) {
                    for (const auto& t : tracks) {
                        if (t.track_id == *sel) { owner = &t; break; }
                    }
                }
                bool is_locked = (lock_state == LockState::LOCKED);
                narrow.update(owner, is_locked);
                BBox crop = narrow.narrow_crop();

                std::optional<AngularOffset> angular_offset;
                if (owner) {
                    std::optional<GimbalSnapshot> g;
                    if (gimbal_csv_source) {
                        g = gimbal_csv_source->lookup(qf.frame_idx, frame_w, frame_h);
                    } else if (a.fov_h_deg > 0.0f) {
                        GimbalSnapshot s;
                        s.fov_h_rad = deg_to_rad(a.fov_h_deg);
                        s.fov_v_rad = fov_v_from_h(s.fov_h_rad, frame_w, frame_h);
                        s.axis_az_mrad = a.axis_az_mrad;
                        s.axis_el_mrad = a.axis_el_mrad;
                        g = s;
                    }
                    if (g) {
                        float cx = 0.5f * (owner->bbox.x1 + owner->bbox.x2);
                        float cy = 0.5f * (owner->bbox.y1 + owner->bbox.y2);
                        angular_offset = pixel_to_angular(cx, cy, frame_w, frame_h, *g);
                    }
                }

                auto snap = std::make_shared<DisplaySnapshot>();
                snap->frame_idx = qf.frame_idx;
                snap->tracks.reserve(tracks.size());
                for (const auto& t : tracks) snap->tracks.push_back(t.clone());
                snap->sel_id = sel ? *sel : -1;
                snap->lock_state = lock_state;
                snap->narrow_state = narrow.state();
                snap->narrow_crop = crop;
                snap->angular = angular_offset;
                snap->persistent_owner_id = tm.persistent_owner_id();
                {
                    std::lock_guard<std::mutex> lk(snap_mtx);
                    latest_snap = snap;
                }
            }
        });

        // === Display loop ===
        auto t_disp_start = std::chrono::steady_clock::now();
        int disp_frame_idx = 0;
        bool quit_disp = false;
        int dropped_frames = 0;
        // Pacing do source fps. Bez tego file playback leci as-fast-as-possible
        // (np. 71 fps na 30-fps source = 2.4x speed). Real-time camera ma natural
        // pacing przez blocking read, file source nie ma — sztucznie throttle.
        const double frame_period_us = (fps > 0.0) ? (1e6 / fps) : (1e6 / 30.0);
        std::cout << "\n=== DISPLAY THREAD MODE === (target " << fps << " fps)\n";
        while (!quit_disp) {
            auto t_frame0 = std::chrono::steady_clock::now();
            dtracker::io::Frame fd;
            if (!source->read(fd) || fd.image.empty()) break;
            ++disp_frame_idx;
            if (a.max_frames > 0 && disp_frame_idx >= a.max_frames) break;

            // Queue for worker (drop if backed up)
            {
                std::lock_guard<std::mutex> lk(queue_mtx);
                if (work_queue.size() < 3) {
                    work_queue.push({disp_frame_idx, fd.image.clone()});
                    queue_cv.notify_one();
                } else {
                    ++dropped_frames;
                }
            }

            std::shared_ptr<DisplaySnapshot> snap;
            {
                std::lock_guard<std::mutex> lk(snap_mtx);
                snap = latest_snap;
            }

            int key = -1;
            if (a.gui) {
                key = dashboard.render(fd.image, snap->tracks, snap->sel_id, snap->lock_state,
                                        snap->narrow_state, snap->narrow_crop, snap->angular,
                                        snap->persistent_owner_id);
            }
            if (key == 'q' || key == 'Q' || key == 27) {
                quit_disp = true;
            }

            // Pacing: spij do nastepnej target frame time (ale nie cofamy gdy behind).
            auto t_target = t_disp_start + std::chrono::microseconds(
                static_cast<long long>(disp_frame_idx * frame_period_us));
            auto now = std::chrono::steady_clock::now();
            if (now < t_target) {
                std::this_thread::sleep_until(t_target);
            }
        }

        // Cleanup worker
        stop_worker.store(true);
        queue_cv.notify_all();
        worker_thread.join();

        auto t_disp_end = std::chrono::steady_clock::now();
        double disp_s = std::chrono::duration<double>(t_disp_end - t_disp_start).count();
        std::shared_ptr<DisplaySnapshot> final_snap;
        {
            std::lock_guard<std::mutex> lk(snap_mtx);
            final_snap = latest_snap;
        }
        std::cout << "Display frames: " << disp_frame_idx << " / " << disp_s << "s = "
                  << (disp_frame_idx > 0 ? disp_frame_idx / disp_s : 0.0) << " fps (display thread)\n";
        std::cout << "Worker processed up to frame: " << final_snap->frame_idx << "\n";
        std::cout << "Dropped frames (queue full): " << dropped_frames << "\n";

        if (video_writer.isOpened()) video_writer.release();
        source->close();
        if (narrow_source) narrow_source->close();
        cv::destroyAllWindows();
        telemetry.close();
        return 0;
    }
    // ====================================================================
    // === LEGACY SINGLE-THREAD MODE (below) ===
    // ====================================================================

    while (!quit) {
        auto t_cycle0 = std::chrono::steady_clock::now();
        if (a.async) {
            // Read NEXT frame, enqueue (worker robi preprocess równolegle z wait_get).
            // Process io_frame (= prev-read), którego detekcje są in-flight.
            if (async_eof) break;
            bool has_next = source->read(io_frame_next) && !io_frame_next.image.empty();
            if (has_next && dual_camera_mode) {
                if (!narrow_source->read(io_frame_narrow_next) || io_frame_narrow_next.image.empty()) {
                    std::cout << "Narrow EOF at frame " << frame_idx << " -- stopping\n";
                    has_next = false;
                }
            }
            if (has_next) detector.enqueue(io_frame_next.image);
            else async_eof = true;
            frame = io_frame.image;
            if (dual_camera_mode) narrow_frame = io_frame_narrow.image;
        } else {
            if (!source->read(io_frame) || io_frame.image.empty()) break;
            frame = io_frame.image;
            if (dual_camera_mode) {
                if (!narrow_source->read(io_frame_narrow) || io_frame_narrow.image.empty()) {
                    std::cout << "Narrow EOF at frame " << frame_idx << " -- stopping\n";
                    break;
                }
                narrow_frame = io_frame_narrow.image;
            }
        }
        if (a.max_frames > 0 && frame_idx >= a.max_frames) break;
        ++frame_idx;

        auto t_inf0 = std::chrono::steady_clock::now();
        double read_ms = std::chrono::duration<double, std::milli>(t_inf0 - t_cycle0).count();
        Detections raw = a.async ? detector.wait_get() : detector.detect(frame);
        auto t_inf1 = std::chrono::steady_clock::now();
        double inf_ms = std::chrono::duration<double, std::milli>(t_inf1 - t_inf0).count();
        total_inf_ms += inf_ms;

        Detections filtered = filter_and_pad(raw, frame_w, frame_h, a.min_area, a.min_side);
        // Dedup po filter_and_pad: scala duplikaty YOLO (per-class NMS nie laczy
        // cross-class, plus IoU 0.45 jest noisy dla malych dronow).
        size_t pre_dedup_n = filtered.size();
        filtered = nms_dedup(filtered, dedup_iou, dedup_center_px);
        int dedup_dropped = static_cast<int>(pre_dedup_n) - static_cast<int>(filtered.size());

        // ROI search fallback: gdy primary YOLO traci wszystkie targety LUB
        // jest drop_streak >= 1, a narrow ma last_good_center, robimy 2-gi
        // inference na ROI 4.8x wokol last_good. Wyniki merge'owane z primary
        // BEFORE MTT update (zeby track lifecycle dalej dzialal naturalnie).
        bool roi_search_used = false;
        int  roi_search_added = 0;
        if (filtered.empty() || drop_streak >= roi_required_drop) {
            const auto& nstate = narrow.state();
            if (nstate.last_good_center) {
                float dyn_expand = roi_expand * (1.0f + std::min(1.2f, 0.10f * static_cast<float>(std::max(0, drop_streak))));
                RoiRect roi = build_reacquire_roi(frame_w, frame_h, *nstate.last_good_center,
                                                   nstate.last_good_bbox, dyn_expand,
                                                   roi_min_size, roi_max_size);
                if (roi.valid) {
                    Detections roi_dets = detect_in_roi(detector, frame, roi, roi_conf);
                    Detections roi_filtered = filter_and_pad(roi_dets, frame_w, frame_h, a.min_area, a.min_side);
                    if (!roi_filtered.empty()) {
                        Detections merged = merge_detection_lists(filtered, roi_filtered,
                                                                   roi_merge_iou, roi_merge_center_px);
                        // Drugi dedup po merge ROI -- merge_detection_lists ma luzne progi
                        // (IoU 0.16 / 48 px), wiec moze przepuscic mniej-overlapping kandydatow
                        // ktorzy razem stworza duplikaty w MTT.
                        size_t pre_merge_dedup = merged.size();
                        merged = nms_dedup(merged, dedup_iou, dedup_center_px);
                        dedup_dropped += static_cast<int>(pre_merge_dedup) - static_cast<int>(merged.size());
                        roi_search_added = static_cast<int>(merged.size()) - static_cast<int>(filtered.size());
                        filtered = merged;
                        roi_search_used = true;
                    }
                }
            }
        }

        auto t_trk0 = std::chrono::steady_clock::now();
        std::vector<Track> tracks = mtt.update(filtered, frame);  // CMC enabled
        drop_streak = filtered.empty() ? (drop_streak + 1) : 0;
        std::optional<int> sel = tm.select(tracks);
        LockState lock_state = lock.step(sel, tracks);
        auto t_mtt_tm_lock1 = std::chrono::steady_clock::now();

        const Track* owner = nullptr;
        if (sel) {
            for (const auto& t : tracks) {
                if (t.track_id == *sel) { owner = &t; break; }
            }
        }
        bool is_locked = (lock_state == LockState::LOCKED);

        // ---------- LocalTargetTracker (CSRT) lifecycle ----------
        // Init/re-init gdy zdrowy real owner. Lazy update: skip gdy YOLO ma
        // zdrowego ownera (missed=0, conf>=0.18). Inaczej update co klatka.
        // Fallback synthetic gdy YOLO traci ownera (port src/core/app.py:1066-1095).
        Track synthetic_csrt_owner;          // wypelniany gdy CSRT da result
        bool have_synthetic_owner = false;
        bool csrt_updated_this_frame = false;
        bool csrt_synthetic_used = false;
        bool csrt_refresh_event = false;
        float csrt_score_seen = 0.0f;
        bool owner_healthy = (owner && owner->missed_frames == 0 && owner->confidence >= 0.18f);

        // Init/re-init przy widocznym + healthy owner. Refresh przy stabilnym
        // ownerze co CSRT_REFRESH_INTERVAL klatek (anti-drift: appearance model
        // CSRT zostaje swiezy nawet gdy YOLO trzyma owner_id przez setki klatek).
        if (owner && owner->missed_frames <= 1) {
            int sid = owner->track_id;
            bool need_init = !local_tracker.is_active() || local_tracker_owner_id != sid;
            bool need_refresh = local_tracker.is_active()
                                && local_tracker_owner_id == sid
                                && owner->confidence >= CSRT_REFRESH_MIN_CONF
                                && (frame_idx - local_tracker_init_frame) >= CSRT_REFRESH_INTERVAL;
            if (need_init || need_refresh) {
                if (local_tracker.init(frame, owner->bbox)) {
                    local_tracker_owner_id = sid;
                    local_tracker_init_frame = frame_idx;
                    if (need_refresh && !need_init) csrt_refresh_event = true;
                }
            }
        } else if (!sel && !tm.state().last_selected_center) {
            local_tracker.reset();
            local_tracker_owner_id = -1;
            local_tracker_init_frame = -10000;
        }

        // Lazy CSRT: update tylko gdy YOLO degraded/missing (oszczedza ~28 ms/klatka
        // gdy YOLO trzyma zdrowego ownera). Trade-off: appearance model CSRT
        // moze byc N klatek stary gdy gap arrives, ale CSRT robust na to.
        if (!owner_healthy && local_tracker.is_active()) {
            LocalTrackResult lr = local_tracker.update(frame);
            csrt_updated_this_frame = true;
            csrt_score_seen = lr.score;
            // Synthetic fallback gdy YOLO calkiem zgubil ownera
            if (!owner && sel && lr.bbox && lr.center
                && (lr.ok || lr.score >= local_tracker_min_score)) {
                synthetic_csrt_owner.track_id = *sel;
                synthetic_csrt_owner.raw_id = local_tracker_owner_id;
                synthetic_csrt_owner.bbox = *lr.bbox;
                synthetic_csrt_owner.center = *lr.center;
                synthetic_csrt_owner.confidence = std::max(0.12f, lr.score * 0.35f);
                synthetic_csrt_owner.is_confirmed = true;
                synthetic_csrt_owner.is_active_target = true;
                synthetic_csrt_owner.missed_frames = 0;
                synthetic_csrt_owner.hits = 1;
                have_synthetic_owner = true;
                csrt_synthetic_used = true;
                owner = &synthetic_csrt_owner;
            }
        }
        // owner_healthy == true: pomijamy update -- CSRT model zostaje ostatnio init'owany

        auto t_csrt1 = std::chrono::steady_clock::now();
        narrow.update(owner, is_locked);
        auto t_trk1 = std::chrono::steady_clock::now();
        double trk_ms = std::chrono::duration<double, std::milli>(t_trk1 - t_trk0).count();
        total_track_ms += trk_ms;

        BBox crop = narrow.narrow_crop();
        int sel_id = sel ? *sel : -1;

        // Fix 5 diagnostic: czy crop bedzie renderowalny
        bool narrow_rendered_flag = false;
        {
            int dx1 = std::max(0, static_cast<int>(crop.x1));
            int dy1 = std::max(0, static_cast<int>(crop.y1));
            int dx2 = std::min(frame_w, static_cast<int>(crop.x2));
            int dy2 = std::min(frame_h, static_cast<int>(crop.y2));
            if (dx2 > dx1 && dy2 > dy1 && narrow.state().has_owner) {
                narrow_rendered_flag = true;
            }
        }

        // Angular target position (faza B+C): gimbal aktywny + mamy ownera.
        // Source priority: CSV per-frame > static args > OFF.
        // Liczymy PRZED render zeby przekazac do dashboard overlay (faza D).
        std::optional<AngularOffset> angular_offset;
        if (owner) {
            std::optional<GimbalSnapshot> g;
            if (gimbal_csv_source) {
                g = gimbal_csv_source->lookup(frame_idx, frame_w, frame_h);
            } else if (a.fov_h_deg > 0.0f) {
                GimbalSnapshot s;
                s.fov_h_rad = deg_to_rad(a.fov_h_deg);
                s.fov_v_rad = fov_v_from_h(s.fov_h_rad, frame_w, frame_h);
                s.axis_az_mrad = a.axis_az_mrad;
                s.axis_el_mrad = a.axis_el_mrad;
                g = s;
            }
            if (g) {
                float cx = 0.5f * (owner->bbox.x1 + owner->bbox.x2);
                float cy = 0.5f * (owner->bbox.y1 + owner->bbox.y2);
                angular_offset = pixel_to_angular(cx, cy, frame_w, frame_h, *g);
            }
        }

        // GUI render (cv::imshow + key)
        int key = -1;
        if (a.gui) {
            key = dashboard.render(frame, tracks, sel_id, lock_state,
                                    narrow.state(), crop, angular_offset,
                                    tm.persistent_owner_id());
        }

        // Telemetry (Track is move-only -> clone)
        FrameTelemetry rec;
        rec.frame_idx = frame_idx;
        rec.time_s = static_cast<double>(frame_idx) / fps;
        rec.selected_id = sel;
        rec.persistent_owner_id = tm.persistent_owner_id();    // Fix 2
        if (owner) rec.active_track = owner->clone();
        if (angular_offset) {
            rec.target_delta_az_mrad = angular_offset->delta_az_mrad;
            rec.target_delta_el_mrad = angular_offset->delta_el_mrad;
            rec.target_az_mrad = angular_offset->target_az_mrad;
            rec.target_el_mrad = angular_offset->target_el_mrad;
            rec.target_angular_dist_mrad = angular_offset->theta_mrad;
        }
        rec.lock_state = lock_state;
        rec.multi_tracks.reserve(tracks.size());
        for (const auto& t : tracks) rec.multi_tracks.push_back(t.clone());
        if (narrow.state().smooth_center) rec.narrow_center = narrow.state().smooth_center;
        rec.center_lock = is_locked;
        rec.narrow_synthetic_hold = narrow.state().is_synthetic;  // Fix 1
        rec.narrow_hold_count = narrow.state().hold_count;
        rec.narrow_has_owner = narrow.state().has_owner;
        rec.narrow_smooth_size = narrow.state().smooth_size;
        rec.narrow_crop_x1 = crop.x1;
        rec.narrow_crop_y1 = crop.y1;
        rec.narrow_crop_x2 = crop.x2;
        rec.narrow_crop_y2 = crop.y2;
        rec.narrow_rendered = narrow_rendered_flag;
        rec.csrt_active = local_tracker.is_active();
        rec.csrt_updated_this_frame = csrt_updated_this_frame;
        rec.csrt_synthetic_used = csrt_synthetic_used;
        rec.csrt_refresh_event = csrt_refresh_event;
        rec.csrt_score = csrt_score_seen;
        Point2 cm = mtt.last_camera_motion();
        rec.cmc_dx = static_cast<float>(cm.x);
        rec.cmc_dy = static_cast<float>(cm.y);
        rec.cmc_inliers = mtt.last_camera_motion_inliers();
        rec.inference_ms = inf_ms;
        rec.tracker_ms = trk_ms;
        rec.dedup_dropped = dedup_dropped;
        if (telemetry_active) telemetry.write(rec);

        // VideoWriter — composite wide + narrow crop
        if (video_writer.isOpened() && recording_active) {
            cv::Mat wide_vis = draw_wide_overlays(frame, tracks, sel_id,
                                                    tm.persistent_owner_id(),
                                                    lock_state, crop, narrow.state(),
                                                    angular_offset);
            cv::Mat narrow_vis;
            if (dual_camera_mode) {
                // Phase 3: narrow = physical stream z osobnej kamery (np. PTZ
                // optical zoom). Wysylamy go bezposrednio na panel, nadal pokazujac
                // synthetic-hold banner gdy narrow tracker nie ma swiezej detekcji.
                narrow_vis = narrow_frame.clone();
                if (narrow.state().has_owner && narrow.state().is_synthetic) {
                    std::ostringstream syn;
                    syn << "HOLD " << narrow.state().hold_count;
                    cv::putText(narrow_vis, syn.str(), cv::Point(10, 30),
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                }
            } else {
                int nx1 = std::max(0, static_cast<int>(crop.x1));
                int ny1 = std::max(0, static_cast<int>(crop.y1));
                int nx2 = std::min(frame_w, static_cast<int>(crop.x2));
                int ny2 = std::min(frame_h, static_cast<int>(crop.y2));
                // Fix 1: narrow.state().has_owner pozostaje true podczas synthetic hold
                // -> renderujemy crop z ostatniej dobrej pozycji zamiast czarnego ekranu
                if (nx2 > nx1 && ny2 > ny1 && narrow.state().has_owner) {
                    narrow_vis = frame(cv::Rect(nx1, ny1, nx2 - nx1, ny2 - ny1)).clone();
                    if (narrow.state().is_synthetic) {
                        // Zolty banner "HOLD N/max" zeby uzytkownik widzial ze to synthetic
                        std::ostringstream syn;
                        syn << "HOLD " << narrow.state().hold_count;
                        cv::putText(narrow_vis, syn.str(), cv::Point(10, 30),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                    }
                }
            }
            cv::Mat composite = make_composite(wide_vis, narrow_vis,
                                                composite_w, composite_h);
            video_writer.write(composite);
        }

        // Keyboard
        if (a.gui && key >= 0) {
            if (key == 27 || key == 'q' || key == 'Q') {
                quit = true;
            } else if (key == 's' || key == 'S') {
                std::ostringstream snap;
                snap << "snap_" << frame_idx << ".png";
                fs::path snap_path = images_dir / snap.str();
                cv::imwrite(snap_path.string(), frame);
                std::cout << "SHOT: " << snap_path.string() << "\n";
            } else if (key == 'r' || key == 'R') {
                recording_active = !recording_active;
                std::cout << "RECORDING " << (recording_active ? "ON" : "OFF") << "\n";
            } else if (key == 't' || key == 'T') {
                telemetry_active = !telemetry_active;
                std::cout << "TELEMETRY " << (telemetry_active ? "ON" : "OFF") << "\n";
            } else if (key == '0') {
                tm.clear_manual_lock();
                std::cout << "MANUAL LOCK CLEARED\n";
            } else if (key >= '1' && key <= '9') {
                int idx = key - '1';
                std::vector<Track> sorted;
                sorted.reserve(tracks.size());
                for (const auto& t : tracks) sorted.push_back(t.clone());
                std::sort(sorted.begin(), sorted.end(),
                          [](const Track& a, const Track& b) {
                              return a.track_id < b.track_id;
                          });
                if (idx < static_cast<int>(sorted.size())) {
                    tm.set_manual_lock(sorted[idx].track_id);
                    std::cout << "MANUAL LOCK -> id " << sorted[idx].track_id << "\n";
                }
            } else if (key == ',' || key == '.') {
                std::vector<Track> sorted;
                sorted.reserve(tracks.size());
                for (const auto& t : tracks) sorted.push_back(t.clone());
                std::sort(sorted.begin(), sorted.end(),
                          [](const Track& a, const Track& b) {
                              return a.track_id < b.track_id;
                          });
                if (!sorted.empty()) {
                    int cur_idx = 0;
                    if (sel) {
                        for (size_t i = 0; i < sorted.size(); ++i) {
                            if (sorted[i].track_id == *sel) { cur_idx = static_cast<int>(i); break; }
                        }
                    }
                    int step = (key == ',') ? -1 : 1;
                    int n = static_cast<int>(sorted.size());
                    int next_idx = ((cur_idx + step) % n + n) % n;
                    tm.set_manual_lock(sorted[next_idx].track_id);
                    std::cout << "MANUAL LOCK -> id " << sorted[next_idx].track_id << "\n";
                }
            }
        }

        // Async pipeline: shift NEXT->CURRENT for next iteration's processing.
        if (a.async && !async_eof) {
            io_frame = io_frame_next;
            if (dual_camera_mode) io_frame_narrow = io_frame_narrow_next;
        }

        auto t_cycle1 = std::chrono::steady_clock::now();
        double cycle_ms = std::chrono::duration<double, std::milli>(t_cycle1 - t_cycle0).count();
        // post_inf_ms = od konca inference do konca trackera (ROI search + filter + MTT + TM + lock + CSRT + narrow)
        double post_inf_ms = std::chrono::duration<double, std::milli>(t_trk1 - t_inf1).count();
        // post_trk_ms = od konca trackera do konca cyklu (telemetry + dashboard render + video write + shift)
        double post_trk_ms = std::chrono::duration<double, std::milli>(t_cycle1 - t_trk1).count();
        // Breakdown (2026-05-13 diag) — szukamy ktory podetap z post_inf zera p99
        double filter_roi_ms  = std::chrono::duration<double, std::milli>(t_trk0 - t_inf1).count();
        double mtt_lock_ms    = std::chrono::duration<double, std::milli>(t_mtt_tm_lock1 - t_trk0).count();
        double csrt_ms        = std::chrono::duration<double, std::milli>(t_csrt1 - t_mtt_tm_lock1).count();
        double narrow_only_ms = std::chrono::duration<double, std::milli>(t_trk1 - t_csrt1).count();
        total_cycle_ms += cycle_ms;
        all_inf_ms.push_back(inf_ms);
        all_cycle_ms.push_back(cycle_ms);
        all_read_ms.push_back(read_ms);
        all_post_inf_ms.push_back(post_inf_ms);
        all_post_trk_ms.push_back(post_trk_ms);
        all_filter_roi_ms.push_back(filter_roi_ms);
        all_mtt_lock_ms.push_back(mtt_lock_ms);
        all_csrt_ms.push_back(csrt_ms);
        all_narrow_ms.push_back(narrow_only_ms);

        if (frame_idx % 30 == 0) {
            std::cout << "frame " << frame_idx << "/" << total
                      << "  lock=" << to_string(lock_state)
                      << "  owner=" << (sel ? std::to_string(*sel) : std::string("-"))
                      << "  tracks=" << tracks.size()
                      << "  inf=" << std::fixed << std::setprecision(1) << inf_ms
                      << "ms cycle=" << std::fixed << std::setprecision(1) << cycle_ms << "ms"
                      << "\n";
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "\n=== DONE ===\n";
    std::cout << "Frames: " << frame_idx << " / " << total_s << "s = "
              << (frame_idx > 0 ? frame_idx / total_s : 0.0) << " fps\n";
    if (frame_idx > 0) {
        auto stats = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            double sum = 0.0; for (double x : v) sum += x;
            size_t n = v.size();
            return std::tuple<double,double,double,double,double>{
                v[0], v[(n*50)/100], v[(n*90)/100], v[(n*99)/100], sum / n
            };
        };
        auto print_stats = [&](const char* label, const std::vector<double>& v) {
            auto [mn, p50, p90, p99, mean] = stats(v);
            std::cout << label
                      << "  mean=" << std::fixed << std::setprecision(2) << mean
                      << "  p50=" << p50
                      << "  p90=" << p90
                      << "  p99=" << p99
                      << "  min=" << mn
                      << "  n=" << v.size() << "\n";
        };
        std::cout << "\n=== ALL FRAMES ===\n";
        print_stats("inference (ms)   ", all_inf_ms);
        print_stats("cycle     (ms)   ", all_cycle_ms);
        // Steady state: pomijamy pierwsze 1/3 (GPU warmup, file caches).
        if (frame_idx >= 30) {
            size_t skip = static_cast<size_t>(frame_idx) / 3;
            std::vector<double> inf_ss(all_inf_ms.begin() + skip, all_inf_ms.end());
            std::vector<double> cyc_ss(all_cycle_ms.begin() + skip, all_cycle_ms.end());
            std::vector<double> read_ss(all_read_ms.begin() + skip, all_read_ms.end());
            std::vector<double> postinf_ss(all_post_inf_ms.begin() + skip, all_post_inf_ms.end());
            std::vector<double> posttrk_ss(all_post_trk_ms.begin() + skip, all_post_trk_ms.end());
            std::cout << "\n=== STEADY STATE (skip first " << skip << " frames) ===\n";
            print_stats("read+enqueue (ms)", read_ss);
            print_stats("inference   (ms)", inf_ss);
            print_stats("post_inf    (ms)", postinf_ss);
            print_stats("post_trk    (ms)", posttrk_ss);
            print_stats("cycle       (ms)", cyc_ss);
            // post_inf breakdown
            std::vector<double> filter_ss(all_filter_roi_ms.begin() + skip, all_filter_roi_ms.end());
            std::vector<double> mtt_ss(all_mtt_lock_ms.begin() + skip, all_mtt_lock_ms.end());
            std::vector<double> csrt_ss(all_csrt_ms.begin() + skip, all_csrt_ms.end());
            std::vector<double> narrow_ss(all_narrow_ms.begin() + skip, all_narrow_ms.end());
            std::cout << "  -- post_inf breakdown --\n";
            print_stats("  filter+ROI", filter_ss);
            print_stats("  MTT+TM+lock", mtt_ss);
            print_stats("  CSRT       ", csrt_ss);
            print_stats("  narrow     ", narrow_ss);
            auto [mn, p50, p90, p99, mean] = stats(cyc_ss);
            std::cout << "throughput: " << (1000.0 / mean) << " fps (steady-state mean)\n";
        }
        std::cout << "Avg tracker:   " << (total_track_ms / frame_idx) << " ms\n";
        std::cout << "mode:          " << (a.async ? "ASYNC (Fala 1a)" : "SYNC")
                  << (a.async ? "  (inference = wait_get: DML + wait, preprocess overlapped)" : "") << "\n";
    }

    if (video_writer.isOpened()) video_writer.release();
    source->close();
    if (narrow_source) narrow_source->close();
    cv::destroyAllWindows();
    telemetry.close();

    fs::path summary_path = run_dir / "run_summary.json";
    write_run_summary(summary_path, frame_idx, lock.state().owner_id,
                       lock.current(), lock.state());
    std::cout << "RUN SUMMARY: " << summary_path.string() << "\n";

    return 0;
}
