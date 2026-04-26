// dtracker_main: pelny pipeline end-to-end.
// video -> YOLO ONNX detect -> filter+pad -> MTT -> TM -> Lock -> Narrow ->
// dashboard + telemetry + recording.
// Parity z Python src/main.py + src/core/app.py (D6 plan).
#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

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
    // Default: FP16 ONNX (1.48x szybszy vs FP32 z zerowa utrata accuracy na v3).
    // Patrz tools/export_v3_to_onnx_fp16.py + raport benchmarka 2026-04-25.
    std::string model = "../../../data/weights/v3_best_fp16_imgsz960.onnx";
    std::string out_dir = "../../../artifacts/runs";
    bool gui = true;
    bool record = true;
    int imgsz = 960;
    float conf = 0.20f;
    int max_frames = -1;
    bool use_directml = true;
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
        if (take_int("--max-frames", a.max_frames)) continue;
        if (s == "--no-gui") { a.gui = false; continue; }
        if (s == "--no-record") { a.record = false; continue; }
        if (s == "--cpu") { a.use_directml = false; continue; }
        if (s == "-h" || s == "--help") {
            std::cout << "Usage: dtracker_main [--video PATH] [--model PATH] [--out-dir PATH]"
                      << " [--imgsz N] [--conf F] [--max-frames N] [--no-gui] [--no-record] [--cpu]\n";
            std::exit(0);
        }
    }
    return a;
}

// ====================== parse_tracks port ======================

// Filter z app.py:parse_tracks. Drops noise + maly drone padding.
// area>200, aspect 0.10..10, drop bottom 18% (samolot/dron rzadko leci nisko),
// padding 15% horizontal + 20% vertical (propellery YOLO odcina).
static Detections filter_and_pad(const Detections& raw, int frame_w, int frame_h) {
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
        if (bw < 8.0f || bh < 8.0f) continue;
        if (area < 200.0f) continue;
        if (area > max_area) continue;
        if (aspect < 0.10f || aspect > 10.0f) continue;

        float pad_w = bw * 0.15f;
        float pad_h = bh * 0.20f;
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

// Wide frame z overlays (per-track bbox, narrow crop rect, status banner) -- na
// recording. Dashboard::render robi wlasne imshow, my potrzebujemy "to-Mat" wersji.
static cv::Mat draw_wide_overlays(const cv::Mat& frame, const std::vector<Track>& tracks,
                                   int sel_id, int persistent_id,
                                   LockState lock_state, const BBox& crop,
                                   const NarrowState& nstate) {
    cv::Mat vis = frame.clone();
    cv::Scalar lock_color(0, 255, 0);
    if (lock_state == LockState::ACQUIRE)        lock_color = cv::Scalar(0, 200, 255);
    else if (lock_state == LockState::HOLD)      lock_color = cv::Scalar(0, 255, 255);
    else if (lock_state == LockState::REACQUIRE) lock_color = cv::Scalar(0, 100, 255);
    else if (lock_state == LockState::LOCKED)    lock_color = cv::Scalar(0, 255, 0);
    else                                          lock_color = cv::Scalar(120, 120, 120);

    for (const auto& t : tracks) {
        cv::Scalar col = (t.track_id == sel_id) ? lock_color : cv::Scalar(120, 120, 120);
        cv::rectangle(vis,
                      cv::Point(static_cast<int>(t.bbox.x1), static_cast<int>(t.bbox.y1)),
                      cv::Point(static_cast<int>(t.bbox.x2), static_cast<int>(t.bbox.y2)),
                      col, 2);
        std::ostringstream ss;
        // Fix 4c: "tid=" zamiast "id=" -- disambiguation z persistent owner
        // ID w bannerze (#X). Per-track label = raw track_id z MTT.
        ss << "tid=" << t.track_id << " c=" << std::fixed << std::setprecision(2) << t.confidence;
        cv::putText(vis, ss.str(),
                    cv::Point(static_cast<int>(t.bbox.x1), static_cast<int>(t.bbox.y1) - 6),
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

    MTTConfig mtt_cfg;
    mtt_cfg.max_missed_frames = 36;
    mtt_cfg.confirm_hits = 2;
    mtt_cfg.max_center_distance = 220.0f;
    mtt_cfg.velocity_alpha = 0.65f;
    MultiTargetTracker mtt(mtt_cfg);

    TargetManager tm;
    LockPipeline lock;

    // LocalTargetTracker (CSRT) jako wizualny fallback gdy YOLO traci ownera.
    // Init przy zdrowej detekcji ownera, update co klatka, fallback przy gapie.
    LocalTargetTracker local_tracker(/*max_lost_frames=*/20);
    int local_tracker_owner_id = -1;  // raw track_id na ktorym CSRT byl init'owany
    const float local_tracker_min_score = 0.55f;

    NarrowConfig narrow_cfg;
    narrow_cfg.display_center_alpha = 0.78f;
    narrow_cfg.display_size_alpha = 0.50f;
    narrow_cfg.display_max_size_step = 50.0f;
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
    cv::Mat narrow_frame;  // Phase 3: dual-camera physical narrow stream
    dtracker::io::Frame io_frame;
    dtracker::io::Frame io_frame_narrow;
    auto t_start = std::chrono::steady_clock::now();
    double total_inf_ms = 0.0;
    double total_track_ms = 0.0;
    bool quit = false;

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

    // Runtime toggles (R = recording, T = telemetry). Domyslny stan z CLI flag.
    bool recording_active = a.record;
    bool telemetry_active = true;

    while (!quit) {
        if (!source->read(io_frame) || io_frame.image.empty()) break;
        frame = io_frame.image;
        // Phase 3 dual-camera: czytaj narrow stream sync per frame index. Jesli
        // narrow short-circuits (EOF wczesniej niz wide), zatrzymaj caly pipeline
        // -- inaczej rozjedzie sie sync.
        if (dual_camera_mode) {
            if (!narrow_source->read(io_frame_narrow) || io_frame_narrow.image.empty()) {
                std::cout << "Narrow EOF at frame " << frame_idx << " -- stopping\n";
                break;
            }
            narrow_frame = io_frame_narrow.image;
        }
        if (a.max_frames > 0 && frame_idx >= a.max_frames) break;
        ++frame_idx;

        auto t_inf0 = std::chrono::steady_clock::now();
        Detections raw = detector.detect(frame);
        auto t_inf1 = std::chrono::steady_clock::now();
        double inf_ms = std::chrono::duration<double, std::milli>(t_inf1 - t_inf0).count();
        total_inf_ms += inf_ms;

        Detections filtered = filter_and_pad(raw, frame_w, frame_h);

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
                    Detections roi_filtered = filter_and_pad(roi_dets, frame_w, frame_h);
                    if (!roi_filtered.empty()) {
                        Detections merged = merge_detection_lists(filtered, roi_filtered,
                                                                   roi_merge_iou, roi_merge_center_px);
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
        float csrt_score_seen = 0.0f;
        bool owner_healthy = (owner && owner->missed_frames == 0 && owner->confidence >= 0.18f);

        // Init/re-init przy widocznym + healthy owner
        if (owner && owner->missed_frames <= 1) {
            int sid = owner->track_id;
            if (!local_tracker.is_active() || local_tracker_owner_id != sid) {
                if (local_tracker.init(frame, owner->bbox)) {
                    local_tracker_owner_id = sid;
                }
            }
        } else if (!sel && !tm.state().last_selected_center) {
            local_tracker.reset();
            local_tracker_owner_id = -1;
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

        // GUI render (cv::imshow + key)
        int key = -1;
        if (a.gui) {
            key = dashboard.render(frame, tracks, sel_id, lock_state,
                                    narrow.state(), crop);
        }

        // Telemetry (Track is move-only -> clone)
        FrameTelemetry rec;
        rec.frame_idx = frame_idx;
        rec.time_s = static_cast<double>(frame_idx) / fps;
        rec.selected_id = sel;
        rec.persistent_owner_id = tm.persistent_owner_id();    // Fix 2
        if (owner) rec.active_track = owner->clone();
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
        rec.csrt_score = csrt_score_seen;
        Point2 cm = mtt.last_camera_motion();
        rec.cmc_dx = static_cast<float>(cm.x);
        rec.cmc_dy = static_cast<float>(cm.y);
        rec.cmc_inliers = mtt.last_camera_motion_inliers();
        rec.inference_ms = inf_ms;
        rec.tracker_ms = trk_ms;
        if (telemetry_active) telemetry.write(rec);

        // VideoWriter — composite wide + narrow crop
        if (video_writer.isOpened() && recording_active) {
            cv::Mat wide_vis = draw_wide_overlays(frame, tracks, sel_id,
                                                    tm.persistent_owner_id(),
                                                    lock_state, crop, narrow.state());
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

        if (frame_idx % 30 == 0) {
            std::cout << "frame " << frame_idx << "/" << total
                      << "  lock=" << to_string(lock_state)
                      << "  owner=" << (sel ? std::to_string(*sel) : std::string("-"))
                      << "  tracks=" << tracks.size()
                      << "  inf=" << std::fixed << std::setprecision(1) << inf_ms << "ms"
                      << "\n";
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_s = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "\n=== DONE ===\n";
    std::cout << "Frames: " << frame_idx << " / " << total_s << "s = "
              << (frame_idx > 0 ? frame_idx / total_s : 0.0) << " fps\n";
    if (frame_idx > 0) {
        std::cout << "Avg inference: " << (total_inf_ms / frame_idx) << " ms\n";
        std::cout << "Avg tracker:   " << (total_track_ms / frame_idx) << " ms\n";
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
