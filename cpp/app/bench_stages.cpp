#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "dtracker/inference.hpp"

using dtracker::StageTimings;
using dtracker::YoloConfig;
using dtracker::YoloOnnxDetector;

struct Args {
    std::string video = "../../artifacts/test_videos/video_test_wide.mp4";
    std::string model = "../../data/weights/v7_best_fp16_imgsz1280.onnx";
    int imgsz = 1280;
    int n_frames = 200;
    int warmup = 10;
    bool use_directml = true;
    std::string mode = "sync";  // "sync" lub "async"
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto take_str = [&](const char* flag, std::string& dst) {
            if (s == flag && i + 1 < argc) { dst = argv[++i]; return true; }
            return false;
        };
        auto take_int = [&](const char* flag, int& dst) {
            if (s == flag && i + 1 < argc) { dst = std::atoi(argv[++i]); return true; }
            return false;
        };
        if (take_str("--video", a.video)) continue;
        if (take_str("--model", a.model)) continue;
        if (take_int("--imgsz", a.imgsz)) continue;
        if (take_int("--n", a.n_frames)) continue;
        if (take_int("--warmup", a.warmup)) continue;
        if (take_str("--mode", a.mode)) continue;
        if (s == "--cpu") { a.use_directml = false; continue; }
    }
    return a;
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * (v.size() - 1));
    return v[idx];
}

struct StageStats {
    const char* name;
    std::vector<double> samples;
};

static void print_stage(const StageStats& s) {
    if (s.samples.empty()) {
        std::cout << std::left << std::setw(18) << s.name << "  (no samples)\n";
        return;
    }
    double sum = 0.0, mx = 0.0, mn = s.samples[0];
    for (double v : s.samples) { sum += v; mx = std::max(mx, v); mn = std::min(mn, v); }
    double mean = sum / s.samples.size();
    double p50 = percentile(s.samples, 0.50);
    double p90 = percentile(s.samples, 0.90);
    double p99 = percentile(s.samples, 0.99);
    std::cout << std::left << std::setw(18) << s.name
              << std::right << std::fixed << std::setprecision(2)
              << "  mean=" << std::setw(7) << mean
              << "  p50=" << std::setw(7) << p50
              << "  p90=" << std::setw(7) << p90
              << "  p99=" << std::setw(7) << p99
              << "  max=" << std::setw(7) << mx
              << "  min=" << std::setw(7) << mn
              << "  n=" << s.samples.size() << "\n";
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);

    std::cout << "video:    " << a.video << "\n";
    std::cout << "model:    " << a.model << "\n";
    std::cout << "imgsz:    " << a.imgsz << "\n";
    std::cout << "frames:   " << a.n_frames << " (warmup=" << a.warmup << ")\n";
    std::cout << "EP:       " << (a.use_directml ? "DirectML" : "CPU") << "\n";
    std::cout << "mode:     " << a.mode << "\n\n";

    cv::VideoCapture cap(a.video);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: nie udalo sie otworzyc video: " << a.video << "\n";
        return 1;
    }

    int total_needed = a.warmup + a.n_frames;
    std::vector<cv::Mat> frames;
    frames.reserve(total_needed);
    cv::Mat f;
    while (frames.size() < static_cast<size_t>(total_needed) && cap.read(f)) {
        frames.push_back(f.clone());
    }
    cap.release();
    if (frames.size() < static_cast<size_t>(a.warmup + 1)) {
        std::cerr << "ERROR: za malo klatek w video (" << frames.size() << "), wymagane >="
                  << (a.warmup + 1) << "\n";
        return 1;
    }
    int actual_n = static_cast<int>(frames.size()) - a.warmup;
    std::cout << "loaded:   " << frames.size() << " klatek (bench=" << actual_n << ")\n";
    std::cout << "frame:    " << frames[0].cols << "x" << frames[0].rows << "\n\n";

    YoloConfig cfg;
    cfg.model_path = a.model;
    cfg.imgsz = a.imgsz;
    cfg.conf_threshold = 0.20f;
    cfg.nms_iou_threshold = 0.45f;
    cfg.use_directml = a.use_directml;

    std::cout << "init detector..." << std::flush;
    YoloOnnxDetector det(cfg);
    std::cout << " OK\n";

    StageStats pre{"preprocess", {}};
    StageStats tensor{"create_tensor", {}};
    StageStats run{"run (DML)", {}};
    StageStats decode{"decode", {}};
    StageStats nms_s{"nms", {}};
    StageStats unproj{"unproject", {}};
    StageStats total{"TOTAL", {}};
    StageStats cycle{"CYCLE (wallclock)", {}};

    pre.samples.reserve(actual_n);
    tensor.samples.reserve(actual_n);
    run.samples.reserve(actual_n);
    decode.samples.reserve(actual_n);
    nms_s.samples.reserve(actual_n);
    unproj.samples.reserve(actual_n);
    total.samples.reserve(actual_n);
    cycle.samples.reserve(actual_n);

    using clk = std::chrono::high_resolution_clock;

    if (a.mode == "async") {
        std::cout << "warmup " << a.warmup << " (async)..." << std::flush;
        det.enqueue(frames[0]);
        for (int i = 0; i < a.warmup; ++i) {
            if (i + 1 < a.warmup) det.enqueue(frames[i + 1]);
            else det.enqueue(frames[a.warmup]);
            det.wait_get();
        }
        std::cout << " OK\n";

        std::cout << "bench " << actual_n << " klatek (async)..." << std::flush;
        // pre-condition: po warmupie 1 frame jest in-flight (frames[a.warmup])
        for (int i = 0; i < actual_n; ++i) {
            int next_idx = a.warmup + i + 1;
            if (next_idx < static_cast<int>(frames.size())) det.enqueue(frames[next_idx]);
            auto t0 = clk::now();
            det.wait_get();
            auto t1 = clk::now();
            double cycle_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            cycle.samples.push_back(cycle_ms);
            StageTimings t = det.last_stage_timings();
            pre.samples.push_back(t.preprocess_ms);
            tensor.samples.push_back(t.create_tensor_ms);
            run.samples.push_back(t.run_ms);
            decode.samples.push_back(t.decode_ms);
            nms_s.samples.push_back(t.nms_ms);
            unproj.samples.push_back(t.unproject_ms);
            total.samples.push_back(t.total_ms);
        }
        std::cout << " OK\n\n";
    } else {
        std::cout << "warmup " << a.warmup << "..." << std::flush;
        for (int i = 0; i < a.warmup; ++i) det.detect(frames[i]);
        std::cout << " OK\n";

        std::cout << "bench " << actual_n << " klatek..." << std::flush;
        for (int i = 0; i < actual_n; ++i) {
            auto t0 = clk::now();
            det.detect(frames[a.warmup + i]);
            auto t1 = clk::now();
            double cycle_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            cycle.samples.push_back(cycle_ms);
            StageTimings t = det.last_stage_timings();
            pre.samples.push_back(t.preprocess_ms);
            tensor.samples.push_back(t.create_tensor_ms);
            run.samples.push_back(t.run_ms);
            decode.samples.push_back(t.decode_ms);
            nms_s.samples.push_back(t.nms_ms);
            unproj.samples.push_back(t.unproject_ms);
            total.samples.push_back(t.total_ms);
        }
        std::cout << " OK\n\n";
    }

    std::cout << "=== STAGE TIMINGS (ms) ===\n";
    print_stage(pre);
    print_stage(tensor);
    print_stage(run);
    print_stage(decode);
    print_stage(nms_s);
    print_stage(unproj);
    print_stage(total);
    print_stage(cycle);

    double sum_cycle = 0.0;
    for (double v : cycle.samples) sum_cycle += v;
    double mean_cycle = sum_cycle / cycle.samples.size();
    std::cout << "\nthroughput: " << std::fixed << std::setprecision(1)
              << (1000.0 / mean_cycle) << " fps (wallclock cycle, mode=" << a.mode << ")\n";

    return 0;
}
