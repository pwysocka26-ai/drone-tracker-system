// POC: load ONNX, run inference na 1 klatce, dump bboxy.
#include <chrono>
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "dtracker/inference.hpp"

int main(int argc, char** argv) {
    using namespace dtracker;

    std::string model_path = "../../data/weights/v3_best.onnx";
    std::string image_path = "../../artifacts/cvat_import/obj_train_data/frame_000050.jpg";
    bool use_directml = true;

    int imgsz = 960;
    if (argc > 1) image_path = argv[1];
    if (argc > 2) model_path = argv[2];
    if (argc > 3) {
        std::string a3 = argv[3];
        if (a3 == "cpu") use_directml = false;
        else imgsz = std::atoi(argv[3]);
    }
    if (argc > 4 && std::string(argv[4]) == "cpu") use_directml = false;

    std::cout << "Model:   " << model_path << "\n";
    std::cout << "Image:   " << image_path << "\n";
    std::cout << "imgsz:   " << imgsz << "\n";
    std::cout << "DirectML: " << (use_directml ? "TAK" : "NIE") << "\n";

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "ERROR: nie udalo sie wczytac obrazu: " << image_path << "\n";
        return 1;
    }
    std::cout << "Image size: " << img.cols << "x" << img.rows << "\n";

    YoloConfig cfg;
    cfg.model_path = model_path;
    cfg.imgsz = imgsz;
    cfg.conf_threshold = 0.01f;  // very low dla debug
    cfg.nms_iou_threshold = 0.45f;
    cfg.use_directml = use_directml;

    std::cout << "Init detector..." << std::flush;
    YoloOnnxDetector det(cfg);
    std::cout << " OK\n";

    // warmup 3 runs
    std::cout << "Warmup..." << std::flush;
    for (int i = 0; i < 3; ++i) det.detect(img);
    std::cout << " OK\n";

    // benchmark 10 runs
    int n_runs = 10;
    double total_ms = 0.0;
    Detections last_dets;
    for (int i = 0; i < n_runs; ++i) {
        last_dets = det.detect(img);
        total_ms += det.last_inference_ms();
    }
    double avg_ms = total_ms / n_runs;

    std::cout << "\n=== BENCHMARK ===\n";
    std::cout << "Avg inference time: " << avg_ms << " ms (" << 1000.0 / avg_ms << " fps)\n";

    std::cout << "\n=== DETECTIONS (" << last_dets.size() << ") ===\n";
    for (size_t i = 0; i < last_dets.size(); ++i) {
        const auto& d = last_dets[i];
        std::cout << "  [" << i << "] cls=" << d.cls << " conf=" << d.conf
                  << " bbox=(" << d.bbox.x1 << "," << d.bbox.y1 << ","
                  << d.bbox.x2 << "," << d.bbox.y2 << ")"
                  << " area=" << d.bbox.area() << "\n";
    }

    // Zapisz wizualizacje
    cv::Mat vis = img.clone();
    for (const auto& d : last_dets) {
        cv::rectangle(vis,
                      cv::Point(static_cast<int>(d.bbox.x1), static_cast<int>(d.bbox.y1)),
                      cv::Point(static_cast<int>(d.bbox.x2), static_cast<int>(d.bbox.y2)),
                      cv::Scalar(0, 255, 0), 2);
        char label[64];
        snprintf(label, sizeof(label), "%.2f", d.conf);
        cv::putText(vis, label,
                    cv::Point(static_cast<int>(d.bbox.x1), static_cast<int>(d.bbox.y1) - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("poc_inference_out.jpg", vis);
    std::cout << "\nZapisano: poc_inference_out.jpg\n";

    return 0;
}
