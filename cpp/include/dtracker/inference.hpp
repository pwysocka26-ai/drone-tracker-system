// YOLO ONNX inference przez ONNX Runtime z DirectML EP (AMD iGPU / NPU).
// Kompatybilny z v3 best.onnx (YOLOv8 single-class) i v4 po dotrenowaniu.
#pragma once

#include <memory>
#include <string>

#include <onnxruntime_cxx_api.h>
#include <opencv2/core/mat.hpp>

#include "dtracker/types.hpp"

namespace dtracker {

struct YoloConfig {
    std::string model_path;
    int imgsz = 640;
    float conf_threshold = 0.20f;
    float nms_iou_threshold = 0.45f;
    bool use_directml = true;  // fallback na CPU jesli false
    int directml_device_id = 0;
};

class YoloOnnxDetector : public IDetector {
public:
    explicit YoloOnnxDetector(const YoloConfig& cfg);
    ~YoloOnnxDetector() override;

    // Run inference na jednej klatce (BGR, dowolny rozmiar — auto-resize).
    // Zwraca detekcje w kordynatach ORYGINALNEJ klatki.
    Detections detect(const cv::Mat& frame) override;

    // ROI search wariant: nizszy conf threshold dla "widmowych" detekcji
    // podczas reacquire. Wywolanie z crop'em z oryginalnej klatki -- mapping
    // back do globalnych wspolrzednych zostaje na callerze (offset x1, y1).
    Detections detect_with_conf(const cv::Mat& frame, float conf_override);

    int imgsz() const override { return cfg_.imgsz; }

    // Diagnostyka: czas ostatniego inference w ms.
    double last_inference_ms() const { return last_inference_ms_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    YoloConfig cfg_;
    double last_inference_ms_ = 0.0;
};

}  // namespace dtracker
