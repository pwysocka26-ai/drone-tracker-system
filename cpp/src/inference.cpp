#include "dtracker/inference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <dml_provider_factory.h>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace dtracker {

// Konwersja std::string -> std::wstring (wymagane przez Ort::Session na Windows).
static std::wstring widen(const std::string& s) {
    return std::wstring(s.begin(), s.end());
}

struct YoloOnnxDetector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dtracker"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> input_shapes;  // cache z modelu
    std::vector<std::vector<int64_t>> output_shapes;
};

YoloOnnxDetector::YoloOnnxDetector(const YoloConfig& cfg) : impl_(std::make_unique<Impl>()), cfg_(cfg) {
    auto& opts = impl_->session_options;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    opts.SetIntraOpNumThreads(1);  // DirectML pcha rownolegle; CPU fallback uzyje intra

    if (cfg_.use_directml) {
        // AMD iGPU (Radeon 8060S) albo Intel iGPU albo NVIDIA — DirectML agnostic.
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(opts, cfg_.directml_device_id));
    }
    // CPU fallback zawsze appendowany automatycznie przez ORT jesli DirectML nie akceptuje node'a.

    std::wstring wpath = widen(cfg_.model_path);
    impl_->session = std::make_unique<Ort::Session>(impl_->env, wpath.c_str(), opts);

    // Pobierz nazwy wejsc/wyjsc
    size_t n_in = impl_->session->GetInputCount();
    size_t n_out = impl_->session->GetOutputCount();
    for (size_t i = 0; i < n_in; ++i) {
        auto name = impl_->session->GetInputNameAllocated(i, impl_->allocator);
        impl_->input_names.emplace_back(name.get());
        auto shape = impl_->session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        impl_->input_shapes.push_back(shape);
    }
    for (size_t i = 0; i < n_out; ++i) {
        auto name = impl_->session->GetOutputNameAllocated(i, impl_->allocator);
        impl_->output_names.emplace_back(name.get());
        auto shape = impl_->session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        impl_->output_shapes.push_back(shape);
    }
    if (impl_->input_names.empty() || impl_->output_names.empty()) {
        throw std::runtime_error("YoloOnnxDetector: model bez wejsc lub wyjsc");
    }
}

YoloOnnxDetector::~YoloOnnxDetector() = default;

// Preprocess: BGR -> letterbox pad -> RGB -> /255 -> CHW float32 (manual).
// Korzystamy bezposrednio z blobFromImage: BGR uint8 + swapRB=true + scalefactor=1/255.
static void preprocess(const cv::Mat& bgr, int imgsz, cv::Mat& out_blob,
                       float& scale, int& pad_x, int& pad_y) {
    int h = bgr.rows, w = bgr.cols;
    scale = std::min(static_cast<float>(imgsz) / h, static_cast<float>(imgsz) / w);
    int new_w = static_cast<int>(std::round(w * scale));
    int new_h = static_cast<int>(std::round(h * scale));
    pad_x = (imgsz - new_w) / 2;
    pad_y = (imgsz - new_h) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // letterbox (padding szarym 114 zgodnie z YOLOv8 default)
    cv::Mat padded(imgsz, imgsz, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

    // blobFromImage: BGR uint8 -> scale /255 -> swapRB -> CHW float32 batch
    // (Ultralytics preprocess robi dokladnie to w Python)
    out_blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0, cv::Size(imgsz, imgsz),
                                       cv::Scalar(), true /*swapRB*/, false /*crop*/);
}

// Compute IoU
static float iou(const BBox& a, const BBox& b) {
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

// Non-maximum suppression per class
static Detections nms(const Detections& dets, float iou_thresh) {
    Detections out;
    std::vector<int> idx(dets.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
    // sort by conf desc
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return dets[a].conf > dets[b].conf; });
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < idx.size(); ++i) {
        int ii = idx[i];
        if (suppressed[ii]) continue;
        out.push_back(dets[ii]);
        for (size_t j = i + 1; j < idx.size(); ++j) {
            int jj = idx[j];
            if (suppressed[jj]) continue;
            if (dets[ii].cls != dets[jj].cls) continue;
            if (iou(dets[ii].bbox, dets[jj].bbox) > iou_thresh) suppressed[jj] = true;
        }
    }
    return out;
}

// Decode YOLOv8 output tensor. Output shape: [1, num_classes+4, num_anchors]
// num_anchors dla imgsz=640: 8400 (80*80 + 40*40 + 20*20)
// Channels: [cx, cy, w, h, cls0_score, cls1_score, ...]
static Detections decode_yolov8(const float* data, const std::vector<int64_t>& shape,
                                 int imgsz, float conf_thresh) {
    if (shape.size() != 3) throw std::runtime_error("decode: oczekuje 3D tensor");
    int64_t batch = shape[0];
    int64_t dims = shape[1];     // 4 + num_classes
    int64_t n = shape[2];        // num_anchors
    if (batch != 1) throw std::runtime_error("decode: batch != 1");
    int num_classes = static_cast<int>(dims - 4);
    if (num_classes <= 0) throw std::runtime_error("decode: num_classes <= 0");

    Detections dets;
    dets.reserve(128);
    for (int64_t i = 0; i < n; ++i) {
        // data layout: [cx, cy, w, h, cls0, cls1, ...] po kolumnach
        // dostep: data[c * n + i]
        float cx = data[0 * n + i];
        float cy = data[1 * n + i];
        float w = data[2 * n + i];
        float h = data[3 * n + i];
        float best_score = 0.0f;
        int best_cls = 0;
        for (int c = 0; c < num_classes; ++c) {
            float s = data[(4 + c) * n + i];
            if (s > best_score) {
                best_score = s;
                best_cls = c;
            }
        }
        if (best_score < conf_thresh) continue;
        Detection d;
        d.bbox.x1 = cx - w / 2.0f;
        d.bbox.y1 = cy - h / 2.0f;
        d.bbox.x2 = cx + w / 2.0f;
        d.bbox.y2 = cy + h / 2.0f;
        d.conf = best_score;
        d.cls = best_cls;
        dets.push_back(d);
    }
    return dets;
}

Detections YoloOnnxDetector::detect(const cv::Mat& frame) {
    return detect_with_conf(frame, -1.0f);
}

Detections YoloOnnxDetector::detect_with_conf(const cv::Mat& frame, float conf_override) {
    if (frame.empty()) return {};
    float conf = (conf_override > 0.0f) ? conf_override : cfg_.conf_threshold;

    // Preprocess
    cv::Mat blob;
    float scale = 1.0f;
    int pad_x = 0, pad_y = 0;
    preprocess(frame, cfg_.imgsz, blob, scale, pad_x, pad_y);

    // Prep ORT input tensor (NCHW float32)
    std::array<int64_t, 4> input_shape{1, 3, cfg_.imgsz, cfg_.imgsz};
    size_t input_size = 1 * 3 * cfg_.imgsz * cfg_.imgsz;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        impl_->memory_info, reinterpret_cast<float*>(blob.data), input_size,
        input_shape.data(), input_shape.size());

    // Inference (timed)
    std::vector<const char*> in_names_c = {impl_->input_names[0].c_str()};
    std::vector<const char*> out_names_c = {impl_->output_names[0].c_str()};
    auto t0 = std::chrono::high_resolution_clock::now();
    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, in_names_c.data(),
                                        &input_tensor, 1, out_names_c.data(), 1);
    auto t1 = std::chrono::high_resolution_clock::now();
    last_inference_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Decode
    if (outputs.empty()) return {};
    auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* data = outputs[0].GetTensorData<float>();
    Detections raw = decode_yolov8(data, shape, cfg_.imgsz, conf);

    // NMS
    Detections nmsed = nms(raw, cfg_.nms_iou_threshold);

    // Unproject bboxy do rozmiaru oryginalnej klatki (odwrocenie letterbox)
    for (auto& d : nmsed) {
        d.bbox.x1 = (d.bbox.x1 - pad_x) / scale;
        d.bbox.y1 = (d.bbox.y1 - pad_y) / scale;
        d.bbox.x2 = (d.bbox.x2 - pad_x) / scale;
        d.bbox.y2 = (d.bbox.y2 - pad_y) / scale;
        // clamp do ramek
        d.bbox.x1 = std::max(0.0f, std::min(d.bbox.x1, static_cast<float>(frame.cols)));
        d.bbox.y1 = std::max(0.0f, std::min(d.bbox.y1, static_cast<float>(frame.rows)));
        d.bbox.x2 = std::max(0.0f, std::min(d.bbox.x2, static_cast<float>(frame.cols)));
        d.bbox.y2 = std::max(0.0f, std::min(d.bbox.y2, static_cast<float>(frame.rows)));
    }

    return nmsed;
}

}  // namespace dtracker
