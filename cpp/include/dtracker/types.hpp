// Wspolne typy/aliasy dla pipeline.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

// Forward declaration — pelen include cv::Mat tylko gdzie potrzebne.
namespace cv { class Mat; }

namespace dtracker {

struct BBox {
    float x1, y1, x2, y2;

    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area() const { return width() * height(); }
    float cx() const { return 0.5f * (x1 + x2); }
    float cy() const { return 0.5f * (y1 + y2); }
};

struct Detection {
    BBox bbox;
    float conf;
    int cls;
};

using Detections = std::vector<Detection>;

// Abstract interface — detector loosely coupled od konkretnego modelu
// (YOLOv8, RT-DETR, cokolwiek).
class IDetector {
public:
    virtual ~IDetector() = default;
    virtual Detections detect(const cv::Mat& frame) = 0;
    virtual int imgsz() const = 0;
};

}  // namespace dtracker
