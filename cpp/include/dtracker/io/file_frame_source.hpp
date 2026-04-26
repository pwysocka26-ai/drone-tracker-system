// FileFrameSource — IFrameSource adapter dla plików video (cv::VideoCapture).
//
// Reference implementation -- pokazuje vendorowi jak napisać własny adapter.
// Wsparcie URI:
//   "/path/to/video.mp4"
//   "file:///path/to/video.mp4"  (z prefixem)
//   "0", "1"  (kamera index, OpenCV CAP_ANY)
#pragma once

#include <opencv2/videoio.hpp>

#include "dtracker/io/frame_source.hpp"

namespace dtracker::io {

class FileFrameSource : public IFrameSource {
public:
    FileFrameSource() = default;
    ~FileFrameSource() override { close(); }

    bool open(const std::string& uri) override;
    bool read(Frame& out) override;
    const FrameSourceInfo& info() const override { return info_; }
    void close() override;
    bool is_open() const override { return cap_.isOpened(); }

private:
    cv::VideoCapture cap_;
    FrameSourceInfo info_;
    long frame_idx_ = 0;
};

}  // namespace dtracker::io
