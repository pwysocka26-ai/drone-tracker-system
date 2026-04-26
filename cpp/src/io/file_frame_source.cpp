#include "dtracker/io/file_frame_source.hpp"

#include <algorithm>

namespace dtracker::io {

static bool is_digit_only(const std::string& s) {
    if (s.empty()) return false;
    return std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; });
}

bool FileFrameSource::open(const std::string& uri) {
    close();

    std::string parsed = uri;
    constexpr char kFilePrefix[] = "file://";
    if (parsed.rfind(kFilePrefix, 0) == 0) {
        parsed = parsed.substr(sizeof(kFilePrefix) - 1);
        // Tolerancja "file:///path" -- strip wiodacy "/"
        if (!parsed.empty() && parsed[0] == '/' && parsed.size() > 1 && parsed[1] != '/') {
            // pozostaw -- POSIX absolute path
        }
    }

    bool opened = false;
    if (is_digit_only(parsed)) {
        int idx = std::stoi(parsed);
        opened = cap_.open(idx, cv::CAP_ANY);
    } else {
        opened = cap_.open(parsed);
    }
    if (!opened) return false;

    info_.uri = uri;
    info_.width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    info_.height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    info_.fps = cap_.get(cv::CAP_PROP_FPS);
    if (info_.fps <= 0.0 || info_.fps > 240.0) info_.fps = 30.0;
    int total = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
    info_.total_frames = (total > 0) ? total : -1;  // streams nie maja count
    int fourcc = static_cast<int>(cap_.get(cv::CAP_PROP_FOURCC));
    char fcc[5] = { static_cast<char>(fourcc & 0xff),
                    static_cast<char>((fourcc >> 8) & 0xff),
                    static_cast<char>((fourcc >> 16) & 0xff),
                    static_cast<char>((fourcc >> 24) & 0xff), 0 };
    info_.codec = fcc;
    frame_idx_ = 0;
    return true;
}

bool FileFrameSource::read(Frame& out) {
    if (!cap_.isOpened()) return false;
    if (!cap_.read(out.image) || out.image.empty()) return false;
    out.frame_idx = frame_idx_++;
    out.timestamp_s = (info_.fps > 0.0) ? (out.frame_idx / info_.fps) : 0.0;
    out.is_keyframe = false;  // OpenCV cv::VideoCapture nie eksponuje, pomijamy
    return true;
}

void FileFrameSource::close() {
    if (cap_.isOpened()) cap_.release();
    info_ = {};
    frame_idx_ = 0;
}

}  // namespace dtracker::io
