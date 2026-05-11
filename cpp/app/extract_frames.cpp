// Helper: extract specific frames from a video to PNG.
// Usage: extract_frames <video> <out_dir> <frame_idx> [<frame_idx>...]
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: extract_frames <video> <out_dir> <frame_idx> [<frame_idx>...]\n";
        return 2;
    }
    std::string video_path = argv[1];
    std::string out_dir = argv[2];
    std::vector<int> frame_indices;
    for (int i = 3; i < argc; ++i) frame_indices.push_back(std::atoi(argv[i]));

    std::filesystem::create_directories(out_dir);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: cannot open " << video_path << "\n";
        return 1;
    }
    int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Video: " << w << "x" << h << ", " << total << " frames\n";

    for (int idx : frame_indices) {
        if (idx < 0 || idx >= total) {
            std::cerr << "skip frame " << idx << " (out of range)\n";
            continue;
        }
        cap.set(cv::CAP_PROP_POS_FRAMES, idx);
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "skip frame " << idx << " (read failed)\n";
            continue;
        }
        std::string out_path = out_dir + "/frame_" + std::to_string(idx) + ".png";
        if (cv::imwrite(out_path, frame)) {
            std::cout << "wrote " << out_path << "\n";
        } else {
            std::cerr << "ERROR writing " << out_path << "\n";
        }
    }
    return 0;
}
