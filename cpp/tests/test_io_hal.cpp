// Parity testy dla Hardware Abstraction Layer (Phase 1):
// - FileFrameSource (real impl on cv::VideoCapture)
// - MockPTZController (header-only, dla testow)
//
// Pokazuje ze interfejsy IFrameSource + IPTZController dzialaja
// jako blackbox -- aplikacja moze pisac do nich abstrakcyjnie.
#include "test_framework.hpp"

#include "dtracker/io/file_frame_source.hpp"
#include "dtracker/io/mock_ptz_controller.hpp"

#include <cstdio>
#include <thread>
#include <chrono>

using namespace dtracker::io;

// ---------- FileFrameSource ----------

TEST(IO_FileFrameSource_OpenNonExistentFails) {
    FileFrameSource fs;
    ASSERT_FALSE(fs.open("/nonexistent/path/video.mp4"));
    ASSERT_FALSE(fs.is_open());
}

TEST(IO_FileFrameSource_OpenRealVideo) {
    FileFrameSource fs;
    bool opened = fs.open("artifacts/test_videos/video_test_wide_short.mp4");
    if (!opened) {
        // Skip if test asset missing -- not a unit-test failure
        std::cout << "  (skip: video_test_wide_short.mp4 unavailable)\n";
        return;
    }
    ASSERT_TRUE(fs.is_open());
    const auto& info = fs.info();
    ASSERT_EQ(info.width, 1920);
    ASSERT_EQ(info.height, 1080);
    ASSERT_TRUE(info.fps > 49.0 && info.fps < 51.0);
    ASSERT_TRUE(info.total_frames > 0);
}

TEST(IO_FileFrameSource_ReadIncrementsFrameIdx) {
    FileFrameSource fs;
    if (!fs.open("artifacts/test_videos/video_test_wide_short.mp4")) {
        std::cout << "  (skip)\n";
        return;
    }
    Frame f1, f2, f3;
    ASSERT_TRUE(fs.read(f1));
    ASSERT_TRUE(fs.read(f2));
    ASSERT_TRUE(fs.read(f3));
    ASSERT_EQ(f1.frame_idx, 0);
    ASSERT_EQ(f2.frame_idx, 1);
    ASSERT_EQ(f3.frame_idx, 2);
    // Timestamps monotonicznie rosnace
    ASSERT_TRUE(f2.timestamp_s > f1.timestamp_s);
    ASSERT_TRUE(f3.timestamp_s > f2.timestamp_s);
    // Image niepusty
    ASSERT_FALSE(f1.image.empty());
}

TEST(IO_FileFrameSource_CloseIdempotent) {
    FileFrameSource fs;
    if (!fs.open("artifacts/test_videos/video_test_wide_short.mp4")) {
        std::cout << "  (skip)\n";
        return;
    }
    fs.close();
    fs.close();  // 2nd time = no crash
    ASSERT_FALSE(fs.is_open());
}

// ---------- MockPTZController ----------

TEST(IO_MockPTZ_NotConnectedRejects) {
    MockPTZController ptz;
    ASSERT_FALSE(ptz.is_connected());
    ASSERT_FALSE(ptz.set_pan_tilt_velocity(10.0, 0.0));  // not connected
    ASSERT_FALSE(ptz.absolute_pan_tilt(45.0, 10.0));
    ASSERT_FALSE(ptz.set_zoom(2.0));
}

TEST(IO_MockPTZ_ConnectThenCommand) {
    MockPTZController ptz;
    ASSERT_TRUE(ptz.connect("mock://"));
    ASSERT_TRUE(ptz.is_connected());
    ASSERT_TRUE(ptz.absolute_pan_tilt(45.0, 10.0));
    auto s = ptz.get_state();
    ASSERT_NEAR(s.pan_deg, 45.0, 0.001);
    ASSERT_NEAR(s.tilt_deg, 10.0, 0.001);
    ASSERT_TRUE(s.valid);
}

TEST(IO_MockPTZ_VelocityIntegratesPosition) {
    MockPTZController ptz;
    ptz.connect("mock://");
    ptz.absolute_pan_tilt(0.0, 0.0);
    ptz.set_pan_tilt_velocity(60.0, 0.0);  // 60 deg/s w prawo
    std::this_thread::sleep_for(std::chrono::milliseconds(150));  // ~9 deg ruch
    auto s = ptz.get_state();
    // Po ~150ms przy 60 dps powinno byc ~9 deg pan
    ASSERT_TRUE(s.pan_deg > 5.0);
    ASSERT_TRUE(s.pan_deg < 15.0);
    ASSERT_TRUE(s.moving);
}

TEST(IO_MockPTZ_ZoomSetters) {
    MockPTZController ptz;
    ptz.connect("mock://");
    ASSERT_TRUE(ptz.set_zoom(5.5));
    auto s = ptz.get_state();
    ASSERT_NEAR(s.zoom_x, 5.5, 0.001);
}

TEST(IO_MockPTZ_DisconnectInvalidates) {
    MockPTZController ptz;
    ptz.connect("mock://");
    ptz.absolute_pan_tilt(30.0, 5.0);
    ptz.disconnect();
    ASSERT_FALSE(ptz.is_connected());
    auto s = ptz.get_state();
    ASSERT_FALSE(s.valid);
}

// ---------- Polymorphism / blackbox usage ----------

TEST(IO_PolymorphicUsage_AppHoldsAbstractInterface) {
    // Testuje ze aplikacja moze trzymac abstract pointer + uzywac komend
    // bez wiedzy o konkretnym hardware
    PTZControllerPtr ptz = std::make_shared<MockPTZController>();
    ASSERT_TRUE(ptz->connect("mock://"));
    ASSERT_TRUE(ptz->absolute_pan_tilt(20.0, 5.0));
    auto s = ptz->get_state();
    ASSERT_NEAR(s.pan_deg, 20.0, 0.001);
}
