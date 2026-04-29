// Unit testy dla GimbalCsvSource — CSV loader + per-frame lookup.
#include "test_framework.hpp"
#include "dtracker/gimbal_source.hpp"

#include <cstdio>
#include <fstream>
#include <string>

using namespace dtracker;

// Helper: utworz tmp CSV z podana zawartoscia, zwroc path.
static std::string write_tmp_csv(const std::string& content) {
    static int counter = 0;
    std::string path = "test_gimbal_tmp_" + std::to_string(counter++) + ".csv";
    std::ofstream f(path);
    f << content;
    f.close();
    return path;
}

TEST(GimbalCsv_BasicLoad_ThreeRows) {
    std::string csv =
        "frame_idx,fov_h_deg,axis_az_mrad,axis_el_mrad\n"
        "0,8.0,1500.0,200.0\n"
        "1,8.0,1502.5,200.1\n"
        "2,8.0,1505.0,200.2\n";
    std::string path = write_tmp_csv(csv);
    GimbalCsvSource src(path);
    ASSERT_EQ(src.size(), 3);
    ASSERT_FALSE(src.empty());
    std::remove(path.c_str());
}

TEST(GimbalCsv_ExactMatch_ReturnsCorrect) {
    std::string path = write_tmp_csv(
        "frame_idx,fov_h_deg,axis_az_mrad,axis_el_mrad\n"
        "10,8.0,1500.0,200.0\n"
        "20,4.0,1600.0,210.0\n");
    GimbalCsvSource src(path);
    auto g10 = src.lookup(10, 1920, 1080);
    ASSERT_TRUE(g10.has_value());
    ASSERT_NEAR(g10->axis_az_mrad, 1500.0f, 1e-3);
    ASSERT_NEAR(g10->axis_el_mrad, 200.0f, 1e-3);
    auto g20 = src.lookup(20, 1920, 1080);
    ASSERT_TRUE(g20.has_value());
    ASSERT_NEAR(g20->axis_az_mrad, 1600.0f, 1e-3);
    std::remove(path.c_str());
}

TEST(GimbalCsv_NearestPrevious_ForGapsInFrameIdx) {
    // Gimbal raportuje co 10 klatek, video 50fps -> lookup dla frame 13 zwraca entry frame 10.
    std::string path = write_tmp_csv(
        "frame_idx,fov_h_deg,axis_az_mrad,axis_el_mrad\n"
        "0,8.0,1000.0,100.0\n"
        "10,8.0,1100.0,110.0\n"
        "20,8.0,1200.0,120.0\n");
    GimbalCsvSource src(path);
    auto g13 = src.lookup(13, 1920, 1080);
    ASSERT_TRUE(g13.has_value());
    ASSERT_NEAR(g13->axis_az_mrad, 1100.0f, 1e-3);  // frame 10
    auto g19 = src.lookup(19, 1920, 1080);
    ASSERT_NEAR(g19->axis_az_mrad, 1100.0f, 1e-3);  // wciaz frame 10
    auto g20 = src.lookup(20, 1920, 1080);
    ASSERT_NEAR(g20->axis_az_mrad, 1200.0f, 1e-3);  // exact match nowego
    auto g999 = src.lookup(999, 1920, 1080);
    ASSERT_NEAR(g999->axis_az_mrad, 1200.0f, 1e-3);  // nearest-prev = ostatni entry
    std::remove(path.c_str());
}

TEST(GimbalCsv_BeforeFirstEntry_ReturnsNullopt) {
    // CSV zaczyna sie od frame 100, pytanie o frame 50 -> nullopt
    std::string path = write_tmp_csv(
        "frame_idx,fov_h_deg,axis_az_mrad,axis_el_mrad\n"
        "100,8.0,1000.0,100.0\n");
    GimbalCsvSource src(path);
    auto g50 = src.lookup(50, 1920, 1080);
    ASSERT_FALSE(g50.has_value());
    auto g100 = src.lookup(100, 1920, 1080);
    ASSERT_TRUE(g100.has_value());
    std::remove(path.c_str());
}

TEST(GimbalCsv_FovVDerivedFromAspect) {
    // fov_h=8 deg, image 1920x1080 -> fov_v ~= 4.51 deg = atan(tan(4)/aspect) * 2
    std::string path = write_tmp_csv(
        "frame_idx,fov_h_deg,axis_az_mrad,axis_el_mrad\n"
        "0,8.0,0.0,0.0\n");
    GimbalCsvSource src(path);
    auto g = src.lookup(0, 1920, 1080);
    ASSERT_TRUE(g.has_value());
    float expected_fov_v = fov_v_from_h(deg_to_rad(8.0f), 1920, 1080);
    ASSERT_NEAR(g->fov_v_rad, expected_fov_v, 1e-6);
    // sanity: fov_v < fov_h
    ASSERT_TRUE(g->fov_v_rad < g->fov_h_rad);
    std::remove(path.c_str());
}

TEST(GimbalCsv_HeaderColumnOrderInsensitive) {
    // Kolejnosc kolumn dowolna w headerze.
    std::string path = write_tmp_csv(
        "axis_el_mrad,axis_az_mrad,fov_h_deg,frame_idx\n"
        "300.0,2000.0,16.0,0\n"
        "310.0,2010.0,16.0,5\n");
    GimbalCsvSource src(path);
    auto g0 = src.lookup(0, 1920, 1080);
    ASSERT_TRUE(g0.has_value());
    ASSERT_NEAR(g0->axis_az_mrad, 2000.0f, 1e-3);
    ASSERT_NEAR(g0->axis_el_mrad, 300.0f, 1e-3);
    ASSERT_NEAR(g0->fov_h_rad, deg_to_rad(16.0f), 1e-6);
    std::remove(path.c_str());
}

TEST(GimbalCsv_MissingColumn_Throws) {
    std::string path = write_tmp_csv(
        "frame_idx,fov_h_deg,axis_az_mrad\n"  // brak axis_el_mrad
        "0,8.0,100.0\n");
    bool threw = false;
    try {
        GimbalCsvSource src(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    ASSERT_TRUE(threw);
    std::remove(path.c_str());
}

TEST(GimbalCsv_UnsortedInputGetsSorted) {
    // Wpisy w odwrotnej kolejnosci -> klasa sortuje wewnetrznie.
    std::string path = write_tmp_csv(
        "frame_idx,fov_h_deg,axis_az_mrad,axis_el_mrad\n"
        "20,8.0,1200.0,120.0\n"
        "0,8.0,1000.0,100.0\n"
        "10,8.0,1100.0,110.0\n");
    GimbalCsvSource src(path);
    // Lookup frame 5 powinno zwrocic frame 0 (nearest-prev), nie frame 20.
    auto g5 = src.lookup(5, 1920, 1080);
    ASSERT_TRUE(g5.has_value());
    ASSERT_NEAR(g5->axis_az_mrad, 1000.0f, 1e-3);
    std::remove(path.c_str());
}

TEST(GimbalCsv_EndToEndWithPixelToAngular) {
    // Symulacja realnego use case: CSV per-frame + pixel_to_angular.
    std::string path = write_tmp_csv(
        "frame_idx,fov_h_deg,axis_az_mrad,axis_el_mrad\n"
        "0,8.0,1500.0,200.0\n");
    GimbalCsvSource src(path);
    int W = 1920, H = 1080;
    auto g = src.lookup(0, W, H);
    ASSERT_TRUE(g.has_value());
    // Target w centrum -> target_az = axis_az
    AngularOffset r = pixel_to_angular(0.5f * W, 0.5f * H, W, H, *g);
    ASSERT_NEAR(r.target_az_mrad, 1500.0f, 1e-3);
    ASSERT_NEAR(r.target_el_mrad, 200.0f, 1e-3);
    ASSERT_NEAR(r.delta_az_mrad, 0.0f, 1e-3);
    ASSERT_NEAR(r.delta_el_mrad, 0.0f, 1e-3);
    std::remove(path.c_str());
}
