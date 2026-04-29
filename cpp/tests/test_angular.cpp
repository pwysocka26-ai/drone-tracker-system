// Unit testy dla dtracker/angular.hpp — pixel -> angular conversion z gimbal data.
#include "test_framework.hpp"
#include "dtracker/angular.hpp"

#include <cmath>

using namespace dtracker;

// Helper: typowy gimbal snapshot dla testow (poziomy FOV 8 deg, pionowy 4.5 deg
// dla aspect 16:9, oba w rad). axis = 0 mrad domyslnie (zerowanie wzgledem osi).
static GimbalSnapshot default_gimbal(float axis_az_mrad = 0.0f, float axis_el_mrad = 0.0f) {
    GimbalSnapshot g;
    g.fov_h_rad = deg_to_rad(8.0f);    // ~140 mrad
    g.fov_v_rad = deg_to_rad(4.5f);    // ~78.5 mrad
    g.axis_az_mrad = axis_az_mrad;
    g.axis_el_mrad = axis_el_mrad;
    return g;
}

TEST(Angular_TargetAtCenter_AllOffsetsZero) {
    GimbalSnapshot g = default_gimbal();
    int W = 1920, H = 1080;
    AngularOffset r = pixel_to_angular(0.5f * W, 0.5f * H, W, H, g);
    ASSERT_NEAR(r.delta_az_mrad, 0.0f, 1e-3);
    ASSERT_NEAR(r.delta_el_mrad, 0.0f, 1e-3);
    ASSERT_NEAR(r.theta_mrad, 0.0f, 1e-3);
    ASSERT_NEAR(r.target_az_mrad, 0.0f, 1e-3);
    ASSERT_NEAR(r.target_el_mrad, 0.0f, 1e-3);
}

TEST(Angular_TargetAtRightEdge_DeltaAzPlusHalfFovH) {
    GimbalSnapshot g = default_gimbal();
    int W = 1920, H = 1080;
    // Prawa krawedz: pixel_x = W (1 piksel za, ale matematycznie tg dx=W/2)
    AngularOffset r = pixel_to_angular(static_cast<float>(W), 0.5f * H, W, H, g);
    // Oczekiwane: atan(W/2 / fx) gdzie fx = (W/2)/tan(fov_h/2) -> atan(tan(fov_h/2)) = fov_h/2
    float expected_mrad = 0.5f * g.fov_h_rad * 1000.0f;
    ASSERT_NEAR(r.delta_az_mrad, expected_mrad, 0.01f);
    ASSERT_NEAR(r.delta_el_mrad, 0.0f, 1e-3);
    ASSERT_NEAR(r.theta_mrad, expected_mrad, 0.01f);
}

TEST(Angular_TargetAtLeftEdge_DeltaAzMinusHalfFovH) {
    GimbalSnapshot g = default_gimbal();
    int W = 1920, H = 1080;
    AngularOffset r = pixel_to_angular(0.0f, 0.5f * H, W, H, g);
    float expected_mrad = -0.5f * g.fov_h_rad * 1000.0f;
    ASSERT_NEAR(r.delta_az_mrad, expected_mrad, 0.01f);
    ASSERT_NEAR(r.delta_el_mrad, 0.0f, 1e-3);
}

TEST(Angular_TargetAtTopEdge_DeltaElPlusHalfFovV) {
    // Top of image: pixel_y = 0. Konwencja: image-y down, ale elewacja up,
    // wiec target NAD osia -> delta_el > 0.
    GimbalSnapshot g = default_gimbal();
    int W = 1920, H = 1080;
    AngularOffset r = pixel_to_angular(0.5f * W, 0.0f, W, H, g);
    float expected_mrad = 0.5f * g.fov_v_rad * 1000.0f;
    ASSERT_NEAR(r.delta_el_mrad, expected_mrad, 0.01f);
    ASSERT_NEAR(r.delta_az_mrad, 0.0f, 1e-3);
}

TEST(Angular_TargetAtBottomEdge_DeltaElMinusHalfFovV) {
    GimbalSnapshot g = default_gimbal();
    int W = 1920, H = 1080;
    AngularOffset r = pixel_to_angular(0.5f * W, static_cast<float>(H), W, H, g);
    float expected_mrad = -0.5f * g.fov_v_rad * 1000.0f;
    ASSERT_NEAR(r.delta_el_mrad, expected_mrad, 0.01f);
}

TEST(Angular_AbsoluteAddsAxisOffset) {
    // axis_az = 1500 mrad (~85.94 deg), axis_el = -100 mrad (~-5.73 deg)
    // Target w centrum -> target_az = axis_az + 0, target_el = axis_el + 0
    GimbalSnapshot g = default_gimbal(1500.0f, -100.0f);
    int W = 1920, H = 1080;
    AngularOffset r = pixel_to_angular(0.5f * W, 0.5f * H, W, H, g);
    ASSERT_NEAR(r.target_az_mrad, 1500.0f, 1e-3);
    ASSERT_NEAR(r.target_el_mrad, -100.0f, 1e-3);
    // delta wciaz zerowe
    ASSERT_NEAR(r.delta_az_mrad, 0.0f, 1e-3);
    ASSERT_NEAR(r.delta_el_mrad, 0.0f, 1e-3);
}

TEST(Angular_AbsoluteWithOffset_AddsCorrectly) {
    // axis_az=2000 mrad, target przy prawej krawedzi -> delta_az = +fov_h/2 mrad,
    // target_az = 2000 + fov_h/2 mrad.
    GimbalSnapshot g = default_gimbal(2000.0f, 500.0f);
    int W = 1920, H = 1080;
    AngularOffset r = pixel_to_angular(static_cast<float>(W), 0.0f, W, H, g);
    float expected_delta_az = 0.5f * g.fov_h_rad * 1000.0f;
    float expected_delta_el = 0.5f * g.fov_v_rad * 1000.0f;
    ASSERT_NEAR(r.delta_az_mrad, expected_delta_az, 0.01f);
    ASSERT_NEAR(r.delta_el_mrad, expected_delta_el, 0.01f);
    ASSERT_NEAR(r.target_az_mrad, 2000.0f + expected_delta_az, 0.01f);
    ASSERT_NEAR(r.target_el_mrad, 500.0f + expected_delta_el, 0.01f);
}

TEST(Angular_TargetInCorner_ThetaIsRadialDistance) {
    GimbalSnapshot g = default_gimbal();
    int W = 1920, H = 1080;
    // Top-right corner
    AngularOffset r = pixel_to_angular(static_cast<float>(W), 0.0f, W, H, g);
    // Expected theta dla pinhole: atan(sqrt(tan(fov_h/2)^2 + tan(fov_v/2)^2)) * 1000
    float t1 = std::tan(0.5f * g.fov_h_rad);
    float t2 = std::tan(0.5f * g.fov_v_rad);
    float expected_theta = 1000.0f * std::atan(std::sqrt(t1 * t1 + t2 * t2));
    ASSERT_NEAR(r.theta_mrad, expected_theta, 0.01f);
    // Theta zawsze nieujemne
    ASSERT_TRUE(r.theta_mrad >= 0.0f);
}

TEST(Angular_AnisotropicFov_DifferentFxFy) {
    // FOV poziomy szeroki, pionowy waski (np. wide tele rectangular)
    GimbalSnapshot g;
    g.fov_h_rad = deg_to_rad(20.0f);
    g.fov_v_rad = deg_to_rad(5.0f);
    int W = 1920, H = 1080;
    // Target 100 px w prawo + 100 px w dol od centrum
    AngularOffset r = pixel_to_angular(0.5f * W + 100.0f, 0.5f * H + 100.0f, W, H, g);
    // delta_az: atan(100/fx), gdzie fx = (W/2)/tan(fov_h/2) — duze fx -> maly katowy offset
    float fx = (0.5f * W) / std::tan(0.5f * g.fov_h_rad);
    float fy = (0.5f * H) / std::tan(0.5f * g.fov_v_rad);
    float expected_delta_az = 1000.0f * std::atan(100.0f / fx);
    float expected_delta_el = -1000.0f * std::atan(100.0f / fy);
    ASSERT_NEAR(r.delta_az_mrad, expected_delta_az, 0.01f);
    ASSERT_NEAR(r.delta_el_mrad, expected_delta_el, 0.01f);
    // Aspect: szeroki fov_h, waski fov_v -> fx < fy -> ten sam pixel offset (100 px)
    // daje WIEKSZY katowy az niz el. Sanity check ze konwencja math jest poprawna.
    ASSERT_TRUE(std::fabs(r.delta_az_mrad) > std::fabs(r.delta_el_mrad));
}

TEST(Angular_FovVfromH_AspectRatio16x9) {
    // FOV poziomy 16 deg, image 16:9 -> fov_v = atan(tan(8 deg)/(16/9)) * 2
    float fov_h = deg_to_rad(16.0f);
    int W = 1920, H = 1080;
    float fov_v = fov_v_from_h(fov_h, W, H);
    float expected = 2.0f * std::atan(std::tan(0.5f * fov_h) / (16.0f / 9.0f));
    ASSERT_NEAR(fov_v, expected, 1e-6);
    // Sanity: fov_v < fov_h dla aspect > 1
    ASSERT_TRUE(fov_v < fov_h);
}

TEST(Angular_SmallAngleApprox_AccurateBelow5Deg) {
    // Dla off-axis < 5 deg, atan(x) ≈ x z bledem <0.5%.
    // Sprawdzamy ze wynik atan jest blisko liniowego mnozenia.
    GimbalSnapshot g = default_gimbal();
    int W = 1920, H = 1080;
    // 1/4 ekranu od centrum (dx = 480 px) na fov_h=8 deg -> ~2 deg = ~35 mrad
    AngularOffset r = pixel_to_angular(0.5f * W + 480.0f, 0.5f * H, W, H, g);
    float fx = (0.5f * W) / std::tan(0.5f * g.fov_h_rad);
    float linear_mrad = 1000.0f * (480.0f / fx);
    float atan_mrad = r.delta_az_mrad;
    // atan vs linear roznica <0.5% dla 2 deg
    float rel_error = std::fabs(atan_mrad - linear_mrad) / std::fabs(atan_mrad);
    ASSERT_TRUE(rel_error < 0.005);
}

TEST(Angular_RealWorldScenario_WideFov_DroneAtBboxCenter) {
    // Realistyczny case z video_test_wide.mp4: drone na (1020, 490)
    // przy 1920x1080, fov_h = 8 deg (typowy tele), axis_az=300 mrad, axis_el=150 mrad.
    GimbalSnapshot g;
    g.fov_h_rad = deg_to_rad(8.0f);
    g.fov_v_rad = fov_v_from_h(g.fov_h_rad, 1920, 1080);
    g.axis_az_mrad = 300.0f;
    g.axis_el_mrad = 150.0f;
    AngularOffset r = pixel_to_angular(1020.0f, 490.0f, 1920, 1080, g);
    // Dron praktycznie w centrum (60 px w prawo, 50 px w gore od (960, 540))
    // -> mala katowa odleglosc, target_az ≈ axis_az
    ASSERT_TRUE(std::fabs(r.delta_az_mrad) < 10.0f);  // <10 mrad off-axis
    ASSERT_TRUE(std::fabs(r.delta_el_mrad) < 10.0f);
    ASSERT_TRUE(r.theta_mrad < 15.0f);
    // Target absolute ~ axis +/- offsetu
    ASSERT_NEAR(r.target_az_mrad, 300.0f + r.delta_az_mrad, 1e-3);
    ASSERT_NEAR(r.target_el_mrad, 150.0f + r.delta_el_mrad, 1e-3);
    // Sanity: target po prawej i wyzej -> delta_az>0 i delta_el>0
    ASSERT_TRUE(r.delta_az_mrad > 0.0f);
    ASSERT_TRUE(r.delta_el_mrad > 0.0f);
}
