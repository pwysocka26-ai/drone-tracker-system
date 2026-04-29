// Pixel -> angular coordinates dla integracji z encoder'ami glowicy gimbal.
//
// Wejscie:
//   - target pixel (cx, cy) z YOLO bbox center (px w window)
//   - image_w, image_h (rozdzielczosc kadru, px)
//   - GimbalSnapshot: FOV kamery (rad) + pozycja osi optycznej (azymut/elewacja w mrad)
// Wyjscie (AngularOffset):
//   - delta_az/el_mrad: offset target od osi optycznej kamery
//   - target_az/el_mrad: absolutny kierunek do obiektu (axis + delta)
//   - theta_mrad: sumaryczna katowa odleglosc target od osi optycznej
//
// Konwencje znakow:
//   - target_pixel_x rosnaca w PRAWO (OpenCV/standard image)
//   - target_pixel_y rosnaca W DOL  (OpenCV/standard image)
//   - delta_az_mrad +: target NA PRAWO od osi  (azymut clockwise from north)
//   - delta_el_mrad +: target NAD osia          (elewacja rosnaca w gore)
//   - W konsekwencji: target przy GORNEJ krawedzi obrazu (pixel_y=0) -> delta_el = +fov_v/2.
//
// Model: pinhole + pelne atan dla dokladnosci (small-angle approximation NIE jest
// uzywana — koszt < 1 us per call, blad atan vs liniowe ujawnia sie dla off-axis >5°).
#pragma once

#include <cmath>

namespace dtracker {

struct GimbalSnapshot {
    float fov_h_rad = 0.0f;       // poziome field of view (rad)
    float fov_v_rad = 0.0f;       // pionowe field of view (rad)
    float axis_az_mrad = 0.0f;    // azymut osi optycznej (mrad, 0=north, +=clockwise)
    float axis_el_mrad = 0.0f;    // elewacja osi optycznej (mrad, 0=horizon, +=up)
};

struct AngularOffset {
    float delta_az_mrad = 0.0f;   // offset target od osi w azymucie (+ na prawo)
    float delta_el_mrad = 0.0f;   // offset target od osi w elewacji (+ na gore)
    float target_az_mrad = 0.0f;  // absolutny azymut obiektu (axis_az + delta_az)
    float target_el_mrad = 0.0f;  // absolutny elewacja obiektu (axis_el + delta_el)
    float theta_mrad = 0.0f;      // odleglosc katowa target od osi optycznej (zawsze >=0)
};

inline AngularOffset pixel_to_angular(float target_pixel_x, float target_pixel_y,
                                       int image_w, int image_h,
                                       const GimbalSnapshot& gimbal) {
    // dx, dy: offset target od centrum ekranu w pikselach
    const float dx = target_pixel_x - 0.5f * static_cast<float>(image_w);
    const float dy = target_pixel_y - 0.5f * static_cast<float>(image_h);

    // Focal length w pikselach (pinhole model)
    const float fx = (0.5f * static_cast<float>(image_w)) / std::tan(0.5f * gimbal.fov_h_rad);
    const float fy = (0.5f * static_cast<float>(image_h)) / std::tan(0.5f * gimbal.fov_v_rad);

    // Konwersja do radianow przez atan
    const float angle_x_rad = std::atan(dx / fx);
    const float angle_y_rad = std::atan(dy / fy);

    AngularOffset out;
    out.delta_az_mrad = 1000.0f * angle_x_rad;
    out.delta_el_mrad = -1000.0f * angle_y_rad;  // znak: image-y down -> elewacja up
    out.target_az_mrad = gimbal.axis_az_mrad + out.delta_az_mrad;
    out.target_el_mrad = gimbal.axis_el_mrad + out.delta_el_mrad;

    // Theta: katowa odleglosc (3D) target od osi optycznej.
    // Dla pinhole: theta = atan(sqrt(dx^2 + dy^2) / f_eff), gdzie f_eff
    // przy izotropowym fov = fx = fy. Przy anizotropii uzywamy projekcji
    // na sfere: tan(theta)^2 ≈ tan(angle_x)^2 + tan(angle_y)^2 (dokladne dla
    // pinhole, mala odbieg <0.01% przy off-axis <10°).
    const float tx = dx / fx;
    const float ty = dy / fy;
    out.theta_mrad = 1000.0f * std::atan(std::sqrt(tx * tx + ty * ty));
    return out;
}

// Wylicz pionowe FOV z poziomego przy assumption square pixels.
// Gdy glowica raportuje tylko fov_h, fov_v wynika z aspect ratio.
inline float fov_v_from_h(float fov_h_rad, int image_w, int image_h) {
    const float aspect = static_cast<float>(image_w) / static_cast<float>(image_h);
    return 2.0f * std::atan(std::tan(0.5f * fov_h_rad) / aspect);
}

// Konwencje pomocnicze
inline float deg_to_rad(float deg) { return deg * 0.017453292519943295f; }
inline float rad_to_mrad(float rad) { return rad * 1000.0f; }
inline float mrad_to_deg(float mrad) { return mrad * 0.0572957795130823f; }

}  // namespace dtracker
