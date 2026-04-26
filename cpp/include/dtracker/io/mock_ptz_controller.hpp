// MockPTZController — header-only IPTZController dla testow.
//
// Pamietajacy stan (pan, tilt, zoom) bez fizycznego hardware. Idealny do:
//   - unit tests (porownanie expected pan delta po komendzie)
//   - smoke tests pipeline'u bez podlaczonej kamery PTZ
//   - deweloperka logiki integracji zanim hardware jest dostepny
//
// Nie symuluje opóźnienia mechanicznego ruchu -- absolute_pan_tilt() ustawia
// state natychmiast. Vendor adapter dla real PTZ bedzie typowo asynchroniczny.
#pragma once

#include <chrono>
#include <mutex>
#include <string>

#include "dtracker/io/ptz_controller.hpp"

namespace dtracker::io {

class MockPTZController : public IPTZController {
public:
    MockPTZController() {
        caps_.supports_velocity_control = true;
        caps_.supports_absolute_position = true;
        caps_.supports_zoom = true;
        caps_.supports_telemetry = true;
        epoch_ = std::chrono::steady_clock::now();
    }

    bool connect(const std::string& uri) override {
        std::lock_guard<std::mutex> lk(mu_);
        connected_ = true;
        last_uri_ = uri;
        return true;
    }

    bool set_pan_tilt_velocity(double pan_dps, double tilt_dps) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (!connected_) return false;
        // Symulacja: integruj velocity do pozycji per upływ czasu od ostatniej komendy
        update_state_from_velocity_();
        pan_velocity_ = pan_dps;
        tilt_velocity_ = tilt_dps;
        moving_ = (pan_dps != 0.0 || tilt_dps != 0.0);
        return true;
    }

    bool absolute_pan_tilt(double pan_deg, double tilt_deg) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (!connected_) return false;
        pan_ = pan_deg;
        tilt_ = tilt_deg;
        pan_velocity_ = 0.0;
        tilt_velocity_ = 0.0;
        moving_ = false;
        return true;
    }

    bool set_zoom(double zoom_x) override {
        std::lock_guard<std::mutex> lk(mu_);
        if (!connected_) return false;
        zoom_ = zoom_x;
        return true;
    }

    PTZState get_state() override {
        std::lock_guard<std::mutex> lk(mu_);
        update_state_from_velocity_();
        PTZState s;
        s.pan_deg = pan_;
        s.tilt_deg = tilt_;
        s.zoom_x = zoom_;
        s.timestamp_s = elapsed_s_();
        s.moving = moving_;
        s.valid = connected_;
        return s;
    }

    const PTZCapabilities& capabilities() const override { return caps_; }

    void disconnect() override {
        std::lock_guard<std::mutex> lk(mu_);
        connected_ = false;
        moving_ = false;
        pan_velocity_ = 0.0;
        tilt_velocity_ = 0.0;
    }

    bool is_connected() const override {
        std::lock_guard<std::mutex> lk(mu_);
        return connected_;
    }

private:
    double elapsed_s_() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - epoch_).count();
    }

    void update_state_from_velocity_() {
        // Integruj velocity od ostatniego update_
        double now = elapsed_s_();
        if (last_velocity_update_s_ == 0.0) {
            last_velocity_update_s_ = now;
            return;
        }
        double dt = now - last_velocity_update_s_;
        last_velocity_update_s_ = now;
        if (pan_velocity_ != 0.0) pan_ += pan_velocity_ * dt;
        if (tilt_velocity_ != 0.0) tilt_ += tilt_velocity_ * dt;
        // Clamp do range
        if (pan_ > caps_.pan_max_deg) pan_ = caps_.pan_max_deg;
        if (pan_ < caps_.pan_min_deg) pan_ = caps_.pan_min_deg;
        if (tilt_ > caps_.tilt_max_deg) tilt_ = caps_.tilt_max_deg;
        if (tilt_ < caps_.tilt_min_deg) tilt_ = caps_.tilt_min_deg;
    }

    mutable std::mutex mu_;
    PTZCapabilities caps_{};
    bool connected_ = false;
    std::string last_uri_;
    double pan_ = 0.0;
    double tilt_ = 0.0;
    double zoom_ = 1.0;
    double pan_velocity_ = 0.0;
    double tilt_velocity_ = 0.0;
    bool moving_ = false;
    std::chrono::steady_clock::time_point epoch_;
    double last_velocity_update_s_ = 0.0;
};

}  // namespace dtracker::io
