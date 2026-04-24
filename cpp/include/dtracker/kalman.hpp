// Lekki constant-velocity Kalman 2D (port 1:1 z SimpleKalman2D w Python).
// State: [x, y, vx, vy], observation: [x, y].
#pragma once

#include <array>

namespace dtracker {

struct Point2 {
    double x, y;
};

class SimpleKalman2D {
public:
    SimpleKalman2D(double process_noise = 0.03, double measurement_noise = 0.20);

    void init_state(double x, double y, double vx = 0.0, double vy = 0.0);
    Point2 predict();
    Point2 correct(double mx, double my);

    bool initialized() const { return initialized_; }
    Point2 position() const { return {x_[0], x_[1]}; }
    Point2 velocity() const { return {x_[2], x_[3]}; }

    SimpleKalman2D clone() const;

private:
    double q_;
    double r_;
    bool initialized_ = false;
    std::array<double, 4> x_{0, 0, 0, 0};
    std::array<std::array<double, 4>, 4> P_{};
};

}  // namespace dtracker
