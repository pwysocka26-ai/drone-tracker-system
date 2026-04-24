#include "dtracker/kalman.hpp"

#include <cmath>

namespace dtracker {

SimpleKalman2D::SimpleKalman2D(double process_noise, double measurement_noise)
    : q_(process_noise), r_(measurement_noise) {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            P_[i][j] = (i == j) ? 1.0 : 0.0;
}

void SimpleKalman2D::init_state(double x, double y, double vx, double vy) {
    x_ = {x, y, vx, vy};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            P_[i][j] = 0.0;
    P_[0][0] = 1.0;
    P_[1][1] = 1.0;
    P_[2][2] = 4.0;
    P_[3][3] = 4.0;
    initialized_ = true;
}

Point2 SimpleKalman2D::predict() {
    if (!initialized_) return {0.0, 0.0};
    // state: x = F*x, F = [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]
    x_[0] += x_[2];
    x_[1] += x_[3];
    // x_[2], x_[3] bez zmian (stala prędkosc)

    // P = F * P * F^T + Q
    // F*P: row i, col j = sum_k F[i][k] * P[k][j]
    // F nie-trywialnie: F[0]=[1,0,1,0], F[1]=[0,1,0,1], F[2]=[0,0,1,0], F[3]=[0,0,0,1]
    std::array<std::array<double, 4>, 4> FP{};
    for (int j = 0; j < 4; ++j) {
        FP[0][j] = P_[0][j] + P_[2][j];
        FP[1][j] = P_[1][j] + P_[3][j];
        FP[2][j] = P_[2][j];
        FP[3][j] = P_[3][j];
    }
    // FP * F^T: F^T[k][j] = F[j][k]
    // (FP*F^T)[i][j] = sum_k FP[i][k] * F^T[k][j] = sum_k FP[i][k] * F[j][k]
    // F[0][k]=[1,0,1,0], F[1][k]=[0,1,0,1], F[2][k]=[0,0,1,0], F[3][k]=[0,0,0,1]
    std::array<std::array<double, 4>, 4> Pnew{};
    for (int i = 0; i < 4; ++i) {
        Pnew[i][0] = FP[i][0] + FP[i][2];
        Pnew[i][1] = FP[i][1] + FP[i][3];
        Pnew[i][2] = FP[i][2];
        Pnew[i][3] = FP[i][3];
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            P_[i][j] = Pnew[i][j];

    for (int i = 0; i < 4; ++i) P_[i][i] += q_;
    return {x_[0], x_[1]};
}

Point2 SimpleKalman2D::correct(double mx, double my) {
    if (!initialized_) {
        init_state(mx, my);
        return {mx, my};
    }
    double yx = mx - x_[0];
    double yy = my - x_[1];

    // S = H*P*H^T + R, H = [[1,0,0,0],[0,1,0,0]], R = r_ * I_2
    double s00 = P_[0][0] + r_;
    double s01 = P_[0][1];
    double s10 = P_[1][0];
    double s11 = P_[1][1] + r_;

    double det = s00 * s11 - s01 * s10;
    if (std::fabs(det) < 1e-9) return {x_[0], x_[1]};

    double inv00 = s11 / det;
    double inv01 = -s01 / det;
    double inv10 = -s10 / det;
    double inv11 = s00 / det;

    // K = P * H^T * S^{-1}, P*H^T bierze pierwsze 2 kolumny P
    // K[i] = [P[i][0]*inv00 + P[i][1]*inv10, P[i][0]*inv01 + P[i][1]*inv11]
    std::array<std::array<double, 2>, 4> K{};
    for (int i = 0; i < 4; ++i) {
        K[i][0] = P_[i][0] * inv00 + P_[i][1] * inv10;
        K[i][1] = P_[i][0] * inv01 + P_[i][1] * inv11;
    }

    for (int i = 0; i < 4; ++i) {
        x_[i] += K[i][0] * yx + K[i][1] * yy;
    }

    // P = (I - K*H) * P
    // K*H: (K*H)[i][j] = K[i][0]*H[0][j] + K[i][1]*H[1][j]
    // H[0]=[1,0,0,0], H[1]=[0,1,0,0] -> K*H[i][0]=K[i][0], K*H[i][1]=K[i][1], K*H[i][2,3]=0
    std::array<std::array<double, 4>, 4> IKH{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double kh = 0.0;
            if (j == 0) kh = K[i][0];
            else if (j == 1) kh = K[i][1];
            IKH[i][j] = (i == j ? 1.0 : 0.0) - kh;
        }
    }
    std::array<std::array<double, 4>, 4> Pnew{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double s = 0.0;
            for (int k = 0; k < 4; ++k) s += IKH[i][k] * P_[k][j];
            Pnew[i][j] = s;
        }
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            P_[i][j] = Pnew[i][j];

    return {x_[0], x_[1]};
}

SimpleKalman2D SimpleKalman2D::clone() const {
    SimpleKalman2D k(q_, r_);
    k.initialized_ = initialized_;
    k.x_ = x_;
    k.P_ = P_;
    return k;
}

}  // namespace dtracker
