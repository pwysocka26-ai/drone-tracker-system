// Parity testy SimpleKalman2D vs Python src/core/multi_target_tracker.py:SimpleKalman2D.
// Tolerancja +/-5% (memory: project_cpp_port_plan.md).
//
// Reference values wyliczone z Python implementacji recznie / albo na zasadzie
// matematycznej spojnosci (constant velocity model, init state). NIE laczymy z
// Python runtime -- testy samodzielne.

#include "test_framework.hpp"
#include "dtracker/kalman.hpp"

using namespace dtracker;

TEST(KalmanInit_DefaultsAreSane) {
    SimpleKalman2D k;
    ASSERT_FALSE(k.initialized());
}

TEST(KalmanInit_SetsState) {
    SimpleKalman2D k;
    k.init_state(100.0, 200.0, 5.0, 3.0);
    ASSERT_TRUE(k.initialized());
    Point2 pos = k.position();
    Point2 vel = k.velocity();
    ASSERT_NEAR(pos.x, 100.0, 0.001);
    ASSERT_NEAR(pos.y, 200.0, 0.001);
    ASSERT_NEAR(vel.x, 5.0, 0.001);
    ASSERT_NEAR(vel.y, 3.0, 0.001);
}

TEST(KalmanPredict_ConstantVelocity) {
    // Model: x' = x + vx, y' = y + vy. Velocity nie zmienia sie w predict.
    SimpleKalman2D k;
    k.init_state(100.0, 200.0, 5.0, 3.0);
    Point2 p1 = k.predict();
    ASSERT_NEAR(p1.x, 105.0, 0.001);  // 100 + 5
    ASSERT_NEAR(p1.y, 203.0, 0.001);  // 200 + 3
    Point2 p2 = k.predict();
    ASSERT_NEAR(p2.x, 110.0, 0.001);  // 105 + 5
    ASSERT_NEAR(p2.y, 206.0, 0.001);  // 203 + 3
    Point2 vel = k.velocity();
    ASSERT_NEAR(vel.x, 5.0, 0.001);   // velocity stable
    ASSERT_NEAR(vel.y, 3.0, 0.001);
}

TEST(KalmanCorrect_PullsTowardMeasurement) {
    SimpleKalman2D k;
    k.init_state(100.0, 200.0, 0.0, 0.0);
    // Predict -> (100, 200). Correct measurement (110, 210).
    k.predict();
    Point2 corrected = k.correct(110.0, 210.0);
    // Correction powinna przyciagnac stan ku pomiarowi (ale nie 100% bo R != 0).
    ASSERT_TRUE(corrected.x > 100.0 && corrected.x <= 110.0);
    ASSERT_TRUE(corrected.y > 200.0 && corrected.y <= 210.0);
}

TEST(KalmanCorrect_LearnsVelocity) {
    // Po kilku correct'ach z konsekwentnym ruchem velocity powinno sie ujawnic.
    SimpleKalman2D k;
    k.init_state(0.0, 0.0, 0.0, 0.0);
    for (int i = 1; i <= 10; ++i) {
        k.predict();
        k.correct(static_cast<double>(i * 10), static_cast<double>(i * 5));
    }
    Point2 vel = k.velocity();
    // Po 10 krokach velocity powinno byc bliskie (10, 5) -- tolerancja na process noise.
    ASSERT_NEAR(vel.x, 10.0, 1.5);
    ASSERT_NEAR(vel.y, 5.0, 1.0);
}

TEST(KalmanCorrect_FirstCallInitializes) {
    // Bez init_state correct() powinien zainicjalizowac na measurement.
    SimpleKalman2D k;
    Point2 c = k.correct(50.0, 75.0);
    ASSERT_TRUE(k.initialized());
    ASSERT_NEAR(c.x, 50.0, 0.001);
    ASSERT_NEAR(c.y, 75.0, 0.001);
}

TEST(KalmanClone_IndependentState) {
    SimpleKalman2D a;
    a.init_state(100.0, 200.0, 5.0, 3.0);
    a.predict();
    SimpleKalman2D b = a.clone();
    // Zmiana w b nie powinna wpłynąć na a.
    b.predict();
    b.predict();
    Point2 pa = a.position();
    Point2 pb = b.position();
    // a po 1 predict (100,200)+5,3 = (105, 203). b po 3 predict = (115, 209).
    ASSERT_NEAR(pa.x, 105.0, 0.001);
    ASSERT_NEAR(pa.y, 203.0, 0.001);
    ASSERT_NEAR(pb.x, 115.0, 0.001);
    ASSERT_NEAR(pb.y, 209.0, 0.001);
}

TEST(KalmanCovariance_GrowsOnPredict) {
    // P powinno rosnąć z process noise Q. Po wielu predict'ach wariancja
    // pozycji rośnie, więc kolejne correct powinny przyciagac silniej.
    SimpleKalman2D k1, k2;
    k1.init_state(0.0, 0.0);
    k2.init_state(0.0, 0.0);
    // k1: 1 predict -> correct
    k1.predict();
    Point2 c1 = k1.correct(10.0, 10.0);
    // k2: 10 predicts -> correct (P wieksze)
    for (int i = 0; i < 10; ++i) k2.predict();
    Point2 c2 = k2.correct(10.0, 10.0);
    // Po 10 predict P jest wieksze, gain Kalmana wyzszy -> blizej pomiaru.
    // c2 powinno byc blizej (10,10) niz c1.
    double dist1 = std::hypot(c1.x - 10.0, c1.y - 10.0);
    double dist2 = std::hypot(c2.x - 10.0, c2.y - 10.0);
    ASSERT_TRUE(dist2 < dist1);
}
