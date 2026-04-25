// Parity testy NarrowTracker — PID smoothing + adaptive feedforward + last_good
// persistence (synthetic hold). Pokrywa Fix 1 + Fix 3 + Fix 4 z 2026-04-25.
#include "test_framework.hpp"
#include "dtracker/narrow_tracker.hpp"
#include "dtracker/track.hpp"

using namespace dtracker;

// Helper synthetic Track. Bbox 30x30 wokol (cx, cy).
static Track make_track(int id, double cx, double cy, double vx = 0.0, double vy = 0.0,
                         int missed = 0, float conf = 0.5f) {
    Track t;
    t.track_id = id;
    t.raw_id = id;
    t.bbox = BBox{
        static_cast<float>(cx - 15), static_cast<float>(cy - 15),
        static_cast<float>(cx + 15), static_cast<float>(cy + 15)
    };
    t.center = Point2{cx, cy};
    t.velocity = Point2{vx, vy};
    t.confidence = conf;
    t.is_confirmed = true;
    t.missed_frames = missed;
    t.hits = 10;
    return t;
}

TEST(Narrow_InitialStateNoOwner) {
    NarrowTracker nt;
    const auto& s = nt.state();
    ASSERT_FALSE(s.has_owner);
    ASSERT_FALSE(s.smooth_center.has_value());
    ASSERT_EQ(s.hold_count, 0);
}

TEST(Narrow_FirstUpdateSetsSmoothCenter) {
    NarrowTracker nt;
    Track t = make_track(1, 500, 300);
    nt.update(&t, true);
    ASSERT_TRUE(nt.state().has_owner);
    ASSERT_TRUE(nt.state().smooth_center.has_value());
    // Smooth_center init na pozycje desired (zero velocity, zero feedforward kontrybucji)
    ASSERT_NEAR(nt.state().smooth_center->x, 500.0, 1.0);
    ASSERT_NEAR(nt.state().smooth_center->y, 300.0, 1.0);
}

TEST(Narrow_PIDDeadZoneIgnoresSmallErrors) {
    // Po init smooth_center = (500, 300). Track przesuwa sie o 1px (< dead_zone=4).
    // PID powinien NIE ruszyc smooth_center po deadzone.
    NarrowTracker nt;
    Track t = make_track(1, 500, 300);
    nt.update(&t, true);
    Point2 before = *nt.state().smooth_center;
    Track t2 = make_track(1, 502, 301);  // delta (2, 1) -- pod dead_zone=4
    nt.update(&t2, true);
    Point2 after = *nt.state().smooth_center;
    // Smooth_center NIE powinien znacznie sie ruszyc bo error < dead_zone
    ASSERT_NEAR(after.x, before.x, 1.0);
    ASSERT_NEAR(after.y, before.y, 1.0);
}

TEST(Narrow_AdaptiveFFScale_LowVelocity) {
    // |v| = 1 < ff_v_low=2 -> ff_scale=0.2
    NarrowTracker nt;
    Track t = make_track(1, 500, 300, /*vx*/0.5, /*vy*/0.5);  // |v| ~ 0.7
    nt.update(&t, true);
    const auto& s = nt.state();
    ASSERT_NEAR(s.last_ff_scale, 0.2, 0.01);
    ASSERT_NEAR(s.last_velocity_magnitude, 0.707, 0.05);
}

TEST(Narrow_AdaptiveFFScale_HighVelocity) {
    // |v| = 7 > ff_v_high=5 -> ff_scale=1.0
    NarrowTracker nt;
    Track t = make_track(1, 500, 300, /*vx*/5, /*vy*/4);  // |v| ~ 6.4
    nt.update(&t, true);
    const auto& s = nt.state();
    ASSERT_NEAR(s.last_ff_scale, 1.0, 0.01);
    ASSERT_NEAR(s.last_velocity_magnitude, 6.4, 0.1);
}

TEST(Narrow_AdaptiveFFScale_MidVelocity_LinearInterp) {
    // |v| = 3.5 -> ff_scale = 0.2 + (3.5 - 2)/(5 - 2) * 0.8 = 0.2 + 0.4 = 0.6
    NarrowTracker nt;
    Track t = make_track(1, 500, 300, /*vx*/3.5, /*vy*/0);
    nt.update(&t, true);
    const auto& s = nt.state();
    ASSERT_NEAR(s.last_ff_scale, 0.6, 0.01);
}

TEST(Narrow_HoldPersistenceWhenOwnerLost) {
    // Fix 1: gdy owner == nullptr, narrow trzyma synthetic crop z last_good
    NarrowTracker nt;
    Track t = make_track(1, 500, 300);
    nt.update(&t, true);
    ASSERT_TRUE(nt.state().last_good_center.has_value());

    // Owner zniknal
    nt.update(nullptr, false);
    const auto& s = nt.state();
    // Synthetic hold aktywny: has_owner=true, is_synthetic=true
    ASSERT_TRUE(s.has_owner);
    ASSERT_TRUE(s.is_synthetic);
    ASSERT_TRUE(s.hold_count == 1);
}

TEST(Narrow_HoldTimeoutBlanksAfterMaxFrames) {
    NarrowConfig cfg;
    cfg.max_hold_frames = 5;  // szybki test
    NarrowTracker nt(cfg);
    Track t = make_track(1, 500, 300);
    nt.update(&t, true);

    // 5 klatek bez ownera -- wciaz synthetic
    for (int i = 0; i < 5; ++i) nt.update(nullptr, false);
    ASSERT_TRUE(nt.state().is_synthetic);
    ASSERT_TRUE(nt.state().has_owner);

    // 6 klatka -- przekroczone, wygaszamy
    nt.update(nullptr, false);
    ASSERT_FALSE(nt.state().is_synthetic);
    ASSERT_FALSE(nt.state().has_owner);
}

TEST(Narrow_HoldCountResetsOnOwnerReturn) {
    NarrowTracker nt;
    Track t = make_track(1, 500, 300);
    nt.update(&t, true);
    nt.update(nullptr, false);
    nt.update(nullptr, false);
    ASSERT_EQ(nt.state().hold_count, 2);

    // Owner wraca -> hold_count resetuje
    nt.update(&t, true);
    ASSERT_EQ(nt.state().hold_count, 0);
    ASSERT_FALSE(nt.state().is_synthetic);
}

TEST(Narrow_NarrowCropCoordinatesAreInsideFrame) {
    // Frame 1920x1080, owner na (500, 300), bbox 30x30
    NarrowTracker nt({}, 1920, 1080);
    Track t = make_track(1, 500, 300);
    nt.update(&t, true);
    BBox crop = nt.narrow_crop();
    ASSERT_TRUE(crop.x1 >= 0.0f);
    ASSERT_TRUE(crop.y1 >= 0.0f);
    ASSERT_TRUE(crop.x2 <= 1920.0f);
    ASSERT_TRUE(crop.y2 <= 1080.0f);
    ASSERT_TRUE(crop.x2 > crop.x1);
    ASSERT_TRUE(crop.y2 > crop.y1);
}

TEST(Narrow_DegradedFlagOnHighMissedFrames) {
    NarrowTracker nt;
    Track t = make_track(1, 500, 300, /*vx*/0, /*vy*/0, /*missed*/3);  // missed >= 1 -> degraded
    nt.update(&t, true);
    ASSERT_TRUE(nt.state().last_degraded);
}

TEST(Narrow_DegradedFlagOnLowConfidence) {
    NarrowTracker nt;
    Track t = make_track(1, 500, 300, 0, 0, 0, /*conf*/0.10f);  // < 0.18 -> degraded
    nt.update(&t, true);
    ASSERT_TRUE(nt.state().last_degraded);
}
