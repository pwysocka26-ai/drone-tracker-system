// Parity testy MultiTargetTracker. Synthetic detections -> sprawdz ze:
// - track_id stable przy konsekwentnych detekcjach
// - missed_frames inkrementuje gdy detekcja znika
// - track umiera po max_missed_frames
// - matching wybiera najblizsze detekcje (greedy)
#include "test_framework.hpp"
#include "dtracker/multi_target_tracker.hpp"

using namespace dtracker;

static Detection make_det(float cx, float cy, float w = 30.0f, float h = 30.0f, float conf = 0.5f, int cls = 0) {
    Detection d;
    d.bbox.x1 = cx - w / 2.0f;
    d.bbox.y1 = cy - h / 2.0f;
    d.bbox.x2 = cx + w / 2.0f;
    d.bbox.y2 = cy + h / 2.0f;
    d.conf = conf;
    d.cls = cls;
    return d;
}

TEST(MTT_SpawnFirstDetection) {
    MultiTargetTracker mtt;
    Detections dets = { make_det(100, 100) };
    auto tracks = mtt.update(dets);
    ASSERT_EQ(tracks.size(), static_cast<size_t>(1));
    ASSERT_EQ(tracks[0].track_id, 1);
    ASSERT_EQ(tracks[0].hits, 1);
    ASSERT_EQ(tracks[0].missed_frames, 0);
}

TEST(MTT_TrackIdStableUnderConsistentDetection) {
    MultiTargetTracker mtt;
    int last_id = -1;
    for (int i = 0; i < 10; ++i) {
        // Drone moves slightly each frame
        Detections dets = { make_det(100.0f + i * 2.0f, 100.0f + i * 1.0f) };
        auto tracks = mtt.update(dets);
        ASSERT_EQ(tracks.size(), static_cast<size_t>(1));
        if (last_id < 0) last_id = tracks[0].track_id;
        ASSERT_EQ(tracks[0].track_id, last_id);
    }
}

TEST(MTT_TrackBecomesConfirmedAfterHits) {
    MTTConfig cfg;
    cfg.confirm_hits = 2;
    MultiTargetTracker mtt(cfg);
    Detections dets = { make_det(100, 100) };
    auto t1 = mtt.update(dets);
    ASSERT_FALSE(t1[0].is_confirmed);  // 1 hit, < 2
    auto t2 = mtt.update(dets);
    ASSERT_TRUE(t2[0].is_confirmed);   // 2 hits, >= 2
}

TEST(MTT_MissedFramesIncrement) {
    MultiTargetTracker mtt;
    auto t1 = mtt.update({ make_det(100, 100) });
    ASSERT_EQ(t1[0].missed_frames, 0);
    auto t2 = mtt.update({});  // brak detekcji
    ASSERT_EQ(t2.size(), static_cast<size_t>(1));
    ASSERT_EQ(t2[0].missed_frames, 1);
    auto t3 = mtt.update({});
    ASSERT_EQ(t3[0].missed_frames, 2);
}

TEST(MTT_TrackDiesAfterMaxMissed) {
    MTTConfig cfg;
    cfg.max_missed_frames = 3;
    MultiTargetTracker mtt(cfg);
    mtt.update({ make_det(100, 100) });
    auto t1 = mtt.update({});
    ASSERT_EQ(t1.size(), static_cast<size_t>(1));  // missed=1
    auto t2 = mtt.update({});                       // missed=2
    auto t3 = mtt.update({});                       // missed=3
    auto t4 = mtt.update({});                       // missed=4 > 3 -> remove
    ASSERT_EQ(t4.size(), static_cast<size_t>(0));
}

TEST(MTT_TwoSeparateDetectionsTwoTracks) {
    MultiTargetTracker mtt;
    Detections dets = { make_det(100, 100), make_det(500, 500) };
    auto tracks = mtt.update(dets);
    ASSERT_EQ(tracks.size(), static_cast<size_t>(2));
    ASSERT_TRUE(tracks[0].track_id != tracks[1].track_id);
}

TEST(MTT_GreedyMatchingPicksClosest) {
    MultiTargetTracker mtt;
    // Frame 1: dwa tracki
    auto t1 = mtt.update({ make_det(100, 100), make_det(500, 500) });
    int id_a = t1[0].track_id;
    int id_b = t1[1].track_id;
    // Frame 2: detekcje przeniesione blisko tych pozycji + 1 nowa
    auto t2 = mtt.update({
        make_det(105, 102),  // bliska id_a
        make_det(495, 505),  // bliska id_b
        make_det(900, 900),  // nowa
    });
    ASSERT_EQ(t2.size(), static_cast<size_t>(3));
    // ID powinny pozostac stale dla pierwszych dwoch
    bool found_a = false, found_b = false;
    int new_id = -1;
    for (const auto& t : t2) {
        if (t.track_id == id_a) found_a = true;
        else if (t.track_id == id_b) found_b = true;
        else new_id = t.track_id;
    }
    ASSERT_TRUE(found_a);
    ASSERT_TRUE(found_b);
    ASSERT_TRUE(new_id > 0);
}

TEST(MTT_VelocityConvergesAfterSeveralFrames) {
    MultiTargetTracker mtt;
    // Drone porusza sie z velocity (10, 5). Kalman P_velocity init=4.0,
    // measurement_noise=0.20, process_noise=0.03 -> convergence steady-state
    // wymaga ~15+ frames. Po 15 klatek velocity powinno byc w okolicach (10, 5).
    std::vector<Track> tracks;
    for (int i = 0; i < 20; ++i) {
        Detections dets = { make_det(100.0f + i * 10.0f, 100.0f + i * 5.0f) };
        tracks = mtt.update(dets);
    }
    // Po 20 klatkach Kalman steady-state: velocity convergence w okolicach truth
    ASSERT_NEAR(tracks[0].velocity.x, 10.0, 4.0);  // tolerancja +/- 40%
    ASSERT_NEAR(tracks[0].velocity.y, 5.0, 2.5);
}

TEST(MTT_ResetClearsTracks) {
    MultiTargetTracker mtt;
    mtt.update({ make_det(100, 100) });
    mtt.reset();
    auto tracks = mtt.update({});
    ASSERT_EQ(tracks.size(), static_cast<size_t>(0));
    // Po reset next_id zaczyna od 1 znowu
    auto t1 = mtt.update({ make_det(50, 50) });
    ASSERT_EQ(t1[0].track_id, 1);
}
