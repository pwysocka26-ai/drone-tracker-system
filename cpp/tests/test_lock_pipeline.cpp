// Parity testy LockPipeline — FSM (UNLOCKED → ACQUIRE → LOCKED → HOLD → REACQUIRE).
// Pokrywa transitions + counters (lock_loss_events, reacquire_*).
#include "test_framework.hpp"
#include "dtracker/lock_pipeline.hpp"
#include "dtracker/track.hpp"

#include <ostream>

namespace dtracker {
// ostream operator dla ASSERT_EQ output (LockState to enum class)
inline std::ostream& operator<<(std::ostream& o, LockState s) {
    return o << to_string(s);
}
}

using namespace dtracker;

// Helper: synthetic Track. Domyslnie: confirmed, healthy, hits=10, missed=0.
static Track make_track(int id, float cx, float cy, int missed = 0, int hits = 10,
                         float conf = 0.5f, bool confirmed = true) {
    Track t;
    t.track_id = id;
    t.raw_id = id;
    t.bbox = BBox{cx - 15, cy - 15, cx + 15, cy + 15};
    t.center = Point2{cx, cy};
    t.confidence = conf;
    t.hits = hits;
    t.missed_frames = missed;
    t.is_confirmed = confirmed;
    t.is_valid_target = true;
    return t;
}

// FSM: UNLOCKED -> ACQUIRE (step 1) -> LOCKED (step 2 jesli hits>=min_hits).
// Helper drive'uje pipeline do LOCKED (max 4 step'y), umiernie pasywuje zalozenia.
static void drive_to_locked(LockPipeline& lp, int sel, const std::vector<Track>& tracks) {
    for (int i = 0; i < 4 && lp.current() != LockState::LOCKED; ++i) {
        lp.step(sel, tracks);
    }
}

TEST(Lock_InitialStateIsUnlocked) {
    LockPipeline lp;
    ASSERT_EQ(lp.current(), LockState::UNLOCKED);
}

TEST(Lock_UnlockedToAcquireWhenOwnerProvided) {
    LockPipeline lp;
    std::vector<Track> tracks; tracks.push_back(make_track(1, 100, 100, /*missed*/0, /*hits*/1));
    LockState s = lp.step(1, tracks);
    ASSERT_EQ(s, LockState::ACQUIRE);
}

TEST(Lock_AcquireToLockedWhenHitsReachThreshold) {
    LockPipeline lp;  // acquire_min_hits=5 default
    // hits=10 >= 5 i is_confirmed=true → po 2 stepach LOCKED
    std::vector<Track> tracks; tracks.push_back(make_track(1, 100, 100, 0, 10));
    lp.step(1, tracks);            // UNLOCKED -> ACQUIRE
    LockState s = lp.step(1, tracks);  // ACQUIRE -> LOCKED
    ASSERT_EQ(s, LockState::LOCKED);
}

TEST(Lock_AcquireStaysAcquireBelowThreshold) {
    LockPipelineConfig cfg;
    cfg.acquire_min_hits = 5;
    LockPipeline lp(cfg);
    std::vector<Track> tracks; tracks.push_back(make_track(1, 100, 100, 0, 3));  // hits=3 < 5
    LockState s = lp.step(1, tracks);
    ASSERT_EQ(s, LockState::ACQUIRE);
}

TEST(Lock_LockedToHoldOnSingleMiss) {
    // hold_limit=50 default. missed=1 -> HOLD (1 <= 50).
    LockPipeline lp;
    std::vector<Track> tracks; tracks.push_back(make_track(1, 100, 100, 0, 10));
    drive_to_locked(lp, 1, tracks);
    ASSERT_EQ(lp.current(), LockState::LOCKED);

    tracks[0].missed_frames = 1;
    LockState s = lp.step(1, tracks);
    ASSERT_EQ(s, LockState::HOLD);
}

TEST(Lock_HoldToLockedWhenOwnerReappears) {
    LockPipeline lp;
    std::vector<Track> tracks; tracks.push_back(make_track(1, 100, 100, 0, 10));
    drive_to_locked(lp, 1, tracks);

    tracks[0].missed_frames = 1;
    lp.step(1, tracks);  // HOLD

    tracks[0].missed_frames = 0;
    LockState s = lp.step(1, tracks);
    ASSERT_EQ(s, LockState::LOCKED);
}

TEST(Lock_LockedToReacquireWhenMissedExceedsHoldLimit) {
    LockPipelineConfig cfg;
    cfg.hold_limit = 5;
    LockPipeline lp(cfg);
    std::vector<Track> tracks; tracks.push_back(make_track(1, 100, 100, 0, 10));
    drive_to_locked(lp, 1, tracks);

    tracks[0].missed_frames = 6;  // > hold_limit
    LockState s = lp.step(1, tracks);
    ASSERT_EQ(s, LockState::REACQUIRE);
}

TEST(Lock_ReacquireFindsCandidateNearAnchor) {
    LockPipelineConfig cfg;
    cfg.hold_limit = 5;
    cfg.reacquire_max_dist = 50.0f;
    LockPipeline lp(cfg);
    std::vector<Track> tracks; tracks.push_back(make_track(1, 100, 100));
    drive_to_locked(lp, 1, tracks);
    tracks[0].missed_frames = 6;
    lp.step(1, tracks);  // REACQUIRE

    // Owner zniknal, pojawia sie nowy track id=2 blisko anchora
    tracks.clear();
    tracks.push_back(make_track(2, 110, 105));  // dist ~ 11 (< 50)
    LockState s = lp.step(2, tracks);
    ASSERT_EQ(s, LockState::LOCKED);
}

TEST(Lock_ReacquireTimeoutGoesUnlocked) {
    LockPipelineConfig cfg;
    cfg.reacquire_limit = 3;
    cfg.hold_limit = 5;
    LockPipeline lp(cfg);
    std::vector<Track> tracks; tracks.push_back(make_track(1, 100, 100));
    drive_to_locked(lp, 1, tracks);
    tracks[0].missed_frames = 10;
    LockState s = lp.step(1, tracks);  // REACQUIRE
    ASSERT_EQ(s, LockState::REACQUIRE);

    // Brak kandydatow przez reacquire_limit klatek -> UNLOCKED
    std::vector<Track> empty;
    for (int i = 0; i < 5; ++i) {
        s = lp.step(std::nullopt, empty);
    }
    ASSERT_EQ(s, LockState::UNLOCKED);
}

TEST(Lock_LockLossEventCounter) {
    LockPipeline lp;
    std::vector<Track> tracks; tracks.push_back(make_track(1, 100, 100));
    drive_to_locked(lp, 1, tracks);
    ASSERT_EQ(lp.state().total_lock_loss_events, 0);

    // LOCKED -> REACQUIRE = 1 lock_loss_event
    tracks[0].missed_frames = 100;  // >> hold_limit
    lp.step(1, tracks);  // REACQUIRE
    ASSERT_EQ(lp.state().total_lock_loss_events, 1);
    ASSERT_EQ(lp.state().total_reacquire_starts, 1);
}

TEST(Lock_ReacquireSuccessCounter) {
    LockPipelineConfig cfg;
    cfg.hold_limit = 2;
    LockPipeline lp(cfg);

    std::vector<Track> tracks; tracks.push_back(make_track(1, 100, 100));
    drive_to_locked(lp, 1, tracks);
    tracks[0].missed_frames = 5;
    lp.step(1, tracks);  // -> REACQUIRE

    // Kandydat -> LOCKED (success)
    tracks.clear();
    tracks.push_back(make_track(2, 105, 100));
    lp.step(2, tracks);
    ASSERT_EQ(lp.state().total_reacquire_successes, 1);
}

TEST(Lock_FrameCountersIncrement) {
    LockPipeline lp;
    std::vector<Track> tracks; tracks.push_back(make_track(1, 100, 100));
    drive_to_locked(lp, 1, tracks);  // 2 stepy zazwyczaj
    int locked_before = lp.state().total_frames_locked;
    lp.step(1, tracks);
    lp.step(1, tracks);
    ASSERT_TRUE(lp.state().total_frames_locked > locked_before);
}
