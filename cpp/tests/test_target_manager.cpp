// Parity testy TargetManager. Pokrywa identity_anchor + persistent_owner_id +
// re-acquisition (Fix 2 + Fix 3 z 2026-04-25).
#include "test_framework.hpp"
#include "dtracker/target_manager.hpp"
#include "dtracker/multi_target_tracker.hpp"

using namespace dtracker;

static Detection make_det(float cx, float cy, float w = 30.0f, float h = 30.0f, float conf = 0.5f) {
    Detection d;
    d.bbox.x1 = cx - w / 2.0f;
    d.bbox.y1 = cy - h / 2.0f;
    d.bbox.x2 = cx + w / 2.0f;
    d.bbox.y2 = cy + h / 2.0f;
    d.conf = conf;
    d.cls = 0;
    return d;
}

// Helper: zbuduj realistyczne tracks przez MTT z synthetic detections
static std::vector<Track> drive_mtt(MultiTargetTracker& mtt, const Detections& dets) {
    return mtt.update(dets);
}

TEST(TM_FirstSelectionAssignsPersistentOne) {
    MultiTargetTracker mtt;
    TargetManager tm;

    // Owner musi byc confirmed -- 2 frames z ta sama detekcja
    drive_mtt(mtt, { make_det(100, 100) });
    auto tracks = drive_mtt(mtt, { make_det(100, 100) });
    auto sel = tm.select(tracks);
    ASSERT_TRUE(sel.has_value());
    ASSERT_EQ(tm.persistent_owner_id(), 1);
}

TEST(TM_PersistentStableThroughTrackIdShuffle) {
    // Scenariusz: real owner gubi sie (track_id=1 missed > max_select_missed)
    // ale anchor + last_good zostaje. Pojawia sie blisko track o nowym ID.
    // Re-acquisition adopt -> persistent zostaje #1.
    MultiTargetTracker mtt;
    TMConfig tcfg;
    tcfg.reacquire_persist = 2;     // szybciej do testu
    tcfg.stale_owner_frames = 50;
    TargetManager tm(tcfg);

    // 5 klatek detekcji -> commit owner #1 (track_id=1 z MTT)
    std::vector<Track> last_tracks;
    for (int i = 0; i < 5; ++i) {
        last_tracks = drive_mtt(mtt, { make_det(100.0f + i, 100.0f + i) });
        tm.select(last_tracks);
    }
    int persistent_before = tm.persistent_owner_id();
    ASSERT_TRUE(persistent_before > 0);

    // Klatki 6-15: brak detekcji -> track_id=1 missed_frames++
    for (int i = 0; i < 10; ++i) {
        last_tracks = drive_mtt(mtt, {});
        tm.select(last_tracks);
    }

    // Klatki 16-20: nowa detekcja blisko anchora
    for (int i = 0; i < 5; ++i) {
        last_tracks = drive_mtt(mtt, { make_det(110.0f, 110.0f) });
        tm.select(last_tracks);
    }

    // Persistent_owner_id powinno byc niezmienione
    ASSERT_EQ(tm.persistent_owner_id(), persistent_before);
}

TEST(TM_StaleOwnerEscapeIncrementsPersistent) {
    // Po stale_owner_frames bez re-acquisition -> fresh persistent.
    MultiTargetTracker mtt;
    TMConfig tcfg;
    tcfg.stale_owner_frames = 5;  // szybciej
    tcfg.reacquire_persist = 100; // praktycznie nigdy nie adopt blisko
    TargetManager tm(tcfg);

    // Commit pierwszego ownera
    std::vector<Track> tracks;
    for (int i = 0; i < 3; ++i) {
        tracks = drive_mtt(mtt, { make_det(100.0f, 100.0f) });
        tm.select(tracks);
    }
    int p1 = tm.persistent_owner_id();
    ASSERT_TRUE(p1 > 0);

    // Wszystkie tracki gina przez 6 klatek bez detekcji (max_missed_frames=36 default - more)
    // Ale stale_owner_frames=5 wyzwala escape juz po 5 klatkach bez ownera
    for (int i = 0; i < 10; ++i) {
        tracks = drive_mtt(mtt, {});
        tm.select(tracks);
    }

    // Pojawia sie zupelnie inna detekcja DALEKO -- nie adopt, nowy owner
    for (int i = 0; i < 3; ++i) {
        tracks = drive_mtt(mtt, { make_det(1500.0f, 800.0f) });
        tm.select(tracks);
    }

    // Persistent powinno byc inkrementowane (nowy fizyczny obiekt)
    int p2 = tm.persistent_owner_id();
    ASSERT_TRUE(p2 > p1);
}

TEST(TM_ManualLockSetsFreshPersistent) {
    MultiTargetTracker mtt;
    TargetManager tm;
    auto tracks = drive_mtt(mtt, { make_det(100, 100) });
    drive_mtt(mtt, { make_det(100, 100) });
    tm.select(tracks);
    int p_before = tm.persistent_owner_id();
    tm.set_manual_lock(42);
    int p_after = tm.persistent_owner_id();
    ASSERT_TRUE(p_after > p_before);
}

TEST(TM_NoTracksReturnsNullopt) {
    MultiTargetTracker mtt;
    TargetManager tm;
    auto sel = tm.select({});
    ASSERT_FALSE(sel.has_value());
    ASSERT_EQ(tm.persistent_owner_id(), -1);
}

TEST(TM_IdentityAnchorSetAfterCommit) {
    MultiTargetTracker mtt;
    TargetManager tm;
    drive_mtt(mtt, { make_det(100, 100) });
    auto tracks = drive_mtt(mtt, { make_det(100, 100) });
    tm.select(tracks);
    ASSERT_TRUE(tm.state().identity_anchor.has_value());
    ASSERT_TRUE(tm.state().last_selected_center.has_value());
}
