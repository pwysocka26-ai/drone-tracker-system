// LockPipeline — high-level FSM zarzadzajacy stanem lock-u na wybranym owner-ze.
// Uproszczony port z Python (na demo 30.04):
//   ACQUIRE -> detekcja pojawila sie, startuje tracking
//   LOCKED  -> stabilne sledzenie
//   HOLD    -> krotka luka detekcji (owner missed_frames > 0, ale <= hold_limit)
//   REACQUIRE -> duza luka (owner missed > hold_limit), probujemy odzyskac
//   UNLOCKED -> initial / utracone
#pragma once

#include <optional>

#include "dtracker/track.hpp"

namespace dtracker {

enum class LockState {
    UNLOCKED,
    ACQUIRE,    // mamy owner-a, potrzebne N hits zeby LOCKED
    LOCKED,
    HOLD,       // owner missed od 1 do hold_limit
    REACQUIRE,  // owner missed > hold_limit
};

const char* to_string(LockState s);

struct LockPipelineConfig {
    // 2026-05-14 sesja stabilizacja: 5 -> 3. Empirically v8 na test.mp4 dawal
    // hits=4 dopiero w klatce 155, hits=5 dopiero 160 (50 klatek od ACQUIRE).
    // 3 hits = stabilna detekcja, ale bez ryzyka fala-as-drone (FP zwykle 1 hit).
    int acquire_min_hits = 3;         // ile hits przed przejsciem ACQUIRE -> LOCKED
    // Fix 4: hold_limit 12 -> 50 sync z TMConfig.stale_owner_frames=50.
    // Bez tego Lock FSM rzucal w REACQUIRE po 12 klatkach mimo ze TM trzymal
    // persistent ownera przez kolejne 38 klatek (telemetria: lock_loss_events
    // 51 -> 100 z fix 3, oczekiwany powrót do <40).
    int hold_limit = 50;
    int reacquire_limit = 36;         // max missed_frames w REACQUIRE
    float reacquire_min_conf = 0.15f; // min conf kandydata do reacquire
    float reacquire_max_dist = 150.0f;// max odleglosc (px) od last-known-pos
    // 2026-05-14: hysteresis na LOCKED. Bez tego state oscyluje LOCKED<->HOLD
    // co 1-3 klatki przy każdej missed detection (czysto kosmetyczne, bo track
    // żyje z hold_limit=50, ale UX indicator chaos). Z hysteresis=3:
    // LOCKED zostaje LOCKED do missed_frames<3, dopiero potem HOLD. Detection
    // recovery natychmiastowa (HOLD->LOCKED na missed_frames==0 bez hysteresis).
    int lock_hysteresis_misses = 3;
};

struct LockPipelineState {
    LockState state = LockState::UNLOCKED;
    std::optional<int> owner_id;
    Point2 last_known_center{0, 0};
    int frames_in_state = 0;
    int total_lock_loss_events = 0;
    int total_reacquire_starts = 0;
    int total_reacquire_successes = 0;
    int total_frames_locked = 0;
    int total_frames_hold = 0;
    int total_frames_reacquire = 0;
};

class LockPipeline {
public:
    explicit LockPipeline(LockPipelineConfig cfg = {});

    // Step: podaj aktualnie wybranego owner-a (z TargetManager.select()),
    // oraz wszystkie tracki (zeby REACQUIRE mogl szukac).
    // Zwraca stan po update.
    LockState step(std::optional<int> selected_id, const std::vector<Track>& tracks);

    const LockPipelineState& state() const { return s_; }
    LockState current() const { return s_.state; }

private:
    // Pomocnicze
    const Track* find_owner_(int owner_id, const std::vector<Track>& tracks) const;
    int find_reacquire_candidate_(const std::vector<Track>& tracks) const;

    void transit_(LockState new_state);

    LockPipelineConfig cfg_;
    LockPipelineState s_;
};

}  // namespace dtracker
