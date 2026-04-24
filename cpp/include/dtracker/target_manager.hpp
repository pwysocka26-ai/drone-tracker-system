// TargetManager (basic) — wybor globalnego owner-a z listy trackow MTT.
// Uproszczony vs Python: bez full identity anchor, bez continuity guard,
// bez selection_freeze (dodajemy post-demo).
#pragma once

#include <optional>
#include <vector>

#include "dtracker/track.hpp"

namespace dtracker {

struct TMConfig {
    int max_select_missed = 4;        // po ilu missed frames owner jest porzucany
    int sticky_frames = 24;           // ile klatek trzymamy current ownera przy braku lepszego
    float switch_margin = 0.55f;      // nowy kandydat musi byc x razy lepszy niz obecny
    int switch_dwell = 8;             // ile klatek nowy kandydat musi wygrywac
    int target_selection_freeze_frames = 14;  // po owner_switch freeze na N klatek
    float current_target_bonus = 5.4f;
    float raw_id_bonus = 2.2f;

    // Score = conf * size_norm + bonus(current) + bonus(raw_id_match)
    // Rozmiar: sqrt(area) / max(frame_size)  -- normalized
};

struct TargetState {
    std::optional<int> selected_id;   // track_id aktualnego owner-a
    int frames_since_switch = 0;
    int candidate_id = -1;
    int candidate_dwell = 0;
    bool manual_lock = false;
    int manual_lock_id = -1;
};

class TargetManager {
public:
    explicit TargetManager(TMConfig cfg = {});

    // Wybierz owner-a (lub zostaw current) na podstawie trackow.
    // Zwraca track_id wybranego owner-a albo nullopt jesli brak.
    std::optional<int> select(const std::vector<Track>& tracks);

    void set_manual_lock(int track_id);
    void clear_manual_lock();
    bool manual_lock() const { return state_.manual_lock; }

    const TargetState& state() const { return state_; }

private:
    float score_candidate_(const Track& tr, bool is_current) const;

    TMConfig cfg_;
    TargetState state_;
};

}  // namespace dtracker
