// TargetManager — wybor globalnego owner-a z listy trackow MTT.
// D2 upgrade: identity anchor + continuity guard (parity z src/core/target_manager.py).
//
// Identity anchor zapisywany przy commit ownera: area, velocity, formation
// signature (top-3 sasiedzi: dx, dy, area_ratio). Continuity guard blokuje
// switch jesli best_continuity < 0.9 * active_continuity -- chroni przed
// przelaczeniem na geometrycznie-zblizonego sasiada w formacji
// identycznych obiektow (klasyczny scenariusz: dron + odbicie w wodzie,
// stado samolotow akrobatycznych).
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

    // D2 continuity guard
    int formation_top_n = 3;            // ile najblizszych sasiadow w signature
    double continuity_motion_sigma = 15.0;
    double continuity_scale_sigma = 0.5;
    double continuity_formation_decay = 50.0;
    double continuity_block_ratio = 0.9;  // best < active * 0.9 -> block

    // D2 re-acquisition (gdy current zniknie z tracks):
    float reacquire_radius_auto = 135.0f;  // promien search wokol anchora
    float reacquire_radius_growth = 0.08f; // 8% wzrostu / klatkę bez ownera
    float reacquire_radius_max_growth = 0.80f;
    // Fix 3: stale_owner zwiekszony 10 -> 50 dla synchronizacji z narrow
    // max_hold_frames=100. Bez tego TM dropuje persistent ownera po 10 klatkach
    // missed mimo ze narrow trzyma synthetic crop przez kolejne 50.
    int   stale_owner_frames = 50;
    // Fix 3: reacquire_persist 2 -> 4 (kandydat musi wygrac 4 klatki)
    int   reacquire_persist = 4;
    double reacquire_min_continuity = 0.35; // continuity_score min do adopt
};

// (dx, dy, area_ratio) wzgledem ownera dla jednego sasiada
struct FormationFeature {
    double dx = 0.0;
    double dy = 0.0;
    double area_ratio = 1.0;
};

struct IdentityAnchor {
    double area = 0.0;
    Point2 velocity{0.0, 0.0};
    std::vector<FormationFeature> formation_sig;
    int frame_idx = 0;
};

struct TargetState {
    std::optional<int> selected_id;   // track_id aktualnego owner-a
    int frames_since_switch = 0;
    int candidate_id = -1;
    int candidate_dwell = 0;
    bool manual_lock = false;
    int manual_lock_id = -1;

    // D2: frame counter + identity tracking
    int frame_id = 0;
    std::optional<IdentityAnchor> identity_anchor;
    int identity_anchor_freshness = 0;
    std::optional<Point2> last_selected_center;
    std::optional<BBox> last_selected_bbox;

    // D2: re-acquisition state -- current zniknal z tracks, czekamy na powrot
    int selected_missing_frames = 0;
    int reacquire_pending_id = -1;
    int reacquire_pending_count = 0;

    // Fix 2: persistent owner ID -- niezalezny od raw track_id z MTT.
    // Inkrementowany przy fresh selection / dwell-switch / stale-escape.
    // Zachowywany przy re-acquisition adopt (ten sam fizyczny obiekt).
    int next_persistent_id = 0;
    int current_persistent_id = -1;

    // Diag: ostatni continuity score (na potrzeby telemetrii / debug)
    std::optional<double> last_continuity_score_active;
    std::optional<double> last_continuity_score_best;
    bool last_continuity_block = false;
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

    // Fix 2: persistent owner ID (logical, niezalezny od raw track_id).
    // -1 gdy brak ownera, inaczej rosnacy licznik nadanego logical ID.
    int persistent_owner_id() const { return state_.current_persistent_id; }

private:
    float score_candidate_(const Track& tr, bool is_current) const;

    // D2: identity anchor + continuity guard
    std::vector<FormationFeature> compute_formation_signature_(
        const Track& tr, const std::vector<Track>& tracks) const;
    double signature_similarity_(
        const std::vector<FormationFeature>& a,
        const std::vector<FormationFeature>& b) const;
    std::optional<double> continuity_score_(
        const Track& tr, const std::vector<Track>& tracks) const;
    void set_identity_anchor_(const Track& tr, const std::vector<Track>& tracks);
    void store_owner_reference_(const Track& tr);
    // fresh_persistent=true -> nowy persistent_owner_id (rozne fizycznie obiekty)
    // fresh_persistent=false -> zachowaj current_persistent_id (re-acquisition adopt)
    void commit_owner_(const Track& tr, const std::vector<Track>& tracks,
                       bool fresh_persistent);

    TMConfig cfg_;
    TargetState state_;
};

}  // namespace dtracker
