#include "dtracker/target_manager.hpp"

#include <algorithm>
#include <cmath>

namespace dtracker {

TargetManager::TargetManager(TMConfig cfg) : cfg_(cfg) {}

void TargetManager::set_manual_lock(int track_id) {
    state_.manual_lock = true;
    state_.manual_lock_id = track_id;
    state_.selected_id = track_id;
    state_.frames_since_switch = 0;
    // Fix 2: manual lock = fresh logical owner (uzytkownik wybral inny cel)
    state_.current_persistent_id = ++state_.next_persistent_id;
}

void TargetManager::clear_manual_lock() {
    state_.manual_lock = false;
    state_.manual_lock_id = -1;
}

// ---------- D2: identity continuity ----------

std::vector<FormationFeature>
TargetManager::compute_formation_signature_(const Track& tr,
                                            const std::vector<Track>& tracks) const {
    Point2 center = tr.center;
    double area = std::max(1.0, static_cast<double>(tr.bbox.area()));
    int tid = tr.track_id;

    // (dist^2, feature) -- zeby posortowac
    std::vector<std::pair<double, FormationFeature>> pool;
    pool.reserve(tracks.size());
    for (const auto& nb : tracks) {
        if (nb.track_id == tid) continue;
        double narea = std::max(1.0, static_cast<double>(nb.bbox.area()));
        FormationFeature f;
        f.dx = nb.center.x - center.x;
        f.dy = nb.center.y - center.y;
        f.area_ratio = narea / area;
        double d2 = f.dx * f.dx + f.dy * f.dy;
        pool.emplace_back(d2, f);
    }
    std::sort(pool.begin(), pool.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<FormationFeature> out;
    int n = std::min(static_cast<int>(pool.size()), cfg_.formation_top_n);
    out.reserve(n);
    for (int i = 0; i < n; ++i) out.push_back(pool[i].second);
    return out;
}

double TargetManager::signature_similarity_(
        const std::vector<FormationFeature>& a,
        const std::vector<FormationFeature>& b) const {
    if (a.empty() || b.empty()) return 0.5;  // neutralny gdy ktorys pusty
    int n = static_cast<int>(std::min(a.size(), b.size()));
    if (n == 0) return 0.5;
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        double dist_diff = std::hypot(a[i].dx - b[i].dx, a[i].dy - b[i].dy);
        double ar_diff = std::abs(std::log(std::max(0.01, a[i].area_ratio) /
                                             std::max(0.01, b[i].area_ratio)));
        total += std::exp(-dist_diff / cfg_.continuity_formation_decay) * std::exp(-ar_diff);
    }
    return total / n;
}

std::optional<double>
TargetManager::continuity_score_(const Track& tr, const std::vector<Track>& tracks) const {
    if (!state_.identity_anchor) return std::nullopt;
    const auto& anchor = *state_.identity_anchor;

    // Motion: velocity match (gauss exp)
    double dvx = tr.velocity.x - anchor.velocity.x;
    double dvy = tr.velocity.y - anchor.velocity.y;
    double dv = std::hypot(dvx, dvy);
    double sigma_m = cfg_.continuity_motion_sigma;
    double motion = std::exp(-(dv * dv) / (2.0 * sigma_m * sigma_m));

    // Scale: log area ratio (gauss exp)
    double area = std::max(1.0, static_cast<double>(tr.bbox.area()));
    double anchor_area = std::max(1.0, anchor.area);
    double log_ratio = std::log(area / anchor_area);
    double sigma_s = cfg_.continuity_scale_sigma;
    double scale = std::exp(-(log_ratio * log_ratio) / (2.0 * sigma_s * sigma_s));

    // Formation: top-N nearest neighbours signature
    auto current_sig = compute_formation_signature_(tr, tracks);
    double formation = signature_similarity_(current_sig, anchor.formation_sig);

    return 0.35 * motion + 0.20 * scale + 0.45 * formation;
}

void TargetManager::set_identity_anchor_(const Track& tr,
                                          const std::vector<Track>& tracks) {
    IdentityAnchor a;
    a.area = std::max(1.0, static_cast<double>(tr.bbox.area()));
    a.velocity = tr.velocity;
    a.formation_sig = compute_formation_signature_(tr, tracks);
    a.frame_idx = state_.frame_id;
    state_.identity_anchor = a;
    state_.identity_anchor_freshness = 0;
}

void TargetManager::store_owner_reference_(const Track& tr) {
    state_.last_selected_center = tr.center;
    state_.last_selected_bbox = tr.bbox;
}

void TargetManager::commit_owner_(const Track& tr, const std::vector<Track>& tracks,
                                    bool fresh_persistent) {
    state_.selected_id = tr.track_id;
    state_.frames_since_switch = 0;
    state_.candidate_id = -1;
    state_.candidate_dwell = 0;
    set_identity_anchor_(tr, tracks);
    store_owner_reference_(tr);
    // Fix 2: nowy logical ID dla fizycznie nowego ownera, keep dla re-acquisition.
    if (fresh_persistent || state_.current_persistent_id < 0) {
        state_.current_persistent_id = ++state_.next_persistent_id;
    }
}

float TargetManager::score_candidate_(const Track& tr, bool is_current) const {
    // Rozmiar bbox normalizowany (proxy — wiekszy dron = wyrazniejszy cel)
    float w = std::max(1.0f, tr.bbox.x2 - tr.bbox.x1);
    float h = std::max(1.0f, tr.bbox.y2 - tr.bbox.y1);
    float size_score = std::sqrt(w * h) / 100.0f;

    float score = tr.confidence * 10.0f + size_score;
    if (is_current) score += cfg_.current_target_bonus;

    // Penalty za missed frames
    score -= 0.5f * tr.missed_frames;

    // Preferuj potwierdzone
    if (!tr.is_confirmed) score *= 0.5f;

    return score;
}

// D2: promien re-acquisition rosnie z czasem braku ownera (Python:_recovery_radius)
static double recovery_radius(const TMConfig& cfg, int missing_frames) {
    double growth = 1.0 + std::min(static_cast<double>(cfg.reacquire_radius_max_growth),
                                     static_cast<double>(cfg.reacquire_radius_growth) *
                                         static_cast<double>(missing_frames));
    return static_cast<double>(cfg.reacquire_radius_auto) * growth;
}

std::optional<int> TargetManager::select(const std::vector<Track>& tracks) {
    // D2: frame counter + identity_anchor freshness
    ++state_.frame_id;
    if (state_.identity_anchor) {
        ++state_.identity_anchor_freshness;
    }
    state_.last_continuity_block = false;
    state_.last_continuity_score_active.reset();
    state_.last_continuity_score_best.reset();

    // Manual lock: trzymaj forsownie, nawet jak owner zniknal
    if (state_.manual_lock) {
        state_.selected_id = state_.manual_lock_id;
        // Sprawdz czy track istnieje; jesli nie, zostaw selected (freeze) — recovery zrobi
        for (const auto& t : tracks) {
            if (t.track_id == state_.manual_lock_id) {
                store_owner_reference_(t);
                if (!state_.identity_anchor) set_identity_anchor_(t, tracks);
                break;
            }
        }
        return state_.selected_id;
    }

    if (tracks.empty()) {
        // Trzymaj selected_id ale increment selected_missing_frames -- gdy tracki
        // wroca, reacquisition mechanism sprobuje odzyskac tego samego ownera.
        if (state_.selected_id) ++state_.selected_missing_frames;
        return state_.selected_id;
    }

    // Znajdz current track w liscie
    const Track* current = nullptr;
    if (state_.selected_id) {
        for (const auto& t : tracks) {
            if (t.track_id == *state_.selected_id) { current = &t; break; }
        }
    }

    // D2: NIE porzucaj selected_id natychmiast po missed > max_select_missed.
    // Trzymaj go (anchor zostaje aktywny) i probuj re-acquire na innym track_id
    // ktory pasuje geometrycznie/continuity. Zerwij selected_id dopiero gdy
    // stale_owner_frames upłynęło i nikogo blisko nie ma.
    if (current && current->missed_frames > cfg_.max_select_missed) {
        // Owner technicznie nadal w tracks ale martwy z punktu widzenia matchera.
        // Traktuj jak missing -- przechodzimy do re-acquisition w nizszej galezi.
        current = nullptr;
        ++state_.selected_missing_frames;
    } else if (current) {
        state_.selected_missing_frames = 0;
    } else if (state_.selected_id) {
        // selected_id istnieje ale current==nullptr (wcale nie w tracks)
        ++state_.selected_missing_frames;
    }

    // D2: Re-acquisition path -- selected_id istnieje, ale current jest gone/stale.
    // Szukamy w tracks kandydata z najwyzszym continuity_score do anchora,
    // w promieniu reacquire_radius od last_selected_center.
    if (state_.selected_id && !current && state_.last_selected_center && state_.identity_anchor) {
        const double radius = recovery_radius(cfg_, state_.selected_missing_frames);
        const double radius_sq = radius * radius;
        const Point2 anchor_pos = *state_.last_selected_center;

        int best_reacq_idx = -1;
        double best_cont = cfg_.reacquire_min_continuity;
        for (size_t i = 0; i < tracks.size(); ++i) {
            const auto& t = tracks[i];
            if (t.missed_frames > cfg_.max_select_missed) continue;
            if (t.confidence < 0.08f) continue;

            double dx = t.center.x - anchor_pos.x;
            double dy = t.center.y - anchor_pos.y;
            if (dx * dx + dy * dy > radius_sq) continue;

            auto cont = continuity_score_(t, tracks);
            if (!cont) continue;
            if (*cont > best_cont) {
                best_cont = *cont;
                best_reacq_idx = static_cast<int>(i);
            }
        }

        if (best_reacq_idx >= 0) {
            const Track& cand = tracks[best_reacq_idx];
            // Persist gate: kandydat musi powtorzyc sie N klatek pod rzad
            if (state_.reacquire_pending_id == cand.track_id) {
                ++state_.reacquire_pending_count;
            } else {
                state_.reacquire_pending_id = cand.track_id;
                state_.reacquire_pending_count = 1;
            }
            if (state_.reacquire_pending_count >= cfg_.reacquire_persist) {
                // Adopt as same-owner (re-acquisition) -- update anchor zamiast tworzyc nowy
                state_.selected_id = cand.track_id;
                state_.selected_missing_frames = 0;
                state_.candidate_id = -1;
                state_.candidate_dwell = 0;
                state_.reacquire_pending_id = -1;
                state_.reacquire_pending_count = 0;
                store_owner_reference_(cand);
                set_identity_anchor_(cand, tracks);  // refresh formation_sig
                return state_.selected_id;
            }
            // Czekamy na persist -- zwracamy stary selected_id
            return state_.selected_id;
        }

        // Nikt blisko anchora -- sprawdz czy stale_owner timeout
        if (state_.selected_missing_frames < cfg_.stale_owner_frames) {
            state_.reacquire_pending_id = -1;
            state_.reacquire_pending_count = 0;
            return state_.selected_id;  // jeszcze nie zwalniam ownera
        }
        // stale_owner_frames upłynęło -> escape do best global. Falls through.
    }

    // Score kazdego confirmed track-a
    int best_idx = -1;
    float best_score = -1e9f;
    for (size_t i = 0; i < tracks.size(); ++i) {
        const auto& t = tracks[i];
        if (!t.is_confirmed) continue;
        bool is_cur = current && t.track_id == current->track_id;
        float s = score_candidate_(t, is_cur);
        if (s > best_score) {
            best_score = s;
            best_idx = static_cast<int>(i);
        }
    }

    // Brak confirmed -> bierz najwiekszy hits
    if (best_idx < 0) {
        int max_hits = -1;
        for (size_t i = 0; i < tracks.size(); ++i) {
            if (tracks[i].hits > max_hits) {
                max_hits = tracks[i].hits;
                best_idx = static_cast<int>(i);
            }
        }
    }
    if (best_idx < 0) {
        // Brak kandydata. Trzymaj selected_id (anchor) — wracamy do re-acquisition
        // w nastepnej klatce gdy tracki sie pojawia.
        return state_.selected_id;
    }

    const Track& best = tracks[best_idx];

    // Pierwszy wybor (pustka po starcie) lub stale_owner escape -> nowy commit + nowy anchor
    if (!state_.selected_id || state_.selected_missing_frames >= cfg_.stale_owner_frames) {
        state_.selected_missing_frames = 0;
        state_.reacquire_pending_id = -1;
        state_.reacquire_pending_count = 0;
        // Fix 2: fresh persistent ID -- nowy fizyczny obiekt (start lub stale escape).
        commit_owner_(best, tracks, /*fresh_persistent=*/true);
        return state_.selected_id;
    }

    ++state_.frames_since_switch;

    // Current wygrywa -> reset candidate, refresh reference
    if (current && best.track_id == current->track_id) {
        state_.candidate_id = -1;
        state_.candidate_dwell = 0;
        store_owner_reference_(*current);
        return state_.selected_id;
    }

    // Freeze przez N klatek po switch
    if (state_.frames_since_switch < cfg_.target_selection_freeze_frames) {
        if (current) store_owner_reference_(*current);
        return state_.selected_id;
    }

    // Sticky: nie przelaczaj dopóki sticky_frames nie minie po initial lub ostatnim switch
    if (state_.frames_since_switch < cfg_.sticky_frames) {
        // Candidate must win by margin
        float cur_score = current ? score_candidate_(*current, true) : -1e9f;
        if (best_score < cur_score * (1.0f + cfg_.switch_margin)) {
            if (current) store_owner_reference_(*current);
            return state_.selected_id;
        }
    }

    // Dwell: kandydat musi wygrywac N klatek pod rzad
    if (best.track_id != state_.candidate_id) {
        state_.candidate_id = best.track_id;
        state_.candidate_dwell = 1;
    } else {
        ++state_.candidate_dwell;
    }
    if (state_.candidate_dwell < cfg_.switch_dwell) {
        if (current) store_owner_reference_(*current);
        return state_.selected_id;
    }

    // D2 continuity guard: best.continuity musi byc >= active.continuity * 0.9.
    // Inaczej -- block switch (zostawiamy current, cofamy candidate dwell).
    // Chroni przed scenariuszem "wrong-adjacent-object" w formacji identycznych
    // dronow / samolotow.
    if (current && state_.identity_anchor) {
        auto best_cont = continuity_score_(best, tracks);
        auto active_cont = continuity_score_(*current, tracks);
        state_.last_continuity_score_best = best_cont;
        state_.last_continuity_score_active = active_cont;
        if (best_cont && active_cont) {
            if (*best_cont < *active_cont * cfg_.continuity_block_ratio) {
                state_.last_continuity_block = true;
                // Zresetuj kandydata — wyglada jak inny obiekt mimo wysokiego score'a
                state_.candidate_id = -1;
                state_.candidate_dwell = 0;
                store_owner_reference_(*current);
                return state_.selected_id;
            }
        }
    }

    // Switch! Update anchor do nowego ownera.
    // Fix 2: fresh persistent ID -- dwell-based switch oznacza ze wygral inny
    // fizyczny kandydat (continuity guard mial szanse zablokowac, ale przepuscil).
    commit_owner_(best, tracks, /*fresh_persistent=*/true);
    return state_.selected_id;
}

}  // namespace dtracker
