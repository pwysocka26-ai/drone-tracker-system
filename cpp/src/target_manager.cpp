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
}

void TargetManager::clear_manual_lock() {
    state_.manual_lock = false;
    state_.manual_lock_id = -1;
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

std::optional<int> TargetManager::select(const std::vector<Track>& tracks) {
    // Manual lock: trzymaj forsownie, nawet jak owner zniknal
    if (state_.manual_lock) {
        state_.selected_id = state_.manual_lock_id;
        // Sprawdz czy track istnieje; jesli nie, zostaw selected (freeze) — recovery zrobi
        return state_.selected_id;
    }

    if (tracks.empty()) {
        state_.selected_id = std::nullopt;
        state_.candidate_id = -1;
        state_.candidate_dwell = 0;
        return std::nullopt;
    }

    // Znajdz current track w liscie
    const Track* current = nullptr;
    if (state_.selected_id) {
        for (const auto& t : tracks) {
            if (t.track_id == *state_.selected_id) { current = &t; break; }
        }
    }

    // Porzuc current jesli za duzo missed
    if (current && current->missed_frames > cfg_.max_select_missed) {
        state_.selected_id = std::nullopt;
        current = nullptr;
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
        state_.selected_id = std::nullopt;
        return std::nullopt;
    }

    const Track& best = tracks[best_idx];

    // Pierwszy wybor
    if (!state_.selected_id) {
        state_.selected_id = best.track_id;
        state_.frames_since_switch = 0;
        state_.candidate_id = -1;
        state_.candidate_dwell = 0;
        return state_.selected_id;
    }

    ++state_.frames_since_switch;

    // Current wygrywa -> reset candidate
    if (current && best.track_id == current->track_id) {
        state_.candidate_id = -1;
        state_.candidate_dwell = 0;
        return state_.selected_id;
    }

    // Freeze przez N klatek po switch
    if (state_.frames_since_switch < cfg_.target_selection_freeze_frames) {
        return state_.selected_id;
    }

    // Sticky: nie przelaczaj dopóki sticky_frames nie minie po initial lub ostatnim switch
    if (state_.frames_since_switch < cfg_.sticky_frames) {
        // Candidate must win by margin
        float cur_score = current ? score_candidate_(*current, true) : -1e9f;
        if (best_score < cur_score * (1.0f + cfg_.switch_margin)) {
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
        return state_.selected_id;
    }

    // Switch!
    state_.selected_id = best.track_id;
    state_.frames_since_switch = 0;
    state_.candidate_id = -1;
    state_.candidate_dwell = 0;
    return state_.selected_id;
}

}  // namespace dtracker
