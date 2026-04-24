#include "dtracker/lock_pipeline.hpp"

#include <cmath>

namespace dtracker {

const char* to_string(LockState s) {
    switch (s) {
        case LockState::UNLOCKED:  return "UNLOCKED";
        case LockState::ACQUIRE:   return "ACQUIRE";
        case LockState::LOCKED:    return "LOCKED";
        case LockState::HOLD:      return "HOLD";
        case LockState::REACQUIRE: return "REACQUIRE";
    }
    return "?";
}

LockPipeline::LockPipeline(LockPipelineConfig cfg) : cfg_(cfg) {}

const Track* LockPipeline::find_owner_(int owner_id, const std::vector<Track>& tracks) const {
    for (const auto& t : tracks) if (t.track_id == owner_id) return &t;
    return nullptr;
}

int LockPipeline::find_reacquire_candidate_(const std::vector<Track>& tracks) const {
    // Najblizszy track do last_known_center z confidence >= min i is_confirmed.
    int best = -1;
    double best_dist = cfg_.reacquire_max_dist;
    for (const auto& t : tracks) {
        if (!t.is_confirmed || t.confidence < cfg_.reacquire_min_conf) continue;
        double d = std::hypot(t.center.x - s_.last_known_center.x, t.center.y - s_.last_known_center.y);
        if (d < best_dist) { best_dist = d; best = t.track_id; }
    }
    return best;
}

void LockPipeline::transit_(LockState new_state) {
    if (new_state != s_.state) {
        // Metryki event-based
        if ((s_.state == LockState::LOCKED || s_.state == LockState::HOLD) &&
            (new_state == LockState::REACQUIRE || new_state == LockState::UNLOCKED)) {
            ++s_.total_lock_loss_events;
        }
        if (new_state == LockState::REACQUIRE) ++s_.total_reacquire_starts;
        if (s_.state == LockState::REACQUIRE && new_state == LockState::LOCKED) {
            ++s_.total_reacquire_successes;
        }
        s_.state = new_state;
        s_.frames_in_state = 0;
    } else {
        ++s_.frames_in_state;
    }

    // Liczniki frame-based
    if (s_.state == LockState::LOCKED)    ++s_.total_frames_locked;
    if (s_.state == LockState::HOLD)      ++s_.total_frames_hold;
    if (s_.state == LockState::REACQUIRE) ++s_.total_frames_reacquire;
}

LockState LockPipeline::step(std::optional<int> selected_id, const std::vector<Track>& tracks) {
    switch (s_.state) {
        case LockState::UNLOCKED: {
            if (selected_id) {
                s_.owner_id = selected_id;
                if (const Track* o = find_owner_(*selected_id, tracks)) {
                    s_.last_known_center = o->center;
                }
                transit_(LockState::ACQUIRE);
            } else {
                transit_(LockState::UNLOCKED);
            }
            break;
        }
        case LockState::ACQUIRE: {
            if (!selected_id) {
                transit_(LockState::UNLOCKED);
                break;
            }
            s_.owner_id = selected_id;
            const Track* o = find_owner_(*selected_id, tracks);
            if (!o) {
                transit_(LockState::HOLD);
                break;
            }
            s_.last_known_center = o->center;
            if (o->hits >= cfg_.acquire_min_hits && o->is_confirmed) {
                transit_(LockState::LOCKED);
            } else {
                transit_(LockState::ACQUIRE);
            }
            break;
        }
        case LockState::LOCKED: {
            if (!selected_id) {
                transit_(LockState::REACQUIRE);
                break;
            }
            s_.owner_id = selected_id;
            const Track* o = find_owner_(*selected_id, tracks);
            if (!o) {
                transit_(LockState::HOLD);
                break;
            }
            s_.last_known_center = o->center;
            if (o->missed_frames == 0) {
                transit_(LockState::LOCKED);
            } else if (o->missed_frames <= cfg_.hold_limit) {
                transit_(LockState::HOLD);
            } else {
                transit_(LockState::REACQUIRE);
            }
            break;
        }
        case LockState::HOLD: {
            if (!selected_id) {
                transit_(LockState::REACQUIRE);
                break;
            }
            s_.owner_id = selected_id;
            const Track* o = find_owner_(*selected_id, tracks);
            if (!o) {
                transit_(LockState::REACQUIRE);
                break;
            }
            s_.last_known_center = o->center;
            if (o->missed_frames == 0) {
                transit_(LockState::LOCKED);
            } else if (o->missed_frames <= cfg_.hold_limit) {
                transit_(LockState::HOLD);
            } else {
                transit_(LockState::REACQUIRE);
            }
            break;
        }
        case LockState::REACQUIRE: {
            // Szukamy nowego kandydata w okolicy last_known_center
            int cand = find_reacquire_candidate_(tracks);
            if (cand > 0) {
                s_.owner_id = cand;
                if (const Track* o = find_owner_(cand, tracks)) {
                    s_.last_known_center = o->center;
                }
                transit_(LockState::LOCKED);
            } else if (s_.frames_in_state >= cfg_.reacquire_limit) {
                transit_(LockState::UNLOCKED);
                s_.owner_id.reset();
            } else {
                transit_(LockState::REACQUIRE);
            }
            break;
        }
    }
    return s_.state;
}

}  // namespace dtracker
