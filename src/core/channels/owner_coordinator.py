from __future__ import annotations

from .models import OwnerDecision


class OwnerCoordinator:
    def __init__(self, switch_commit_frames=5, min_keep_conf=0.18, keep_missed_frames=1):
        self.switch_commit_frames = max(2, int(switch_commit_frames))
        self.min_keep_conf = float(min_keep_conf)
        self.keep_missed_frames = int(keep_missed_frames)
        self.pending_id = None
        self.pending_count = 0

    def reset(self):
        self.pending_id = None
        self.pending_count = 0

    def _is_healthy(self, track):
        if track is None:
            return False
        missed = int(getattr(track, 'missed_frames', 0) or 0)
        conf = float(getattr(track, 'confidence', 0.0) or 0.0)
        return missed <= self.keep_missed_frames and conf >= self.min_keep_conf

    def _find(self, tracks, tid):
        if tid is None:
            return None
        for tr in tracks or []:
            if int(getattr(tr, 'track_id', -1)) == int(tid):
                return tr
        return None

    def choose_requested_track(self, wide_state, narrow_snapshot):
        current = getattr(wide_state, 'selected_track', None)
        if self._is_healthy(current):
            return current

        tracks = getattr(wide_state, 'selection_tracks', None) or getattr(wide_state, 'tracks', None) or []
        owner_track = self._find(tracks, getattr(narrow_snapshot, 'owner_id', None))
        if self._is_healthy(owner_track):
            return owner_track

        return current or owner_track

    def finalize(self, wide_state, narrow_state):
        wide_track = getattr(wide_state, 'selected_track', None)
        narrow_track = getattr(narrow_state, 'owner_track', None)

        if wide_track is not None and narrow_track is not None and int(getattr(wide_track, 'track_id', -1)) == int(getattr(narrow_track, 'track_id', -2)):
            self.reset()
            return OwnerDecision(owner_track=wide_track, mode='aligned', reason='wide_narrow_aligned')

        if self._is_healthy(wide_track):
            self.reset()
            return OwnerDecision(owner_track=wide_track, mode='wide_keep', reason='wide_owner_healthy')

        candidate = narrow_track or wide_track
        if candidate is None:
            self.reset()
            return OwnerDecision(owner_track=None, mode='no_owner', reason='no_candidate')

        candidate_id = int(getattr(candidate, 'track_id', -1))
        if candidate_id < 0:
            self.reset()
            return OwnerDecision(owner_track=None, mode='invalid_candidate', reason='candidate_id_invalid')

        if self.pending_id == candidate_id:
            self.pending_count += 1
        else:
            self.pending_id = candidate_id
            self.pending_count = 1

        if self.pending_count >= self.switch_commit_frames:
            self.reset()
            return OwnerDecision(owner_track=candidate, mode='commit_candidate', reason='candidate_persisted', pending_id=candidate_id)

        return OwnerDecision(owner_track=candidate, mode='pending', reason='awaiting_commit', pending_id=candidate_id)

    def apply(self, decision, target_manager, narrow_channel, freeze_frames=8):
        if decision is None:
            return
        track = getattr(decision, 'owner_track', None)
        if track is None:
            return

        selected_id = getattr(target_manager, 'selected_id', None)
        track_id = int(getattr(track, 'track_id', -1))

        if selected_id != track_id and not getattr(target_manager, 'manual_lock', False):
            if hasattr(target_manager, 'commit_owner'):
                target_manager.commit_owner(track, freeze_frames=freeze_frames, reason=decision.reason)
            else:
                target_manager.selected_id = track_id
                try:
                    target_manager.freeze_to(track_id, freeze_frames)
                except Exception:
                    pass
                target_manager.lock_age = 0

        if getattr(narrow_channel.narrow_tracker, 'owner_id', None) != track_id:
            narrow_channel.bind_owner(track)
