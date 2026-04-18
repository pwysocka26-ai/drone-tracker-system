class OperatorSlotManager:
    def __init__(self, max_slots=9, lost_ttl=240):
        self.max_slots = int(max_slots)
        self.lost_ttl = int(lost_ttl)
        self.track_to_slot = {}
        self.slot_to_track = {}
        self.missing = {}

    def reset(self):
        self.track_to_slot = {}
        self.slot_to_track = {}
        self.missing = {}

    def _free_slots(self):
        used = set(self.slot_to_track.keys())
        return [slot for slot in range(1, self.max_slots + 1) if slot not in used]

    def update(self, tracks):
        seen_ids = {int(tr.track_id) for tr in tracks}

        # starzenie nieobecnych trackow
        for tid in list(self.track_to_slot.keys()):
            if tid not in seen_ids:
                self.missing[tid] = self.missing.get(tid, 0) + 1
                if self.missing[tid] > self.lost_ttl:
                    slot = self.track_to_slot.pop(tid)
                    self.slot_to_track.pop(slot, None)
                    self.missing.pop(tid, None)
            else:
                self.missing.pop(tid, None)

        # najpierw przypisz istniejace sloty
        used_slots = set()
        new_tracks = []

        for tr in tracks:
            tid = int(tr.track_id)
            if tid in self.track_to_slot:
                slot = self.track_to_slot[tid]
                tr.operator_id = slot
                used_slots.add(slot)
            else:
                new_tracks.append(tr)

        # nowe tracki dostaja wolne sloty, stabilnie od lewej do prawej
        free_slots = [slot for slot in self._free_slots() if slot not in used_slots]
        new_tracks.sort(key=lambda t: t.center_xy[0])

        for tr, slot in zip(new_tracks, free_slots):
            tid = int(tr.track_id)
            self.track_to_slot[tid] = slot
            self.slot_to_track[slot] = tid
            tr.operator_id = slot
            used_slots.add(slot)

        # fallback gdyby zabraklo slotow
        next_fallback = self.max_slots + 1
        for tr in tracks:
            if not hasattr(tr, "operator_id"):
                tr.operator_id = next_fallback
                next_fallback += 1

        tracks.sort(key=lambda t: getattr(t, "operator_id", 999))
        return tracks
