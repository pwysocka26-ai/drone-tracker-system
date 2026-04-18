from __future__ import annotations

import inspect

from .models import NarrowSnapshot, NarrowState


class NarrowHandoffState:
    def __init__(self):
        self.track = None
        self.center = None
        self.bbox = None
        self.zoom = 1.8
        self.missed = 9999
        self.age = 9999
        self.last_good_center = None
        self.last_good_bbox = None
        self.last_good_zoom = 1.8
        self.gap_len = 0
        self.max_gap_len = 0

    def reset(self):
        self.track = None
        self.center = None
        self.bbox = None
        self.zoom = 1.8
        self.missed = 9999
        self.age = 9999
        self.last_good_center = None
        self.last_good_bbox = None
        self.last_good_zoom = 1.8
        self.gap_len = 0
        self.max_gap_len = 0

    def update_from_track(self, tr, zoom=None):
        self.track = tr
        self.center = tuple(float(v) for v in tr.center_xy)
        self.bbox = tuple(float(v) for v in tr.bbox_xyxy)
        if zoom is not None:
            self.zoom = float(zoom)
            self.last_good_zoom = float(zoom)
        self.last_good_center = self.center
        self.last_good_bbox = self.bbox
        self.missed = 0
        self.age = 0
        self.gap_len = 0

    def mark_missed(self):
        self.missed += 1
        self.age += 1
        self.gap_len += 1
        self.max_gap_len = max(self.max_gap_len, self.gap_len)


def _bbox_size(bbox):
    x1, y1, x2, y2 = bbox
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def _distance(a, b):
    if a is None or b is None:
        return float('inf')
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return (dx * dx + dy * dy) ** 0.5


def _bbox_similarity(a, b):
    if a is None or b is None:
        return 0.0
    aw, ah = _bbox_size(a)
    bw, bh = _bbox_size(b)
    dw = abs(aw - bw) / max(aw, bw)
    dh = abs(ah - bh) / max(ah, bh)
    return max(0.0, 1.0 - 0.5 * (dw + dh))


def _choose_soft_handoff_track(tracks, selected_id, handoff_state, radius_px=140.0):
    if not tracks:
        return None

    if selected_id is not None:
        for tr in tracks:
            if int(getattr(tr, 'track_id', -1)) == int(selected_id):
                return tr

    anchor = handoff_state.last_good_center or handoff_state.center
    if anchor is None:
        return None

    best = None
    best_score = -1e9
    for tr in tracks:
        dist = _distance(tuple(tr.center_xy), anchor)
        if dist > radius_px:
            continue
        conf = float(getattr(tr, 'confidence', 0.0) or 0.0)
        sim = _bbox_similarity(getattr(tr, 'bbox_xyxy', None), handoff_state.last_good_bbox or handoff_state.bbox)
        score = conf * 8.0 + sim * 4.0 - dist / max(1.0, radius_px)
        if best is None or score > best_score:
            best = tr
            best_score = score
    return best


def _blend_track_with_handoff(tr, handoff_state, center_alpha=0.76, size_alpha=0.84):
    if tr is None:
        return None
    ref_center = handoff_state.last_good_center or handoff_state.center
    ref_bbox = handoff_state.last_good_bbox or handoff_state.bbox
    if ref_center is None or ref_bbox is None:
        return tr

    hx, hy = ref_center
    tx, ty = tr.center_xy
    bw, bh = _bbox_size(tr.bbox_xyxy)
    hw, hh = _bbox_size(ref_bbox)

    smx = center_alpha * hx + (1.0 - center_alpha) * tx
    smy = center_alpha * hy + (1.0 - center_alpha) * ty
    smw = size_alpha * hw + (1.0 - size_alpha) * bw
    smh = size_alpha * hh + (1.0 - size_alpha) * bh

    tr.center_xy = (smx, smy)
    tr.bbox_xyxy = (smx - smw * 0.5, smy - smh * 0.5, smx + smw * 0.5, smy + smh * 0.5)
    return tr


def _narrow_update_compat(narrow_tracker, frame, active_track, tracks=None, manual_switch=False, selected_track=None, handoff_track=None):
    try:
        sig = inspect.signature(narrow_tracker.update)
        kwargs = {}
        if 'tracks' in sig.parameters:
            kwargs['tracks'] = tracks
        if 'manual_switch' in sig.parameters:
            kwargs['manual_switch'] = manual_switch
        if 'selected_track' in sig.parameters:
            kwargs['selected_track'] = selected_track
        if 'handoff_track' in sig.parameters:
            kwargs['handoff_track'] = handoff_track
        result = narrow_tracker.update(frame, active_track, **kwargs)
    except TypeError:
        result = narrow_tracker.update(frame, active_track)

    if not isinstance(result, tuple):
        raise ValueError('NarrowTracker.update returned unsupported result type')
    if len(result) == 7:
        return result
    if len(result) == 6:
        predicted_center, smooth_center, smooth_zoom, hold_count, pan_speed, tilt_speed = result
        return predicted_center, smooth_center, smooth_zoom, hold_count, pan_speed, tilt_speed, active_track
    raise ValueError(f'NarrowTracker.update returned unexpected tuple length: {len(result)}')


class NarrowChannel:
    def __init__(self, narrow_tracker, *, hold_frames, soft_active_max_missed, handoff_reacquire_radius, handoff_hold_frames):
        self.narrow_tracker = narrow_tracker
        self.handoff_state = NarrowHandoffState()
        self.hold_frames = int(hold_frames)
        self.soft_active_max_missed = int(soft_active_max_missed)
        self.handoff_reacquire_radius = float(handoff_reacquire_radius)
        self.handoff_hold_frames = int(handoff_hold_frames)

    def reset(self):
        self.narrow_tracker.reset()
        self.handoff_state.reset()

    def initialize_zoom(self, zoom_value):
        z = float(zoom_value)
        self.handoff_state.zoom = z
        self.handoff_state.last_good_zoom = z
        self.narrow_tracker.smooth_zoom = z
        self.narrow_tracker.last_good_zoom = z

    def bind_owner(self, track):
        if track is not None:
            self.narrow_tracker.bind_owner(track)

    def predict_center(self):
        try:
            return self.narrow_tracker.kalman.predict()
        except Exception:
            return getattr(self.narrow_tracker, 'smooth_center', None)

    def snapshot(self):
        return NarrowSnapshot(
            owner_id=getattr(self.narrow_tracker, 'owner_id', None),
            owner_track=getattr(self.narrow_tracker, 'owner_track', None),
            smooth_center=getattr(self.narrow_tracker, 'smooth_center', None),
            smooth_zoom=float(getattr(self.narrow_tracker, 'smooth_zoom', self.handoff_state.zoom)),
            hold_count=int(getattr(self.narrow_tracker, 'hold_count', 0)),
        )

    def step(self, frame, requested_track, wide_state):
        tracks = list(getattr(wide_state, 'tracks', []) or [])
        selected_track = getattr(wide_state, 'selected_track', None)
        target_manager = getattr(wide_state, 'target_manager', None)
        selected_id = getattr(target_manager, 'selected_id', None)

        active_track = requested_track
        if active_track is not None and int(getattr(active_track, 'missed_frames', 0) or 0) > self.soft_active_max_missed:
            active_track = None

        if selected_track is not None and int(getattr(selected_track, 'missed_frames', 0) or 0) <= self.soft_active_max_missed:
            self.handoff_state.update_from_track(selected_track, zoom=self.handoff_state.zoom)
        elif active_track is not None and int(getattr(active_track, 'missed_frames', 0) or 0) <= self.soft_active_max_missed:
            self.handoff_state.update_from_track(active_track, zoom=self.handoff_state.zoom)
        else:
            self.handoff_state.mark_missed()

        soft_track = active_track
        reused_last_good = False
        if soft_track is None and self.handoff_state.missed <= self.handoff_hold_frames:
            reacquired = _choose_soft_handoff_track(
                tracks,
                selected_id,
                self.handoff_state,
                self.handoff_reacquire_radius,
            )
            if reacquired is not None:
                soft_track = _blend_track_with_handoff(reacquired, self.handoff_state)
                self.handoff_state.update_from_track(soft_track, zoom=self.handoff_state.zoom)
            elif self.handoff_state.last_good_center is not None and self.handoff_state.last_good_bbox is not None:
                soft_track = selected_track
                reused_last_good = True

        predicted_center, smooth_center, smooth_zoom, hold_count, pan_speed, tilt_speed, owner_track = _narrow_update_compat(
            self.narrow_tracker,
            frame,
            soft_track,
            tracks=tracks,
            manual_switch=False,
            selected_track=selected_track,
            handoff_track=self.handoff_state.track,
        )

        effective_track = owner_track if owner_track is not None else soft_track
        edge_limit_active = bool(getattr(self.narrow_tracker, 'zoom_mode', '') != 'TARGET_FILL')

        if effective_track is not None and smooth_center is not None:
            self.handoff_state.last_good_center = tuple(float(v) for v in smooth_center)
            self.handoff_state.last_good_bbox = tuple(float(v) for v in effective_track.bbox_xyxy)
            self.handoff_state.last_good_zoom = float(smooth_zoom)
            self.handoff_state.zoom = float(smooth_zoom)

        return NarrowState(
            predicted_center=predicted_center,
            smooth_center=smooth_center,
            smooth_zoom=float(smooth_zoom),
            hold_count=int(hold_count),
            pan_speed=float(pan_speed),
            tilt_speed=float(tilt_speed),
            owner_track=owner_track,
            effective_track=effective_track,
            reused_last_good=bool(reused_last_good),
            edge_limit_active=edge_limit_active,
        )
