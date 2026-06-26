"""
BarTracker — single-object-of-interest tracker for barbell plates.

Replaces Ultralytics' built-in ByteTrack with an explicit pipeline we control:

  1. KALMAN FILTER   — one constant-velocity filter per track predicts where
                       the box should be; a physically illogical jump fails the
                       Mahalanobis gate and never enters the path.
  2. HUNGARIAN       — `linear_sum_assignment` on 1 - IoU cost matches predicted
                       track boxes to detections, preventing ID swaps.
  3. BYTETRACK LOGIC — two-stage association: high-confidence detections match
                       first, then low-confidence leftovers rescue still-
                       unmatched confirmed tracks (blur/shadow keeps its lock).
                       Lost tracks coast on the Kalman prediction up to
                       `max_age` frames before being dropped.

Per frame the tracker emits the smoothed `TrackSnapshot` of each confirmed
track.
"""
from __future__ import annotations

import numpy as np

from .association import associate, to_measurement
from .track import Track, TrackSnapshot


class BarTracker:
    def __init__(self, *, high_thresh: float = 0.45, low_thresh: float = 0.10,
                 match_iou: float = 0.30, low_match_iou: float = 0.20,
                 max_age: int = 30, min_hits: int = 3):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_iou = match_iou
        self.low_match_iou = low_match_iou
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self._next_id = 1
        # id -> (hits, avg_area) over the whole run, incl. deleted tracks —
        # used to pick the primary (bar) track stably.
        self._history: dict[int, tuple[int, float]] = {}

    def update(self, detections: np.ndarray) -> list[TrackSnapshot]:
        """Advance one frame. `detections` is (N,5) [cx, cy, w, h, conf]."""
        dets = np.asarray(detections, dtype=float).reshape(-1, 5)

        for t in self.tracks:
            t.kf.predict()
            t.age += 1
            t.time_since_update += 1   # reset below on match

        high = dets[dets[:, 4] >= self.high_thresh]
        low = dets[(dets[:, 4] >= self.low_thresh)
                   & (dets[:, 4] < self.high_thresh)]

        # Stage 1: every track vs high-confidence detections.
        matches, um_t, um_d = associate(self.tracks, high, self.match_iou)
        for ti, di in matches:
            self._apply_match(self.tracks[ti], high[di])

        # Stage 2 (ByteTrack): unmatched *confirmed* tracks get a second chance
        # against low-confidence detections — keeps the lock through blur.
        leftovers = [self.tracks[i] for i in um_t if self.tracks[i].confirmed]
        m2, _, _ = associate(leftovers, low, self.low_match_iou)
        for ti, di in m2:
            self._apply_match(leftovers[ti], low[di])

        # Birth: unmatched high-conf detections only (never low-conf).
        for di in um_d:
            self.tracks.append(Track(self._next_id, to_measurement(high[di, :4]),
                                     float(high[di, 4])))
            self._next_id += 1

        # Death: stale tentatives quickly, confirmed tracks after max_age.
        kept = []
        for t in self.tracks:
            limit = self.max_age if t.confirmed else 2
            if t.time_since_update <= limit:
                kept.append(t)
        self.tracks = kept

        snaps = []
        for t in self.tracks:
            if not t.confirmed:
                continue
            self._history[t.id] = (t.hits, t.avg_area)
            snaps.append(t.snapshot())
        return snaps

    def _apply_match(self, t: Track, det: np.ndarray) -> None:
        t.kf.update(to_measurement(det[:4]))
        t.hits += 1
        t.time_since_update = 0
        t.last_conf = float(det[4])
        t.sum_area += float(det[2]) * float(det[3])
        if t.hits >= self.min_hits:
            t.confirmed = True

    def primary_id(self) -> int | None:
        """The bar track: longest-lived, then largest. One barbell — the
        camera-side plate matches most frames and is biggest on screen."""
        if not self._history:
            return None
        return max(self._history, key=lambda i: self._history[i])
