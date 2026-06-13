#!/usr/bin/env python3
"""
bartracker.py — custom single-object-of-interest tracker for barbell plates.

Replaces Ultralytics' built-in ByteTrack with an explicit pipeline we control:

  1. KALMAN FILTER   — one constant-velocity filter per track. It predicts
                       where the box should be and how fast it is moving, so a
                       physically illogical jump fails the Mahalanobis gate and
                       never enters the path.
  2. HUNGARIAN       — scipy.optimize.linear_sum_assignment matches predicted
                       track boxes to detections (1 - IoU cost), preventing
                       ID swaps between plates.
  3. BYTETRACK LOGIC — two-stage association: high-confidence detections are
                       matched first, then low-confidence leftovers rescue
                       still-unmatched tracks (blurry/shadowed plate keeps its
                       track). Lost tracks coast on the Kalman prediction for
                       up to `max_age` frames before being dropped.

Per frame the tracker emits the *Kalman-smoothed* state of each confirmed
track — that smoothed state is what the bar path and velocity stages consume.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

# chi-square 0.95 quantile, 4 degrees of freedom — standard association gate.
CHI2_GATE = 9.4877
_BIG_COST = 1e5


class KalmanFilter:
    """Constant-velocity Kalman filter over [cx, cy, a, h], a = w/h.

    State: [cx, cy, a, h, vcx, vcy, va, vh], dt = 1 frame. Noise scales with
    box height (the ByteTrack/SORT convention).
    """

    _WP = 1.0 / 20    # position noise weight
    _WV = 1.0 / 160   # velocity noise weight

    def __init__(self, z: np.ndarray):
        self.x = np.zeros(8)
        self.x[:4] = z
        h = float(z[3])
        std = [2 * self._WP * h, 2 * self._WP * h, 1e-2, 2 * self._WP * h,
               10 * self._WV * h, 10 * self._WV * h, 1e-5, 10 * self._WV * h]
        self.P = np.diag(np.square(std))
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0
        self.H = np.eye(4, 8)

    def predict(self) -> None:
        h = float(self.x[3])
        std = [self._WP * h, self._WP * h, 1e-2, self._WP * h,
               self._WV * h, self._WV * h, 1e-5, self._WV * h]
        Q = np.diag(np.square(std))
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q

    def _innovation(self, z: np.ndarray):
        h = float(self.x[3])
        std = [self._WP * h, self._WP * h, 1e-1, self._WP * h]
        R = np.diag(np.square(std))
        S = self.H @ self.P @ self.H.T + R
        y = np.asarray(z, dtype=float) - self.H @ self.x
        return y, S

    def update(self, z: np.ndarray) -> None:
        y, S = self._innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def gating_distance(self, z: np.ndarray) -> float:
        """Squared Mahalanobis distance of measurement z = [cx, cy, a, h]."""
        y, S = self._innovation(z)
        return float(y @ np.linalg.solve(S, y))

    @property
    def box(self) -> tuple[float, float, float, float]:
        """Smoothed (cx, cy, w, h)."""
        cx, cy, a, h = self.x[:4]
        h = max(float(h), 1.0)
        return float(cx), float(cy), max(float(a) * h, 1.0), h


@dataclass(frozen=True)
class TrackSnapshot:
    track_id: int
    cx: float
    cy: float
    w: float
    h: float
    vx: float          # px/frame, Kalman velocity
    vy: float          # px/frame, +down (image coords)
    conf: float        # matched detection conf, 0.0 while coasting
    coasting: bool     # True = no detection this frame, pure prediction
    hits: int
    age: int


class _Track:
    def __init__(self, tid: int, z: np.ndarray, conf: float):
        self.id = tid
        self.kf = KalmanFilter(z)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.confirmed = False
        self.last_conf = conf
        _, _, w, h = self.kf.box
        self.sum_area = w * h

    @property
    def avg_area(self) -> float:
        return self.sum_area / max(self.hits, 1)


def _to_z(box_cxcywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = box_cxcywh
    h = max(float(h), 1.0)
    return np.array([cx, cy, w / h, h], dtype=float)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (M,4) and (N,4) boxes in cxcywh format -> (M,N)."""
    ax1 = a[:, 0] - a[:, 2] / 2
    ay1 = a[:, 1] - a[:, 3] / 2
    ax2 = a[:, 0] + a[:, 2] / 2
    ay2 = a[:, 1] + a[:, 3] / 2
    bx1 = b[:, 0] - b[:, 2] / 2
    by1 = b[:, 1] - b[:, 3] / 2
    bx2 = b[:, 0] + b[:, 2] / 2
    by2 = b[:, 1] + b[:, 3] / 2

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _associate(tracks: list[_Track], dets: np.ndarray, iou_thresh: float):
    """Hungarian matching of tracks to detections (N,5) with IoU + Mahalanobis
    gating. Returns (matches, unmatched_track_idx, unmatched_det_idx)."""
    if not tracks or len(dets) == 0:
        return [], list(range(len(tracks))), list(range(len(dets)))

    t_boxes = np.array([t.kf.box for t in tracks])
    iou = iou_matrix(t_boxes, dets[:, :4])
    cost = 1.0 - iou
    for ti, t in enumerate(tracks):
        for di in range(len(dets)):
            if iou[ti, di] < iou_thresh or \
                    t.kf.gating_distance(_to_z(dets[di, :4])) > CHI2_GATE:
                cost[ti, di] = _BIG_COST

    rows, cols = linear_sum_assignment(cost)
    matches = [(ti, di) for ti, di in zip(rows, cols)
               if cost[ti, di] < _BIG_COST]
    matched_t = {ti for ti, _ in matches}
    matched_d = {di for _, di in matches}
    unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
    unmatched_d = [i for i in range(len(dets)) if i not in matched_d]
    return matches, unmatched_t, unmatched_d


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
        self.tracks: list[_Track] = []
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
        matches, um_t, um_d = _associate(self.tracks, high, self.match_iou)
        for ti, di in matches:
            self._apply_match(self.tracks[ti], high[di])

        # Stage 2 (ByteTrack): unmatched *confirmed* tracks get a second chance
        # against low-confidence detections — keeps the lock through blur.
        leftovers = [self.tracks[i] for i in um_t if self.tracks[i].confirmed]
        m2, _, _ = _associate(leftovers, low, self.low_match_iou)
        for ti, di in m2:
            self._apply_match(leftovers[ti], low[di])

        # Birth: unmatched high-conf detections only (never low-conf).
        for di in um_d:
            self.tracks.append(_Track(self._next_id, _to_z(high[di, :4]),
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
            cx, cy, w, h = t.kf.box
            coasting = t.time_since_update > 0
            snaps.append(TrackSnapshot(
                track_id=t.id, cx=cx, cy=cy, w=w, h=h,
                vx=float(t.kf.x[4]), vy=float(t.kf.x[5]),
                conf=0.0 if coasting else t.last_conf,
                coasting=coasting, hits=t.hits, age=t.age,
            ))
        return snaps

    def _apply_match(self, t: _Track, det: np.ndarray) -> None:
        t.kf.update(_to_z(det[:4]))
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


if __name__ == "__main__":
    # Self-test: noisy sinusoidal bar + dropout + spurious far detections.
    rng = np.random.default_rng(0)
    tracker = BarTracker(high_thresh=0.45, low_thresh=0.10, max_age=30)
    true_path, smooth_path = [], []
    for f in range(300):
        cy = 600 + 250 * np.sin(f / 30)
        dets = []
        if f % 17 != 0:                       # occasional total dropout
            blur = 0.3 if abs(np.cos(f / 30)) < 0.15 else 0.9   # turnaround blur
            dets.append([640 + rng.normal(0, 3), cy + rng.normal(0, 3),
                         180 + rng.normal(0, 5), 180 + rng.normal(0, 5), blur])
        if f % 23 == 0:                       # spurious high-conf ghost
            dets.append([200, 200, 150, 150, 0.95])
        snaps = tracker.update(np.array(dets).reshape(-1, 5))
        pid = tracker.primary_id()
        for s in snaps:
            if s.track_id == pid:
                true_path.append(cy)
                smooth_path.append(s.cy)
    err = np.abs(np.array(smooth_path) - np.array(true_path))
    pid = tracker.primary_id()
    hits, _ = tracker._history[pid]
    print(f"primary id={pid} hits={hits} samples={len(smooth_path)} "
          f"mean|err|={err.mean():.2f}px max|err|={err.max():.2f}px")
    assert pid is not None and hits > 250, "primary track lost"
    assert err.mean() < 6, "smoothed path drifted from truth"
    assert err.max() < 60, "a ghost detection captured the primary track"
    print("bartracker self-test OK")
