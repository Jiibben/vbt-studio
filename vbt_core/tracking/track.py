"""
Track value types: the immutable per-frame `TrackSnapshot` the pipeline reads,
and the mutable `Track` the tracker maintains internally.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .association import to_measurement
from .kalman import KalmanFilter


@dataclass(frozen=True)
class TrackSnapshot:
    """What the tracker emits per confirmed track each frame (smoothed state)."""

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


class Track:
    """One live hypothesis: a Kalman filter plus the bookkeeping the
    ByteTrack-style lifecycle (tentative → confirmed → coasting → dead) needs."""

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

    def snapshot(self) -> TrackSnapshot:
        cx, cy, w, h = self.kf.box
        coasting = self.time_since_update > 0
        return TrackSnapshot(
            track_id=self.id, cx=cx, cy=cy, w=w, h=h,
            vx=float(self.kf.x[4]), vy=float(self.kf.x[5]),
            conf=0.0 if coasting else self.last_conf,
            coasting=coasting, hits=self.hits, age=self.age,
        )

    @staticmethod
    def measurement(box_cxcywh: np.ndarray) -> np.ndarray:
        return to_measurement(box_cxcywh)
