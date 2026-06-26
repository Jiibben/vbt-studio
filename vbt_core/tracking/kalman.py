"""
Constant-velocity Kalman filter over a barbell-plate box.

State (8-dim): [cx, cy, a, h, vcx, vcy, va, vh], a = w/h aspect, dt = 1 frame.
Process/measurement noise scale with box height (the ByteTrack/SORT convention)
so the gate is tight on a small far plate and loose on a big near one. The
filter's smoothed state — not the raw detection — is what every downstream
stage (bar path, velocity) consumes.
"""
from __future__ import annotations

import numpy as np


class KalmanFilter:
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
