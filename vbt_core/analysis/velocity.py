"""
Velocity from a calibrated bar path, and the per-phase "active span" mask.

Velocity is computed frame-by-frame (Savitzky–Golay derivative of the
calibrated vertical position), not from rep endpoints, so the per-rep mean
concentric velocity (MCV) is the average of the instantaneous velocity over the
up phase — matching standard VBT practice. Sign is flipped so +v = up.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def frame_velocity(y_m: np.ndarray, times: np.ndarray, fps: float) -> np.ndarray:
    """Frame-by-frame velocity (+up) of vertical position `y_m` (metres, +down).

    Savitzky–Golay differentiates while smoothing over a ~quarter-second window;
    falls back to a plain gradient when the series is shorter than the window."""
    y_m = np.asarray(y_m, dtype=float)
    times = np.asarray(times, dtype=float)
    eff_fps = fps if fps > 0 else 30.0
    dt = 1.0 / eff_fps
    win = max(5, int(round(0.25 * eff_fps)) | 1)
    if len(y_m) >= win:
        return -savgol_filter(y_m, win, polyorder=2, deriv=1, delta=dt)
    return -np.gradient(y_m, times)


def active_span(mask, v, times, thresh: float = 0.05, max_gap_s: float = 0.6):
    """The longest contiguous movement inside `mask`: samples with
    |v| > thresh, where sub-threshold gaps shorter than `max_gap_s` are bridged
    (a mid-rep sticking point stays in the rep) but long stationary dwell (a
    pause at lockout, the bar resting on the floor) is cut off."""
    active = np.where(mask & (np.abs(v) > thresh))[0]
    out = np.zeros(len(mask), dtype=bool)
    if len(active) == 0:
        return out
    runs = []
    start = prev = active[0]
    for i in active[1:]:
        if times[i] - times[prev] > max_gap_s:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    s, e = max(runs, key=lambda r: times[r[1]] - times[r[0]])
    out[s:e + 1] = True
    return out
