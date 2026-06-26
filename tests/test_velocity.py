"""Tests for vbt_core.analysis.velocity: frame_velocity and active_span."""
from __future__ import annotations

import numpy as np
import pytest

from vbt_core.analysis import active_span, frame_velocity


# --------------------------------------------------------------------------- #
# frame_velocity
# --------------------------------------------------------------------------- #
def test_output_length_matches_input():
    ts = np.arange(0.0, 3.0, 1.0 / 30.0)
    y = np.sin(ts)
    v = frame_velocity(y, ts, 30.0)
    assert len(v) == len(y)


def test_sign_up_is_positive():
    # y is +down; a steadily DECREASING y means the bar moves UP => +v.
    ts = np.arange(0.0, 2.0, 1.0 / 30.0)
    y = -0.5 * ts                      # decreasing
    v = frame_velocity(y, ts, 30.0)
    assert np.all(v > 0)
    assert np.mean(v) == pytest.approx(0.5, abs=1e-6)


def test_sign_down_is_negative():
    ts = np.arange(0.0, 2.0, 1.0 / 30.0)
    y = 0.4 * ts                       # increasing => moving down
    v = frame_velocity(y, ts, 30.0)
    assert np.all(v < 0)
    assert np.mean(v) == pytest.approx(-0.4, abs=1e-6)


def test_constant_position_zero_velocity():
    ts = np.arange(0.0, 3.0, 1.0 / 30.0)
    y = np.full(len(ts), 0.3)
    v = frame_velocity(y, ts, 30.0)
    assert np.allclose(v, 0.0, atol=1e-9)


@pytest.mark.parametrize("fps", [24, 30, 60, 120, 240])
def test_known_slope_various_fps(fps):
    ts = np.arange(0.0, 2.0, 1.0 / fps)
    y = -1.2 * ts                      # up at 1.2 m/s
    v = frame_velocity(y, ts, fps)
    assert len(v) == len(y)
    assert np.mean(v) == pytest.approx(1.2, abs=1e-6)


def test_short_series_falls_back_to_gradient():
    # 3 samples is shorter than the Savitzky-Golay window (>=5): must not raise.
    y = np.array([0.0, 0.1, 0.2])
    ts = np.array([0.0, 1 / 30, 2 / 30])
    v = frame_velocity(y, ts, 30.0)
    assert len(v) == 3
    # y increasing -> downward -> negative velocity; gradient = 3 m/s downward.
    assert np.all(v < 0)
    assert v[0] == pytest.approx(-3.0)


def test_short_constant_series_zero():
    y = np.array([0.5, 0.5, 0.5, 0.5])     # len 4 < window 5
    ts = np.arange(4) / 30.0
    v = frame_velocity(y, ts, 30.0)
    assert np.allclose(v, 0.0)


def test_zero_or_negative_fps_defaults_to_30():
    ts = np.arange(0.0, 2.0, 1.0 / 30.0)
    y = -0.5 * ts
    v_default = frame_velocity(y, ts, 30.0)
    v_zero = frame_velocity(y, ts, 0.0)
    v_neg = frame_velocity(y, ts, -10.0)
    assert np.allclose(v_zero, v_default)
    assert np.allclose(v_neg, v_default)


def test_window_is_odd_and_at_least_five():
    # Series exactly window length must work (uses savgol, not gradient).
    fps = 30.0
    win = max(5, int(round(0.25 * fps)) | 1)   # mirror the implementation
    ts = np.arange(win) / fps
    y = -0.5 * ts
    v = frame_velocity(y, ts, fps)
    assert len(v) == win
    assert np.mean(v) == pytest.approx(0.5, abs=1e-3)


def test_returns_ndarray():
    ts = np.arange(0.0, 2.0, 1.0 / 30.0)
    v = frame_velocity(np.sin(ts), ts, 30.0)
    assert isinstance(v, np.ndarray)


# --------------------------------------------------------------------------- #
# active_span
# --------------------------------------------------------------------------- #
def _times(n, fps=30.0):
    return np.arange(n) / fps


def test_all_quiet_returns_all_false():
    n = 60
    v = np.zeros(n)
    mask = np.ones(n, dtype=bool)
    out = active_span(mask, v, _times(n))
    assert out.dtype == bool
    assert not out.any()
    assert len(out) == n


def test_single_active_run_selected():
    n = 60
    v = np.zeros(n)
    v[10:30] = 0.2
    out = active_span(np.ones(n, bool), v, _times(n))
    idx = np.where(out)[0]
    assert idx.min() == 10 and idx.max() == 29


def test_bridges_short_subthreshold_gap():
    # Two active runs separated by a 3-sample (0.1 s) gap < max_gap 0.6 s.
    n = 60
    v = np.zeros(n)
    v[5:15] = 0.2
    v[18:28] = 0.2          # gap indices 15,16,17 -> 0.1 s
    out = active_span(np.ones(n, bool), v, _times(n))
    idx = np.where(out)[0]
    assert idx.min() == 5 and idx.max() == 27   # bridged into one span


def test_long_gap_picks_longest_run():
    n = 90
    v = np.zeros(n)
    v[2:8] = 0.2            # short run (6)
    v[40:70] = 0.2         # long run (30), separated by a long gap
    out = active_span(np.ones(n, bool), v, _times(n))
    idx = np.where(out)[0]
    assert idx.min() == 40 and idx.max() == 69


def test_long_dwell_in_middle_is_cut():
    # Active, then a long stationary dwell, then active again. The dwell exceeds
    # max_gap so the two halves are separate; the longer half is kept.
    n = 120
    v = np.zeros(n)
    v[5:20] = 0.2          # 15-sample run
    # dwell 20..80 (2 s) quiet
    v[80:110] = 0.2        # 30-sample run -> the longer one
    out = active_span(np.ones(n, bool), v, _times(n))
    idx = np.where(out)[0]
    assert idx.min() == 80 and idx.max() == 109


def test_threshold_respected():
    n = 60
    v = np.full(n, 0.04)   # below default thresh 0.05 everywhere
    out = active_span(np.ones(n, bool), v, _times(n))
    assert not out.any()


def test_negative_velocity_counts_by_magnitude():
    n = 60
    v = np.zeros(n)
    v[10:30] = -0.2        # downward, |v| > thresh
    out = active_span(np.ones(n, bool), v, _times(n))
    idx = np.where(out)[0]
    assert idx.min() == 10 and idx.max() == 29


def test_mask_limits_consideration():
    # Activity outside the mask is ignored.
    n = 60
    v = np.full(n, 0.2)
    mask = np.zeros(n, dtype=bool)
    mask[20:40] = True
    out = active_span(mask, v, _times(n))
    idx = np.where(out)[0]
    assert idx.min() == 20 and idx.max() == 39


def test_custom_thresh_and_gap():
    n = 60
    v = np.zeros(n)
    v[10:20] = 0.1
    v[25:35] = 0.1
    # gap 20..24 = 5 samples = 0.166 s. With max_gap_s 0.1 it is NOT bridged.
    out = active_span(np.ones(n, bool), v, _times(n), thresh=0.05,
                      max_gap_s=0.1)
    idx = np.where(out)[0]
    # the two runs are equal length; max() picks the first encountered (10..19)
    assert idx.min() == 10 and idx.max() == 19


def test_returns_independent_array_length():
    n = 45
    v = np.zeros(n)
    v[5:10] = 0.3
    out = active_span(np.ones(n, bool), v, _times(n))
    assert len(out) == n
