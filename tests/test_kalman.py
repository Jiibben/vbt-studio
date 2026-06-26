"""Characterization + edge-case tests for vbt_core.tracking.kalman.KalmanFilter.

The filter state is 8-dim [cx, cy, a, h, vcx, vcy, va, vh] with a = w/h aspect
and dt = 1 frame. `.box` reconstructs (cx, cy, w, h) with w = a * h and clamps
h, w to >= 1.
"""
from __future__ import annotations

import numpy as np
import pytest

from vbt_core.tracking.association import CHI2_GATE, to_measurement
from vbt_core.tracking.kalman import KalmanFilter


def make(cx=100.0, cy=200.0, a=0.5, h=40.0):
    return KalmanFilter(np.array([cx, cy, a, h], dtype=float))


# --------------------------------------------------------------------------- #
# Construction / .box reconstruction
# --------------------------------------------------------------------------- #
def test_box_reconstructs_w_from_aspect_times_h():
    kf = make(cx=100.0, cy=200.0, a=0.5, h=40.0)
    cx, cy, w, h = kf.box
    assert (cx, cy) == (100.0, 200.0)
    assert h == 40.0
    assert w == pytest.approx(0.5 * 40.0)  # 20.0


def test_initial_state_layout():
    kf = make()
    # position carried into x[:4], velocity components start at zero
    assert np.allclose(kf.x[:4], [100.0, 200.0, 0.5, 40.0])
    assert np.allclose(kf.x[4:], 0.0)


def test_F_is_constant_velocity():
    kf = make()
    expected = np.eye(8)
    for i in range(4):
        expected[i, i + 4] = 1.0
    assert np.allclose(kf.F, expected)


def test_H_extracts_position():
    kf = make()
    assert kf.H.shape == (4, 8)
    assert np.allclose(kf.H @ kf.x, kf.x[:4])


@pytest.mark.parametrize("cx,cy,w,h", [
    (100.0, 200.0, 20.0, 40.0),
    (5.0, 6.0, 30.0, 10.0),
    (0.0, 0.0, 80.0, 80.0),
    (640.0, 360.0, 12.0, 100.0),
])
def test_box_matches_cxcywh_via_to_measurement(cx, cy, w, h):
    """Build a measurement from a cxcywh box, then .box recovers cxcywh."""
    kf = KalmanFilter(to_measurement(np.array([cx, cy, w, h], dtype=float)))
    bcx, bcy, bw, bh = kf.box
    assert bcx == pytest.approx(cx)
    assert bcy == pytest.approx(cy)
    assert bw == pytest.approx(w)
    assert bh == pytest.approx(h)


# --------------------------------------------------------------------------- #
# h / w clamping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("h", [0.0, 0.4, 1e-9, -3.0])
def test_box_clamps_height_to_at_least_one(h):
    kf = KalmanFilter(np.array([10.0, 10.0, 1.0, h], dtype=float))
    _, _, w, bh = kf.box
    assert bh >= 1.0
    assert w >= 1.0


def test_box_clamps_width_to_at_least_one():
    # aspect ~0 would give w ~0 -> clamped to 1.0
    kf = KalmanFilter(np.array([10.0, 10.0, 1e-6, 40.0], dtype=float))
    _, _, w, _ = kf.box
    assert w == 1.0


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #
def test_predict_advances_position_by_velocity():
    kf = make()
    kf.x[4] = 5.0   # vcx
    kf.x[5] = -3.0  # vcy
    kf.x[7] = 2.0   # vh
    before = kf.x[:4].copy()
    kf.predict()
    assert kf.x[0] == pytest.approx(before[0] + 5.0)
    assert kf.x[1] == pytest.approx(before[1] - 3.0)
    assert kf.x[3] == pytest.approx(before[3] + 2.0)
    # velocity components are unchanged by the constant-velocity transition
    assert np.allclose(kf.x[4:], [5.0, -3.0, 0.0, 2.0])


def test_predict_grows_covariance():
    kf = make()
    trace_before = np.trace(kf.P)
    kf.predict()
    assert np.trace(kf.P) > trace_before


def test_predict_with_zero_velocity_keeps_position():
    kf = make()
    pos = kf.x[:4].copy()
    kf.predict()
    assert np.allclose(kf.x[:4], pos)


# --------------------------------------------------------------------------- #
# gating_distance
# --------------------------------------------------------------------------- #
def test_gating_distance_zero_for_just_set_measurement():
    z = np.array([100.0, 200.0, 0.5, 40.0])
    kf = KalmanFilter(z)
    assert kf.gating_distance(z) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("far", [
    [1000.0, 200.0, 0.5, 40.0],
    [100.0, 2000.0, 0.5, 40.0],
    [1000.0, 2000.0, 0.5, 40.0],
    [100.0, 200.0, 0.5, 400.0],
])
def test_gating_distance_large_for_far_box(far):
    kf = make()
    assert kf.gating_distance(np.array(far)) > CHI2_GATE


def test_gating_distance_is_nonnegative():
    kf = make()
    for dx in (-50.0, -1.0, 0.0, 1.0, 50.0):
        z = np.array([100.0 + dx, 200.0, 0.5, 40.0])
        assert kf.gating_distance(z) >= 0.0


# --------------------------------------------------------------------------- #
# update
# --------------------------------------------------------------------------- #
def test_update_reduces_gating_for_discrepant_measurement():
    kf = make()
    kf.predict()
    z = np.array([108.0, 205.0, 0.5, 40.0])  # within-gate offset
    g_before = kf.gating_distance(z)
    assert g_before > 0.0
    kf.update(z)
    g_after = kf.gating_distance(z)
    assert g_after < g_before


def test_update_pulls_state_toward_measurement():
    kf = make()
    kf.predict()
    z = np.array([110.0, 210.0, 0.5, 40.0])
    before_cx, before_cy = kf.x[0], kf.x[1]
    kf.update(z)
    # state moves toward (but not all the way to) the measurement
    assert before_cx < kf.x[0] <= 110.0
    assert before_cy < kf.x[1] <= 210.0


def test_update_shrinks_covariance():
    kf = make()
    kf.predict()
    trace_before = np.trace(kf.P)
    kf.update(np.array([100.0, 200.0, 0.5, 40.0]))
    assert np.trace(kf.P) < trace_before


def test_repeated_consistent_updates_keep_gating_near_zero():
    kf = make()
    z = np.array([100.0, 200.0, 0.5, 40.0])
    for _ in range(5):
        kf.predict()
        kf.update(z)
    assert kf.gating_distance(z) < 1.0


def test_box_clamps_even_for_zero_height_filter():
    # The clamp lives only in `.box`; the internal noise model uses the raw h.
    kf = KalmanFilter(np.array([10.0, 10.0, 1.0, 0.0]))
    _, _, w, h = kf.box
    assert h == 1.0 and w == 1.0


def test_raw_zero_height_update_is_singular():
    """Characterization: a filter built directly from h=0 has a degenerate
    (height-scaled) noise model, so update() inverts a singular S and raises.
    Production code avoids this by going through to_measurement (clamps h>=1).
    """
    kf = KalmanFilter(np.array([10.0, 10.0, 1.0, 0.0]))
    kf.predict()
    with pytest.raises(np.linalg.LinAlgError):
        kf.update(np.array([10.0, 10.0, 1.0, 0.0]))


def test_to_measurement_clamp_keeps_filter_well_conditioned():
    """The production path clamps h>=1, so a zero-height detection is safe."""
    z = to_measurement(np.array([10.0, 10.0, 40.0, 0.0]))
    kf = KalmanFilter(z)
    kf.predict()
    kf.update(z)
    assert np.all(np.isfinite(kf.x))
    assert np.all(np.isfinite(kf.P))
    _, _, w, h = kf.box
    assert h >= 1.0 and w >= 1.0
