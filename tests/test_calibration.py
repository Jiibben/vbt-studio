"""Tests for vbt_core.analysis.calibration.calibrate_m_per_px."""
from __future__ import annotations

import numpy as np
import pytest

from vbt_core.analysis import calibrate_m_per_px


def test_returns_tuple_of_mpp_and_size():
    out = calibrate_m_per_px([100] * 6, [0.9] * 6, [False] * 6, 0.45)
    assert isinstance(out, tuple) and len(out) == 2


def test_median_of_confident_non_coasting_frames():
    # 5 confident non-coasting frames of size 100; two coasting frames ignored.
    sizes = [100, 100, 100, 100, 100, 200, 200]
    confs = [0.9] * 7
    coasting = [False] * 5 + [True] * 2
    mpp, size_px = calibrate_m_per_px(sizes, confs, coasting, 0.45)
    assert size_px == 100.0
    assert mpp == pytest.approx(0.45 / 100.0)


def test_hand_computed_value():
    # diameter 0.45 m, median box 90 px -> 0.005 m/px.
    mpp, size_px = calibrate_m_per_px([90] * 5, [1.0] * 5, [False] * 5, 0.45)
    assert size_px == 90.0
    assert mpp == pytest.approx(0.005)


def test_low_confidence_frames_excluded_when_enough_confident():
    # 6 confident @100, 3 unconfident @1000; with >= min_samples confident the
    # low-conf ones never enter the median.
    sizes = [100] * 6 + [1000] * 3
    confs = [0.9] * 6 + [0.1] * 3
    coasting = [False] * 9
    mpp, size_px = calibrate_m_per_px(sizes, confs, coasting, 0.45)
    assert size_px == 100.0


def test_fallback_to_all_non_coasting_when_few_confident():
    # Only 3 confident frames (< default min_samples=5) -> fall back to ALL
    # non-coasting frames (confident + unconfident), so the unconfident sizes
    # now influence the median.
    sizes = [200, 200, 200, 100, 100, 100, 100]
    confs = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]
    coasting = [False] * 7
    mpp, size_px = calibrate_m_per_px(sizes, confs, coasting, 0.45)
    # median of all seven non-coasting sizes = 100 (not 200 from confident-only)
    assert size_px == 100.0
    assert mpp == pytest.approx(0.45 / 100.0)


def test_fallback_still_excludes_coasting():
    # Few confident -> fallback, but coasting frames are excluded in both paths.
    sizes = [120, 120, 120, 80, 999, 999]
    confs = [0.9, 0.9, 0.9, 0.2, 0.2, 0.2]
    coasting = [False, False, False, False, True, True]
    mpp, size_px = calibrate_m_per_px(sizes, confs, coasting, 0.45)
    # non-coasting sizes = [120,120,120,80] -> median 120
    assert size_px == 120.0


def test_all_coasting_returns_none():
    assert calibrate_m_per_px([100, 100], [0.9, 0.9], [True, True], 0.45) == (None, None)


def test_empty_returns_none():
    assert calibrate_m_per_px([], [], [], 0.45) == (None, None)


def test_median_size_non_positive_returns_none():
    mpp, size_px = calibrate_m_per_px([0] * 6, [0.9] * 6, [False] * 6, 0.45)
    assert (mpp, size_px) == (None, None)


def test_custom_conf_thresh():
    # All confs 0.6: with conf_thresh 0.7 none pass -> fall back to non-coasting.
    sizes = [50, 50, 50, 50, 50, 50]
    confs = [0.6] * 6
    coasting = [False] * 6
    mpp, size_px = calibrate_m_per_px(sizes, confs, coasting, 0.45,
                                      conf_thresh=0.7)
    assert size_px == 50.0  # fallback to all non-coasting still yields data


def test_custom_min_samples():
    # min_samples=3: 3 confident @100 is enough, low-conf @999 excluded.
    sizes = [100, 100, 100, 999, 999]
    confs = [0.9, 0.9, 0.9, 0.1, 0.1]
    coasting = [False] * 5
    mpp, size_px = calibrate_m_per_px(sizes, confs, coasting, 0.45,
                                      min_samples=3)
    assert size_px == 100.0


def test_conf_thresh_boundary_inclusive():
    # conf exactly == thresh passes (>=).
    mpp, size_px = calibrate_m_per_px([100] * 5, [0.5] * 5, [False] * 5, 0.45,
                                      conf_thresh=0.5)
    assert size_px == 100.0


def test_accepts_numpy_arrays():
    mpp, size_px = calibrate_m_per_px(
        np.full(6, 100.0), np.full(6, 0.9), np.zeros(6, dtype=bool), 0.45)
    assert size_px == 100.0


@pytest.mark.parametrize("diameter_m,size_px", [
    (0.45, 90.0),
    (0.45, 150.0),
    (0.50, 100.0),
    (0.35, 70.0),
])
def test_parametrized_mpp(diameter_m, size_px):
    mpp, sz = calibrate_m_per_px([size_px] * 6, [0.9] * 6, [False] * 6,
                                 diameter_m)
    assert sz == size_px
    assert mpp == pytest.approx(diameter_m / size_px)


def test_returns_python_floats():
    mpp, size_px = calibrate_m_per_px([100] * 6, [0.9] * 6, [False] * 6, 0.45)
    assert type(mpp) is float and type(size_px) is float


def test_even_count_uses_average_of_two_middle():
    # numpy median of an even set averages the two central values.
    mpp, size_px = calibrate_m_per_px([100, 100, 200, 200, 100, 200],
                                      [0.9] * 6, [False] * 6, 0.45)
    # sorted: 100,100,100,200,200,200 -> median 150
    assert size_px == 150.0
