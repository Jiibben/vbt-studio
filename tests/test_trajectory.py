"""Tests for vbt_core.analysis.trajectory.analyze_trajectory."""
from __future__ import annotations

import json

import numpy as np
import pytest

from vbt_core.analysis import AnalysisResult, RepMetrics, analyze_trajectory

from conftest import deadlift_path, squat_path

PLATE_MM = 450.0
SIZE_PX = 150.0
MPP = (PLATE_MM / 1000.0) / SIZE_PX        # metres per pixel for the fixtures


def _to_px(y_m, base_px=300.0):
    """Convert a +down metres path into pixel centre-y plus a base offset."""
    return base_px + np.asarray(y_m) / MPP


def _detection_arrays(n, size=SIZE_PX, conf=0.9, coasting=False):
    return (np.full(n, size), np.full(n, conf),
            np.full(n, coasting, dtype=bool))


def squat_px(n_reps=3, rom=0.30, fps=30.0):
    ts, y_m = squat_path(n_reps=n_reps, rom=rom, fps=fps)
    return ts, _to_px(y_m)


def deadlift_px(n_reps=3, rom=0.30, fps=30.0):
    ts, y_m = deadlift_path(n_reps=n_reps, rom=rom, fps=fps)
    return ts, _to_px(y_m)


# --------------------------------------------------------------------------- #
# Guard clauses
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [0, 1, 4])
def test_too_few_samples_uncalibrated(n):
    res = analyze_trajectory(
        list(range(n)), list(range(n)), [1] * n, [1] * n, [False] * n,
        plate_diameter_mm=PLATE_MM, fps=30.0)
    assert isinstance(res, AnalysisResult)
    assert res.calibrated is False
    assert res.rep_count == 0


def test_five_samples_passes_length_guard():
    # Exactly 5 samples clears the length guard (calibration may still run).
    ts = np.arange(5) / 30.0
    cy = _to_px(np.zeros(5))
    sizes, confs, coast = _detection_arrays(5)
    res = analyze_trajectory(ts, cy, sizes, confs, coast,
                             plate_diameter_mm=PLATE_MM, fps=30.0)
    assert res.calibrated is True   # 5 confident frames -> calibration succeeds


def test_calibration_failure_uncalibrated():
    # All frames coasting -> calibration returns None -> uncalibrated result.
    ts, cy = squat_px(2)
    n = len(ts)
    sizes, confs, _ = _detection_arrays(n)
    coast = np.ones(n, dtype=bool)
    res = analyze_trajectory(ts, cy, sizes, confs, coast,
                             plate_diameter_mm=PLATE_MM, fps=30.0)
    assert res.calibrated is False
    assert res.rep_count == 0


# --------------------------------------------------------------------------- #
# Happy path — squat
# --------------------------------------------------------------------------- #
@pytest.fixture
def squat_result():
    ts, cy = squat_px(3, 0.30, 30.0)
    n = len(ts)
    sizes, confs, coast = _detection_arrays(n)
    return analyze_trajectory(ts, cy, sizes, confs, coast,
                              plate_diameter_mm=PLATE_MM, fps=30.0)


def test_squat_calibrated_fields(squat_result):
    res = squat_result
    assert res.calibrated is True
    assert res.m_per_px == pytest.approx(MPP, rel=1e-6)
    assert res.plate_diameter_mm == PLATE_MM
    assert res.plate_size_px == pytest.approx(SIZE_PX, abs=0.1)
    assert res.rep_style == "down-first"


def test_squat_rep_count(squat_result):
    assert squat_result.rep_count == 3
    assert len(squat_result.reps) == 3


def test_series_keys_and_equal_length(squat_result):
    series = squat_result.series
    assert set(series) == {"t", "v", "y_m", "phase", "rep"}
    lengths = {len(series[k]) for k in series}
    assert len(lengths) == 1            # all equal


def test_series_phase_values(squat_result):
    assert set(squat_result.series["phase"]) <= {"idle", "down", "up"}


def test_series_rep_indices(squat_result):
    reps = set(squat_result.series["rep"])
    # rep index 0 = "not in a rep", plus 1..3 for the three reps.
    assert reps <= {0, 1, 2, 3}
    assert {1, 2, 3} <= reps


def test_reps_are_repmetrics_shaped(squat_result):
    expected = set(RepMetrics.__dataclass_fields__)
    for rep in squat_result.reps:
        assert set(rep) == expected
        assert rep["mean_concentric_v"] > 0
        assert rep["peak_concentric_v"] >= rep["mean_concentric_v"]
        assert rep["rom_m"] > 0
        assert rep["duration_s"] > 0


def test_summary_keys_and_values(squat_result):
    s = squat_result.summary
    assert set(s) == {"avg_mcv", "best_mcv", "last_mcv", "velocity_loss_pct"}
    mcvs = [r["mean_concentric_v"] for r in squat_result.reps]
    assert s["best_mcv"] == pytest.approx(max(mcvs))
    assert s["last_mcv"] == pytest.approx(mcvs[-1])
    assert s["avg_mcv"] == pytest.approx(round(float(np.mean(mcvs)), 3))
    assert s["velocity_loss_pct"] >= 0.0


def test_rom_matches_expected(squat_result):
    # ROM should be close to the 0.30 m built into the path.
    for rep in squat_result.reps:
        assert rep["rom_m"] == pytest.approx(0.30, abs=0.03)


# --------------------------------------------------------------------------- #
# Deadlift (up-first)
# --------------------------------------------------------------------------- #
def test_deadlift_style_and_reps():
    ts, cy = deadlift_px(2, 0.30, 30.0)
    n = len(ts)
    sizes, confs, coast = _detection_arrays(n)
    res = analyze_trajectory(ts, cy, sizes, confs, coast,
                             plate_diameter_mm=PLATE_MM, fps=30.0)
    assert res.calibrated is True
    assert res.rep_style == "up-first"
    assert res.rep_count == 2
    for rep in res.reps:
        assert rep["mean_concentric_v"] > 0


# --------------------------------------------------------------------------- #
# Flat path: calibrated but no reps
# --------------------------------------------------------------------------- #
def test_flat_path_no_reps():
    ts = np.arange(0.0, 5.0, 1.0 / 30.0)
    rng = np.random.default_rng(1)
    y_m = rng.normal(0.0, 0.01, len(ts))
    cy = _to_px(y_m)
    n = len(ts)
    sizes, confs, coast = _detection_arrays(n)
    res = analyze_trajectory(ts, cy, sizes, confs, coast,
                             plate_diameter_mm=PLATE_MM, fps=30.0)
    assert res.calibrated is True
    assert res.rep_count == 0
    assert res.reps == []
    assert res.summary == {}


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def test_to_dict_is_json_serialisable(squat_result):
    d = squat_result.to_dict()
    assert isinstance(d, dict)
    text = json.dumps(d)               # must not raise
    round_tripped = json.loads(text)
    assert round_tripped["rep_count"] == squat_result.rep_count
    assert round_tripped["calibrated"] is True


def test_uncalibrated_to_dict_serialisable():
    res = analyze_trajectory([], [], [], [], [],
                             plate_diameter_mm=PLATE_MM, fps=30.0)
    json.dumps(res.to_dict())


# --------------------------------------------------------------------------- #
# Parametrized families
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_reps", [1, 2, 4])
def test_various_rep_counts(n_reps):
    ts, cy = squat_px(n_reps, 0.30, 30.0)
    n = len(ts)
    sizes, confs, coast = _detection_arrays(n)
    res = analyze_trajectory(ts, cy, sizes, confs, coast,
                             plate_diameter_mm=PLATE_MM, fps=30.0)
    assert res.rep_count == n_reps


@pytest.mark.parametrize("fps", [24, 30, 60])
def test_various_fps(fps):
    ts, cy = squat_px(2, 0.30, fps)
    n = len(ts)
    sizes, confs, coast = _detection_arrays(n)
    res = analyze_trajectory(ts, cy, sizes, confs, coast,
                             plate_diameter_mm=PLATE_MM, fps=fps)
    assert res.rep_count == 2
    assert res.calibrated is True


@pytest.mark.parametrize("rep_style", ["auto", "down-first"])
def test_forced_style_matches_detection_on_squat(rep_style):
    ts, cy = squat_px(2, 0.30, 30.0)
    n = len(ts)
    sizes, confs, coast = _detection_arrays(n)
    res = analyze_trajectory(ts, cy, sizes, confs, coast,
                             plate_diameter_mm=PLATE_MM, fps=30.0,
                             rep_style=rep_style)
    assert res.rep_style == "down-first"
    assert res.rep_count == 2


def test_min_displacement_override_suppresses_small_reps():
    # ROM 0.20 m: a 0.30 m threshold suppresses it; default 0.15 detects it.
    ts, cy = squat_px(2, 0.20, 30.0)
    n = len(ts)
    sizes, confs, coast = _detection_arrays(n)
    strict = analyze_trajectory(ts, cy, sizes, confs, coast,
                                plate_diameter_mm=PLATE_MM, fps=30.0,
                                min_displacement_m=0.30)
    lax = analyze_trajectory(ts, cy, sizes, confs, coast,
                             plate_diameter_mm=PLATE_MM, fps=30.0,
                             min_displacement_m=0.15)
    assert strict.rep_count == 0
    assert lax.rep_count == 2


def test_default_result_shape():
    res = AnalysisResult(calibrated=False)
    assert res.rep_count == 0
    assert res.reps == [] and res.summary == {} and res.series == {}
    assert res.m_per_px is None
