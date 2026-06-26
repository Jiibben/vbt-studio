"""Tests for vbt_core.analysis.rep_counter.RepCounter."""
from __future__ import annotations

import numpy as np
import pytest

from vbt_core.analysis import (PHASE_DOWN, PHASE_IDLE, PHASE_UP, REP_STYLES,
                               RepCounter)


def _run(rc, ts, ys, finalize=True):
    for t, y in zip(ts, ys):
        rc.update(float(t), float(y))
    if finalize:
        rc.finalize()
    return rc


# --------------------------------------------------------------------------- #
# Constants / construction
# --------------------------------------------------------------------------- #
def test_phase_constants():
    assert (PHASE_IDLE, PHASE_DOWN, PHASE_UP) == ("idle", "down", "up")


def test_rep_styles_tuple():
    assert REP_STYLES == ("auto", "down-first", "up-first")


@pytest.mark.parametrize("style", ["auto", "down-first", "up-first"])
def test_valid_styles_construct(style):
    rc = RepCounter(style=style)
    assert rc.style == style


def test_invalid_style_raises():
    with pytest.raises(ValueError):
        RepCounter(style="bogus")


def test_initial_state():
    rc = RepCounter()
    assert rc.state == PHASE_IDLE
    assert rc.rep_count == 0
    assert rc.reps == []
    assert rc.detected_style is None
    assert rc.style_in_effect == "down-first"   # default when unresolved


def test_forced_style_resolves_immediately():
    rc = RepCounter(style="up-first")
    assert rc.detected_style == "up-first"
    assert rc.style_in_effect == "up-first"


# --------------------------------------------------------------------------- #
# Auto-detection + rep counting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_reps", [1, 2, 3, 5])
def test_squat_down_first_count(make_squat, n_reps):
    ts, ys = make_squat(n_reps=n_reps, rom=0.30)
    rc = _run(RepCounter(), ts, ys)
    assert rc.detected_style == "down-first"
    assert rc.rep_count == n_reps
    assert len(rc.reps) == n_reps


@pytest.mark.parametrize("n_reps", [1, 2, 3])
def test_deadlift_up_first_count(make_deadlift, n_reps):
    ts, ys = make_deadlift(n_reps=n_reps, rom=0.30)
    rc = _run(RepCounter(), ts, ys)
    assert rc.detected_style == "up-first"
    assert rc.rep_count == n_reps


@pytest.mark.parametrize("rom", [0.20, 0.30, 0.45, 0.60])
def test_various_rom_sizes_detected(make_squat, rom):
    ts, ys = make_squat(n_reps=2, rom=rom)
    rc = _run(RepCounter(), ts, ys)
    assert rc.rep_count == 2


@pytest.mark.parametrize("fps", [24, 30, 60, 120])
def test_various_fps(make_squat, fps):
    ts, ys = make_squat(n_reps=2, rom=0.30, fps=fps)
    rc = _run(RepCounter(), ts, ys)
    assert rc.rep_count == 2
    assert rc.detected_style == "down-first"


# --------------------------------------------------------------------------- #
# Rep structure (eccentric/concentric ordering)
# --------------------------------------------------------------------------- #
def test_down_first_rep_has_eccentric_before_concentric(make_squat):
    ts, ys = make_squat(n_reps=2, rom=0.30)
    rc = _run(RepCounter(), ts, ys)
    for rep in rc.reps:
        # eccentric (down) precedes concentric (up)
        assert rep["ecc_start"] is not None
        assert rep["ecc_end"] <= rep["conc_start"] + 1e-9
        assert rep["conc_start"] < rep["conc_end"]
        assert rep["start_t"] == rep["ecc_start"]


def test_up_first_rep_concentric_then_eccentric(make_deadlift):
    ts, ys = make_deadlift(n_reps=2, rom=0.30)
    rc = _run(RepCounter(), ts, ys)
    for rep in rc.reps:
        # concentric (up) precedes the eccentric (down) lowering
        assert rep["conc_start"] < rep["conc_end"]
        assert rep["ecc_start"] is not None
        assert rep["conc_end"] <= rep["ecc_start"] + 1e-9
        assert rep["end_t"] == rep["ecc_end"]


def test_rep_dict_keys(make_squat):
    ts, ys = make_squat(n_reps=1, rom=0.30)
    rc = _run(RepCounter(), ts, ys)
    rep = rc.reps[0]
    assert set(rep) == {"rep", "start_t", "end_t", "conc_start", "conc_end",
                        "ecc_start", "ecc_end"}
    assert rep["rep"] == 1


def test_rep_numbers_are_sequential(make_squat):
    ts, ys = make_squat(n_reps=4, rom=0.30)
    rc = _run(RepCounter(), ts, ys)
    assert [r["rep"] for r in rc.reps] == [1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# Hysteresis / noise rejection
# --------------------------------------------------------------------------- #
def test_flat_noisy_path_no_reps(make_flat):
    ts, ys = make_flat(seconds=5.0, noise=0.01, seed=0)
    rc = _run(RepCounter(), ts, ys)
    assert rc.rep_count == 0
    assert rc.reps == []


def test_jitter_below_min_displacement_no_reps():
    # Oscillation with peak-to-peak 0.10 m < min_displacement 0.15 m.
    ts = np.arange(0.0, 6.0, 1.0 / 30.0)
    ys = 0.05 * np.sin(2 * np.pi * 0.5 * ts)   # amplitude 0.05 -> swing 0.10
    rc = _run(RepCounter(min_displacement_m=0.15), ts, ys)
    assert rc.rep_count == 0


def test_displacement_just_above_threshold_counts():
    # Single down-up cycle with swing 0.20 m > 0.15 m -> exactly one rep.
    ts = np.arange(0.0, 2.0, 1.0 / 30.0)
    ys = 0.10 * (1 - np.cos(2 * np.pi * 0.5 * ts))  # 0 -> 0.20 -> 0
    rc = _run(RepCounter(min_displacement_m=0.15), ts, ys)
    assert rc.rep_count == 1


def test_smaller_min_displacement_detects_small_reps():
    ts = np.arange(0.0, 2.0, 1.0 / 30.0)
    ys = 0.05 * (1 - np.cos(2 * np.pi * 0.5 * ts))  # swing 0.10
    assert _run(RepCounter(min_displacement_m=0.15), ts, ys).rep_count == 0
    assert _run(RepCounter(min_displacement_m=0.05), ts, ys).rep_count == 1


# --------------------------------------------------------------------------- #
# min_phase_s rejection
# --------------------------------------------------------------------------- #
def test_too_short_up_phase_not_counted():
    # A fast up phase shorter than min_phase_s is rejected even if displacement
    # exceeds the threshold.
    ts = np.arange(0.0, 2.0, 1.0 / 30.0)
    ys = 0.10 * (1 - np.cos(2 * np.pi * 0.5 * ts))   # ~1 s up phase
    # min_phase_s huge -> the up phase is "too short" -> no rep.
    rc = _run(RepCounter(min_displacement_m=0.15, min_phase_s=10.0), ts, ys)
    assert rc.rep_count == 0


# --------------------------------------------------------------------------- #
# Forcing style overrides auto-detection
# --------------------------------------------------------------------------- #
def test_force_up_first_on_squat_path(make_squat):
    ts, ys = make_squat(n_reps=2, rom=0.30)
    rc = _run(RepCounter(style="up-first"), ts, ys)
    assert rc.detected_style == "up-first"


def test_force_down_first_on_deadlift_path(make_deadlift):
    ts, ys = make_deadlift(n_reps=2, rom=0.30)
    rc = _run(RepCounter(style="down-first"), ts, ys)
    assert rc.detected_style == "down-first"


# --------------------------------------------------------------------------- #
# phase_at / transitions
# --------------------------------------------------------------------------- #
def test_phase_at_reflects_transitions(make_squat):
    ts, ys = make_squat(n_reps=1, rom=0.30, fps=30)
    rc = _run(RepCounter(), ts, ys)
    # one rep: down ~[0,1], up ~[1,2]
    assert rc.phase_at(-1.0) == PHASE_IDLE
    assert rc.phase_at(0.5) == PHASE_DOWN
    assert rc.phase_at(1.5) == PHASE_UP


def test_phase_at_before_any_transition_is_idle(make_flat):
    ts, ys = make_flat()
    rc = _run(RepCounter(), ts, ys)
    assert rc.transitions == []
    assert rc.phase_at(0.0) == PHASE_IDLE
    assert rc.phase_at(100.0) == PHASE_IDLE


def test_phase_at_boundary_inclusive(make_squat):
    ts, ys = make_squat(n_reps=1, rom=0.30, fps=30)
    rc = _run(RepCounter(), ts, ys)
    first_t, first_state = rc.transitions[0]
    # exactly at the transition timestamp the new state already applies.
    assert rc.phase_at(first_t) == first_state


# --------------------------------------------------------------------------- #
# finalize
# --------------------------------------------------------------------------- #
def test_finalize_closes_dangling_up_phase(make_squat):
    ts, ys = make_squat(n_reps=1, rom=0.30)
    rc = _run(RepCounter(), ts, ys, finalize=False)
    # Path ends mid-lockout: the last 'up' phase is still open.
    assert rc.state == PHASE_UP
    assert rc.rep_count == 0
    rc.finalize()
    assert rc.rep_count == 1


def test_finalize_idempotent_after_completed_reps(make_squat):
    ts, ys = make_squat(n_reps=2, rom=0.30)
    rc = _run(RepCounter(), ts, ys)
    count = rc.rep_count
    rc.finalize()   # second call should not double-count
    assert rc.rep_count == count


def test_finalize_returns_none_when_nothing_to_close(make_flat):
    ts, ys = make_flat()
    rc = _run(RepCounter(), ts, ys, finalize=False)
    assert rc.finalize() is None


def test_update_returns_completed_rep(make_squat):
    # update() returns the rep dict the moment a rep closes.
    ts, ys = make_squat(n_reps=2, rom=0.30)
    rc = RepCounter()
    completed = [rc.update(float(t), float(y)) for t, y in zip(ts, ys)]
    closed = [c for c in completed if c is not None]
    # second rep closes at the start of the next descent during the pass;
    # first rep certainly closes mid-stream.
    assert any(c["rep"] == 1 for c in closed)


# --------------------------------------------------------------------------- #
# Dwell tolerance: a bottom pause stays in the eccentric, not the concentric.
# --------------------------------------------------------------------------- #
def test_bottom_pause_stays_eccentric():
    fps = 30.0
    dt = 1.0 / fps
    per = int(fps)            # 1 s phases
    rom = 0.30
    ys, ts, t = [], [], 0.0
    # descend 0 -> rom over 1 s
    for i in range(per):
        ph = i / per
        ys.append(rom * 0.5 * (1 - np.cos(np.pi * ph))); ts.append(t); t += dt
    # pause at the bottom for 0.5 s (held at rom, within dwell band)
    for _ in range(int(0.5 * fps)):
        ys.append(rom); ts.append(t); t += dt
    # ascend rom -> 0 over 1 s
    for i in range(per):
        ph = i / per
        ys.append(rom * 0.5 * (1 + np.cos(np.pi * ph))); ts.append(t); t += dt
    ys.append(0.0); ts.append(t)
    rc = _run(RepCounter(min_displacement_m=0.15, dwell_tolerance_m=0.02),
              np.asarray(ts), np.asarray(ys))
    assert rc.rep_count == 1
    rep = rc.reps[0]
    # Nominal descent ends at t≈1.0; the pause (to ≈1.5) must remain eccentric,
    # so the concentric only starts well after 1.0.
    assert rep["conc_start"] > 1.4
    # eccentric end coincides with concentric start (no gap, no overlap).
    assert rep["ecc_end"] == pytest.approx(rep["conc_start"])


# --------------------------------------------------------------------------- #
# Misc input handling
# --------------------------------------------------------------------------- #
def test_single_sample_no_crash():
    rc = RepCounter()
    assert rc.update(0.0, 0.0) is None
    rc.finalize()
    assert rc.rep_count == 0


def test_phases_list_populated(make_squat):
    ts, ys = make_squat(n_reps=1, rom=0.30)
    rc = _run(RepCounter(), ts, ys)
    kinds = [p["phase"] for p in rc.phases]
    assert PHASE_DOWN in kinds and PHASE_UP in kinds
