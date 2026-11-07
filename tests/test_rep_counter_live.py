"""Tests for RepCounter.live_rep_count — the lockout-timed live/HUD counter.

The structured `rep_count`/`reps` are covered in test_rep_counter.py; here we pin
the NEW behavior: the live counter ticks at the top of the concentric (lockout),
earlier than the structured count, is displacement-based (slow reps still count),
counts each rep exactly once (sticking points don't double), and never fires on
idle jitter.
"""
from __future__ import annotations

import numpy as np
import pytest

from vbt_core.analysis import RepCounter


def _run(rc, ts, ys, finalize=True):
    for t, y in zip(ts, ys):
        rc.update(float(t), float(y))
    if finalize:
        rc.finalize()
    return rc


def _step_record(rc, ts, ys):
    """Run sample-by-sample, returning per-sample (live, structured) counts."""
    live, struct = [], []
    for t, y in zip(ts, ys):
        rc.update(float(t), float(y))
        live.append(rc.live_rep_count)
        struct.append(rc.rep_count)
    return np.asarray(live), np.asarray(struct)


# --------------------------------------------------------------------------- #
# Live count totals match the number of reps (both styles, incl. finalize)
# --------------------------------------------------------------------------- #
def test_initial_live_count_zero():
    assert RepCounter().live_rep_count == 0


@pytest.mark.parametrize("n_reps", [1, 2, 3, 5])
def test_live_count_squat(make_squat, n_reps):
    ts, ys = make_squat(n_reps=n_reps, rom=0.30)
    rc = _run(RepCounter(), ts, ys)
    assert rc.live_rep_count == n_reps
    assert rc.live_rep_count == rc.rep_count   # agrees with the structured total


@pytest.mark.parametrize("n_reps", [1, 2, 3])
def test_live_count_deadlift(make_deadlift, n_reps):
    ts, ys = make_deadlift(n_reps=n_reps, rom=0.30)
    rc = _run(RepCounter(), ts, ys)
    assert rc.live_rep_count == n_reps
    assert rc.live_rep_count == rc.rep_count


# --------------------------------------------------------------------------- #
# The live tick happens at lockout — earlier than the structured count
# --------------------------------------------------------------------------- #
def test_live_tick_precedes_structured_count(make_squat):
    # 2 reps: rep 1's lockout is detected early in rep 2's descent (a small
    # reversal), while the structured count waits for the full 0.15 m hysteresis.
    ts, ys = make_squat(n_reps=2, rom=0.30, seconds_per_phase=1.0, fps=30.0)
    live, struct = _step_record(RepCounter(), ts, ys)

    i_live = int(np.argmax(live >= 1))     # first index where live hits 1
    i_struct = int(np.argmax(struct >= 1))  # first index where structured hits 1
    assert live.max() >= 1 and struct.max() >= 1
    assert i_live < i_struct                # live registers strictly earlier
    # ...and it fires near the top of rep 1 (~2.0 s), not a phase later.
    assert ts[i_live] > 1.5
    assert ts[i_live] < ts[i_struct]


def _squat_with_lockout_hold(fps=30.0, hold_s=0.5, per=30):
    """Two reps where the bar HOLDS standing at the top of rep 1 for `hold_s`
    before descending into rep 2 — the real video19 pattern (a paused lockout)."""
    def seg(a, b, n):
        return list(np.linspace(a, b, n, endpoint=False))
    ys = []
    ys += seg(0.0, 0.30, per)            # rep1 eccentric
    ys += seg(0.30, 0.0, per)            # rep1 concentric -> top at t = 2*per/fps
    ys += [0.0] * int(round(hold_s * fps))  # HOLD at lockout
    ys += seg(0.0, 0.30, per)            # rep2 eccentric
    ys += seg(0.30, 0.0, per)            # rep2 concentric
    ys.append(0.0)
    ts = np.arange(len(ys)) / fps
    return ts, np.asarray(ys)


def test_pause_at_lockout_ticks_during_hold():
    # The bar reaches the top at t=2.0 s and stands there until t=2.5 s. The live
    # tick must fire DURING that hold (the rep is done), not when the next
    # descent finally crosses the hysteresis.
    ts, ys = _squat_with_lockout_hold(fps=30.0, hold_s=0.5)
    live, struct = _step_record(RepCounter(), ts, ys)
    i_live = int(np.argmax(live >= 1))
    i_struct = int(np.argmax(struct >= 1))
    assert live.max() >= 1
    assert 2.0 <= ts[i_live] < 2.5          # ticked during the lockout hold...
    assert i_live < i_struct                # ...well before the next descent


def test_single_rep_ticks_via_finalize_at_lockout(make_squat):
    # A 1-rep squat ends held at the top (no next descent) -> the bar never
    # reverses, so the live tick comes from finalize(), not mid-pass.
    ts, ys = make_squat(n_reps=1, rom=0.30)
    rc = _run(RepCounter(), ts, ys, finalize=False)
    assert rc.live_rep_count == 0            # nothing yet — still at lockout
    rc.finalize()
    assert rc.live_rep_count == 1


# --------------------------------------------------------------------------- #
# Slow reps: displacement-based, so a long grind still counts (no time penalty)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seconds_per_phase", [0.5, 1.0, 3.0, 5.0])
def test_slow_reps_still_count(make_squat, seconds_per_phase):
    ts, ys = make_squat(n_reps=2, rom=0.30, seconds_per_phase=seconds_per_phase)
    rc = _run(RepCounter(), ts, ys)
    assert rc.live_rep_count == 2


# --------------------------------------------------------------------------- #
# Sticking point mid-ascent: counts exactly once (the latch prevents doubles)
# --------------------------------------------------------------------------- #
def _squat_with_sticking_point(fps=30.0, dip_m=0.06):
    """One squat whose ascent has a small downward dip (a grind), large enough
    (> rep_confirm_reversal) to trip the lockout detector mid-concentric."""
    def seg(a, b, n):
        return list(np.linspace(a, b, n, endpoint=False))
    ys = []
    ys += seg(0.0, 0.30, 30)            # eccentric: top -> bottom
    ys += seg(0.30, 0.15, 15)           # ascent, first half
    ys += seg(0.15, 0.15 + dip_m, 6)    # DIP (bar drops back down a little)
    ys += seg(0.15 + dip_m, 0.0, 20)    # ascent resumes to the true top
    ys.append(0.0)
    ts = np.arange(len(ys)) / fps
    return ts, np.asarray(ys)


def test_sticking_point_counts_once():
    ts, ys = _squat_with_sticking_point()
    rc = RepCounter()
    rc_live, _ = _step_record(rc, ts, ys)   # no finalize
    # The dip trips the (single) tick mid-pass; the later true turnover does NOT
    # add a second because the per-up-phase latch is set.
    assert rc.live_rep_count == 1
    rc.finalize()
    assert rc.live_rep_count == 1            # finalize doesn't double-count
    assert rc_live.max() == 1


# --------------------------------------------------------------------------- #
# Noise / robustness
# --------------------------------------------------------------------------- #
def test_idle_noise_no_live_tick(make_flat):
    ts, ys = make_flat(seconds=5.0, noise=0.01)
    rc = _run(RepCounter(), ts, ys)
    assert rc.live_rep_count == 0
    assert rc.rep_count == 0


def test_sub_threshold_oscillation_no_live_tick():
    # ±0.04 m wobble (0.08 m swing) < min_disp: never even enters an up phase.
    fps = 30.0
    ts = np.arange(0.0, 6.0, 1.0 / fps)
    ys = 0.04 * np.sin(2 * np.pi * 0.5 * ts)
    rc = _run(RepCounter(), ts, ys)
    assert rc.live_rep_count == 0


def test_configurable_reversal_threshold(make_squat):
    # A larger reversal threshold still counts correctly (just confirms later).
    ts, ys = make_squat(n_reps=2, rom=0.30)
    rc = _run(RepCounter(rep_confirm_reversal_m=0.10), ts, ys)
    assert rc.live_rep_count == 2
