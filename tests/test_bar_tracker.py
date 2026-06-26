"""Characterization + edge-case tests for vbt_core.tracking.bar_tracker.BarTracker.

Lifecycle: tentative -> confirmed (after min_hits) -> coasting (no detection,
pure prediction) -> dead (after max_age for confirmed, 2 misses for tentative).
`primary_id()` returns the longest-lived-then-largest track over the whole run.
"""
from __future__ import annotations

import numpy as np
import pytest

from vbt_core.tracking.bar_tracker import BarTracker
from vbt_core.tracking.track import TrackSnapshot

EMPTY = np.zeros((0, 5))


def det(cx=100.0, cy=100.0, w=40.0, h=40.0, conf=0.9):
    return np.array([[cx, cy, w, h, conf]], dtype=float)


# --------------------------------------------------------------------------- #
# Confirmation lifecycle
# --------------------------------------------------------------------------- #
def test_no_snapshots_before_min_hits():
    bt = BarTracker()
    d = det()
    assert bt.update(d) == []   # frame 1: hits=1, tentative
    assert bt.update(d) == []   # frame 2: hits=2, tentative
    snaps = bt.update(d)        # frame 3: hits=3 -> confirmed
    assert len(snaps) == 1
    assert isinstance(snaps[0], TrackSnapshot)


def test_confirm_after_exactly_min_hits():
    bt = BarTracker(min_hits=3)
    d = det()
    for _ in range(2):
        bt.update(d)
    assert not bt.tracks[0].confirmed
    bt.update(d)
    assert bt.tracks[0].confirmed


@pytest.mark.parametrize("min_hits", [2, 3, 5])
def test_min_hits_parametrized(min_hits):
    bt = BarTracker(min_hits=min_hits)
    d = det()
    for i in range(1, min_hits):
        assert bt.update(d) == []
    assert len(bt.update(d)) == 1


def test_primary_id_returns_confirmed_track():
    bt = BarTracker()
    d = det()
    assert bt.primary_id() is None
    for _ in range(3):
        bt.update(d)
    assert bt.primary_id() == 1


def test_snapshot_fields_on_steady_track():
    bt = BarTracker()
    d = det(cx=100.0, cy=100.0, w=40.0, h=40.0, conf=0.9)
    snap = None
    for _ in range(4):
        snaps = bt.update(d)
        if snaps:
            snap = snaps[0]
    assert snap.track_id == 1
    assert snap.cx == pytest.approx(100.0, abs=2.0)
    assert snap.cy == pytest.approx(100.0, abs=2.0)
    assert snap.w == pytest.approx(40.0, abs=2.0)
    assert snap.h == pytest.approx(40.0, abs=2.0)
    assert snap.conf == pytest.approx(0.9)
    assert snap.coasting is False
    assert snap.hits >= 3


# --------------------------------------------------------------------------- #
# Coasting
# --------------------------------------------------------------------------- #
def test_confirmed_track_coasts_on_empty_frame():
    bt = BarTracker()
    for _ in range(4):
        bt.update(det())
    snaps = bt.update(EMPTY)
    assert len(snaps) == 1
    s = snaps[0]
    assert s.coasting is True
    assert s.conf == 0.0
    assert s.track_id == 1


def test_coasting_track_keeps_moving_with_velocity():
    bt = BarTracker()
    # feed a track moving steadily right so it has +x velocity
    for i in range(6):
        bt.update(det(cx=100.0 + 10.0 * i))
    last_real = bt.tracks[0].kf.box[0]
    s = bt.update(EMPTY)[0]
    # pure prediction continues advancing cx in the motion direction
    assert s.cx > last_real
    assert s.coasting is True


def test_confirmed_track_survives_until_max_age_then_dropped():
    bt = BarTracker(max_age=5)
    for _ in range(4):
        bt.update(det())
    # coast: survives while time_since_update <= max_age
    for i in range(1, 6):
        snaps = bt.update(EMPTY)
        assert len(bt.tracks) == 1, f"dropped too early at miss {i}"
        assert len(snaps) == 1
    # the 6th miss pushes time_since_update past max_age -> dropped
    snaps = bt.update(EMPTY)
    assert bt.tracks == []
    assert snaps == []


def test_coasting_then_redetect_resumes_and_clears_coasting():
    bt = BarTracker(max_age=10)
    for _ in range(4):
        bt.update(det())
    bt.update(EMPTY)
    bt.update(EMPTY)
    snaps = bt.update(det())
    assert len(snaps) == 1
    assert snaps[0].coasting is False
    assert snaps[0].conf == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# Tentative death
# --------------------------------------------------------------------------- #
def test_tentative_track_survives_two_misses_dies_on_third():
    bt = BarTracker()
    bt.update(det())                 # 1 hit, tentative
    assert len(bt.tracks) == 1
    bt.update(EMPTY)                 # miss 1 -> tsu=1 (<=2) kept
    assert len(bt.tracks) == 1
    bt.update(EMPTY)                 # miss 2 -> tsu=2 (<=2) kept
    assert len(bt.tracks) == 1
    bt.update(EMPTY)                 # miss 3 -> tsu=3 (>2) dropped
    assert bt.tracks == []


def test_tentative_track_emits_no_snapshots():
    bt = BarTracker()
    snaps = bt.update(det())
    assert snaps == []
    assert bt.primary_id() is None   # never confirmed -> no history


# --------------------------------------------------------------------------- #
# Birth rules
# --------------------------------------------------------------------------- #
def test_low_conf_detections_never_birth_a_track():
    bt = BarTracker(high_thresh=0.45, low_thresh=0.10)
    low = det(conf=0.2)
    for _ in range(6):
        assert bt.update(low) == []
    assert bt.tracks == []
    assert bt.primary_id() is None


def test_below_low_thresh_ignored_entirely():
    bt = BarTracker(low_thresh=0.10)
    for _ in range(6):
        bt.update(det(conf=0.05))
    assert bt.tracks == []


def test_high_conf_unmatched_births_new_track():
    bt = BarTracker()
    for _ in range(3):
        bt.update(det(cx=100.0))
    n_before = len(bt.tracks)
    # a second, spatially separate high-conf detection arrives
    two = np.array([
        [100.0, 100.0, 40.0, 40.0, 0.9],
        [400.0, 400.0, 40.0, 40.0, 0.9],
    ])
    bt.update(two)
    assert len(bt.tracks) == n_before + 1


# --------------------------------------------------------------------------- #
# Primary stability
# --------------------------------------------------------------------------- #
def test_spurious_far_high_conf_box_does_not_steal_primary():
    bt = BarTracker()
    big = det(cx=100.0, cy=100.0, w=80.0, h=80.0, conf=0.9)
    for _ in range(20):
        bt.update(big)
    assert bt.primary_id() == 1
    for _ in range(4):
        combo = np.array([
            [100.0, 100.0, 80.0, 80.0, 0.9],     # the long-lived big plate
            [500.0, 500.0, 20.0, 20.0, 0.99],    # spurious far high-conf box
        ])
        bt.update(combo)
    # the big, long-lived plate keeps the primary slot
    assert bt.primary_id() == 1


def test_primary_prefers_more_hits_then_larger_area():
    bt = BarTracker()
    # track 1: many frames, big
    big = det(cx=100.0, cy=100.0, w=80.0, h=80.0, conf=0.9)
    for _ in range(10):
        bt.update(big)
    # introduce track 2: fewer frames, even bigger box -> still loses on hits
    for _ in range(4):
        combo = np.array([
            [100.0, 100.0, 80.0, 80.0, 0.9],
            [600.0, 100.0, 200.0, 200.0, 0.9],
        ])
        bt.update(combo)
    hits1 = bt._history[1][0]
    hits2 = bt._history[2][0]
    assert hits1 > hits2
    assert bt.primary_id() == 1


def test_history_records_confirmed_tracks_only():
    bt = BarTracker()
    bt.update(det())            # tentative only
    assert bt._history == {}
    for _ in range(3):
        bt.update(det())
    assert 1 in bt._history
    hits, area = bt._history[1]
    assert hits >= 3
    assert area > 0.0


# --------------------------------------------------------------------------- #
# Sinusoidal motion (realistic) — split focused checks
# --------------------------------------------------------------------------- #
def _sinusoidal_run(seed=0, frames=40, noise=0.5):
    rng = np.random.default_rng(seed)
    bt = BarTracker()
    ids_seen = []
    coasting_count = 0
    for f in range(frames):
        cy = 200.0 + 40.0 * np.sin(2 * np.pi * f / 40.0)
        cx = 320.0 + rng.normal(0, noise)
        cy += rng.normal(0, noise)
        d = np.array([[cx, cy, 80.0, 80.0, 0.9]])
        snaps = bt.update(d)
        for s in snaps:
            ids_seen.append(s.track_id)
            if s.coasting:
                coasting_count += 1
    return bt, ids_seen, coasting_count


def test_sinusoidal_motion_single_stable_id():
    bt, ids_seen, coasting = _sinusoidal_run(seed=42)
    # one barbell -> exactly one track id ever emitted
    assert set(ids_seen) == {1}
    # a fully-detected steady stream never coasts
    assert coasting == 0
    assert bt.primary_id() == 1


def test_sinusoidal_motion_tracks_vertical_position():
    rng = np.random.default_rng(7)
    bt = BarTracker()
    errs = []
    for f in range(60):
        cy = 200.0 + 40.0 * np.sin(2 * np.pi * f / 40.0)
        d = np.array([[320.0, cy + rng.normal(0, 0.3), 80.0, 80.0, 0.9]])
        snaps = bt.update(d)
        if snaps:
            errs.append(abs(snaps[0].cy - cy))
    # smoothed estimate stays close to the true sinusoid after warm-up
    assert np.median(errs[5:]) < 15.0


def test_velocity_sign_follows_motion_direction():
    bt = BarTracker()
    # rising cy (moving down in image coords) -> positive vy
    for i in range(8):
        bt.update(det(cx=100.0, cy=100.0 + 8.0 * i))
    snap = bt.update(det(cx=100.0, cy=100.0 + 8.0 * 8))[0]
    assert snap.vy > 0.0
    assert abs(snap.vx) < abs(snap.vy)


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def test_update_accepts_flat_single_detection():
    bt = BarTracker()
    flat = np.array([100.0, 100.0, 40.0, 40.0, 0.9])  # 1-D, reshaped internally
    for _ in range(3):
        snaps = bt.update(flat)
    assert len(snaps) == 1


def test_update_accepts_python_list():
    bt = BarTracker()
    for _ in range(3):
        snaps = bt.update([[100.0, 100.0, 40.0, 40.0, 0.9]])
    assert len(snaps) == 1
