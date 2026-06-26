"""Shared fixtures + global safety isolation for the test suite.

The most important guarantee here is that **no test ever touches the user's
real ``data/plate_config.json``**. ``config_path()`` reads ``VBT_PLATE_CONFIG``
on every call, so an autouse fixture redirects it into a throwaway temp file for
every test. Individual config tests still set their own path explicitly; this is
a backstop in case one forgets.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _isolate_plate_config(tmp_path, monkeypatch):
    """Point VBT_PLATE_CONFIG at a per-test temp file so the real store is safe."""
    monkeypatch.setenv("VBT_PLATE_CONFIG", str(tmp_path / "plate_config.json"))
    yield


def squat_path_frames(n_frames: int = 120, cycles: int = 2, *,
                      width: int = 240, height: int = 180,
                      box: float = 40.0, conf: float = 0.9,
                      center_y: float = 90.0, amplitude: float = 30.0):
    """Yield synthetic ``FrameDetections`` of one plate moving in a smooth
    squat path (down-then-up, ``cycles`` times) over a small black image.

    The motion is large enough (amplitude in px → > 0.15 m after calibration)
    that the analysis post-pass segments real reps.
    """
    from vbt_core.detection import FrameDetections

    cx = width / 2.0
    for i in range(n_frames):
        phase = 2.0 * np.pi * cycles * i / n_frames
        cy = center_y - amplitude * np.cos(phase)   # i=0 -> top, mid -> bottom
        img = np.zeros((height, width, 3), dtype=np.uint8)
        boxes = np.array([[cx, cy, box, box, conf]], dtype=float)
        yield FrameDetections(index=i, image=img, boxes=boxes)


# --------------------------------------------------------------------------- #
# Synthetic 1-D bar-path builders for the analysis module tests.
#
# Convention: vertical position y is in metres, +down (image coords). A cosine
# "bump" gives a smooth descent/ascent with zero velocity at the turnarounds,
# matching the real smoothed bar path.
# --------------------------------------------------------------------------- #
def _half_cos(rom: float, per: int, descending: bool) -> list:
    """One half-cycle of `per` samples over [0, rom] (descending) or [rom, 0]
    (ascending), excluding the final endpoint."""
    out = []
    for i in range(per):
        ph = i / per
        if descending:                       # 0 -> rom
            out.append(rom * 0.5 * (1 - np.cos(np.pi * ph)))
        else:                                 # rom -> 0
            out.append(rom * 0.5 * (1 + np.cos(np.pi * ph)))
    return out


def squat_path(n_reps: int = 3, rom: float = 0.30, fps: float = 30.0,
               seconds_per_phase: float = 1.0, base: float = 0.0):
    """down-first (squat/bench): each rep descends then ascends.

    Returns (times, y_metres) with y +down, starting/ending at `base`."""
    dt = 1.0 / fps
    per = int(round(seconds_per_phase * fps))
    ys, ts, t = [], [], 0.0
    for _ in range(n_reps):
        for y in _half_cos(rom, per, descending=True):
            ys.append(base + y); ts.append(t); t += dt
        for y in _half_cos(rom, per, descending=False):
            ys.append(base + y); ts.append(t); t += dt
    ys.append(base); ts.append(t)
    return np.asarray(ts), np.asarray(ys)


def deadlift_path(n_reps: int = 3, rom: float = 0.30, fps: float = 30.0,
                  seconds_per_phase: float = 1.0, base: float = 0.0):
    """up-first (deadlift/row): each rep ascends off the floor then lowers.

    The bar starts on the floor at y = base + rom (lowest = largest +down),
    rises to `base`, lowers back, and ends on the floor."""
    dt = 1.0 / fps
    per = int(round(seconds_per_phase * fps))
    ys, ts, t = [], [], 0.0
    for _ in range(n_reps):
        for y in _half_cos(rom, per, descending=False):   # rom -> 0  (lift)
            ys.append(base + y); ts.append(t); t += dt
        for y in _half_cos(rom, per, descending=True):     # 0 -> rom  (lower)
            ys.append(base + y); ts.append(t); t += dt
    ys.append(base + rom); ts.append(t)
    return np.asarray(ts), np.asarray(ys)


def flat_path(seconds: float = 5.0, fps: float = 30.0, noise: float = 0.01,
              seed: int = 0):
    """A stationary bar with sub-rep jitter — must produce zero reps."""
    rng = np.random.default_rng(seed)
    ts = np.arange(0.0, seconds, 1.0 / fps)
    ys = rng.normal(0.0, noise, size=len(ts))
    return ts, ys


@pytest.fixture
def make_squat():
    return squat_path


@pytest.fixture
def make_deadlift():
    return deadlift_path


@pytest.fixture
def make_flat():
    return flat_path
