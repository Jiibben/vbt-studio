"""Characterization + edge-case tests for vbt_core.tracking.association:
to_measurement, iou_matrix, associate, CHI2_GATE.
"""
from __future__ import annotations

import numpy as np
import pytest

from vbt_core.tracking.association import (
    CHI2_GATE,
    associate,
    iou_matrix,
    to_measurement,
)
from vbt_core.tracking.track import Track


# --------------------------------------------------------------------------- #
# Small Track-like doubles for associate()
# --------------------------------------------------------------------------- #
class _FakeKF:
    def __init__(self, box, gating=0.0):
        self._box = box
        self._gating = gating

    @property
    def box(self):
        return self._box

    def gating_distance(self, z):
        return self._gating


class _FakeTrack:
    """Minimal stand-in exposing the .kf.box / .kf.gating_distance interface."""

    def __init__(self, box, gating=0.0):
        self.kf = _FakeKF(box, gating)


# --------------------------------------------------------------------------- #
# to_measurement
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cx,cy,w,h,a", [
    (5.0, 6.0, 20.0, 10.0, 2.0),
    (0.0, 0.0, 10.0, 40.0, 0.25),
    (100.0, 200.0, 50.0, 50.0, 1.0),
])
def test_to_measurement_aspect(cx, cy, w, h, a):
    out = to_measurement(np.array([cx, cy, w, h], dtype=float))
    assert out.tolist() == [cx, cy, a, h]
    assert out.dtype == float


@pytest.mark.parametrize("h", [0.0, 0.5, -2.0])
def test_to_measurement_clamps_height(h):
    out = to_measurement(np.array([5.0, 6.0, 20.0, h], dtype=float))
    assert out[3] == 1.0
    # aspect computed against the clamped height
    assert out[2] == pytest.approx(20.0 / 1.0)


# --------------------------------------------------------------------------- #
# iou_matrix
# --------------------------------------------------------------------------- #
def test_iou_identical_boxes_is_one():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[100.0, 100.0, 10.0, 10.0]])
    assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)


def test_iou_partial_overlap_handcomputed():
    # a: center (0,0) 10x10 -> [-5,-5]..[5,5]
    # c: center (5,0) 10x10 -> [0,-5]..[10,5]
    # intersection x[0,5] y[-5,5] = 5*10 = 50, union = 100+100-50 = 150 -> 1/3
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    c = np.array([[5.0, 0.0, 10.0, 10.0]])
    assert iou_matrix(a, c)[0, 0] == pytest.approx(1.0 / 3.0)


def test_iou_contained_box():
    # small box fully inside big box: IoU = area_small / area_big
    big = np.array([[0.0, 0.0, 20.0, 20.0]])      # area 400
    small = np.array([[0.0, 0.0, 10.0, 10.0]])    # area 100, fully inside
    assert iou_matrix(big, small)[0, 0] == pytest.approx(100.0 / 400.0)


def test_iou_matrix_shape_MxN():
    a = np.array([[0.0, 0.0, 10.0, 10.0],
                  [100.0, 100.0, 10.0, 10.0],
                  [0.0, 0.0, 20.0, 20.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0],
                  [100.0, 100.0, 10.0, 10.0]])
    m = iou_matrix(a, b)
    assert m.shape == (3, 2)
    assert m[0, 0] == pytest.approx(1.0)   # a0 == b0
    assert m[1, 1] == pytest.approx(1.0)   # a1 == b1
    assert m[0, 1] == pytest.approx(0.0)   # disjoint


def test_iou_zero_area_no_div_by_zero():
    za = np.array([[0.0, 0.0, 0.0, 0.0]])
    with np.errstate(invalid="ignore", divide="ignore"):
        m = iou_matrix(za, za)
    assert np.all(np.isfinite(m))
    assert m[0, 0] == 0.0


def test_iou_is_symmetric_for_same_boxes():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    c = np.array([[5.0, 0.0, 10.0, 10.0]])
    assert iou_matrix(a, c)[0, 0] == pytest.approx(iou_matrix(c, a)[0, 0])


def test_iou_bounds_random():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 100, size=(4, 2))
    wh = rng.uniform(5, 30, size=(4, 2))
    boxes_a = np.hstack([a, wh])
    b = rng.uniform(0, 100, size=(3, 2))
    whb = rng.uniform(5, 30, size=(3, 2))
    boxes_b = np.hstack([b, whb])
    out = iou_matrix(boxes_a, boxes_b)
    assert out.min() >= 0.0 - 1e-12
    assert out.max() <= 1.0 + 1e-12


# --------------------------------------------------------------------------- #
# associate — empties
# --------------------------------------------------------------------------- #
def test_associate_empty_tracks():
    dets = np.array([[100.0, 100.0, 20.0, 20.0, 0.9]])
    matches, um_t, um_d = associate([], dets, 0.3)
    assert matches == []
    assert um_t == []
    assert um_d == [0]


def test_associate_empty_dets():
    t = _FakeTrack((100.0, 100.0, 20.0, 20.0))
    matches, um_t, um_d = associate([t], np.zeros((0, 5)), 0.3)
    assert matches == []
    assert um_t == [0]
    assert um_d == []


def test_associate_both_empty():
    assert associate([], np.zeros((0, 5)), 0.3) == ([], [], [])


# --------------------------------------------------------------------------- #
# associate — matching
# --------------------------------------------------------------------------- #
def test_associate_single_match():
    t = _FakeTrack((100.0, 100.0, 20.0, 20.0))
    dets = np.array([[100.0, 100.0, 20.0, 20.0, 0.9]])
    matches, um_t, um_d = associate([t], dets, 0.3)
    assert matches == [(0, 0)]
    assert um_t == [] and um_d == []


def test_associate_real_track_matches_itself():
    rt = Track(1, to_measurement(np.array([100.0, 100.0, 20.0, 20.0])), 0.9)
    dets = np.array([[100.0, 100.0, 20.0, 20.0, 0.9]])
    matches, um_t, um_d = associate([rt], dets, 0.3)
    assert matches == [(0, 0)]


def test_associate_iou_gate_leaves_far_det_unmatched():
    """Only candidate, but disjoint -> IoU below threshold -> no match."""
    t = _FakeTrack((100.0, 100.0, 20.0, 20.0))
    dets = np.array([[500.0, 500.0, 20.0, 20.0, 0.9]])
    matches, um_t, um_d = associate([t], dets, 0.3)
    assert matches == []
    assert um_t == [0]
    assert um_d == [0]


def test_associate_mahalanobis_gate_blocks_overlapping_det():
    """IoU passes (identical boxes) but the Kalman gate fails -> no match."""
    t = _FakeTrack((100.0, 100.0, 20.0, 20.0), gating=CHI2_GATE + 1.0)
    dets = np.array([[100.0, 100.0, 20.0, 20.0, 0.9]])
    matches, um_t, um_d = associate([t], dets, 0.3)
    assert matches == []
    assert um_t == [0] and um_d == [0]


def test_associate_mahalanobis_boundary_inclusive():
    """gate is `> CHI2_GATE`, so exactly CHI2_GATE still matches."""
    t = _FakeTrack((100.0, 100.0, 20.0, 20.0), gating=CHI2_GATE)
    dets = np.array([[100.0, 100.0, 20.0, 20.0, 0.9]])
    matches, _, _ = associate([t], dets, 0.3)
    assert matches == [(0, 0)]


def test_associate_iou_threshold_boundary_exclusive():
    """match requires iou >= iou_thresh; iou exactly == thresh matches,
    strictly below does not."""
    # construct overlap of exactly 1/3 (see hand-computed test above)
    t = _FakeTrack((0.0, 0.0, 10.0, 10.0))
    dets = np.array([[5.0, 0.0, 10.0, 10.0, 0.9]])
    # threshold below 1/3 -> match
    assert associate([t], dets, 0.3)[0] == [(0, 0)]
    # threshold above 1/3 -> no match
    assert associate([t], dets, 0.4)[0] == []
    # threshold exactly 1/3 -> match (>= comparison; cost gate uses `<`)
    assert associate([t], dets, 1.0 / 3.0)[0] == [(0, 0)]


def test_associate_picks_best_assignment_of_two():
    t0 = _FakeTrack((100.0, 100.0, 20.0, 20.0))
    t1 = _FakeTrack((300.0, 300.0, 20.0, 20.0))
    dets = np.array([
        [300.0, 300.0, 20.0, 20.0, 0.9],   # belongs to t1
        [100.0, 100.0, 20.0, 20.0, 0.9],   # belongs to t0
    ])
    matches, um_t, um_d = associate([t0, t1], dets, 0.3)
    assert set(matches) == {(0, 1), (1, 0)}
    assert um_t == [] and um_d == []


def test_associate_more_tracks_than_dets_leaves_extra_track():
    t0 = _FakeTrack((100.0, 100.0, 20.0, 20.0))
    t1 = _FakeTrack((300.0, 300.0, 20.0, 20.0))
    dets = np.array([[100.0, 100.0, 20.0, 20.0, 0.9]])
    matches, um_t, um_d = associate([t0, t1], dets, 0.3)
    assert matches == [(0, 0)]
    assert um_t == [1]
    assert um_d == []
