"""
Parametrized iou_matrix tests.

Bootstrapped with the `aider-test-gen` skill (aider + local Ollama
qwen2.5-coder:14b), then human-verified and corrected: the model's hand-computed
IoU values were wrong (it treated containment as IoU 1.0 and the half-overlaps
as 0.25), and its "empty input" cases passed 1-D / nested arrays that iou_matrix
is not contracted to accept (emptiness is guarded upstream in `associate`). The
geometry below is recomputed by hand.
"""
import numpy as np
import pytest

from vbt_core.tracking.association import iou_matrix


def box(cx, cy, w, h):
    return np.array([cx, cy, w, h], dtype=float)


# Identical boxes -> IoU 1.0.
identical_cases = [
    (box(0.5, 0.5, 1, 1), box(0.5, 0.5, 1, 1)),
    (box(2, 3, 4, 5), box(2, 3, 4, 5)),
]

# Completely disjoint boxes -> IoU 0.0.
disjoint_cases = [
    (box(0, 0, 1, 1), box(2, 2, 1, 1)),
    (box(1, 1, 1, 1), box(3, 3, 1, 1)),
]

# Half-step diagonal overlap of two equal 2x2 boxes: intersection 1x1 = 1,
# union 4 + 4 - 1 = 7  ->  IoU = 1/7.
partial_overlap_cases = [
    (box(0.5, 0.5, 2, 2), box(1.5, 1.5, 2, 2), 1 / 7),
    (box(0, 0, 2, 2), box(1, 1, 2, 2), 1 / 7),
]

# One box fully inside another: intersection = inner area; union = outer area.
# (0.5,0.5,2,2) area 4 contains (0.75,0.75,1,1) area 1 -> 1/4.
# (1,1,3,3) area 9 contains (1.5,1.5,1,1) area 1 -> 1/9.
fully_contained_cases = [
    (box(0.5, 0.5, 2, 2), box(0.75, 0.75, 1, 1), 1 / 4),
    (box(1, 1, 3, 3), box(1.5, 1.5, 1, 1), 1 / 9),
]


@pytest.mark.parametrize("a, b", identical_cases)
def test_iou_identical(a, b):
    assert iou_matrix(np.array([a]), np.array([b]))[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize("a, b", disjoint_cases)
def test_iou_disjoint(a, b):
    assert iou_matrix(np.array([a]), np.array([b]))[0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize("a, b, expected", partial_overlap_cases)
def test_iou_partial_overlap(a, b, expected):
    assert iou_matrix(np.array([a]), np.array([b]))[0, 0] == pytest.approx(expected)


@pytest.mark.parametrize("a, b, expected", fully_contained_cases)
def test_iou_fully_contained(a, b, expected):
    assert iou_matrix(np.array([a]), np.array([b]))[0, 0] == pytest.approx(expected)


# IoU is symmetric: IoU(a, b) == IoU(b, a).
@pytest.mark.parametrize("a, b, _expected",
                         partial_overlap_cases + fully_contained_cases)
def test_iou_symmetric(a, b, _expected):
    ab = iou_matrix(np.array([a]), np.array([b]))[0, 0]
    ba = iou_matrix(np.array([b]), np.array([a]))[0, 0]
    assert ab == pytest.approx(ba)


# Non-square MxN: one box vs several, and several vs one.
single_vs_many_cases = [
    (box(0.5, 0.5, 1, 1), [box(0.5, 0.5, 1, 1), box(2, 2, 1, 1)]),
    (box(0, 0, 1, 1),
     [box(0, 0, 1, 1), box(1, 1, 1, 1), box(2, 2, 1, 1)]),
]


@pytest.mark.parametrize("a, many", single_vs_many_cases)
def test_iou_single_vs_many(a, many):
    result = iou_matrix(np.array([a]), np.array(many))
    assert result.shape == (1, len(many))
    assert result[0, 0] == pytest.approx(1.0)         # first is identical
    assert np.all((result >= 0.0) & (result <= 1.0))   # always a valid IoU


def test_iou_rectangular_shape():
    a = np.array([box(0, 0, 1, 1), box(1, 1, 1, 1)])
    b = np.array([box(2, 2, 1, 1)])
    assert iou_matrix(a, b).shape == (2, 1)
