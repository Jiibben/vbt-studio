"""
Detection ↔ track association: IoU geometry + Hungarian matching with a
Mahalanobis gate.

A pair is only allowed to match if it clears BOTH the IoU threshold and the
Kalman gate (chi-square 0.95, 4 dof) — a physically illogical jump fails the
gate and can never capture the track, which is what fixed the old
"most-confident box wins" path glitches.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# chi-square 0.95 quantile, 4 degrees of freedom — standard association gate.
CHI2_GATE = 9.4877
_BIG_COST = 1e5


def to_measurement(box_cxcywh: np.ndarray) -> np.ndarray:
    """[cx, cy, w, h] detection -> [cx, cy, a, h] Kalman measurement (a = w/h)."""
    cx, cy, w, h = box_cxcywh
    h = max(float(h), 1.0)
    return np.array([cx, cy, w / h, h], dtype=float)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (M,4) and (N,4) boxes in cxcywh format -> (M,N)."""
    ax1 = a[:, 0] - a[:, 2] / 2
    ay1 = a[:, 1] - a[:, 3] / 2
    ax2 = a[:, 0] + a[:, 2] / 2
    ay2 = a[:, 1] + a[:, 3] / 2
    bx1 = b[:, 0] - b[:, 2] / 2
    by1 = b[:, 1] - b[:, 3] / 2
    bx2 = b[:, 0] + b[:, 2] / 2
    by2 = b[:, 1] + b[:, 3] / 2

    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def associate(tracks, dets: np.ndarray, iou_thresh: float):
    """Hungarian matching of `tracks` to detections (N,5) with IoU + Mahalanobis
    gating. Returns (matches, unmatched_track_idx, unmatched_det_idx)."""
    if not tracks or len(dets) == 0:
        return [], list(range(len(tracks))), list(range(len(dets)))

    t_boxes = np.array([t.kf.box for t in tracks])
    iou = iou_matrix(t_boxes, dets[:, :4])
    cost = 1.0 - iou
    for ti, t in enumerate(tracks):
        for di in range(len(dets)):
            if iou[ti, di] < iou_thresh or \
                    t.kf.gating_distance(to_measurement(dets[di, :4])) > CHI2_GATE:
                cost[ti, di] = _BIG_COST

    rows, cols = linear_sum_assignment(cost)
    matches = [(ti, di) for ti, di in zip(rows, cols)
               if cost[ti, di] < _BIG_COST]
    matched_t = {ti for ti, _ in matches}
    matched_d = {di for _, di in matches}
    unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
    unmatched_d = [i for i in range(len(dets)) if i not in matched_d]
    return matches, unmatched_t, unmatched_d
