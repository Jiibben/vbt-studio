"""
Pixel→metre calibration.

The largest plate on the bar has a known diameter, and its bounding box is the
same circle in pixels, so m/px = diameter_m / median(box size px) over the
confident, non-coasted frames.
"""
from __future__ import annotations

import numpy as np


def calibrate_m_per_px(sizes_px, confs, coasting, plate_diameter_m,
                       conf_thresh: float = 0.5, min_samples: int = 5):
    """metres-per-pixel from the median box size of confident, non-coasted
    frames; falls back to all detected frames; (None, None) if nothing usable.

    Returns (m_per_px, size_px)."""
    sizes = np.asarray(sizes_px, dtype=float)
    confs = np.asarray(confs, dtype=float)
    coasting = np.asarray(coasting, dtype=bool)
    good = sizes[(confs >= conf_thresh) & ~coasting]
    if len(good) < min_samples:
        good = sizes[~coasting]
    if len(good) == 0:
        return None, None
    size_px = float(np.median(good))
    if size_px <= 0:
        return None, None
    return plate_diameter_m / size_px, size_px
