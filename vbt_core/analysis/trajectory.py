"""
Trajectory analysis — the post-pass that turns the primary track's smoothed
path into calibration, rep segmentation and per-rep velocity metrics.

Inputs are per-frame arrays for the frames where the primary track existed. The
output `AnalysisResult` is JSON-serialisable (`to_dict`) and is what the webapp
and the metrics file consume.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .calibration import calibrate_m_per_px
from .rep_counter import RepCounter
from .velocity import active_span, frame_velocity


@dataclass
class RepMetrics:
    rep: int
    start_t: float
    end_t: float
    duration_s: float
    rom_m: float
    mean_concentric_v: float
    peak_concentric_v: float
    mean_eccentric_v: float
    peak_eccentric_v: float


@dataclass
class AnalysisResult:
    calibrated: bool
    m_per_px: float | None = None
    plate_diameter_mm: float | None = None
    plate_size_px: float | None = None
    rep_style: str | None = None        # "down-first" | "up-first" (resolved)
    rep_count: int = 0
    reps: list = field(default_factory=list)          # list[RepMetrics-dict]
    summary: dict = field(default_factory=dict)
    series: dict = field(default_factory=dict)        # t / v / y_m / phase / rep

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_trajectory(times, cy_px, sizes_px, confs, coasting, *,
                       plate_diameter_mm: float, fps: float,
                       min_displacement_m: float = 0.15,
                       rep_style: str = "auto") -> AnalysisResult:
    """Post-pass analysis of the primary track. All inputs are per-frame arrays
    for frames where the primary track existed (smoothed values)."""
    times = np.asarray(times, dtype=float)
    cy_px = np.asarray(cy_px, dtype=float)
    if len(times) < 5:
        return AnalysisResult(calibrated=False)

    m_per_px, size_px = calibrate_m_per_px(
        sizes_px, confs, coasting, plate_diameter_mm / 1000.0)
    if m_per_px is None:
        return AnalysisResult(calibrated=False)

    y_m = cy_px * m_per_px
    v = frame_velocity(y_m, times, fps)

    rc = RepCounter(min_displacement_m=min_displacement_m, style=rep_style)
    for t, y in zip(times, y_m):
        rc.update(float(t), float(y))
    rc.finalize()

    phase_arr = [rc.phase_at(float(t)) for t in times]
    rep_arr = np.zeros(len(times), dtype=int)
    reps_out: list[dict] = []
    for rep in rc.reps:
        conc = active_span((times >= rep["conc_start"])
                           & (times <= rep["conc_end"]), v, times)
        if rep["ecc_start"] is not None:
            ecc = active_span((times >= rep["ecc_start"])
                              & (times <= rep["ecc_end"]), v, times)
        else:
            ecc = np.zeros(len(times), dtype=bool)
        if not conc.any():
            continue
        span = conc | ecc
        start_t = float(times[span].min())
        end_t = float(times[span].max())
        in_rep = (times >= start_t) & (times <= end_t)
        rep_arr[in_rep] = rep["rep"]
        rom = float(y_m[in_rep].max() - y_m[in_rep].min())
        ecc_v = -v[ecc]    # downward speed as a positive magnitude
        reps_out.append(asdict(RepMetrics(
            rep=rep["rep"],
            start_t=round(start_t, 3),
            end_t=round(end_t, 3),
            duration_s=round(end_t - start_t, 3),
            rom_m=round(rom, 3),
            mean_concentric_v=round(float(v[conc].mean()), 3),
            peak_concentric_v=round(float(v[conc].max()), 3),
            mean_eccentric_v=round(float(ecc_v.mean()), 3) if ecc.any() else 0.0,
            peak_eccentric_v=round(float(ecc_v.max()), 3) if ecc.any() else 0.0,
        )))

    summary = {}
    if reps_out:
        mcvs = [r["mean_concentric_v"] for r in reps_out]
        best = max(mcvs)
        summary = {
            "avg_mcv": round(float(np.mean(mcvs)), 3),
            "best_mcv": round(best, 3),
            "last_mcv": mcvs[-1],
            "velocity_loss_pct": round((1 - mcvs[-1] / best) * 100, 1)
                                 if best > 0 else 0.0,
        }

    return AnalysisResult(
        calibrated=True,
        m_per_px=round(m_per_px, 6),
        plate_diameter_mm=plate_diameter_mm,
        plate_size_px=round(size_px, 1),
        rep_style=rc.detected_style,
        rep_count=len(reps_out),
        reps=reps_out,
        summary=summary,
        series={
            "t": [round(float(t), 4) for t in times],
            "v": [round(float(x), 4) for x in v],
            "y_m": [round(float(x), 4) for x in y_m],
            "phase": phase_arr,
            "rep": rep_arr.tolist(),
        },
    )
