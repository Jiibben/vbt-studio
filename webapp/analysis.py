#!/usr/bin/env python3
"""
analysis.py — calibration, rep counting and velocity metrics for a bar path.

Calibration: the largest plate on the bar has a known diameter (configured in
the webapp), and its bounding box is the same thing in pixels, so
m/px = diameter_m / median(box size px) over confident frames.

Rep counting: the smoothed vertical position is segmented into alternating
down/up phases with a displacement hysteresis (a phase flips only after the
bar has reversed by `min_displacement_m`), so jitter never counts. Concentric
is always the upward phase, but rep *structure* depends on the movement:
  - down-first (squat, bench): rep = eccentric (down) then concentric (up);
    it completes at the end of the up phase.
  - up-first (deadlift, row): rep = concentric (up) then eccentric (down);
    the count ticks at lockout (end of up), and the rep's eccentric is the
    following down phase.
The style is auto-detected from the first movement direction, or forced.

Velocity: computed frame-by-frame (Savitzky–Golay derivative of the calibrated
position), not from rep endpoints — per-rep mean concentric velocity (MCV) is
the average of the instantaneous velocity over the up phase, and peak velocity
is its maximum, matching standard VBT practice.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.signal import savgol_filter

PHASE_IDLE, PHASE_DOWN, PHASE_UP = "idle", "down", "up"


def calibrate_m_per_px(sizes_px, confs, coasting, plate_diameter_m,
                       conf_thresh: float = 0.5, min_samples: int = 5):
    """metres-per-pixel from the median box size of confident, non-coasted
    frames; falls back to all detected frames; None if nothing usable."""
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


class RepCounter:
    """Streaming direction-change rep counter with displacement hysteresis.

    Feed (t, y) samples with y in metres, +down (image convention). No
    lookahead — usable live during the video pass. Call finalize() after the
    last sample to close a set that ends at lockout.
    """

    def __init__(self, *, min_displacement_m: float = 0.15,
                 min_phase_s: float = 0.25, style: str = "auto",
                 dwell_tolerance_m: float = 0.02):
        if style not in ("auto", "down-first", "up-first"):
            raise ValueError(f"Unknown rep style: {style}")
        self.min_disp = min_displacement_m
        self.min_phase_s = min_phase_s
        # While the bar sits within this band of an extreme (a pause at the
        # bottom, on the floor, or before the lift), the extreme's timestamp
        # keeps advancing, so the dwell attaches to the eccentric/idle side
        # of the boundary — never to the concentric phase.
        self.dwell_tol = dwell_tolerance_m
        self.style = style
        # Resolved on the first movement when style == "auto".
        self.detected_style = style if style != "auto" else None
        self.state = PHASE_IDLE
        self.rep_count = 0
        self.phases: list[dict] = []        # closed phases
        # Completed reps: {rep, start_t, end_t, conc_start, conc_end,
        #                  ecc_start, ecc_end} (ecc_* None if no eccentric).
        self.reps: list[dict] = []
        self.transitions: list[tuple[float, str]] = []  # (t, new_state)
        self._open_rep: dict | None = None  # up-first: awaiting its eccentric
        self._started = False
        self._phase_start_t = self._phase_start_y = 0.0
        self._min_y = self._min_t = 0.0
        self._max_y = self._max_t = 0.0
        self._last_t = self._last_y = 0.0

    @property
    def style_in_effect(self) -> str:
        return self.detected_style or "down-first"

    def update(self, t: float, y: float):
        """Returns the completed rep dict when one closes, else None."""
        if not self._started:
            self._started = True
            self._phase_start_t, self._phase_start_y = t, y
            self._min_y = self._max_y = y
            self._min_t = self._max_t = t
            self._last_t, self._last_y = t, y
            return None

        completed = None
        if self.state == PHASE_IDLE:
            if y < self._min_y:
                self._min_y, self._min_t = y, t
            elif y < self._min_y + self.dwell_tol:
                self._min_t = t          # still parked at the top — dwell
            if y > self._max_y:
                self._max_y, self._max_t = y, t
            elif y > self._max_y - self.dwell_tol:
                self._max_t = t          # still parked at the bottom — dwell
            if y - self._min_y >= self.min_disp:
                self._enter(PHASE_DOWN, self._min_t, self._min_y, t, y)
            elif self._max_y - y >= self.min_disp:
                self._enter(PHASE_UP, self._max_t, self._max_y, t, y)
        elif self.state == PHASE_DOWN:
            if y > self._max_y:
                self._max_y, self._max_t = y, t
            elif y > self._max_y - self.dwell_tol:
                self._max_t = t          # bottom/floor pause stays eccentric
            if self._max_y - y >= self.min_disp:
                completed = self._enter(PHASE_UP, self._max_t, self._max_y,
                                        t, y)
        elif self.state == PHASE_UP:
            # Strict minimum (no dwell tracking): a lockout pause must NOT
            # extend the concentric phase.
            if y < self._min_y:
                self._min_y, self._min_t = y, t
            if y - self._min_y >= self.min_disp:
                completed = self._enter(PHASE_DOWN, self._min_t, self._min_y,
                                        t, y)
        self._last_t, self._last_y = t, y
        return completed

    def finalize(self):
        """Close a dangling final phase: a down-first set usually ends at
        lockout (mid-'up'), an up-first set after the last lowering (or at
        lockout with no lowering at all). Returns a final rep or None."""
        completed = None
        if self.state == PHASE_UP:
            end_y = min(self._min_y, self._last_y)
            end_t = self._min_t if self._min_y <= self._last_y else self._last_t
            if self._phase_start_y - end_y >= self.min_disp:
                completed = self._enter(PHASE_IDLE, end_t, end_y,
                                        self._last_t, self._last_y)
        elif self.state == PHASE_DOWN:
            completed = self._enter(PHASE_IDLE, self._max_t, self._max_y,
                                    self._last_t, self._last_y)
        if self._open_rep is not None:
            # Up-first set ended at lockout — close the rep without eccentric.
            rep = self._open_rep
            self._open_rep = None
            self.reps.append(rep)
            completed = completed or rep
        return completed

    def _enter(self, new_state, boundary_t, boundary_y, t, y):
        if self.state == PHASE_IDLE and new_state != PHASE_IDLE \
                and self.detected_style is None:
            self.detected_style = ("down-first" if new_state == PHASE_DOWN
                                   else "up-first")
        completed = None
        if self.state != PHASE_IDLE:
            phase = {
                "phase": self.state,
                "start_t": self._phase_start_t, "start_y": self._phase_start_y,
                "end_t": boundary_t, "end_y": boundary_y,
            }
            self.phases.append(phase)
            completed = self._on_phase_closed(phase)
        self.transitions.append((boundary_t, new_state))
        self.state = new_state
        self._phase_start_t, self._phase_start_y = boundary_t, boundary_y
        self._min_y = self._max_y = y
        self._min_t = self._max_t = t
        return completed

    def _on_phase_closed(self, phase):
        if phase["phase"] == PHASE_UP:
            if phase["end_t"] - phase["start_t"] < self.min_phase_s:
                return None
            self.rep_count += 1   # both styles count the rep at lockout
            rep = {"rep": self.rep_count,
                   "start_t": phase["start_t"], "end_t": phase["end_t"],
                   "conc_start": phase["start_t"], "conc_end": phase["end_t"],
                   "ecc_start": None, "ecc_end": None}
            phase["in_rep"] = True
            if self.style_in_effect == "down-first":
                # Eccentric was the down phase just before this up.
                if len(self.phases) >= 2:
                    prev = self.phases[-2]
                    if prev["phase"] == PHASE_DOWN and not prev.get("in_rep"):
                        prev["in_rep"] = True
                        rep["start_t"] = prev["start_t"]
                        rep["ecc_start"] = prev["start_t"]
                        rep["ecc_end"] = prev["end_t"]
                self.reps.append(rep)
                return rep
            # Up-first: the eccentric is the *next* down phase — keep the
            # rep open until it closes.
            self._open_rep = rep
            return None
        # A down phase closed: in up-first style it is the open rep's eccentric.
        if self.style_in_effect == "up-first" and self._open_rep is not None:
            phase["in_rep"] = True
            rep = self._open_rep
            self._open_rep = None
            rep["ecc_start"] = phase["start_t"]
            rep["ecc_end"] = phase["end_t"]
            rep["end_t"] = phase["end_t"]
            self.reps.append(rep)
            return rep
        return None

    def phase_at(self, t: float) -> str:
        """Phase label for a timestamp, from the recorded transitions."""
        current = PHASE_IDLE
        for tt, state in self.transitions:
            if tt > t:
                break
            current = state
        return current


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


def _active_span(mask, v, times, thresh: float = 0.05,
                 max_gap_s: float = 0.6):
    """The longest contiguous movement inside `mask`: samples with
    |v| > thresh, where sub-threshold gaps shorter than `max_gap_s` are
    bridged (a mid-rep sticking point stays in the rep) but long stationary
    dwell (a pause at lockout, the bar resting on the floor) is cut off."""
    active = np.where(mask & (np.abs(v) > thresh))[0]
    out = np.zeros(len(mask), dtype=bool)
    if len(active) == 0:
        return out
    runs = []
    start = prev = active[0]
    for i in active[1:]:
        if times[i] - times[prev] > max_gap_s:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    s, e = max(runs, key=lambda r: times[r[1]] - times[r[0]])
    out[s:e + 1] = True
    return out


def analyze_trajectory(times, cy_px, sizes_px, confs, coasting, *,
                       plate_diameter_mm: float, fps: float,
                       min_displacement_m: float = 0.15,
                       rep_style: str = "auto") -> AnalysisResult:
    """Post-pass analysis of the primary track. All inputs are per-frame
    arrays for frames where the primary track existed (smoothed values)."""
    times = np.asarray(times, dtype=float)
    cy_px = np.asarray(cy_px, dtype=float)
    if len(times) < 5:
        return AnalysisResult(calibrated=False)

    m_per_px, size_px = calibrate_m_per_px(
        sizes_px, confs, coasting, plate_diameter_mm / 1000.0)
    if m_per_px is None:
        return AnalysisResult(calibrated=False)

    y_m = cy_px * m_per_px
    dt = 1.0 / (fps if fps > 0 else 30.0)

    # Frame-by-frame velocity, +up. Savitzky–Golay differentiates while
    # smoothing; window ≈ a quarter second.
    win = max(5, int(round(0.25 * (fps if fps > 0 else 30.0))) | 1)
    if len(y_m) >= win:
        v = -savgol_filter(y_m, win, polyorder=2, deriv=1, delta=dt)
    else:
        v = -np.gradient(y_m, times)

    rc = RepCounter(min_displacement_m=min_displacement_m, style=rep_style)
    for t, y in zip(times, y_m):
        rc.update(float(t), float(y))
    rc.finalize()

    phase_arr = [rc.phase_at(float(t)) for t in times]
    rep_arr = np.zeros(len(times), dtype=int)
    reps_out: list[dict] = []
    for rep in rc.reps:
        conc = _active_span((times >= rep["conc_start"])
                            & (times <= rep["conc_end"]), v, times)
        if rep["ecc_start"] is not None:
            ecc = _active_span((times >= rep["ecc_start"])
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


if __name__ == "__main__":
    # Self-test: 5 squat-style reps (down first) + jitter, then deadlift-style.
    fps = 30.0
    t = np.arange(0, 20, 1 / fps)
    y = 1.0 + 0.3 * (1 - np.cos(2 * np.pi * t / 4)) / 2 * 2   # 0.3 m ROM, 4 s/rep
    y += np.random.default_rng(1).normal(0, 0.002, len(t))    # 2 mm noise
    sizes = np.full(len(t), 180.0)
    confs = np.full(len(t), 0.9)
    coast = np.zeros(len(t), dtype=bool)
    res = analyze_trajectory(t, y / 0.0025, sizes, confs, coast,
                             plate_diameter_mm=450.0, fps=fps)
    print(f"squat-style: style={res.rep_style} reps={res.rep_count} "
          f"summary={res.summary}")
    assert res.calibrated and res.rep_count == 5, res.rep_count
    assert res.rep_style == "down-first", res.rep_style
    for r in res.reps:
        assert r["mean_eccentric_v"] > 0, r   # every squat rep has its descent

    y2 = 1.0 - 0.5 * np.clip(np.sin(2 * np.pi * t / 5), 0, None)  # up-first pulls
    res2 = analyze_trajectory(t, y2 / 0.0025, sizes, confs, coast,
                              plate_diameter_mm=450.0, fps=fps)
    print(f"deadlift-style: style={res2.rep_style} reps={res2.rep_count}")
    assert res2.rep_count == 4, res2.rep_count
    assert res2.rep_style == "up-first", res2.rep_style
    for r in res2.reps:
        # Each pull owns its own lowering: concentric first, eccentric after,
        # within the same rep window.
        assert r["mean_eccentric_v"] > 0, r
        assert abs(r["start_t"] - (r["rep"] - 1) * 5) < 0.3, r
        assert r["duration_s"] < 4.0, r

    flat = np.full(len(t), 1.0) + np.random.default_rng(2).normal(0, 0.01, len(t))
    res3 = analyze_trajectory(t, flat / 0.0025, sizes, confs, coast,
                              plate_diameter_mm=450.0, fps=fps)
    assert res3.rep_count == 0, res3.rep_count
    print("analysis self-test OK")
