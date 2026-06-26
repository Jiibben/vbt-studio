"""
Streaming rep counter with displacement hysteresis.

The smoothed vertical position is segmented into alternating down/up phases. A
phase only flips after the bar reverses by `min_displacement_m`, so jitter never
counts. Concentric is always the upward phase, but rep *structure* depends on
the movement direction:

  - down-first (squat, bench): rep = eccentric (down) then concentric (up);
    it completes at the end of the up phase.
  - up-first (deadlift, row): rep = concentric (up) then eccentric (down); the
    count ticks at lockout (end of up), and the rep's eccentric is the
    following down phase.

The style is auto-detected from the first movement direction, or forced. No
lookahead — usable live during the video pass; call `finalize()` after the last
sample to close a set that ends at lockout.
"""
from __future__ import annotations

PHASE_IDLE, PHASE_DOWN, PHASE_UP = "idle", "down", "up"
REP_STYLES = ("auto", "down-first", "up-first")


class RepCounter:
    def __init__(self, *, min_displacement_m: float = 0.15,
                 min_phase_s: float = 0.25, style: str = "auto",
                 dwell_tolerance_m: float = 0.02):
        if style not in REP_STYLES:
            raise ValueError(f"Unknown rep style: {style}")
        self.min_disp = min_displacement_m
        self.min_phase_s = min_phase_s
        # While the bar sits within this band of an extreme (a pause at the
        # bottom, on the floor, or before the lift), the extreme's timestamp
        # keeps advancing, so the dwell attaches to the eccentric/idle side of
        # the boundary — never to the concentric phase.
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
        """Feed one (t, y) sample, y in metres +down. Returns the completed rep
        dict when one closes, else None."""
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
                completed = self._enter(PHASE_UP, self._max_t, self._max_y, t, y)
        elif self.state == PHASE_UP:
            # Strict minimum (no dwell tracking): a lockout pause must NOT
            # extend the concentric phase.
            if y < self._min_y:
                self._min_y, self._min_t = y, t
            if y - self._min_y >= self.min_disp:
                completed = self._enter(PHASE_DOWN, self._min_t, self._min_y, t, y)
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
            # Up-first: the eccentric is the *next* down phase — keep the rep
            # open until it closes.
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
