"""
TrackingPipeline — wires detection → tracking → analysis → rendering into one
streaming pass over a clip.

In a single pass it produces:
  1. <stem>_tracked.mp4  — BOXED preview (smoothed track boxes + id + conf;
                           primary green, others grey, coasting orange)
  2. <stem>_barpath.mp4  — BAR-PATH preview: sticky trail through the
                           Kalman-smoothed plate centre, phase-coloured, plus a
                           live HUD (rep count, velocity m/s, phase)
  3. <stem>_track.csv    — per-frame smoothed trajectory of confirmed tracks
  4. <stem>_metrics.json — calibration, rep count, per-rep velocity metrics and
                           the full velocity series (analysis)

cx/cy/w/h in the CSV are the Kalman-smoothed states, not raw detector boxes —
downstream consumers want the cleaned trajectory.
"""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from ..analysis import AnalysisResult, RepCounter, analyze_trajectory
from ..detection import Detector, YOLODetector
from ..rendering import (COLOR_DOWN, COLOR_IDLE, COLOR_UP, FFmpegWriter,
                         draw_box, draw_hud, draw_trail)
from ..tracking import BarTracker

_CSV_FIELDS = ["frame", "time_s", "track_id", "cx", "cy", "w", "h", "conf",
               "coasting", "primary", "vy_mps"]


@dataclass
class TrackResult:
    boxed_video: Path
    barpath_video: Path
    csv_path: Path
    metrics_path: Path
    fps: float
    n_frames: int
    n_with_box: int
    track_ids: list = field(default_factory=list)
    analysis: AnalysisResult | None = None

    @property
    def coverage(self) -> float:
        return (self.n_with_box / self.n_frames * 100) if self.n_frames else 0.0


def _read_fps(source: Path) -> float:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def _read_frame_count(source: Path) -> int:
    """Best-effort total frame count for progress reporting (0 if unknown)."""
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return max(n, 0)


class TrackingPipeline:
    """Run a `Detector` + `BarTracker` + analysis over a clip and render outputs.

    `conf` is the high-confidence threshold of the two-stage association.
    `trail_style` is "green" (single colour) or "phase" (green = up, orange =
    down, grey = idle, recoloured retroactively as turnarounds are confirmed)."""

    def __init__(self, detector: Detector, *, conf: float = 0.25,
                 plate_diameter_mm: float = 450.0, plate_kg: str | None = None,
                 trail_style: str = "green", rep_style: str = "auto",
                 trail_thickness: int = 3):
        self.detector = detector
        self.conf = conf
        self.plate_diameter_mm = plate_diameter_mm
        self.plate_kg = plate_kg
        self.trail_style = trail_style
        self.rep_style = rep_style
        self.trail_thickness = trail_thickness

    def run(self, source, out_boxed, out_barpath, out_csv,
            out_metrics, *, progress_cb=None) -> TrackResult:
        """``progress_cb(fraction)``, if given, is called once per frame with the
        fraction of the clip processed (0..1) — used by batch runs for live
        per-clip progress. Optional and cheap; omit it for the single-clip path."""
        source = Path(source)
        out_boxed, out_barpath = Path(out_boxed), Path(out_barpath)
        out_csv, out_metrics = Path(out_csv), Path(out_metrics)
        for p in (out_boxed, out_barpath, out_csv, out_metrics):
            p.parent.mkdir(parents=True, exist_ok=True)

        fps = _read_fps(source)
        total_frames = _read_frame_count(source) if progress_cb is not None else 0
        tracker = BarTracker(
            high_thresh=max(self.conf, 0.15),
            low_thresh=0.10,
            max_age=int(round(fps)),       # coast up to ~1 s through occlusion
        )
        plate_diameter_m = self.plate_diameter_mm / 1000.0

        boxed_writer = barpath_writer = None
        trail: list[tuple[int, int, float]] = []   # (x, y, time_s)
        rows: list[dict] = []
        n_frames = n_with_box = 0

        # Primary-track buffers for the post-pass analysis.
        p_times: list[float] = []
        p_cy: list[float] = []
        p_sizes: list[float] = []
        p_confs: list[float] = []
        p_coast: list[bool] = []

        # Live state: running-median calibration + streaming rep counter. The
        # post-pass analysis recomputes both; these only feed the burned-in HUD.
        calib_sizes: list[float] = []
        m_per_px: float | None = None
        rep_counter = RepCounter(style=self.rep_style)

        for frame in self.detector.stream(source):
            frame_idx = frame.index
            t = frame_idx / fps
            boxed = frame.image.copy()
            barpath = frame.image.copy()

            if boxed_writer is None:
                h, w = boxed.shape[:2]
                boxed_writer = FFmpegWriter(out_boxed, fps, w, h)
                barpath_writer = FFmpegWriter(out_barpath, fps, w, h)

            dets = frame.boxes
            if len(dets) > 0:
                n_with_box += 1

            snaps = tracker.update(dets)
            primary_id = tracker.primary_id()
            primary = next((s for s in snaps if s.track_id == primary_id), None)

            v_live: float | None = None
            if primary is not None:
                if not primary.coasting and primary.conf >= 0.5:
                    calib_sizes.append((primary.w + primary.h) / 2)
                if len(calib_sizes) >= 5:
                    m_per_px = plate_diameter_m / statistics.median(calib_sizes)

                if m_per_px is not None:
                    rep_counter.update(t, primary.cy * m_per_px)
                    # Kalman vy is px/frame, +down; flip so positive = upward.
                    v_live = -primary.vy * fps * m_per_px

                p_times.append(t)
                p_cy.append(primary.cy)
                p_sizes.append((primary.w + primary.h) / 2)
                p_confs.append(primary.conf)
                p_coast.append(primary.coasting)
                trail.append((int(round(primary.cx)), int(round(primary.cy)), t))

            for s in snaps:
                is_primary = s.track_id == primary_id
                color = (COLOR_DOWN if s.coasting else COLOR_UP) if is_primary \
                    else COLOR_IDLE
                draw_box(boxed, s, color, is_primary)
                row_v = ""
                if is_primary and m_per_px is not None:
                    row_v = round(-s.vy * fps * m_per_px, 4)
                rows.append({
                    "frame": frame_idx,
                    "time_s": round(t, 4),
                    "track_id": s.track_id,
                    "cx": round(s.cx, 2), "cy": round(s.cy, 2),
                    "w": round(s.w, 2), "h": round(s.h, 2),
                    "conf": round(s.conf, 4),
                    "coasting": int(s.coasting),
                    "primary": int(is_primary),
                    "vy_mps": row_v,
                })

            draw_trail(barpath, trail, rep_counter.phase_at, self.trail_style,
                       self.trail_thickness)
            draw_hud(barpath, rep_counter.live_rep_count, v_live,
                     rep_counter.state)

            boxed_writer.write(boxed)
            barpath_writer.write(barpath)
            n_frames += 1
            if progress_cb is not None and total_frames:
                progress_cb(min(n_frames / total_frames, 1.0))

        if boxed_writer is not None:
            boxed_writer.close()
            barpath_writer.close()

        with open(out_csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            wr.writeheader()
            wr.writerows(rows)

        # Canonical metrics from the full trajectory (the live HUD values were
        # incremental approximations of these).
        analysis = analyze_trajectory(
            p_times, p_cy, p_sizes, p_confs, p_coast,
            plate_diameter_mm=self.plate_diameter_mm, fps=fps,
            rep_style=self.rep_style,
        )
        track_ids = sorted({row["track_id"] for row in rows})
        metrics = {
            "source": source.name,
            "fps": round(fps, 3),
            "n_frames": n_frames,
            "coverage_pct": round((n_with_box / n_frames * 100)
                                  if n_frames else 0, 1),
            "primary_track_id": tracker.primary_id(),
            "plate_kg": self.plate_kg,
            **analysis.to_dict(),
        }
        with open(out_metrics, "w") as f:
            json.dump(metrics, f)

        return TrackResult(
            boxed_video=out_boxed,
            barpath_video=out_barpath,
            csv_path=out_csv,
            metrics_path=out_metrics,
            fps=fps,
            n_frames=n_frames,
            n_with_box=n_with_box,
            track_ids=track_ids,
            analysis=analysis,
        )


def run_tracking(*, source, weights, out_boxed, out_barpath, out_csv,
                 out_metrics, conf: float = 0.25,
                 plate_diameter_mm: float = 450.0, plate_kg: str | None = None,
                 trail_style: str = "green", rep_style: str = "auto",
                 trail_thickness: int = 3) -> TrackResult:
    """Convenience wrapper: build a YOLO detector from `weights` and run the
    pipeline once. Kept signature-compatible with the original webapp call."""
    detector = YOLODetector(weights)
    pipeline = TrackingPipeline(
        detector, conf=conf, plate_diameter_mm=plate_diameter_mm,
        plate_kg=plate_kg, trail_style=trail_style, rep_style=rep_style,
        trail_thickness=trail_thickness,
    )
    return pipeline.run(source, out_boxed, out_barpath, out_csv, out_metrics)
