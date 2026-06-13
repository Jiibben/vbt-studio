#!/usr/bin/env python3
"""
tracking.py  —  shared tracking core for the debug webapp.

Given a clip + trained weights, runs the detector once per frame and feeds the
raw detections through our own tracking pipeline (bartracker.py: Kalman filter
→ Hungarian association → ByteTrack-style low-confidence recovery), emitting
in a single streaming pass:

  1. <stem>_tracked.mp4  — the BOXED preview (smoothed track boxes + id + conf;
                           primary track green, others grey, coasting orange)
  2. <stem>_barpath.mp4  — the BAR-PATH preview: sticky trail through the
                           Kalman-smoothed plate centre, coloured by phase
                           (green = concentric/up, orange = eccentric/down),
                           plus a live HUD (rep count, velocity m/s, phase)
  3. <stem>_track.csv    — per-frame smoothed trajectory of confirmed tracks
  4. <stem>_metrics.json — calibration, rep count, per-rep velocity metrics
                           and the full velocity series (analysis.py)

Both videos are written through ffmpeg as H.264 / yuv420p so they play directly
in a browser <video> element (OpenCV's mp4v output does not).

Note: cx/cy/w/h in the CSV are the *Kalman-smoothed* states, not raw detector
boxes — that is intentional; downstream consumers want the cleaned trajectory.
"""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from analysis import AnalysisResult, RepCounter, analyze_trajectory
from bartracker import BarTracker, TrackSnapshot

# BGR colours.
COLOR_UP = (0, 255, 0)        # concentric trail / primary box
COLOR_DOWN = (0, 165, 255)    # eccentric trail / coasting box
COLOR_IDLE = (160, 160, 160)  # no movement yet / secondary boxes
PHASE_COLORS = {"up": COLOR_UP, "down": COLOR_DOWN, "idle": COLOR_IDLE}

# Detection floor passed to YOLO — low on purpose so the ByteTrack second
# stage has low-confidence candidates to rescue tracks with.
DET_FLOOR = 0.10


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


class _FFmpegWriter:
    """Pipe raw BGR frames into ffmpeg, encode to browser-playable H.264 mp4."""

    def __init__(self, path: Path, fps: float, width: int, height: int):
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps}",
                "-i", "-",
                "-an",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "veryfast", "-movflags", "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()


def _draw_box(img, snap: TrackSnapshot, color, primary: bool):
    x1 = int(round(snap.cx - snap.w / 2))
    y1 = int(round(snap.cy - snap.h / 2))
    x2 = int(round(snap.cx + snap.w / 2))
    y2 = int(round(snap.cy + snap.h / 2))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if primary else 2)
    label = f"id {snap.track_id}" + (
        " ~" if snap.coasting else f" {snap.conf:.2f}")
    cv2.putText(img, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def _draw_hud(img, rep_count: int, v_mps: float | None, phase: str):
    h = img.shape[0]
    scale = h / 720.0
    pad = int(14 * scale)
    line_h = int(34 * scale)
    box_w = int(230 * scale)
    box_h = pad * 2 + line_h * 3
    overlay = img[0:box_h + pad, 0:box_w + pad].copy()
    cv2.rectangle(img, (pad // 2, pad // 2), (box_w, box_h), (20, 22, 28), -1)
    cv2.addWeighted(overlay, 0.25, img[0:box_h + pad, 0:box_w + pad], 0.75, 0,
                    img[0:box_h + pad, 0:box_w + pad])
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.85 * scale
    th = max(2, int(2 * scale))
    x = pad
    v_txt = f"{v_mps:+.2f} m/s" if v_mps is not None else "--.- m/s"
    cv2.putText(img, f"REP {rep_count}", (x, pad + line_h - 8), font,
                fs * 1.15, (255, 255, 255), th, cv2.LINE_AA)
    cv2.putText(img, v_txt, (x, pad + line_h * 2 - 8), font, fs,
                (235, 235, 235), th, cv2.LINE_AA)
    cv2.putText(img, phase.upper(), (x, pad + line_h * 3 - 8), font, fs,
                PHASE_COLORS[phase], th, cv2.LINE_AA)


def run_tracking(
    *,
    source: Path,
    weights: Path,
    out_boxed: Path,
    out_barpath: Path,
    out_csv: Path,
    out_metrics: Path,
    conf: float = 0.25,
    plate_diameter_mm: float = 450.0,
    plate_kg: str | None = None,
    trail_style: str = "green",
    rep_style: str = "auto",
    trail_thickness: int = 3,
) -> TrackResult:
    """Run detection + custom tracking on `source`, writing the two preview
    videos, the trajectory CSV and the metrics JSON. `conf` is the
    high-confidence threshold of the two-stage association. `trail_style` is
    "green" (single colour) or "phase" (green = up, orange = down, grey =
    idle, recoloured retroactively as turnarounds are confirmed)."""
    source, weights = Path(source), Path(weights)
    for p in (out_boxed, out_barpath, out_csv, out_metrics):
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    model = YOLO(str(weights))
    results = model.predict(
        source=str(source),
        stream=True,
        conf=DET_FLOOR,
        verbose=False,
    )

    tracker = BarTracker(
        high_thresh=max(conf, 0.15),
        low_thresh=DET_FLOOR,
        max_age=int(round(fps)),       # coast up to ~1 s through occlusion
    )
    plate_diameter_m = plate_diameter_mm / 1000.0

    boxed_writer = None
    barpath_writer = None
    trail: list[tuple[int, int, float]] = []   # (x, y, time_s)
    rows = []
    n_frames = 0
    n_with_box = 0

    # Primary-track buffers for the post-pass analysis.
    p_times: list[float] = []
    p_cy: list[float] = []
    p_sizes: list[float] = []
    p_confs: list[float] = []
    p_coast: list[bool] = []

    # Live state: running-median calibration + streaming rep counter. The
    # post-pass analysis recomputes both over the full clip; these only feed
    # the burned-in HUD.
    calib_sizes: list[float] = []
    m_per_px: float | None = None
    rep_counter = RepCounter(style=rep_style)

    for frame_idx, r in enumerate(results):
        t = frame_idx / fps
        boxed = r.orig_img.copy()
        barpath = r.orig_img.copy()

        if boxed_writer is None:
            h, w = boxed.shape[:2]
            boxed_writer = _FFmpegWriter(out_boxed, fps, w, h)
            barpath_writer = _FFmpegWriter(out_barpath, fps, w, h)

        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            dets = np.hstack([boxes.xywh.cpu().numpy(),
                              boxes.conf.cpu().numpy().reshape(-1, 1)])
            n_with_box += 1
        else:
            dets = np.empty((0, 5))

        snaps = tracker.update(dets)
        primary_id = tracker.primary_id()
        primary = next((s for s in snaps if s.track_id == primary_id), None)

        v_live: float | None = None
        if primary is not None:
            # Running-median calibration from confident frames.
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
            _draw_box(boxed, s, color, is_primary)
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

        # Sticky bar path: redraw the accumulated trail every frame. Phase
        # colours are looked up at draw time from the rep counter's recorded
        # transitions — a turnaround is only confirmed after the hysteresis
        # distance, but its *time* is the true extreme, so segments recolour
        # retroactively instead of lagging by 15 cm.
        if len(trail) >= 2:
            for (x0, y0, _), (x1, y1, t1) in zip(trail, trail[1:]):
                color = PHASE_COLORS[rep_counter.phase_at(t1)] \
                    if trail_style == "phase" else COLOR_UP
                cv2.line(barpath, (x0, y0), (x1, y1), color,
                         trail_thickness, cv2.LINE_AA)
        if trail:
            x1, y1, t1 = trail[-1]
            color = PHASE_COLORS[rep_counter.phase_at(t1)] \
                if trail_style == "phase" else COLOR_UP
            cv2.circle(barpath, (x1, y1), trail_thickness + 3,
                       color, -1, cv2.LINE_AA)
        _draw_hud(barpath, rep_counter.rep_count, v_live, rep_counter.state)

        boxed_writer.write(boxed)
        barpath_writer.write(barpath)
        n_frames += 1

    if boxed_writer is not None:
        boxed_writer.close()
        barpath_writer.close()

    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["frame", "time_s", "track_id",
                                           "cx", "cy", "w", "h", "conf",
                                           "coasting", "primary", "vy_mps"])
        wr.writeheader()
        wr.writerows(rows)

    # Canonical metrics from the full trajectory (the live HUD values were
    # incremental approximations of these).
    analysis = analyze_trajectory(
        p_times, p_cy, p_sizes, p_confs, p_coast,
        plate_diameter_mm=plate_diameter_mm, fps=fps, rep_style=rep_style,
    )
    track_ids = sorted({row["track_id"] for row in rows})
    metrics = {
        "source": source.name,
        "fps": round(fps, 3),
        "n_frames": n_frames,
        "coverage_pct": round((n_with_box / n_frames * 100) if n_frames else 0, 1),
        "primary_track_id": tracker.primary_id(),
        "plate_kg": plate_kg,
        **analysis.to_dict(),
    }
    with open(out_metrics, "w") as f:
        json.dump(metrics, f)

    return TrackResult(
        boxed_video=Path(out_boxed),
        barpath_video=Path(out_barpath),
        csv_path=Path(out_csv),
        metrics_path=Path(out_metrics),
        fps=fps,
        n_frames=n_frames,
        n_with_box=n_with_box,
        track_ids=track_ids,
        analysis=analysis,
    )
