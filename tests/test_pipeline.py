"""Tests for vbt_core.pipeline.runner (TrackingPipeline, run_tracking).

No real video, ffmpeg, or YOLO model is involved:
  * a FAKE Detector yields synthetic FrameDetections of a plate moving in a
    smooth squat path (so the analysis post-pass produces reps),
  * ``runner.FFmpegWriter`` is replaced with a no-op writer that just counts
    frames (no mp4 written),
  * ``runner._read_fps`` is monkeypatched to return 30.0 (no cv2.VideoCapture),
  * for ``run_tracking``, ``runner.YOLODetector`` is replaced with a factory
    returning the fake detector.
"""
from __future__ import annotations

import csv
import json

import pytest

from conftest import squat_path_frames

from vbt_core.detection import Detector
from vbt_core.pipeline.runner import (TrackingPipeline, TrackResult,
                                       _CSV_FIELDS, run_tracking)
import vbt_core.pipeline.runner as runner


class FakeDetector(Detector):
    """Yields a scripted squat path; never decodes a real video."""

    def __init__(self, n_frames: int = 120, cycles: int = 2):
        self.n_frames = n_frames
        self.cycles = cycles
        self.sources = []

    def stream(self, source):
        self.sources.append(source)
        yield from squat_path_frames(self.n_frames, self.cycles)


class FakeFFmpegWriter:
    instances: list = []

    def __init__(self, path, fps, width, height):
        self.path = path
        self.fps = fps
        self.width = width
        self.height = height
        self.frames = 0
        self.closed = False
        FakeFFmpegWriter.instances.append(self)

    def write(self, frame):
        self.frames += 1

    def close(self):
        self.closed = True


@pytest.fixture
def patched_runner(monkeypatch):
    FakeFFmpegWriter.instances = []
    monkeypatch.setattr(runner, "FFmpegWriter", FakeFFmpegWriter)
    monkeypatch.setattr(runner, "_read_fps", lambda source: 30.0)
    return FakeFFmpegWriter


def _out_paths(tmp_path):
    return dict(
        out_boxed=tmp_path / "clip_tracked.mp4",
        out_barpath=tmp_path / "clip_barpath.mp4",
        out_csv=tmp_path / "clip_track.csv",
        out_metrics=tmp_path / "clip_metrics.json",
    )


def test_pipeline_run_produces_results(tmp_path, patched_runner):
    outs = _out_paths(tmp_path)
    pipe = TrackingPipeline(FakeDetector(), plate_kg="20")
    result = pipe.run(tmp_path / "clip.mp4", **outs)

    # --- TrackResult basics ---
    assert isinstance(result, TrackResult)
    assert result.fps == 30.0
    assert result.n_frames == 120
    assert result.n_with_box == 120
    assert result.coverage == 100.0
    assert result.track_ids  # at least one confirmed track
    assert result.analysis is not None

    # --- two video writers, each fed every frame, both closed ---
    assert len(patched_runner.instances) == 2
    for w in patched_runner.instances:
        assert w.frames == 120
        assert w.closed is True

    # --- CSV: documented header + populated rows ---
    assert outs["out_csv"].exists()
    with open(outs["out_csv"], newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        body = list(reader)
    assert header == _CSV_FIELDS
    assert len(body) > 0

    # --- metrics JSON: serialisable + documented keys ---
    assert outs["out_metrics"].exists()
    metrics = json.loads(outs["out_metrics"].read_text())
    for key in ("source", "fps", "n_frames", "coverage_pct",
                "primary_track_id", "plate_kg", "calibrated", "rep_count"):
        assert key in metrics
    assert metrics["plate_kg"] == "20"
    assert metrics["fps"] == 30.0
    assert metrics["n_frames"] == 120
    assert metrics["source"] == "clip.mp4"


def test_pipeline_analysis_is_calibrated_and_counts_reps(tmp_path, patched_runner):
    outs = _out_paths(tmp_path)
    pipe = TrackingPipeline(FakeDetector(n_frames=120, cycles=2),
                            plate_diameter_mm=450.0)
    result = pipe.run(tmp_path / "clip.mp4", **outs)

    analysis = result.analysis
    assert analysis.calibrated is True
    # Two squat cycles -> at least one counted rep.
    assert analysis.rep_count >= 1

    # to_dict must be JSON-serialisable.
    d = analysis.to_dict()
    json.dumps(d)  # raises if any non-serialisable value slips in
    assert d["calibrated"] is True
    assert "series" in d and set(d["series"]) >= {"t", "v", "y_m", "phase", "rep"}


def test_pipeline_csv_rows_match_fields(tmp_path, patched_runner):
    outs = _out_paths(tmp_path)
    TrackingPipeline(FakeDetector()).run(tmp_path / "clip.mp4", **outs)
    with open(outs["out_csv"], newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    sample = rows[0]
    assert set(sample.keys()) == set(_CSV_FIELDS)
    # track_id present in result.track_ids set.
    assert int(sample["track_id"]) >= 1


def test_pipeline_phase_trail_style(tmp_path, patched_runner):
    # "phase" exercises draw_trail's phase_at recolouring branch end-to-end.
    outs = _out_paths(tmp_path)
    pipe = TrackingPipeline(FakeDetector(), trail_style="phase")
    result = pipe.run(tmp_path / "clip.mp4", **outs)
    assert result.n_frames == 120


def test_run_tracking_wires_yolo_factory(tmp_path, patched_runner, monkeypatch):
    captured = {}

    def fake_yolo_factory(weights):
        captured["weights"] = weights
        return FakeDetector()

    monkeypatch.setattr(runner, "YOLODetector", fake_yolo_factory)

    outs = _out_paths(tmp_path)
    result = run_tracking(
        source=tmp_path / "clip.mp4",
        weights=tmp_path / "weights.pt",
        plate_kg="25",
        **outs,
    )

    assert captured["weights"] == tmp_path / "weights.pt"
    assert isinstance(result, TrackResult)
    assert result.n_frames == 120
    assert outs["out_csv"].exists()
    assert outs["out_metrics"].exists()
    metrics = json.loads(outs["out_metrics"].read_text())
    assert metrics["plate_kg"] == "25"


def test_pipeline_coverage_property_zero_frames():
    # Guard the n_frames==0 branch of the coverage property.
    tr = TrackResult(boxed_video=None, barpath_video=None, csv_path=None,
                     metrics_path=None, fps=30.0, n_frames=0, n_with_box=0)
    assert tr.coverage == 0.0
