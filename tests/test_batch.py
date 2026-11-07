"""Tests for vbt_core.pipeline.batch (run_batch, BatchItem).

Like test_pipeline, no real video / ffmpeg / YOLO is touched:
  * ``runner.FFmpegWriter`` -> a no-op writer (reused from test_pipeline),
  * ``runner._read_fps`` -> 30.0,
  * ``batch.YOLODetector`` -> a counting factory returning a fake detector, so
    we can assert the model is constructed exactly once for the whole batch.
"""
from __future__ import annotations

import json

import pytest

from conftest import squat_path_frames
from test_pipeline import FakeDetector, FakeFFmpegWriter

from vbt_core.detection import Detector
from vbt_core.pipeline import BatchItem, run_batch
from vbt_core.pipeline.runner import TrackResult
import vbt_core.pipeline.runner as runner


class FlakyDetector(Detector):
    """Yields a squat path, but raises for any source whose name contains
    'bad' — lets one clip in a batch fail on demand."""

    def __init__(self, n_frames: int = 30, cycles: int = 1):
        self.n_frames = n_frames
        self.cycles = cycles

    def stream(self, source):
        if "bad" in str(source):
            raise RuntimeError("boom: corrupt clip")
        yield from squat_path_frames(self.n_frames, self.cycles)


@pytest.fixture
def patched(monkeypatch):
    """No-op ffmpeg + fixed fps + a counting YOLODetector factory."""
    FakeFFmpegWriter.instances = []
    monkeypatch.setattr(runner, "FFmpegWriter", FakeFFmpegWriter)
    monkeypatch.setattr(runner, "_read_fps", lambda source: 30.0)

    import vbt_core.pipeline.batch as batch
    calls = {"n": 0}

    def factory(weights):
        calls["n"] += 1
        return FlakyDetector()

    monkeypatch.setattr(batch, "YOLODetector", factory)
    return calls


def _item(tmp_path, name):
    base = tmp_path / name
    return BatchItem(
        source=tmp_path / f"{name}.mp4",
        out_boxed=base.with_name(f"{name}_tracked.mp4"),
        out_barpath=base.with_name(f"{name}_barpath.mp4"),
        out_csv=base.with_name(f"{name}_track.csv"),
        out_metrics=base.with_name(f"{name}_metrics.json"),
        plate_kg="20",
    )


def test_detector_built_once_for_whole_batch(tmp_path, patched):
    items = [_item(tmp_path, f"clip{i}") for i in range(3)]
    results = run_batch(tmp_path / "weights.pt", items)

    assert patched["n"] == 1          # model loaded ONCE, not per item
    assert len(results) == 3
    assert all(isinstance(r, TrackResult) for r in results)
    # each clip wrote its own metrics JSON
    for it in items:
        assert it.out_metrics.exists()
        assert json.loads(it.out_metrics.read_text())["plate_kg"] == "20"


def test_callbacks_fire_per_item(tmp_path, patched):
    items = [_item(tmp_path, f"clip{i}") for i in range(3)]
    started, done = [], []
    run_batch(
        tmp_path / "weights.pt", items,
        on_start=lambda i, it: started.append(i),
        on_done=lambda i, it, r: done.append((i, r)),
    )
    assert started == [0, 1, 2]
    assert [i for i, _ in done] == [0, 1, 2]
    assert all(isinstance(r, TrackResult) for _, r in done)


def test_one_failure_does_not_abort_batch(tmp_path, patched):
    items = [_item(tmp_path, "good0"), _item(tmp_path, "bad1"),
             _item(tmp_path, "good2")]
    errors, done = [], []
    results = run_batch(
        tmp_path / "weights.pt", items,
        on_done=lambda i, it, r: done.append(i),
        on_error=lambda i, it, exc: errors.append((i, str(exc))),
    )

    # middle clip errored; the loop continued and finished the rest
    assert [i for i, _ in errors] == [1]
    assert "boom" in errors[0][1]
    assert done == [0, 2]
    assert results[0] is not None and results[2] is not None
    assert results[1] is None


def test_progress_cb_reports_index_and_fraction(tmp_path, patched, monkeypatch):
    monkeypatch.setattr(runner, "_read_frame_count", lambda source: 30)
    items = [_item(tmp_path, "clip0"), _item(tmp_path, "clip1")]
    seen = []
    run_batch(tmp_path / "weights.pt", items,
              progress_cb=lambda i, frac: seen.append((i, frac)))

    assert seen, "progress_cb was never called"
    assert {i for i, _ in seen} == {0, 1}          # both clips reported
    assert all(0.0 < frac <= 1.0 for _, frac in seen)
    assert max(frac for _, frac in seen) == pytest.approx(1.0)
