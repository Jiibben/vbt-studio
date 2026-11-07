"""
Batch processing — run the tracking pipeline over many clips with the YOLO model
loaded once.

This is the engine-side half of the webapp's bulk-import feature: all CV /
orchestration stays in ``vbt_core`` (the webapp only schedules work and persists
state). ``run_batch`` loads a single :class:`YOLODetector` and reuses it across
every clip — only the per-clip :class:`BarTracker` is rebuilt (inside
``TrackingPipeline.run``), so reuse is safe. One clip failing never aborts the
batch: its exception is routed to ``on_error`` and the loop moves on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..detection import YOLODetector
from .runner import TrackingPipeline, TrackResult


@dataclass
class BatchItem:
    """One clip to process plus where its four outputs go.

    Mirrors the per-call knobs of ``TrackingPipeline`` so each clip can carry its
    own plate / rep-style / trail overrides while sharing one detector.
    """

    source: Path
    out_boxed: Path
    out_barpath: Path
    out_csv: Path
    out_metrics: Path
    conf: float = 0.25
    plate_diameter_mm: float = 450.0
    plate_kg: str | None = None
    trail_style: str = "green"
    rep_style: str = "auto"
    trail_thickness: int = 3


def run_batch(weights, items, *, on_start=None, on_done=None, on_error=None,
              progress_cb=None) -> list[TrackResult | None]:
    """Process ``items`` with one shared YOLO detector loaded from ``weights``.

    Callbacks (all optional) report progress as the batch runs:

      * ``on_start(index, item)``            — about to process this clip
      * ``on_done(index, item, result)``     — ``result`` is a :class:`TrackResult`
      * ``on_error(index, item, exc)``       — this clip failed; the loop continues
      * ``progress_cb(index, fraction)``     — 0..1 within the current clip

    Returns a list aligned with ``items``: a :class:`TrackResult` per clip, or
    ``None`` where that clip errored — so callers can use return values instead
    of (or alongside) callbacks.
    """
    detector = YOLODetector(weights)        # the expensive load — done ONCE
    results: list[TrackResult | None] = []
    for index, item in enumerate(items):
        if on_start is not None:
            on_start(index, item)
        try:
            pipeline = TrackingPipeline(
                detector,
                conf=item.conf,
                plate_diameter_mm=item.plate_diameter_mm,
                plate_kg=item.plate_kg,
                trail_style=item.trail_style,
                rep_style=item.rep_style,
                trail_thickness=item.trail_thickness,
            )
            cb = None
            if progress_cb is not None:
                cb = lambda frac, i=index: progress_cb(i, frac)  # noqa: E731
            result = pipeline.run(
                item.source, item.out_boxed, item.out_barpath,
                item.out_csv, item.out_metrics, progress_cb=cb,
            )
            results.append(result)
            if on_done is not None:
                on_done(index, item, result)
        except Exception as exc:            # isolate per-clip failures
            results.append(None)
            if on_error is not None:
                on_error(index, item, exc)
    return results
