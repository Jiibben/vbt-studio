"""
vbt_core — the UI-agnostic engine behind the VBT tracking app.

Barbell-plate velocity-based-training analysis, organised as small modules:

  detection/  YOLO (and future) detectors behind one `Detector` interface
  tracking/   Kalman + Hungarian + ByteTrack-style `BarTracker`
  analysis/   calibration, rep segmentation, per-rep velocity metrics
  rendering/  ffmpeg encode/trim and overlay drawing
  pipeline/   `TrackingPipeline` wiring all of the above into one pass
  config/     persisted plate diameters

Typical use:

    from vbt_core import run_tracking
    result = run_tracking(source=clip, weights=weights, out_boxed=...,
                          out_barpath=..., out_csv=..., out_metrics=...)

or, object-oriented:

    from vbt_core import TrackingPipeline, YOLODetector
    pipeline = TrackingPipeline(YOLODetector(weights), conf=0.25)
    result = pipeline.run(clip, out_boxed, out_barpath, out_csv, out_metrics)
"""
from .analysis import AnalysisResult, RepCounter, analyze_trajectory
from .config import load_plate_config, save_plate_config
from .detection import Detection, Detector, YOLODetector
from .pipeline import (BatchItem, TrackingPipeline, TrackResult, run_batch,
                       run_tracking)
from .tracking import BarTracker, KalmanFilter, TrackSnapshot

__all__ = ["run_tracking", "TrackingPipeline", "TrackResult",
           "run_batch", "BatchItem",
           "YOLODetector", "Detector", "Detection",
           "BarTracker", "KalmanFilter", "TrackSnapshot",
           "analyze_trajectory", "AnalysisResult", "RepCounter",
           "load_plate_config", "save_plate_config"]
