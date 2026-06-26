"""Detection backends — turn a video into per-frame plate detections."""
from .base import Detection, Detector, FrameDetections
from .yolo import DET_FLOOR, YOLODetector

__all__ = ["Detection", "Detector", "FrameDetections", "YOLODetector",
           "DET_FLOOR"]
