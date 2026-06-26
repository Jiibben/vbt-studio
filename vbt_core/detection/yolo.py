"""
YOLO detector backend (Ultralytics YOLO11n, single class: plate).

Runs the model in pure `predict` mode — Ultralytics' own ByteTrack is bypassed
because our `BarTracker` (Kalman → Hungarian → ByteTrack-style recovery) does
the association. The detector floor is deliberately low so the tracker's
second association stage has low-confidence candidates to rescue tracks with.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from ultralytics import YOLO

from .base import Detector, FrameDetections

# Detection floor handed to YOLO — low on purpose (the two-stage tracker
# consumes 0.10–high candidates; see BarTracker).
DET_FLOOR = 0.10


class YOLODetector(Detector):
    """Ultralytics YOLO wrapped behind the `Detector` interface."""

    def __init__(self, weights: str | Path, *, det_floor: float = DET_FLOOR):
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights}")
        self.det_floor = det_floor
        self._model = YOLO(str(self.weights))

    def stream(self, source: str | Path) -> Iterator[FrameDetections]:
        results = self._model.predict(
            source=str(source),
            stream=True,
            conf=self.det_floor,
            verbose=False,
        )
        for index, r in enumerate(results):
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                rows = np.hstack([
                    boxes.xywh.cpu().numpy(),
                    boxes.conf.cpu().numpy().reshape(-1, 1),
                ])
            else:
                rows = np.empty((0, 5))
            yield FrameDetections(index=index, image=r.orig_img, boxes=rows)
