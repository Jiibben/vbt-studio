"""
Detection interfaces — the contract every detector backend implements.

The rest of the pipeline (tracking, analysis, rendering) only ever sees the
small value types defined here, never Ultralytics objects. Swapping YOLO for a
different detector — or a segmentation/mask model later — means writing one new
`Detector` subclass and nothing downstream changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detected box in centre form, pixel units.

    cx, cy = box centre; w, h = box size; conf = detector confidence [0, 1].
    """

    cx: float
    cy: float
    w: float
    h: float
    conf: float

    def as_row(self) -> list[float]:
        """[cx, cy, w, h, conf] — the row layout the tracker consumes."""
        return [self.cx, self.cy, self.w, self.h, self.conf]


@dataclass
class FrameDetections:
    """One decoded video frame plus everything detected in it.

    `image` is the original BGR frame (HxWx3 uint8). `boxes` is an (N, 5) float
    array of [cx, cy, w, h, conf] rows — empty (0, 5) when nothing was found.
    """

    index: int
    image: np.ndarray
    boxes: np.ndarray

    @property
    def detections(self) -> list[Detection]:
        return [Detection(*row) for row in self.boxes]


class Detector(ABC):
    """A detector turns a video source into a stream of per-frame detections.

    Implementations own video decoding (so the backend can letterbox / batch /
    GPU-decode however it likes); the pipeline just iterates the frames.
    """

    @abstractmethod
    def stream(self, source: str | Path) -> Iterator[FrameDetections]:
        """Yield one `FrameDetections` per frame of `source`, in order."""
        raise NotImplementedError
