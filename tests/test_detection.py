"""Tests for vbt_core.detection (base value types + YOLO backend).

No real Ultralytics model is ever loaded: ``YOLODetector.__init__`` is exercised
only with a nonexistent path (raises before touching YOLO), and the ``stream``
path replaces ``vbt_core.detection.yolo.YOLO`` with a fake whose ``predict``
yields fake result objects mimicking the Ultralytics ``.boxes`` API.
"""
from __future__ import annotations

import numpy as np
import pytest

from vbt_core.detection import (DET_FLOOR, Detection, Detector,
                                FrameDetections, YOLODetector)
from vbt_core.detection import yolo as yolo_mod


# --------------------------------------------------------------------------- #
# base value types
# --------------------------------------------------------------------------- #
def test_detection_as_row_order():
    d = Detection(cx=1.0, cy=2.0, w=3.0, h=4.0, conf=0.5)
    assert d.as_row() == [1.0, 2.0, 3.0, 4.0, 0.5]


def test_detection_is_frozen():
    d = Detection(1, 2, 3, 4, 0.5)
    with pytest.raises((AttributeError, Exception)):
        d.cx = 9.0  # frozen dataclass


def test_frame_detections_builds_detection_objects():
    boxes = np.array([[10, 20, 30, 40, 0.9],
                      [50, 60, 70, 80, 0.3]], dtype=float)
    fd = FrameDetections(index=7, image=np.zeros((4, 4, 3), np.uint8), boxes=boxes)
    dets = fd.detections
    assert len(dets) == 2
    assert all(isinstance(d, Detection) for d in dets)
    assert dets[0].as_row() == [10, 20, 30, 40, 0.9]
    assert dets[1].cx == 50 and dets[1].conf == 0.3


def test_frame_detections_empty_boxes_yields_no_detections():
    fd = FrameDetections(index=0, image=np.zeros((2, 2, 3), np.uint8),
                         boxes=np.empty((0, 5)))
    assert fd.detections == []


def test_detector_is_abstract():
    with pytest.raises(TypeError):
        Detector()  # abstract method `stream` unimplemented


# --------------------------------------------------------------------------- #
# YOLO backend — fakes only, never a real model
# --------------------------------------------------------------------------- #
class _FakeTensor:
    """Mimics an Ultralytics tensor: ``.cpu().numpy()`` returns a numpy array."""

    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeBoxes:
    def __init__(self, xywh, conf):
        self.xywh = _FakeTensor(np.asarray(xywh, dtype=float).reshape(-1, 4))
        self.conf = _FakeTensor(np.asarray(conf, dtype=float).reshape(-1))
        self._n = len(self.conf.numpy())

    def __len__(self):
        return self._n


class _FakeResult:
    def __init__(self, boxes, orig_img):
        self.boxes = boxes
        self.orig_img = orig_img


def _make_fake_yolo(results, recorder):
    class _FakeYOLO:
        def __init__(self, weights):
            recorder["weights"] = weights

        def predict(self, **kwargs):
            recorder["predict_kwargs"] = kwargs
            return iter(results)

    return _FakeYOLO


def test_constructor_missing_weights_raises_without_touching_yolo(monkeypatch):
    # If YOLO were instantiated this would explode; assert it is not even called.
    def _boom(*a, **k):
        raise AssertionError("YOLO must not be constructed for a missing file")

    monkeypatch.setattr(yolo_mod, "YOLO", _boom)
    with pytest.raises(FileNotFoundError):
        YOLODetector("definitely_nonexistent_weights.pt")


def test_stream_converts_boxes(tmp_path, monkeypatch):
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"not-a-real-model")

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    results = [
        _FakeResult(_FakeBoxes([[10, 20, 30, 40]], [0.9]), img),          # 1 box
        _FakeResult(_FakeBoxes(np.empty((0, 4)), np.empty((0,))), img),   # none
        _FakeResult(None, img),                                            # None
        _FakeResult(_FakeBoxes([[1, 2, 3, 4], [5, 6, 7, 8]], [0.5, 0.2]), img),
    ]
    rec: dict = {}
    monkeypatch.setattr(yolo_mod, "YOLO", _make_fake_yolo(results, rec))

    det = YOLODetector(weights)
    assert rec["weights"] == str(weights)

    frames = list(det.stream("anything.mp4"))

    # predict called with the right knobs.
    assert rec["predict_kwargs"]["stream"] is True
    assert rec["predict_kwargs"]["conf"] == DET_FLOOR
    assert rec["predict_kwargs"]["verbose"] is False
    assert rec["predict_kwargs"]["source"] == "anything.mp4"

    # index increments across all frames.
    assert [f.index for f in frames] == [0, 1, 2, 3]

    # frame 0: one box -> (1,5) row [cx,cy,w,h,conf]
    assert frames[0].boxes.shape == (1, 5)
    np.testing.assert_allclose(frames[0].boxes[0], [10, 20, 30, 40, 0.9])

    # frame 1 (empty boxes) and frame 2 (None) -> (0,5)
    assert frames[1].boxes.shape == (0, 5)
    assert frames[2].boxes.shape == (0, 5)

    # frame 3: two boxes, conf appended as last column
    assert frames[3].boxes.shape == (2, 5)
    np.testing.assert_allclose(frames[3].boxes[:, 4], [0.5, 0.2])
    np.testing.assert_allclose(frames[3].boxes[0, :4], [1, 2, 3, 4])

    # image passed straight through.
    assert frames[0].image is img


def test_stream_respects_custom_det_floor(tmp_path, monkeypatch):
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"x")
    rec: dict = {}
    monkeypatch.setattr(yolo_mod, "YOLO", _make_fake_yolo([], rec))

    det = YOLODetector(weights, det_floor=0.42)
    list(det.stream("clip.mp4"))
    assert rec["predict_kwargs"]["conf"] == 0.42


def test_det_floor_constant():
    assert DET_FLOOR == 0.10
