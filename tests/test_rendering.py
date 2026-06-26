"""Tests for vbt_core.rendering (ffmpeg helpers + overlay drawing).

No real ffmpeg is ever spawned: ``subprocess.run`` (trim_clip) and
``subprocess.Popen`` (FFmpegWriter) are monkeypatched to record their argv and
piped bytes. Overlay tests run the real OpenCV draw calls on a numpy image and
assert pixels were mutated.
"""
from __future__ import annotations

import subprocess

import numpy as np
import pytest

from vbt_core.rendering import (COLOR_DOWN, COLOR_IDLE, COLOR_UP, FFmpegWriter,
                                PHASE_COLORS, draw_box, draw_hud, draw_trail,
                                trim_clip)
from vbt_core.tracking import TrackSnapshot


def _snap(**kw):
    base = dict(track_id=1, cx=100.0, cy=80.0, w=40.0, h=40.0,
                vx=0.0, vy=0.0, conf=0.9, coasting=False, hits=5, age=5)
    base.update(kw)
    return TrackSnapshot(**base)


# --------------------------------------------------------------------------- #
# trim_clip — argv construction
# --------------------------------------------------------------------------- #
@pytest.fixture
def capture_run(monkeypatch):
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_trim_clip_basic_argv(tmp_path, capture_run):
    src = tmp_path / "in.mp4"
    dst = tmp_path / "out" / "out.mp4"
    trim_clip(src, dst, 0.0, 0.0)

    cmd = capture_run[0]["cmd"]
    assert cmd[0] == "ffmpeg"
    assert str(src) in cmd
    assert str(dst) in cmd
    assert "libx264" in cmd
    assert "yuv420p" in cmd
    assert "+faststart" in cmd
    # check=True is forwarded.
    assert capture_run[0]["kwargs"].get("check") is True
    # destination parent created.
    assert dst.parent.is_dir()


@pytest.mark.parametrize("start,end,want_ss,want_t,t_val", [
    (0.0, 0.0, False, False, None),     # no slice flags
    (0.0, 5.0, False, True, "5.0"),     # start==0 -> no -ss, duration 5
    (2.0, 5.0, True, True, "3.0"),      # -ss 2, -t 3
    (2.0, 2.0, True, False, None),      # zero duration -> no -t
    (3.0, 1.0, True, False, None),      # end<start -> duration clamps to 0
    (0.5, 10.0, True, True, "9.5"),
])
def test_trim_clip_ss_t_flags(tmp_path, capture_run, start, end,
                              want_ss, want_t, t_val):
    src = tmp_path / "in.mp4"
    dst = tmp_path / "out.mp4"
    trim_clip(src, dst, start, end)
    cmd = capture_run[0]["cmd"]

    assert ("-ss" in cmd) is want_ss
    if want_ss:
        assert cmd[cmd.index("-ss") + 1] == f"{start}"
    assert ("-t" in cmd) is want_t
    if want_t:
        assert cmd[cmd.index("-t") + 1] == t_val


# --------------------------------------------------------------------------- #
# FFmpegWriter — piped bytes, no real ffmpeg
# --------------------------------------------------------------------------- #
class _FakeStdin:
    def __init__(self):
        self.writes = []
        self.closed = False

    def write(self, b):
        self.writes.append(b)

    def close(self):
        self.closed = True


class _FakeProc:
    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.stdin = _FakeStdin()
        self.waited = False

    def wait(self):
        self.waited = True
        return 0


@pytest.fixture
def fake_popen(monkeypatch):
    holder = {}

    def fake_popen(argv, **kwargs):
        proc = _FakeProc(argv, **kwargs)
        holder["proc"] = proc
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return holder


def test_ffmpeg_writer_argv(tmp_path, fake_popen):
    out = tmp_path / "vid.mp4"
    FFmpegWriter(out, 30.0, 64, 48)
    argv = fake_popen["proc"].argv
    assert argv[0] == "ffmpeg"
    assert "rawvideo" in argv
    assert "-s" in argv and argv[argv.index("-s") + 1] == "64x48"
    assert "-r" in argv and argv[argv.index("-r") + 1] == "30.0"
    assert "libx264" in argv
    assert "yuv420p" in argv
    assert "+faststart" in argv
    assert str(out) in argv
    assert fake_popen["proc"].kwargs.get("stdin") is subprocess.PIPE


def test_ffmpeg_writer_write_and_close(tmp_path, fake_popen):
    writer = FFmpegWriter(tmp_path / "v.mp4", 30.0, 4, 4)
    proc = fake_popen["proc"]
    frame = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    writer.write(frame)
    assert proc.stdin.writes == [frame.tobytes()]

    writer.close()
    assert proc.stdin.closed is True
    assert proc.waited is True


def test_ffmpeg_writer_context_manager(tmp_path, fake_popen):
    with FFmpegWriter(tmp_path / "v.mp4", 30.0, 4, 4) as w:
        w.write(np.zeros((4, 4, 3), dtype=np.uint8))
    proc = fake_popen["proc"]
    assert proc.stdin.closed is True
    assert proc.waited is True


# --------------------------------------------------------------------------- #
# overlay drawing — real cv2, assert pixels mutated
# --------------------------------------------------------------------------- #
def test_phase_colors_has_expected_keys():
    assert set(PHASE_COLORS) == {"up", "down", "idle"}
    assert PHASE_COLORS["up"] == COLOR_UP
    assert PHASE_COLORS["down"] == COLOR_DOWN
    assert PHASE_COLORS["idle"] == COLOR_IDLE


def test_draw_box_mutates_image():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    draw_box(img, _snap(), COLOR_UP, primary=True)
    assert img.any()


def test_draw_box_coasting_label():
    # coasting snap uses the "~" label branch; should still draw without error.
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    draw_box(img, _snap(coasting=True, conf=0.0), COLOR_DOWN, primary=False)
    assert img.any()


@pytest.mark.parametrize("phase", ["up", "down", "idle"])
def test_draw_hud_mutates_image(phase):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hud(img, rep_count=3, v_mps=0.42, phase=phase)
    assert img.any()


def test_draw_hud_handles_none_velocity():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hud(img, rep_count=0, v_mps=None, phase="idle")
    assert img.any()


@pytest.mark.parametrize("style", ["green", "phase"])
def test_draw_trail_mutates_image(style):
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    trail = [(10, 10, 0.0), (50, 60, 0.1), (90, 120, 0.2)]
    draw_trail(img, trail, phase_at=lambda t: "up", trail_style=style)
    assert img.any()


def test_draw_trail_single_point_draws_circle():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_trail(img, [(50, 50, 0.0)], phase_at=lambda t: "up", trail_style="green")
    assert img.any()


def test_draw_trail_empty_is_noop():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    draw_trail(img, [], phase_at=lambda t: "up", trail_style="green")
    assert not img.any()
