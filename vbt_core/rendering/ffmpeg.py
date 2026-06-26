"""
ffmpeg helpers: a frame writer that pipes raw BGR to a browser-playable H.264
mp4, and a frame-accurate clip trimmer.

OpenCV's own mp4v output does not play in a browser <video> element; piping raw
frames through ffmpeg (libx264 / yuv420p / +faststart) does.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class FFmpegWriter:
    """Pipe raw BGR frames into ffmpeg, encode to browser-playable H.264 mp4."""

    def __init__(self, path: Path, fps: float, width: int, height: int):
        self.path = Path(path)
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

    def write(self, frame) -> None:
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def trim_clip(src: Path, dst: Path, start: float, end: float) -> None:
    """Re-encode the [start, end] slice of `src` into `dst` (frame-accurate)."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end - start)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    if start > 0:
        cmd += ["-ss", f"{start}"]
    if duration > 0:
        cmd += ["-t", f"{duration}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
            "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)
