"""
Overlay drawing: track boxes, the live HUD, and the sticky bar-path trail.

All colours are BGR (OpenCV). The bar path is redrawn from the accumulated
trail every frame so that, when `trail_style == "phase"`, segments recolour
retroactively at confirmed turnarounds — a turnaround is only confirmed after
the hysteresis distance, but its *time* is the true extreme, so the trail does
not lag by 15 cm.
"""
from __future__ import annotations

import cv2

from ..tracking import TrackSnapshot

# BGR colours.
COLOR_UP = (0, 255, 0)        # concentric trail / primary box
COLOR_DOWN = (0, 165, 255)    # eccentric trail / coasting box
COLOR_IDLE = (160, 160, 160)  # no movement yet / secondary boxes
PHASE_COLORS = {"up": COLOR_UP, "down": COLOR_DOWN, "idle": COLOR_IDLE}


def draw_box(img, snap: TrackSnapshot, color, primary: bool) -> None:
    x1 = int(round(snap.cx - snap.w / 2))
    y1 = int(round(snap.cy - snap.h / 2))
    x2 = int(round(snap.cx + snap.w / 2))
    y2 = int(round(snap.cy + snap.h / 2))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if primary else 2)
    label = f"id {snap.track_id}" + (
        " ~" if snap.coasting else f" {snap.conf:.2f}")
    cv2.putText(img, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_hud(img, rep_count: int, v_mps: float | None, phase: str) -> None:
    h = img.shape[0]
    scale = h / 720.0
    pad = int(14 * scale)
    line_h = int(34 * scale)
    box_w = int(230 * scale)
    box_h = pad * 2 + line_h * 3
    overlay = img[0:box_h + pad, 0:box_w + pad].copy()
    cv2.rectangle(img, (pad // 2, pad // 2), (box_w, box_h), (20, 22, 28), -1)
    cv2.addWeighted(overlay, 0.25, img[0:box_h + pad, 0:box_w + pad], 0.75, 0,
                    img[0:box_h + pad, 0:box_w + pad])
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.85 * scale
    th = max(2, int(2 * scale))
    x = pad
    v_txt = f"{v_mps:+.2f} m/s" if v_mps is not None else "--.- m/s"
    cv2.putText(img, f"REP {rep_count}", (x, pad + line_h - 8), font,
                fs * 1.15, (255, 255, 255), th, cv2.LINE_AA)
    cv2.putText(img, v_txt, (x, pad + line_h * 2 - 8), font, fs,
                (235, 235, 235), th, cv2.LINE_AA)
    cv2.putText(img, phase.upper(), (x, pad + line_h * 3 - 8), font, fs,
                PHASE_COLORS[phase], th, cv2.LINE_AA)


def draw_trail(img, trail, phase_at, trail_style: str,
               thickness: int = 3) -> None:
    """Redraw the accumulated bar-path `trail` (list of (x, y, time_s)).

    `phase_at` is a callable t -> phase label; used only when
    `trail_style == "phase"`. In "green" style the whole trail is concentric
    green."""
    if len(trail) >= 2:
        for (x0, y0, _), (x1, y1, t1) in zip(trail, trail[1:]):
            color = PHASE_COLORS[phase_at(t1)] \
                if trail_style == "phase" else COLOR_UP
            cv2.line(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
    if trail:
        x1, y1, t1 = trail[-1]
        color = PHASE_COLORS[phase_at(t1)] \
            if trail_style == "phase" else COLOR_UP
        cv2.circle(img, (x1, y1), thickness + 3, color, -1, cv2.LINE_AA)
