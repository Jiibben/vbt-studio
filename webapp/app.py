#!/usr/bin/env python3
"""
app.py — thin Flask UI for the VBT tracking pipeline.

This module is *only* the web layer: HTTP routing, file upload, request
validation, and serving results. Every piece of computer vision — detection,
tracking, calibration, rep/velocity analysis, rendering — lives in the
`vbt_core` package and is reached through `run_tracking`. Swapping or extending
the engine never touches this file.

Run:
    venv/bin/python webapp/app.py    # then open http://127.0.0.1:5000
"""
from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

from flask import (Flask, abort, jsonify, render_template, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # allow running without `pip install -e .`

from vbt_core import load_plate_config, run_tracking, save_plate_config
from vbt_core.rendering import trim_clip

MODELS_DIR = ROOT / "models"
TRACKED_DIR = ROOT / "data" / "tracked"
UPLOAD_DIR = Path(__file__).resolve().parent / "_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB, debug tool


def list_models() -> list[str]:
    return sorted(p.name for p in MODELS_DIR.glob("*.pt"))


def _default_model() -> str:
    models = list_models()
    for preferred in ("plate_best.pt", "best.pt"):
        if preferred in models:
            return preferred
    return models[0] if models else ""


@app.route("/")
def index():
    return render_template("index.html", models=list_models(),
                           default_model=_default_model(),
                           plate_config=load_plate_config())


@app.route("/plate-config", methods=["GET"])
def plate_config_get():
    return jsonify(load_plate_config())


@app.route("/plate-config", methods=["POST"])
def plate_config_post():
    try:
        return jsonify(save_plate_config(request.get_json(force=True)))
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400


@app.route("/track", methods=["POST"])
def track():
    if "video" not in request.files or request.files["video"].filename == "":
        return jsonify(error="No video uploaded."), 400

    f = request.files["video"]
    model_name = request.form.get("model", _default_model())
    weights = MODELS_DIR / model_name
    if not weights.exists():
        return jsonify(error=f"Model not found: {model_name}"), 400

    try:
        start = float(request.form.get("start", 0) or 0)
        end = float(request.form.get("end", 0) or 0)
    except ValueError:
        return jsonify(error="Invalid trim times."), 400
    conf = float(request.form.get("conf", 0.25) or 0.25)

    plate = request.form.get("plate", "20")
    plates = load_plate_config()
    if plate not in plates:
        return jsonify(error=f"Unknown plate weight: {plate}"), 400
    plate_diameter_mm = plates[plate]

    trail_style = request.form.get("trail_style", "green")
    if trail_style not in ("green", "phase"):
        return jsonify(error=f"Unknown trail style: {trail_style}"), 400

    rep_style = request.form.get("rep_style", "auto")
    if rep_style not in ("auto", "down-first", "up-first"):
        return jsonify(error=f"Unknown rep style: {rep_style}"), 400

    stem = secure_filename(Path(f.filename).stem) or "clip"
    raw_path = UPLOAD_DIR / f"{stem}_raw.mp4"
    f.save(raw_path)

    session = date.today().isoformat()
    outdir = TRACKED_DIR / session
    trimmed = UPLOAD_DIR / f"{stem}_trim.mp4"

    try:
        if end > start:
            trim_clip(raw_path, trimmed, start, end)
            source = trimmed
        else:
            source = raw_path  # no trim requested -> track the whole clip

        result = run_tracking(
            source=source,
            weights=weights,
            out_boxed=outdir / f"{stem}_tracked.mp4",
            out_barpath=outdir / f"{stem}_barpath.mp4",
            out_csv=outdir / f"{stem}_track.csv",
            out_metrics=outdir / f"{stem}_metrics.json",
            conf=conf,
            plate_diameter_mm=plate_diameter_mm,
            plate_kg=plate,
            trail_style=trail_style,
            rep_style=rep_style,
        )
    except subprocess.CalledProcessError:
        return jsonify(error="ffmpeg failed while trimming the clip."), 500
    except Exception as exc:  # surface tracking errors to the browser
        return jsonify(error=f"Tracking failed: {exc}"), 500

    return jsonify(
        boxed_url=url_for("media", session=session,
                          filename=result.boxed_video.name),
        barpath_url=url_for("media", session=session,
                            filename=result.barpath_video.name),
        saved_dir=str(outdir.relative_to(ROOT)),
        boxed_name=result.boxed_video.name,
        barpath_name=result.barpath_video.name,
        csv_name=result.csv_path.name,
        metrics_name=result.metrics_path.name,
        fps=round(result.fps, 2),
        n_frames=result.n_frames,
        coverage=round(result.coverage, 1),
        track_ids=result.track_ids,
        analysis=result.analysis.to_dict() if result.analysis else None,
    )


@app.route("/media/<session>/<path:filename>")
def media(session, filename):
    folder = TRACKED_DIR / secure_filename(session)
    if not (folder / filename).exists():
        abort(404)
    # conditional=True enables HTTP range requests so <video> can seek.
    return send_from_directory(folder, filename, conditional=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
