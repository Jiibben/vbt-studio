# CLAUDE.md — VBT Tracking project guide

Orientation for future Claude Code sessions. Keep edits accurate to the code.

## Core idea

**Velocity-based training (VBT)** auto-regulates barbell training by the *speed*
of the lift instead of fixed %1RM: bar velocity drops predictably as you fatigue,
so per-rep mean concentric velocity (MCV), peak velocity, and **velocity-loss %**
across a set tell you how hard the set really was and when to stop. This app turns
an ordinary phone video of a barbell lift into those numbers:

phone video → **YOLO plate detection** → custom **Kalman / Hungarian / ByteTrack**
tracking of the camera-side plate → **pixel→metre calibration** from a known plate
diameter → **rep segmentation + per-rep velocity metrics** (MCV/peak, velocity-loss%)
→ two rendered preview videos + a trajectory **CSV** + a metrics **JSON**, all
surfaced in a **Flask** web UI ("VBT Studio").

## Codebase structure

The CV engine is the installable package **`vbt_core/`**; **`webapp/` is UI only**.

```
vbt_training_project/
├── vbt_core/                  # THE ENGINE — UI-agnostic, unit-tested, pip-installable (editable)
│   ├── detection/
│   │   ├── base.py            # Detector ABC + Detection / FrameDetections value types (the extension point)
│   │   └── yolo.py            # YOLODetector — Ultralytics YOLO11n, class "plate", DET_FLOOR=0.10
│   ├── tracking/
│   │   ├── kalman.py          # KalmanFilter (8-dim constant-velocity, hand-rolled NumPy)
│   │   ├── association.py     # iou_matrix + associate (IoU + Hungarian + Mahalanobis gate)
│   │   ├── track.py           # Track / TrackSnapshot
│   │   └── bar_tracker.py     # BarTracker — two-stage ByteTrack-style association, primary-track pick
│   ├── analysis/
│   │   ├── calibration.py     # calibrate_m_per_px (median plate size → m/px)
│   │   ├── rep_counter.py     # RepCounter — streaming rep FSM (idle/down/up)
│   │   ├── velocity.py        # frame_velocity (Savitzky–Golay deriv) + active_span
│   │   └── trajectory.py      # analyze_trajectory → AnalysisResult / RepMetrics (post-pass)
│   ├── rendering/
│   │   ├── ffmpeg.py          # FFmpegWriter + trim_clip (libx264 yuv420p faststart)
│   │   └── overlay.py         # draw_box / draw_hud / draw_trail (+ COLOR_* constants)
│   ├── pipeline/
│   │   └── runner.py          # TrackingPipeline (orchestrator) + run_tracking() wrapper + TrackResult
│   ├── config/
│   │   └── plate_config.py    # load/save_plate_config → data/plate_config.json (env VBT_PLATE_CONFIG)
│   └── __init__.py            # public API: run_tracking, TrackingPipeline, YOLODetector, Detector, ...
├── webapp/                    # UI ONLY — no CV logic lives here
│   ├── app.py                 # thin Flask: routing, upload, validation; delegates to vbt_core.run_tracking
│   ├── templates/index.html   # page markup
│   └── static/css/styles.css, static/js/app.js   # "VBT Studio" frontend, vanilla JS + <canvas> velocity chart
├── scripts/                   # offline data/training CLI (not used at inference time)
│   ├── normalize_videos.sh    # raw → processed (ffmpeg, 1080p/30fps/H.264)
│   ├── ls_to_yolo.py          # Label Studio export → YOLO dataset (interpolates keyframes)
│   ├── train_colab.py / train_kaggle.py / yolo_wizard.py   # training helpers (off-machine GPU)
│   └── track_and_velocity.py  # CLI tracking/eval on a clip
├── models/                    # trained weights: plate_best.pt (webapp default, 640), best.pt (960); *.onnx/saved_model are gitignored exports
├── tests/                     # pytest suite (pyproject testpaths = tests/)
├── .claude/skills/aider-test-gen/   # skill: drive aider + local Ollama to mass-generate parametrized tests
├── pyproject.toml             # package vbt-core; extras [webapp] (flask), [test] (pytest); pytest config
├── tech_stack.md              # FULL pipeline / model / tech-stack audit (read this for depth)
└── data/                      # LOCAL-ONLY (gitignored): raw/processed/datasets/tracked + plate_config.json
```

**Architectural rule:** all CV / tracking / analysis / rendering logic lives in
`vbt_core`. `webapp` does HTTP, file I/O, validation, and presentation — nothing
else. The **`Detector` interface** (`vbt_core/detection/base.py`) is the extension
point: a new detector backend (e.g. a segmentation/mask model) is just another
`Detector` subclass, and nothing downstream changes.

## How to run

Python env is **`venv/`** (ultralytics, torch-cpu, scipy, opencv, flask); the
`vbt_core` package is installed **editable** into it.

```bash
# Web UI
venv/bin/python webapp/app.py        # → http://127.0.0.1:5000

# Tests
venv/bin/python -m pytest            # testpaths = tests/

# Engine directly (no Flask)
venv/bin/python -c "from vbt_core import run_tracking"
```

## Key conventions

- **Kalman-smoothed coordinates everywhere downstream** — the CSV, calibration,
  rep FSM, and velocity all consume the smoothed track state, never raw detector
  boxes.
- **`DET_FLOOR` is deliberately low (0.10)** — the two-stage tracker rescues
  tracks from 0.10–high candidates; do not raise it thinking it's a bug.
- **Calibration needs a known plate diameter** (mm), from the per-lift dropdown +
  saved per-plate sizes; no scale reference → no metres.
- **Dates are ISO `YYYY-MM-DD`** so session folders sort chronologically.
- `data/` is **local-only** (gitignored) — datasets, captured/processed/tracked
  video, and `plate_config.json` are not in the repo.

## See also

**See `tech_stack.md` for the full pipeline / model / tech-stack audit** — video
ingestion, dataset, YOLO11n architecture & precision, inference resolution, the
exact Kalman / Hungarian / ByteTrack parameters, and the calibration/rep/velocity
algorithms.
