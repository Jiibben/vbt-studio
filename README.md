# VBT Training Project

Barbell-plate detection & tracking for **velocity-based training (VBT)**. A phone
video of a lift becomes per-rep velocity metrics: detect the plate, track the
camera-side plate through the lift, calibrate pixels→metres from the known plate
diameter, segment reps and compute mean/peak concentric velocity and
velocity-loss %.

> **New here?** Read [`CLAUDE.md`](CLAUDE.md) for the architecture at a glance and
> [`tech_stack.md`](tech_stack.md) for the full pipeline / model audit.

## Architecture

The computer-vision **engine** and the **web UI** are separate:

- **`vbt_core/`** — the UI-agnostic, unit-tested, pip-installable engine. All
  detection, tracking, analysis, rendering and config logic lives here.
  - `detection/` — `Detector` ABC + `YOLODetector` (Ultralytics YOLO11n). The
    `Detector` interface is the extension point for new backends.
  - `tracking/` — `KalmanFilter`, IoU/Hungarian/Mahalanobis `association`,
    `BarTracker` (two-stage ByteTrack-style).
  - `analysis/` — calibration, `RepCounter`, velocity, trajectory metrics.
  - `rendering/` — ffmpeg encode/trim + overlay drawing.
  - `pipeline/` — `TrackingPipeline` (OOP) + `run_tracking()` wrapper.
  - `config/` — persisted plate diameters.
- **`webapp/`** — UI only. A thin Flask app (`app.py`) that delegates to
  `vbt_core`, plus the "VBT Studio" frontend (`templates/` + `static/`).

```
vbt_training_project/
├── vbt_core/      # the engine (detection / tracking / analysis / rendering / pipeline / config)
├── webapp/        # thin Flask UI + static frontend (no CV logic)
├── scripts/       # offline data-prep & training CLIs (normalize, ls_to_yolo, train, track)
├── models/        # trained weights (plate_best.pt default, best.pt); exports gitignored
├── tests/         # pytest suite
├── pyproject.toml # packaging (vbt-core) + pytest config
├── CLAUDE.md      # orientation guide
└── tech_stack.md  # full pipeline / model / tech-stack audit
```

`data/` (raw/processed/datasets/tracked clips, annotations, runtime
`plate_config.json`) is **local-only and gitignored** — it is never committed.
Dates use ISO `YYYY-MM-DD` so session folders sort chronologically.

## Setup

The Python environment lives in `venv/` (ultralytics, torch-cpu, scipy, opencv,
flask). The engine is installed editable:

```bash
venv/bin/pip install -e .          # makes `import vbt_core` work
```

## Run the web UI

```bash
venv/bin/python webapp/app.py      # then open http://127.0.0.1:5000
```

Import a clip, scrub + set trim in/out points, pick the model and plate weight,
hit **Start tracking**. It returns two previews — the **boxed track** and a
**bar-path** view with a live HUD — and saves both videos + the trajectory CSV +
metrics JSON to `data/tracked/<date>/`.

## Tests

```bash
venv/bin/python -m pytest          # testpaths = tests/
```

## Offline data & training workflow

**1. Normalize footage** — drop clips in `data/raw/`, then:
```bash
bash scripts/normalize_videos.sh            # uses today's date
bash scripts/normalize_videos.sh 2026-06-10 # or a specific session date
```
Output lands in `data/processed/<date>/`; originals move to `data/archive/<date>/`.

**2. Annotate** in Label Studio — import from `data/processed/<date>/`, draw +
interpolate plate boxes, then **Export → JSON** into `data/annotations/<date>.json`.

**3. Build the YOLO dataset** (interpolates keyframes to per-frame labels):
```bash
venv/bin/python scripts/ls_to_yolo.py \
  --export data/annotations/2026-06-10.json \
  --videos data/processed/2026-06-10 \
  --out    data/datasets/plate_v1
```

**4. Train** on Colab/Kaggle (T4 GPU), then drop the resulting `best.pt` into `models/`.

**5. Track / evaluate** a held-out clip from the CLI:
```bash
venv/bin/python scripts/track_and_velocity.py \
  --weights models/best.pt \
  --source  data/processed/2026-06-10/video_3.mp4
```

## Notes
- One class: `plate`. Datasets are versioned by folder name (`plate_v1`, `plate_v2`, ...).
- Heavy HSV augmentation in training is intentional, to generalize to plate
  colors that weren't labeled (blue/black). Genuinely new conditions still need
  a few labeled clips.
