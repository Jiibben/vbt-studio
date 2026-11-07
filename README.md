# VBT Studio

**Barbell velocity from a phone video.** Film a lift, get the speed of every rep, with no sensor on the bar.

Velocity-based training (VBT) uses bar speed to steer training load: how fast you move a weight tells you how close you are to your limit. VBT Studio measures that speed with computer vision alone.

## What it does

- **Detects** the barbell plate in every frame with a YOLO11n model trained on a custom dataset.
- **Tracks** the plate through the lift (Kalman filter, Hungarian / IoU association, ByteTrack-style two-stage matching).
- **Calibrates** pixels to metres from the known plate diameter.
- **Counts reps** and reports mean and peak concentric velocity and velocity loss for each rep.
- **Renders** annotated videos (tracked box, bar path, live HUD) and exports the trajectory as CSV and the metrics as JSON.
- **Web app**: analyse a single clip, or import a whole session and process clips in batch.

## How it works

```
phone video ─► normalize (ffmpeg) ─► YOLO11n detection ─► Kalman tracking ─► px→m calibration
           ─► rep segmentation ─► velocity (Savitzky–Golay derivative) ─► CSV / JSON + overlay videos
```

All computer-vision logic lives in `vbt_core/`, a UI-agnostic, unit-tested Python package. The Flask web app in `webapp/` only handles uploads and display.

## Detector results

| | Value |
|---|---|
| Training data | 3,078 images from 15 videos |
| Validation data | 463 images from 3 other videos (split by video, never seen in training) |
| mAP50 / mAP50-95 | 0.995 / 0.884 |
| Precision / recall | 0.998 / 0.999 |
| CPU inference | about 63 ms per frame |

The velocity output has not yet been validated against a reference device (linear encoder or commercial app).

## Getting started

Requirements: Python 3.10+, ffmpeg.

```bash
python -m venv venv
venv/bin/pip install -e ".[webapp,test]"
venv/bin/python webapp/app.py        # then open http://127.0.0.1:5000
```

Import a clip, set the trim points, pick the plate weight, and start tracking. Trained weights are included in `models/`.

From the command line:

```bash
venv/bin/python scripts/track_and_velocity.py --weights models/best.pt --source path/to/lift.mp4
```

Run the tests:

```bash
venv/bin/python -m pytest
```

## Train your own detector

1. Normalize raw clips: `bash scripts/normalize_videos.sh`
2. Label plates in Label Studio and export as JSON.
3. Build a YOLO dataset: `scripts/ls_to_yolo.py` (interpolates keyframes to per-frame labels).
4. Train on a GPU (Colab or Kaggle scripts in `scripts/`) and drop `best.pt` into `models/`.

## Project structure

```
vbt_core/    detection, tracking, analysis, rendering, pipeline, config
webapp/      Flask app and "VBT Studio" front end (single clip and batch sessions)
scripts/     data preparation, training and CLI tracking
models/      trained YOLO weights
tests/       pytest suite
```

More detail: [`tech_stack.md`](tech_stack.md) (full pipeline audit) and [`CLAUDE.md`](CLAUDE.md) (architecture notes used with AI coding agents).

## Roadmap

- Validate velocity against a linear encoder.
- Real-time analysis on a phone (a TFLite export already exists).
- A second sensor: ESP32 + IMU on the bar ([`ideas/esp32-imu-vbt.md`](ideas/esp32-imu-vbt.md)).
