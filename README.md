# VBT Training Project

Barbell-plate detection & tracking for velocity-based training (VBT). The
pipeline goes: **raw video → normalized clips → Label Studio annotations →
YOLO dataset → trained detector → tracking + velocity**.

## Layout

```
vbt_training_project/
├── data/
│   ├── raw/                     # DROP ZONE: drop any video here, then run normalize_videos.sh
│   ├── processed/<date>/        # normalized 1080p / 30fps / H.264 clips (training source)
│   ├── archive/<date>/          # original untouched captures (auto-archived from raw/)
│   ├── annotations/<date>.json  # Label Studio exports (full JSON)
│   ├── datasets/<name>/         # YOLO datasets built from annotations (images/ labels/ data.yaml)
│   └── tracked/<date>/          # tracking outputs: annotated *_tracked.mp4 + *_track.csv
├── scripts/
│   ├── normalize_videos.sh      # raw → processed, AMD GPU (VAAPI) encode
│   ├── ls_to_yolo.py            # annotations → YOLO dataset (interpolates keyframes)
│   ├── train_colab.py           # bombproof, resumable YOLO11 training (run on Colab)
│   └── track_and_velocity.py    # run the detector + tracker on a clip
├── webapp/                      # small Flask debug UI: import → trim → track → preview
├── mobile/                      # self-contained React Native app (on-device tracking) — see mobile/README.md
├── models/                      # trained weights (best.pt) live here
├── papers/                      # reference material
└── venv/                        # Python environment (label-studio, opencv, ...)
```

Dates use ISO format `YYYY-MM-DD` so folders sort chronologically.

## Workflow

**1. Normalize new footage** — drop clips in `data/raw/`, then:
```bash
bash scripts/normalize_videos.sh            # uses today's date
bash scripts/normalize_videos.sh 2026-06-10 # or a specific session date
```
Output lands in `data/processed/<date>/`; originals move to `data/archive/<date>/`.

**2. Annotate** in Label Studio — import the clips from `data/processed/<date>/`,
draw + interpolate plate boxes, then **Export → JSON** into
`data/annotations/<date>.json`.

**3. Build the YOLO dataset** (interpolates keyframes to per-frame labels):
```bash
venv/bin/python scripts/ls_to_yolo.py \
  --export data/annotations/2026-06-10.json \
  --videos data/processed/2026-06-10 \
  --out    data/datasets/plate_v1
```

**4. Train** on Colab (T4 GPU). Upload `data/datasets/plate_v1.zip` + `scripts/train_colab.py`
to Drive, then in a notebook run `!python train_colab.py`. It saves to Drive and
auto-resumes if you get timed out. Drop the resulting `best.pt` into `models/`.

**5. Track / evaluate** a held-out clip:
```bash
venv/bin/python scripts/track_and_velocity.py \
  --weights models/best.pt \
  --source  data/processed/2026-06-10/video_3.mp4
```
(velocity stage TBD — tracking + trajectory CSV are implemented.)

**Debug webapp** — a browser UI for the same tracking, handy for eyeballing
results without the CLI:
```bash
venv/bin/python webapp/app.py     # then open http://127.0.0.1:5000
```
Import a clip, scrub + set trim in/out points, pick the model, hit **Start
tracking**. It returns two previews — the **boxed track** and a **bar-path**
view (a sticky green trail through the plate centre) — and saves both videos +
the trajectory CSV to `data/tracked/<date>/`, same naming as the CLI.

## Notes
- One class: `plate`. Datasets are versioned by folder name (`plate_v1`, `plate_v2`, ...).
- Heavy HSV augmentation in training is intentional, to generalize to plate
  colors that weren't labeled (blue/black). Genuinely new conditions still need
  a few labeled clips.
