# VBT Tracking — Tech Stack & Pipeline Audit

> Audited 2026-06-12 directly from the code and the model checkpoints
> (`torch.load` of the `.pt` files, empirical inference run), not from memory.

```
phone video ──► normalize (ffmpeg) ──► label (Label Studio) ──► dataset (YOLO fmt)
                                                                      │
                                                                  train (Colab/Kaggle)
                                                                      ▼
upload in webapp ──► trim (ffmpeg) ──► YOLO11n detect ──► [Kalman → Hungarian → ByteTrack]
   (webapp/app.py)   (vbt_core.rendering)  (vbt_core.detection)   (vbt_core.tracking)
                                                                      │
                          ┌───────────────────────────────────────────┤
                          ▼                                           ▼
                live overlay videos (H.264)              calibration → rep FSM → velocity
                boxed + bar-path + HUD                   CSV + metrics JSON + webapp charts
                (vbt_core.rendering)                        (vbt_core.analysis)

   All of the above (detect → track → analyse → render) is orchestrated by
   vbt_core.pipeline.TrackingPipeline / run_tracking(); webapp/app.py is a thin
   Flask shell that just validates the upload and calls run_tracking().
```

---

## 0 · Module architecture (post-refactor)

The computer-vision engine lives in an installable package, **`vbt_core/`**, and
is **UI-agnostic and unit-tested** (`tests/`, pytest via `pyproject.toml`). The
Flask webapp is now just one consumer of it. Architectural rule: **all CV /
tracking / analysis / rendering logic is in `vbt_core`; `webapp/` is UI only.**

| Submodule | Holds | Key symbols |
|---|---|---|
| `vbt_core/detection/` | detector backends behind one interface | `Detector` ABC, `Detection`/`FrameDetections` (`base.py`); `YOLODetector` (`yolo.py`) |
| `vbt_core/tracking/` | the custom tracker | `KalmanFilter` (`kalman.py`), `iou_matrix`/`associate` (`association.py`), `Track`/`TrackSnapshot` (`track.py`), `BarTracker` (`bar_tracker.py`) |
| `vbt_core/analysis/` | calibration, reps, velocity | `calibrate_m_per_px` (`calibration.py`), `RepCounter` (`rep_counter.py`), `frame_velocity`/`active_span` (`velocity.py`), `analyze_trajectory`/`AnalysisResult`/`RepMetrics` (`trajectory.py`) |
| `vbt_core/rendering/` | encode + overlay | `FFmpegWriter`/`trim_clip` (`ffmpeg.py`), `draw_box`/`draw_hud`/`draw_trail` (`overlay.py`) |
| `vbt_core/pipeline/` | orchestration | `TrackingPipeline` + `run_tracking()` + `TrackResult` (`runner.py`) |
| `vbt_core/config/` | persisted plate diameters | `load_plate_config`/`save_plate_config` (`plate_config.py`) |

**The `Detector` interface (`vbt_core/detection/base.py`) is the extension
point.** Everything downstream of detection sees only the small `Detection` /
`FrameDetections` value types — never Ultralytics objects — so swapping YOLO for
a different detector, or adding a segmentation/mask backend later, means writing
one new `Detector` subclass and changing nothing else. Packaging: `pyproject.toml`
declares `vbt-core` (extras `[webapp]` → flask, `[test]` → pytest); the package is
installed **editable** into `venv/`.

> `data/` (datasets, raw/processed/tracked video, `plate_config.json`) is
> **local-only and gitignored** — not committed to the repo.

---

## 1 · Video ingestion (`scripts/normalize_videos.sh`)

| Property | Value |
|---|---|
| Resolution | shorter side capped at **1080 px**, aspect kept, never upscaled (phone portrait → **1080×1920**) |
| Frame rate | forced **30 fps** (`fps=30` filter) |
| Codec | H.264 — AMD VAAPI (`h264_vaapi`, qp 24) with CPU fallback (`libx264`, crf 23, yuv420p) |
| Audio | stripped (`-an`) |
| Flow | `data/raw/` → `data/processed/<date>/`; originals kept in `data/archive/<date>/` |

## 2 · Dataset (`data/datasets/plate_v1`, built by `scripts/ls_to_yolo.py`)

| Property | Value |
|---|---|
| Source | Label Studio video-tracking export (`data/annotations/*.json`), keyframe boxes **linearly interpolated** to every frame |
| Image resolution | **1080×1920** JPG (every frame of the normalized clips, full resolution) |
| Train split | **3,078 images** from 15 videos (`video_4…video_23`) |
| Val split | **463 images** from 3 videos (`video_3, video_9, video_12`) — split **by video**, not by frame, so val frames are never seen in training |
| Classes | `nc: 1` — `plate` (the barbell plate face, bbox ≈ the plate circle) |
| Label format | YOLO txt: `class cx cy w h`, normalized [0,1] |
| Note | `data/datasets/plate_v2.zip` exists but is **not extracted** in the repo; `plate_best.pt` was trained on Kaggle from a `data.kaggle.yaml` whose contents aren't recorded here — exact dataset version for that run is not reproducible from the repo alone |

## 3 · Model — YOLO11n (both checkpoints inspected with `torch.load`)

### Architecture (identical for both weights)

| Property | Value |
|---|---|
| Base | **YOLO11-nano** (`yolo11n.yaml`), fine-tuned from COCO-pretrained `yolo11n.pt` |
| Size | **182 layers, 2,590,035 parameters, 6.4 GFLOPs** (at 640) |
| Top-level blocks | 7× `Conv`, 8× `C3k2` (CSP bottlenecks), 1× `SPPF` (spatial pyramid pooling), 1× `C2PSA` (partial self-attention), 2× `Upsample` + 4× `Concat` (FPN/PAN neck), 1× `Detect` head |
| Head | anchor-free, decoupled cls/box, **DFL** box regression, 3 scales at strides **8 / 16 / 32** (P3/P4/P5) |
| Head outputs | single class, so detection = objectness×cls score + 4×16-bin DFL box per location |

### Precision / quantization

| Stage | Precision |
|---|---|
| Training | **AMP (mixed fp16/fp32)** on GPU (`amp: True`) |
| Stored weights | **fp16** (all parameters `torch.float16` in both `.pt` files) |
| Inference (webapp, CPU) | weights are cast up and run in **fp32** — PyTorch CPU inference doesn't use half; the pipeline runs the fp16 `.pt` only, and there is **no INT8/quantized model in the loop** (gitignored `models/plate_best.onnx` + `plate_best_saved_model/` exports exist but are unquantized and unused by the webapp) |
| Measured speed | ~**63 ms/frame** inference on this machine's CPU (≈16 fps, i.e. ~2× slower than real time for 30 fps video) |

### The two checkpoints

| | `plate_best.pt` (webapp default) | `best.pt` |
|---|---|---|
| Trained | 2026-06-11, **Kaggle** | 2026-06-10, **Colab** |
| Train imgsz | **640** | **960** |
| File | 5.4 MB (stripped best-epoch) | 21.3 MB (full ckpt: model+EMA+optimizer) |
| mAP50 / mAP50-95 (val) | 0.995 / **0.884** | 0.995 / 0.865 |
| Precision / recall | 0.998 / 0.999 | 1.000 / 1.000 |
| Color aug | hsv_h **0.3**, hsv_s 0.7, hsv_v 0.4 | hsv_h 0.015 (default) |
| Geometric aug | translate 0.1, scale 0.5, perspective 5e-4, **mosaic 0, fliplr 0** | mosaic 1.0, fliplr 0.5 |
| Optimizer | auto (SGD), lr0 0.01→lrf 0.01, momentum 0.937, wd 5e-4, warmup 3 ep, patience 20, 100 epochs | same recipe |

**Audit findings:**
- `scripts/train_colab.py` in the repo (hsv_h 0.5, mosaic 1.0, fliplr 0.5, Colab paths) does **not** match the args baked into `plate_best.pt` (Kaggle, hsv_h 0.3, mosaic 0, fliplr 0). The script is not the provenance of the production model — the actual Kaggle notebook/config isn't in the repo.
- `best.pt` was trained at **960** but the webapp infers at the Ultralytics default **640** — if you ever select `best.pt` in the UI you're feeding it 33% less resolution than it was trained for. `plate_best.pt` (640/640) is consistent.

## 4 · Inference input resolution (webapp, measured empirically)

| Step | Shape |
|---|---|
| Decoded frame (portrait 1080p clip) | 1080×1920 BGR |
| Ultralytics letterbox (`imgsz=640`, `auto=True`, stride 32) | longest side → 640, short side padded to a /32 multiple → network input **384×640×3** (≈8.5× area reduction) |
| Normalization | /255, RGB, NCHW fp32 |
| Detection floor | `DET_FLOOR=0.10` passed to `predict()` in `vbt_core/detection/yolo.py` (deliberately low — the tracker's second stage consumes 0.10–0.45 candidates) |

The ~390 px plate in the 1080-wide frame becomes ~140 px at network input — still far above what YOLO needs, which is why box jitter is mm-scale after smoothing.

## 5 · Detection → track pipeline (`vbt_core/tracking/`)

Custom tracker replacing Ultralytics ByteTrack (YOLO runs in pure `predict` mode
via `vbt_core/detection/yolo.py`). Files: `kalman.py` (filter),
`association.py` (IoU + Hungarian + gate), `track.py` (`Track`/`TrackSnapshot`),
`bar_tracker.py` (`BarTracker`, the two-stage association + primary-track pick):

### 5.1 Kalman filter (per track, hand-rolled NumPy — `kalman.py`)
- State (8-dim): `[cx, cy, a, h, vcx, vcy, va, vh]` — a = aspect w/h; constant-velocity model, dt = 1 frame.
- Noise scaled by box height (SORT/ByteTrack convention): σ_pos = h/20, σ_vel = h/160; measurement R uses h/20 (1e-1 for aspect).
- `predict()` / `update()` standard KF; smoothed state (not the raw detection) is what every downstream consumer sees.
- **Gating**: squared Mahalanobis distance of the innovation, gate = **χ²₀.₉₅(4 dof) = 9.4877** — a physically illogical jump fails the gate and can never capture the track. This (plus Hungarian) is what fixed the old "most-confident box wins" path glitches.

### 5.2 Hungarian assignment (`association.py`)
- `scipy.optimize.linear_sum_assignment` on cost = 1 − IoU(predicted box, detection).
- Pairs with IoU < threshold **or** failing the Mahalanobis gate get infinite cost (rejected even if "optimal").

### 5.3 ByteTrack-style two-stage association (`bar_tracker.py`)
- Stage 1: all tracks × **high-conf** detections (conf ≥ UI "min confidence", default 0.25, clamped ≥0.15), IoU gate 0.30.
- Stage 2: still-unmatched *confirmed* tracks × **low-conf** detections (0.10–high), looser IoU gate 0.20 — keeps the lock through motion blur/shadow at turnarounds.
- Births only from unmatched high-conf detections (ghost suppression); tentative → confirmed after **3 hits**; tentative dies after 2 missed frames; confirmed track **coasts on prediction up to `max_age` = 1 s of frames** before deletion (bridges occlusions, marked `coasting` in the CSV).
- **Primary track** (the bar) = max over run history of `(hits, avg_area)` — the camera-side plate wins on both.

## 6 · Calibration, reps, velocity (`vbt_core/analysis/`)

Files: `calibration.py` (`calibrate_m_per_px`), `rep_counter.py` (`RepCounter`
streaming FSM), `velocity.py` (`frame_velocity`/`active_span`), `trajectory.py`
(`analyze_trajectory` → `AnalysisResult`/`RepMetrics`, the canonical post-pass).

| Component | Algorithm |
|---|---|
| Pixel→metre | `m/px = plate_diameter / median(box (w+h)/2)` over non-coasting frames with conf ≥ 0.5 (≥5 samples; falls back to all detected frames). Plate diameter comes from the per-lift dropdown + saved per-plate sizes via `vbt_core/config/plate_config.py` (`data/plate_config.json`, overridable with `VBT_PLATE_CONFIG`; defaults 450/450/400/325/230/190 mm for 25→2.5 kg) |
| Velocity (metrics & chart) | **Savitzky–Golay derivative** of calibrated y: window = odd(0.25·fps) ≥ 5 (9 frames @30 fps), polyorder 2, `deriv=1` — frame-by-frame, sign flipped so +v = up |
| Velocity (live HUD) | Kalman state velocity `−v_cy · fps · m/px` (causal, free) |
| Rep segmentation | streaming FSM idle/down/up with **0.15 m displacement hysteresis** (phase flips only after the bar reverses 15 cm), **0.02 m dwell tolerance** so pauses (floor, lockout, rack) attach to eccentric/idle and never inflate the concentric, min concentric duration 0.25 s |
| Rep structure | **auto-detected** from first movement direction (down-first = squat/bench: ecc→conc; up-first = deadlift/row: conc→ecc, the *following* lowering belongs to the same rep), with manual override in the UI; rep count ticks at lockout in both |
| Metric windows | per-phase windows trimmed to the longest active run (|v| > 0.05 m/s, sub-threshold gaps < 0.6 s bridged so sticking points stay inside the rep) |
| Per-rep outputs | duration, ROM (m), mean+peak concentric velocity (MCV/PV — VBT standard), mean+peak eccentric |
| Set summary | avg/best/last MCV, **velocity-loss %** (1 − last/best) |

## 7 · Webapp & outputs

The webapp is now a **thin Flask UI** ("VBT Studio") over `vbt_core`: it does
upload/validation/HTTP and **delegates all CV to `vbt_core.run_tracking`** (and
`vbt_core.rendering.trim_clip` for trims, `vbt_core.config` for plate sizes). No
tracking/analysis code lives in `webapp/` anymore.

| Layer | Stack |
|---|---|
| Server | **Flask** (`webapp/app.py`, port 5000, debug reloader) — endpoints `/`, `/track` (multipart POST → `run_tracking`), `/plate-config` (GET/POST → `load`/`save_plate_config`), `/media/<session>/<file>` (HTTP range streaming). Same endpoints as before |
| Frontend | now split out (not one inline template): `templates/index.html` + `static/css/styles.css` + `static/js/app.js` — **vanilla JS**, professional product look; velocity chart is the same hand-rolled `<canvas>` drawer (phase bands, rep markers, dpr-aware) |
| Trim | `vbt_core.rendering.trim_clip` — ffmpeg re-encode of `[start,end]` (frame-accurate) |
| Video out | `vbt_core.rendering.FFmpegWriter`: raw BGR piped to **ffmpeg → libx264 yuv420p faststart** (browser-playable); 2 previews: boxed tracker debug + bar path (green-only or phase-colored trail, retroactively recolored at confirmed turnarounds) with live HUD (rep / m/s / phase) |
| Persisted per run (`data/tracked/<date>/`) | `<stem>_tracked.mp4`, `<stem>_barpath.mp4`, `<stem>_track.csv` (`frame,time_s,track_id,cx,cy,w,h,conf,coasting,primary,vy_mps` — **Kalman-smoothed** coords), `<stem>_metrics.json` (calibration, rep table, summary, full t/v/y/phase/rep series). All written by `vbt_core.pipeline.TrackingPipeline` |

## 8 · Environment

| | |
|---|---|
| Python | venv, **ultralytics 8.4.64**, **torch 2.12.0+cpu**, scipy 1.17.1, numpy 2.4.6, OpenCV, Flask |
| Inference device | CPU (no CUDA torch installed locally); training happens off-machine (Colab T4 / Kaggle) |
| Video tooling | ffmpeg/ffprobe, AMD VAAPI for normalization encodes |
