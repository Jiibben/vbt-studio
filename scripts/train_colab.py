#!/usr/bin/env python3
"""
Bombproof YOLO11 training for the VBT barbell-plate detector — built for Colab T4.

What it does:
  * Saves everything to Google Drive, so a disconnect never loses progress.
  * AUTO-RESUMES: run the same cell again after a timeout and it picks up from
    the last checkpoint (no flags to remember).
  * Heavy colour augmentation so it generalises to plate colours you never
    labelled (blue / black / etc) without collecting more data.
  * Falls back to a smaller batch automatically on CUDA out-of-memory.

How to use in Colab (one time, in earlier cells):
    !pip -q install ultralytics
    from google.colab import drive; drive.mount('/content/drive')
    !unzip -q "/content/drive/MyDrive/dataset.zip" -d /content
Then run:        !python train_colab.py
Timed out?       just run  !python train_colab.py  again — it resumes.
"""
import os, sys, glob, shutil
import torch
from ultralytics import YOLO

# ----------------------------- config (edit if needed) -----------------------
DATA    = "/content/dataset/data.yaml"               # dataset produced by ls_to_yolo.py
PROJECT = "/content/drive/MyDrive/vbt_runs"          # lives on Drive -> survives disconnects
NAME    = "plate_detector"
MODEL   = "yolo11n.pt"                               # 'yolo11s.pt' if accuracy is low
EPOCHS  = 100
IMGSZ   = 640                                        # plate is large; 640 trains fast & is plenty
BATCH   = 16                                         # auto-drops to 8 on OOM

# Augmentation tuned for COLOUR GENERALISATION (the whole point of your question):
#   hsv_h high  -> randomise hue across the full wheel  => colour-invariant (red/blue/green plates)
#   hsv_s / hsv_v -> desaturate & darken                 => approximates faded / black plates
AUG = dict(
    hsv_h=0.5,      # default 0.015 — deliberately huge so colour stops mattering
    hsv_s=0.7,
    hsv_v=0.5,
    degrees=5.0,    # mild geometric variety
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
)
# -----------------------------------------------------------------------------

RUN_DIR = os.path.join(PROJECT, NAME)
LAST    = os.path.join(RUN_DIR, "weights", "last.pt")
BEST    = os.path.join(RUN_DIR, "weights", "best.pt")


def preflight():
    if not torch.cuda.is_available():
        print("!! No GPU detected. In Colab: Runtime > Change runtime type > T4 GPU.")
        sys.exit(1)
    print(">> GPU:", torch.cuda.get_device_name(0))
    if not os.path.exists("/content/drive/MyDrive"):
        print("!! Google Drive not mounted. Run:\n"
              "   from google.colab import drive; drive.mount('/content/drive')")
        sys.exit(1)
    if not os.path.exists(DATA):
        print(f"!! Dataset not found at {DATA}. Did you unzip dataset.zip to /content ?")
        sys.exit(1)
    os.makedirs(PROJECT, exist_ok=True)


def train_kwargs(batch):
    return dict(
        data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=batch,
        project=PROJECT, name=NAME, exist_ok=True,
        patience=20,            # early-stop once it plateaus
        save_period=5,          # checkpoint every 5 epochs -> minimal loss on timeout
        cache=False,            # RAM cache can crash Colab on 1080p frames; keep it safe
        **AUG,
    )


def main():
    preflight()

    # ---- resume a timed-out run automatically ----
    if os.path.exists(LAST):
        print(f">> Found checkpoint, RESUMING: {LAST}")
        try:
            YOLO(LAST).train(resume=True)
        except Exception as e:
            # resume can fail if the run finished or args changed; continue/eval gracefully
            print(f"!! resume failed ({e}). Treating existing weights as final.")
    else:
        # ---- fresh run, with OOM fallback ----
        model = YOLO(MODEL)
        try:
            model.train(**train_kwargs(BATCH))
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"!! CUDA OOM at batch={BATCH} -> retrying at batch=8")
                YOLO(MODEL).train(**train_kwargs(8))
            else:
                raise

    # ---- copy best.pt somewhere easy to grab ----
    if os.path.exists(BEST):
        out = "/content/drive/MyDrive/plate_best.pt"
        shutil.copy(BEST, out)
        print(f"\n>> DONE. Best weights: {BEST}\n>> Also copied to: {out}")
    else:
        print("\n>> Training ended but no best.pt found yet — re-run to continue.")


if __name__ == "__main__":
    main()
