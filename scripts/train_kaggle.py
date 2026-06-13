#!/usr/bin/env python3
"""
YOLO11 training for the VBT barbell-plate detector — built for KAGGLE GPU.

Why a Kaggle-specific script (vs train_colab.py):
  * Kaggle datasets live under /kaggle/input/<slug>/ (read-only) and the
    data.yaml that ls_to_yolo.py wrote has an ABSOLUTE `path:` from your laptop,
    which is wrong here. This script auto-finds the dataset and rewrites a
    correct data.yaml into /kaggle/working (the only writable place).
  * Outputs go to /kaggle/working so they survive as the notebook's "Output".
  * Within a session it AUTO-RESUMES from last.pt if you re-run after a crash.

Augmentation philosophy (the whole point of this run — generalising to OTHER
gyms/plates without collecting much new data). We keep ONLY the augmentations
that simulate the real distribution shifts you hit, and switch the rest off:

    KEEP — high yield, each maps to a real-world difference between gyms:
      hsv_v  brightness   -> different gym lighting (the #1 shift)
      hsv_h  hue          -> different plate colours / camera white-balance
      hsv_s  saturation   -> washed-out vs vivid footage
      scale  zoom in/out  -> camera nearer/further from the bar
      translate  shift     -> bar framed differently in shot
      perspective (tiny)   -> phone mounted at different heights/angles; a round
                              plate projects as a squashed ellipse. Kept at 0.0005.

    OFF — either unrealistic for a barbell or the "weird nonsense" you flagged:
      fliplr / flipud      -> you asked to drop flips
      degrees / shear      -> a loaded bar is ~level; big warps hurt
      mosaic               -> 4-image collages; high yield in general but does
                              NOT target your shifts. Behind --mosaic if wanted.
      mixup / copy_paste / erasing -> noisy for a single-class small dataset

Usage on Kaggle (in a notebook cell):
    !pip -q install ultralytics
    !python train_kaggle.py                 # auto-finds the dataset in /kaggle/input
    # or be explicit:
    !python train_kaggle.py --data /kaggle/input/plate-v1
Re-run the same cell after a crash to resume.
"""
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    sys.exit("!! ultralytics/torch not installed. Run:  !pip -q install ultralytics")


# ----------------------------- augmentation ----------------------------------
# Curated for maximum generalisation-yield per the docstring. Edit here.
def aug_dict(mosaic: float):
    return dict(
        # --- photometric: the cheap, high-yield wins ---
        hsv_h=0.3,        # hue: plate-colour / white-balance invariance (default 0.015)
        hsv_s=0.7,        # saturation: vivid <-> washed out
        hsv_v=0.4,        # brightness/value: bright gym <-> dim gym  (lighting = top shift)
        # --- geometric: only the realistic ones you named ---
        scale=0.5,        # zoom in/out: camera distance to the bar
        translate=0.1,    # slight shift: different framing in the shot
        perspective=0.0005,  # camera angle/height: a circle plate -> squashed ellipse.
                             # REAL shift (unlike flips). Kept tiny on purpose (range 0..0.001).
        # --- explicitly OFF (no "weird nonsense") ---
        degrees=0.0, shear=0.0,
        fliplr=0.0, flipud=0.0,
        mixup=0.0, copy_paste=0.0, erasing=0.0,
        mosaic=mosaic,    # 0.0 by default; --mosaic turns it on
    )


# ----------------------------- dataset wiring --------------------------------
def find_dataset_root(hint: str | None) -> Path:
    """Locate a dataset root that contains images/train. Searches the hint first
    (a dir or a data.yaml), then all of /kaggle/input."""
    bases = []
    if hint:
        h = Path(hint)
        bases.append(h.parent if h.suffix == ".yaml" else h)
    bases += [Path(p) for p in glob.glob("/kaggle/input/*")]
    bases.append(Path("."))  # local fallback so the script also runs off-Kaggle

    for base in bases:
        if not base.exists():
            continue
        # base itself, or any dir beneath it, that holds images/train
        for cand in [base, *[p.parent.parent for p in base.glob("**/images/train")]]:
            if (cand / "images" / "train").is_dir():
                return cand.resolve()
    sys.exit("!! Could not find a dataset (a folder containing images/train) under "
             f"{hint or '/kaggle/input'}. Add your dataset to the notebook first.")


def write_datayaml(root: Path) -> Path:
    """Write a correct data.yaml into /kaggle/working pointing at `root`. The
    one shipped in the dataset has a stale absolute path from the laptop."""
    val = "images/val" if (root / "images" / "val").is_dir() else "images/train"
    if val == "images/train":
        print("!! No images/val found — using train as val (metrics will be optimistic).")
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    yaml_path = out_dir / "data.kaggle.yaml"
    yaml_path.write_text(
        f"path: {root}\n"
        f"train: images/train\n"
        f"val: {val}\n"
        f"nc: 1\n"
        f"names: ['plate']\n"
    )
    print(f">> dataset root : {root}")
    print(f">> data.yaml    : {yaml_path}")
    return yaml_path


# ----------------------------- training --------------------------------------
def train_kwargs(data, args, batch):
    return dict(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=batch,
        project=args.project, name=args.name, exist_ok=True,
        patience=20,        # early-stop on plateau
        save_period=5,      # checkpoint every 5 epochs -> survive a Kaggle timeout
        cache=False,        # RAM-cache can OOM on big frames; stay safe
        **aug_dict(args.mosaic),
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Train the plate detector on Kaggle.")
    ap.add_argument("--data", default=None,
                    help="Dataset folder or data.yaml. Default: auto-find in /kaggle/input.")
    ap.add_argument("--model", default="yolo11n.pt", help="Base weights (yolo11s.pt if accuracy low)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Train resolution. 640 is plenty if the plate is large; "
                         "raise to 960/1280 if it's small/far (and infer at the same size).")
    ap.add_argument("--batch", type=int, default=16, help="Auto-drops to 8 on CUDA OOM.")
    ap.add_argument("--project", default="/kaggle/working/vbt_runs")
    ap.add_argument("--name", default="plate_detector")
    ap.add_argument("--mosaic", type=float, nargs="?", const=1.0, default=0.0,
                    help="Enable mosaic (e.g. --mosaic or --mosaic 0.5). Off by default.")
    return ap.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("!! No GPU. On Kaggle: Notebook settings > Accelerator > GPU T4 x2 (or P100).")
        sys.exit(1)
    print(">> GPU:", torch.cuda.get_device_name(0))
    if args.mosaic:
        # best practice: drop mosaic for the final epochs so it fine-tunes on real images
        print(f">> mosaic ENABLED at {args.mosaic} (will close for last 10 epochs)")

    data = write_datayaml(find_dataset_root(args.data))

    run_dir = Path(args.project) / args.name
    last = run_dir / "weights" / "last.pt"
    best = run_dir / "weights" / "best.pt"

    # ---- auto-resume a crashed/timed-out run ----
    if last.exists():
        print(f">> Found checkpoint, RESUMING: {last}")
        try:
            YOLO(str(last)).train(resume=True)
        except Exception as e:
            print(f"!! resume failed ({e}). Treating existing weights as final.")
    else:
        # ---- fresh run with OOM fallback ----
        extra = dict(close_mosaic=10) if args.mosaic else {}
        try:
            YOLO(args.model).train(**train_kwargs(data, args, args.batch), **extra)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"!! CUDA OOM at batch={args.batch} -> retrying at batch=8")
                YOLO(args.model).train(**train_kwargs(data, args, 8), **extra)
            else:
                raise

    # ---- surface best.pt at a predictable path for download ----
    if best.exists():
        out = "/kaggle/working/plate_best.pt"
        shutil.copy(best, out)
        print(f"\n>> DONE. Best weights: {best}\n>> Download from: {out}")
    else:
        print("\n>> Training ended but no best.pt found — re-run the cell to continue.")


if __name__ == "__main__":
    main()
