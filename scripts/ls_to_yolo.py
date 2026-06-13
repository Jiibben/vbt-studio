#!/usr/bin/env python3
"""
Convert a Label Studio video-object-tracking export (full JSON) into a YOLO
detection dataset (frames + per-frame .txt labels), interpolating the box
between keyframes exactly like the Label Studio UI does.

Usage (from the project root):
    python scripts/ls_to_yolo.py \
        --export data/annotations/2026-06-10.json \
        --videos data/processed/2026-06-10 \
        --out    data/datasets/plate_v1 \
        --val-frac 0.2

Notes:
- Frames are extracted from the videos in --videos. Those MUST be the same
  clips that were annotated (same fps/resolution) so frame numbers and box
  percentages line up — i.e. your data/processed/<date>/ folder.
- The train/val split is done PER VIDEO (a whole clip goes to train or val) so
  near-identical neighbouring frames don't leak across the split.
"""
import argparse, json, os, re, random
from pathlib import Path
import cv2

CLASSES = ["plate"]  # single class; index 0


def find_video(data_value, videos_dir):
    """Map a Label Studio data path to a local file in videos_dir by its
    trailing 'video_N.mp4' name (strips any upload hash prefix)."""
    base = os.path.basename(str(data_value))
    base = re.sub(r"^[0-9a-fA-F]{6,}-", "", base)  # drop 'a1b2c3d4-' upload prefix
    cand = Path(videos_dir) / base
    if cand.exists():
        return cand
    # fallback: match by the video_N stem anywhere in the name
    m = re.search(r"(video_\d+)\.mp4", base)
    if m:
        for p in Path(videos_dir).glob("*.mp4"):
            if m.group(1) in p.name:
                return p
    return None


def interp_boxes(sequence, frames_count):
    """Return {frame_index(1-based): (x,y,w,h) in 0..1} interpolated between
    keyframes. x,y are the box TOP-LEFT corner; values are fractions."""
    kfs = sorted(sequence, key=lambda k: k["frame"])
    out = {}

    def frac(kf):
        return (kf["x"] / 100.0, kf["y"] / 100.0,
                kf["width"] / 100.0, kf["height"] / 100.0)

    for a, b in zip(kfs, kfs[:0:-1][::-1] if False else kfs[1:]):
        if not a.get("enabled", True):
            # span starting at a disabled keyframe = object hidden here
            continue
        ax, ay, aw, ah = frac(a)
        bx, by, bw, bh = frac(b)
        span = b["frame"] - a["frame"]
        for f in range(a["frame"], b["frame"]):           # [a, b)
            t = 0.0 if span == 0 else (f - a["frame"]) / span
            out[f] = (ax + (bx - ax) * t, ay + (by - ay) * t,
                      aw + (bw - aw) * t, ah + (bh - ah) * t)
    last = kfs[-1]
    if last.get("enabled", True):
        out[last["frame"]] = frac(last)
    return out


def to_yolo_line(box):
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
    w, h = min(max(w, 0), 1), min(max(h, 0), 1)
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def build_dataset(sources, out, val_frac=0.2, seed=0, log=print):
    """Build a YOLO dataset from one or more (export_json, videos_dir) pairs.

    sources : list of (export_path, videos_dir) tuples.
    out     : output dataset directory.
    log     : callable for progress lines (defaults to print). The Qt wizard
              passes a callback that appends to its log panel.
    The train/val split is done ONCE over the combined set of videos, so
    multiple sources can't leak across the split or get an inconsistent split.
    """
    random.seed(seed)

    # collect per-video {frame: [boxes]} across all sources
    per_video = {}
    frames_count_by_video = {}
    for export_path, videos_dir in sources:
        tasks = json.load(open(export_path))
        for task in tasks:
            vid = find_video(task["data"].get("video") or next(iter(task["data"].values())),
                             videos_dir)
            if vid is None:
                log(f"!! could not find video for task {task.get('id')} in {videos_dir}: {task['data']}")
                continue
            anns = task.get("annotations") or []
            if not anns:
                continue
            boxes_by_frame = per_video.setdefault(vid, {})
            for res in anns[0].get("result", []):
                if res.get("type") != "videorectangle":
                    continue
                seq = res["value"]["sequence"]
                fc = res["value"].get("framesCount", max(k["frame"] for k in seq))
                frames_count_by_video[vid] = fc
                for f, box in interp_boxes(seq, fc).items():
                    boxes_by_frame.setdefault(f, []).append(box)

    vids = list(per_video.keys())
    random.shuffle(vids)
    n_val = max(1, int(len(vids) * val_frac)) if vids else 0
    val_set = set(vids[:n_val])

    out = Path(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_img = 0
    for vid, boxes_by_frame in per_video.items():
        split = "val" if vid in val_set else "train"
        cap = cv2.VideoCapture(str(vid))
        real = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = frames_count_by_video.get(vid)
        if want and abs(real - want) > 1:
            log(f"!! SKIP {vid.name}: annotated framesCount={want} but file has "
                f"{real} frames — wrong/un-normalized clip, frames won't align.")
            cap.release()
            continue
        idx0 = 0  # OpenCV frame index (0-based)
        ok, frame = cap.read()
        while ok:
            ls_frame = idx0 + 1  # Label Studio frames are 1-based
            if ls_frame in boxes_by_frame:
                stem = f"{vid.stem}_f{ls_frame:05d}"
                cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), frame)
                lines = [to_yolo_line(b) for b in boxes_by_frame[ls_frame]]
                (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                n_img += 1
            idx0 += 1
            ok, frame = cap.read()
        cap.release()
        log(f"{split:5s}  {vid.name}: {len(boxes_by_frame)} labelled frames")

    yaml = (f"path: {out.resolve()}\n"
            f"train: images/train\nval: images/val\n"
            f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
    (out / "data.yaml").write_text(yaml)
    log(f"\nWrote {n_img} images. Dataset at {out.resolve()}")
    log(f"  train videos: {len(vids) - n_val} | val videos: {n_val}")
    return {"n_images": n_img, "train_videos": len(vids) - n_val, "val_videos": n_val}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", nargs="+", required=True,
                    help="one or more Label Studio export JSON files")
    ap.add_argument("--videos", nargs="+", required=True,
                    help="videos dir for each --export (same order), or a single "
                         "dir searched for all exports")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if len(args.videos) == 1:
        sources = [(e, args.videos[0]) for e in args.export]
    elif len(args.videos) == len(args.export):
        sources = list(zip(args.export, args.videos))
    else:
        ap.error("--videos must be a single dir or have the same count as --export")

    build_dataset(sources, args.out, args.val_frac, args.seed)


if __name__ == "__main__":
    main()
