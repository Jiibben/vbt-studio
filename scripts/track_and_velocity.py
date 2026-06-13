#!/usr/bin/env python3
"""
track_and_velocity.py  —  PART 1: TRACKING (velocity comes later)

Goal of this stage: take the trained detector (best.pt) and a barbell video,
follow the plate across every frame with a tracker, and produce:
  1. an annotated .mp4 you can watch to confirm the box stays locked on the plate
  2. a .csv of the raw per-frame trajectory (frame, track id, box centre, size, conf)
     -> this CSV is exactly the input the velocity stage will consume next.

Why a *tracker* and not just the detector?
  The detector finds boxes independently on each frame with no memory. A tracker
  (ByteTrack here) links those per-frame boxes into consistent "tracks" with a
  stable ID, and bridges short gaps where a detection is briefly missed. That
  continuity is what lets us later read one smooth plate trajectory over time.

Run:
    python track_and_velocity.py --weights best.pt --source some_video.mp4
"""
import argparse
import csv
from pathlib import Path

import cv2                      # only used to read fps / write the output video
from ultralytics import YOLO    # the detector + built-in tracking


def parse_args():
    ap = argparse.ArgumentParser(description="Track the barbell plate in a video.")
    ap.add_argument("--weights", required=True,
                    help="Path to trained YOLO weights, e.g. best.pt")
    ap.add_argument("--source", required=True,
                    help="Input video (ideally a VAL clip the model never trained on)")
    ap.add_argument("--outdir", default=None,
                    help="Output folder (default: data/tracked/<date>/, mirroring the "
                         "processed clip's date folder)")
    ap.add_argument("--out", default=None,
                    help="Explicit annotated-video path (overrides --outdir)")
    ap.add_argument("--csv", default=None,
                    help="Explicit trajectory-CSV path (overrides --outdir)")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="Ultralytics tracker config: 'bytetrack.yaml' or 'botsort.yaml'")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Min detection confidence (lower = more boxes, more false positives)")
    return ap.parse_args()


def main():
    args = parse_args()

    src = Path(args.source)

    # Outputs go to data/tracked/, NOT next to the source clip (keeps processed/
    # clean). If the source lives in data/processed/<date>/, mirror that <date>
    # subfolder under data/tracked/ so runs stay organised by session.
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        root = Path(__file__).resolve().parent.parent          # project root
        outdir = root / "data" / "tracked"
        if src.parent.parent.name == "processed":              # .../processed/<date>/clip.mp4
            outdir = outdir / src.parent.name
    outdir.mkdir(parents=True, exist_ok=True)

    out_video = Path(args.out) if args.out else outdir / (src.stem + "_tracked.mp4")
    out_csv = Path(args.csv) if args.csv else outdir / (src.stem + "_track.csv")

    # --- Read the video's frame rate up front -------------------------------
    # We need the source fps so the annotated output plays back at real speed,
    # and (later) so velocity is in real seconds. We do NOT loop frames here —
    # YOLO's tracker reads the video itself below.
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0   # our clips are 30 fps
    cap.release()
    print(f">> source={src.name}  fps={fps:.3f}")

    # --- Load the trained model --------------------------------------------
    model = YOLO(args.weights)

    # --- Run tracking -------------------------------------------------------
    # stream=True  -> yields results frame-by-frame (constant memory, works on
    #                 long videos instead of loading everything at once).
    # persist=True -> keep track IDs alive across frames (this is what makes it
    #                 "tracking" rather than independent per-frame detection).
    # tracker=...  -> ByteTrack associates boxes between frames + bridges gaps.
    results = model.track(
        source=str(src),
        tracker=args.tracker,
        persist=True,
        stream=True,
        conf=args.conf,
        verbose=False,
    )

    writer = None          # cv2.VideoWriter, created lazily once we know frame size
    rows = []              # collected trajectory rows for the CSV
    n_frames = 0
    n_with_box = 0

    # Iterate over the per-frame tracking results.
    for frame_idx, r in enumerate(results):
        # r.plot() returns the frame as a BGR image with boxes, track IDs and
        # confidences already drawn on it — the simplest reliable visualization.
        annotated = r.plot()

        # Create the video writer on the first frame, now that we know the size.
        if writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))
        writer.write(annotated)
        n_frames += 1

        # --- Pull the raw numbers out for the CSV ---------------------------
        # r.boxes holds this frame's detections. With tracking on, each box may
        # carry an integer track id. xywh gives [centre_x, centre_y, width,
        # height] in PIXELS — centre_x/centre_y is the plate position we will
        # later turn into displacement -> velocity.
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            n_with_box += 1
            ids = boxes.id                       # tensor of track ids, or None if unassigned
            for i in range(len(boxes)):
                cx, cy, bw, bh = boxes.xywh[i].tolist()
                conf = float(boxes.conf[i])
                tid = int(ids[i]) if ids is not None else -1
                rows.append({
                    "frame": frame_idx,
                    "time_s": frame_idx / fps,   # real timestamp, ready for velocity
                    "track_id": tid,
                    "cx": round(cx, 2),
                    "cy": round(cy, 2),
                    "w": round(bw, 2),
                    "h": round(bh, 2),
                    "conf": round(conf, 4),
                })

    if writer is not None:
        writer.release()

    # --- Write the trajectory CSV ------------------------------------------
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "time_s", "track_id",
                                          "cx", "cy", "w", "h", "conf"])
        w.writeheader()
        w.writerows(rows)

    # --- Report -------------------------------------------------------------
    # A quick health check: how often did we actually have a box, and how many
    # distinct track ids appeared. Ideally ONE id covers (almost) every frame.
    track_ids = sorted({row["track_id"] for row in rows})
    coverage = (n_with_box / n_frames * 100) if n_frames else 0
    print(f">> frames processed : {n_frames}")
    print(f">> frames with a box: {n_with_box} ({coverage:.1f}% coverage)")
    print(f">> distinct track ids: {track_ids}  (want this to be basically one id)")
    print(f">> annotated video  : {out_video}")
    print(f">> trajectory csv   : {out_csv}")
    print(">> NEXT: watch the video; the box should stay glued to the plate "
          "with no flicker or id-switches.")

    # =======================================================================
    # TODO (PART 2 — VELOCITY, not implemented yet):
    #   - pick the main plate track from the CSV
    #   - calibrate pixels->metres using the known plate diameter (0.45 m) vs box h
    #   - differentiate cy over time_s to get vertical velocity (m/s)
    #   - smooth, then report peak / mean concentric velocity and plot the curve
    # =======================================================================


if __name__ == "__main__":
    main()
