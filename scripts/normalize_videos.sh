#!/usr/bin/env bash
#
# normalize_videos.sh — turn ANY dropped video into a clean training clip:
#   1080p (long side capped, aspect kept) · 30 fps · H.264 · faststart · no audio
#
# Uses your AMD GPU (VAAPI / renderD128) for the H.264 encode, and falls back to
# CPU (libx264) automatically if the GPU path isn't available.
#
# WORKFLOW
#   1. Drop video files into   data/raw/
#   2. Run                      bash scripts/normalize_videos.sh            (uses today's date)
#      or                       bash scripts/normalize_videos.sh 2026-06-10 (force a date)
#   3. Normalized clips appear in  data/processed/<DATE>/
#      The original is moved to    data/archive/<DATE>/   (kept, not deleted)
#
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
RAW="$ROOT/data/raw"
DATE="${1:-$(date +%F)}"                 # YYYY-MM-DD; default = today
OUT="$ROOT/data/processed/$DATE"
ARCHIVE="$ROOT/data/archive/$DATE"
VAAPI_DEV="/dev/dri/renderD128"          # AMD render node
TARGET=1080                              # cap the SHORTER side at this many px

mkdir -p "$OUT" "$ARCHIVE"

# Gather droppable videos (case-insensitive extensions, empty if none).
shopt -s nullglob nocaseglob
files=("$RAW"/*.mp4 "$RAW"/*.mov "$RAW"/*.avi "$RAW"/*.mkv "$RAW"/*.m4v)
shopt -u nocaseglob
if [ ${#files[@]} -eq 0 ]; then
  echo "Nothing to do — drop video files into $RAW and re-run."
  exit 0
fi

# Use the GPU only if the render node exists; otherwise CPU.
if [ -e "$VAAPI_DEV" ]; then GPU=1; echo ">> AMD GPU encode (VAAPI) via $VAAPI_DEV"
else GPU=0; echo "!! $VAAPI_DEV not found — using CPU (libx264)"; fi

encode_gpu() {  # $1=in $2=out $3=scalefilter
  ffmpeg -hide_banner -loglevel error -y \
    -vaapi_device "$VAAPI_DEV" -i "$1" \
    -vf "fps=30,$3,format=nv12,hwupload" \
    -c:v h264_vaapi -qp 24 -movflags +faststart -an "$2"
}
encode_cpu() {  # $1=in $2=out $3=scalefilter
  ffmpeg -hide_banner -loglevel error -y -i "$1" \
    -vf "fps=30,$3" \
    -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
    -movflags +faststart -an "$2"
}

for f in "${files[@]}"; do
  base="$(basename "$f")"
  out="$OUT/${base%.*}.mp4"
  if [ -e "$out" ]; then echo "SKIP $base (already in $DATE/)"; continue; fi

  # Orientation-aware downscale: cap the shorter side at TARGET, keep aspect,
  # force even dimensions (-2), and never upscale a clip that's already smaller.
  IFS=',' read -r w h < <(ffprobe -v error -select_streams v:0 \
                  -show_entries stream=width,height -of csv=p=0 "$f")
  if [ "$w" -le "$h" ]; then                                  # portrait / square
    if [ "$w" -gt "$TARGET" ]; then scale="scale=$TARGET:-2"; else scale="scale=$w:$h"; fi
  else                                                         # landscape
    if [ "$h" -gt "$TARGET" ]; then scale="scale=-2:$TARGET"; else scale="scale=$w:$h"; fi
  fi

  echo "ENCODE $base (${w}x${h}) -> $DATE/  [$([ "$GPU" = 1 ] && echo GPU || echo CPU)]"
  if [ "$GPU" = 1 ]; then
    if ! encode_gpu "$f" "$out" "$scale"; then
      echo "   GPU encode failed — retrying on CPU"; encode_cpu "$f" "$out" "$scale"
    fi
  else
    encode_cpu "$f" "$out" "$scale"
  fi

  mv -f "$f" "$ARCHIVE/$base"            # preserve original, clear the drop zone
  echo "   done -> $out  (original archived in $DATE/)"
done

echo "ALL DONE. Normalized clips: $OUT"
