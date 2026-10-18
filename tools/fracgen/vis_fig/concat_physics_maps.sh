#!/usr/bin/env bash
# Concatenate reference + strain + stress + damage + flow + fracture horizontally
# into one side-by-side comparison video for a single sample.
#
# Usage:
#   ./concat_physics_maps.sh <reference_video> <phys_map_dir> <sample_name> <out_path>
#
# <phys_map_dir> must contain strain/, stress/, damage/, flow/, fracture/ subfolders,
# each holding "<sample_name>.mp4" (this is the standard infer_step-* output layout).
# <reference_video> is the intact/undamaged reference clip (e.g. from
# benchmarks/<set>/reference_video/<sample_name>.mp4, or gt_vid/videos/<sample_name>.mp4
# for benchmarks that have real PhysGaussian ground truth).
#
# The reference clip is scaled to 480x480/15fps and looped to cover the full
# 81-frame (5.4s) length of the prediction outputs, since reference clips are
# often shot at a different resolution/fps/duration (e.g. 1024x1024 @ 24fps @ 2.5s).

set -euo pipefail

REF="$1"
MAP_DIR="$2"
NAME="$3"
OUT="$4"

FF="${FFMPEG:-ffmpeg}"

"$FF" -y -v error \
  -stream_loop -1 -i "$REF" \
  -i "$MAP_DIR/strain/${NAME}.mp4" \
  -i "$MAP_DIR/stress/${NAME}.mp4" \
  -i "$MAP_DIR/damage/${NAME}.mp4" \
  -i "$MAP_DIR/flow/${NAME}.mp4" \
  -i "$MAP_DIR/fracture/${NAME}.mp4" \
  -filter_complex "[0:v]scale=480:480,fps=15,setsar=1[ref];[1:v]setsar=1[a];[2:v]setsar=1[b];[3:v]setsar=1[c];[4:v]setsar=1[d];[5:v]setsar=1[e];[ref][a][b][c][d][e]hstack=inputs=6[outv]" \
  -map "[outv]" -frames:v 81 -an -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart \
  "$OUT"

echo "Saved: $OUT"
