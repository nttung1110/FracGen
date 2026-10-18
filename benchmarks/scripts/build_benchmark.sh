#!/bin/bash
# Build the fracture-video benchmark from benchmarks/data (imgs/ + annotations/).
#
# Usage:
#   bash benchmarks/scripts/build_benchmark.sh
#   DATA_ROOT=benchmarks/data OUTPUT_DIR=benchmarks/output/my_run FRAMES=81 FPS=25 \
#     bash benchmarks/scripts/build_benchmark.sh
#   OBJECT_IDS=pikachu bash benchmarks/scripts/build_benchmark.sh   # build a subset only
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-benchmarks/data}"          # must contain imgs/ and annotations/
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/output}"
FRAMES="${FRAMES:-81}"
FPS="${FPS:-25}"
SIGMA="${SIGMA:-20.0}"
ALPHA="${ALPHA:-0.85}"
DIM_BG="${DIM_BG:-0.45}"
OBJECT_IDS="${OBJECT_IDS:-}"                       # comma-separated subset, empty = all

EXTRA_ARGS=()
if [ -n "$OBJECT_IDS" ]; then
  EXTRA_ARGS+=(--object-ids "$OBJECT_IDS")
fi

python benchmarks/build_benchmark.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --frames "$FRAMES" \
  --fps "$FPS" \
  --sigma "$SIGMA" \
  --alpha "$ALPHA" \
  --dim-bg "$DIM_BG" \
  "${EXTRA_ARGS[@]}"
