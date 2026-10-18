#!/bin/bash
# Train a 3DGS reconstruction from a multi-view dataset.
# Usage: bash tools/fracsim/gs_recon/train_3dgs.sh <dataset_dir> <output_model_dir> [iterations]
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SRC="$(cd "${1:?dataset dir (transforms_*.json or COLMAP)}" && pwd)"
OUT="${2:?output model dir}"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
ITERS="${3:-30000}"
cd "${REPO_ROOT}/src/fracsim/gaussian-splatting"
python train.py -s "$SRC" --iterations "$ITERS" --model_path "$OUT"
