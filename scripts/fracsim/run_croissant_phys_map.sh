#!/bin/bash
# FracSim demo simulation for croissant: RGB fracture video + physics maps.
# Usage: bash scripts/fracsim/run_croissant_phys_map.sh [output_dir]
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
OUT="${1:-outputs/fracsim/croissant_exp}"

python src/fracsim/simulate.py \
    --model_path  data/fracsim/gaussians/croissant \
    --config      src/fracsim/configs/tear_croissant_config_exp.json \
    --output_path "$OUT" \
    --render_img --compile_video --white_bg \
    --render_width 512 --render_height 512
