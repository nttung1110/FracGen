#!/bin/bash
# FracSim demo simulation for bread: RGB fracture video + physics maps.
# Usage: bash scripts/fracsim/run_bread_phys_map.sh [output_dir]
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
OUT="${1:-outputs/fracsim/bread_exp}"

python src/fracsim/simulate.py \
    --model_path  data/fracsim/gaussians/bread \
    --config      src/fracsim/configs/tear_bread_config_exp.json \
    --output_path "$OUT" \
    --render_img --compile_video --white_bg \
    --render_width 512 --render_height 512
