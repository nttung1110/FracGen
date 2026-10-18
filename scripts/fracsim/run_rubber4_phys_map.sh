#!/bin/bash
# FracSim demo simulation for rubber4: RGB fracture video + physics maps.
# Usage: bash scripts/fracsim/run_rubber4_phys_map.sh [output_dir]
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
OUT="${1:-outputs/fracsim/rubber4_exp}"

python src/fracsim/simulate.py \
    --model_path  data/fracsim/gaussians/rubber4 \
    --config      src/fracsim/configs/tear_rubber4_config_exp.json \
    --output_path "$OUT" \
    --render_img --compile_video --white_bg \
    --render_width 512 --render_height 512
