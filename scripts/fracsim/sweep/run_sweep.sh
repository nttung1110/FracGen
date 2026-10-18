#!/bin/bash
# Run FracSim over a range of generated sweep configs for one object on one GPU.
#   PREFIX=rubber_toy START_ID=0 END_ID=1000 GPU=0 bash scripts/fracsim/sweep/run_sweep.sh
# Configs are read from $CONFIG_DIR/<PREFIX>_config_%04d.json (made by
# tools/fracsim/gen_sweep_configs_with_material_property.py); outputs go to
# $OUT_ROOT/<PREFIX>_config_%04d/. Runs whose mp4 exists are skipped, so it resumes.
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PREFIX="${PREFIX:?set PREFIX, e.g. bread or rubber_toy}"
START_ID="${START_ID:-0}"
END_ID="${END_ID:-3000}"
GPU="${GPU:-0}"
MODEL_PATH="${MODEL_PATH:-data/fracsim/gaussians/${PREFIX}}"
CONFIG_DIR="${CONFIG_DIR:-data/fracsim/sweep_configs}"
OUT_ROOT="${OUT_ROOT:-data/fracsim/sim_output}"

shopt -s nullglob
for ((id=START_ID; id<=END_ID; id++)); do
    printf -v name "%s_config_%04d" "$PREFIX" "$id"
    cfg="${CONFIG_DIR}/${name}.json"
    [[ -f "$cfg" ]] || { echo "Skipping missing $cfg"; continue; }
    out_path="${OUT_ROOT}/${name}"
    mp4_files=("${out_path}"/*.mp4)
    if [[ -d "$out_path" && ${#mp4_files[@]} -gt 0 ]]; then
        echo "Skipping existing video for $name"; continue
    fi
    echo "Running $cfg"
    CUDA_VISIBLE_DEVICES=$GPU python src/fracsim/simulate.py \
        --model_path "${MODEL_PATH}" \
        --output_path "${out_path}" \
        --config "${cfg}" \
        --render_img --compile_video --white_bg \
        --render_width 512 --render_height 512
done
shopt -u nullglob
