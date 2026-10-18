#!/bin/bash
# FracGen inference: generate fracture videos + physics maps from benchmark inputs
# (reference image video + force control video + material parameters in metadata.csv).
#
#   CKPT=ckpts/fracgen/fracgen_stage3_step-21000.safetensors INPUT=data/benchmark/... OUT=outputs/fracgen/demo \
#   bash scripts/fracgen/infer.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_ROOT"

CKPT="${CKPT:-ckpts/fracgen/fracgen_stage3_step-21000.safetensors}"
INPUT="${INPUT:?folder with metadata.csv, reference_video/ and control_video/}"
OUT="${OUT:-outputs/fracgen/infer}"           # infer.py appends _<basename of INPUT> to this
GPU="${GPU:-0}"
MODEL_ID="${MODEL_ID:-alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control}"
NUM_FRAMES="${NUM_FRAMES:-81}"; HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-480}"
STEPS="${STEPS:-50}"
NUM_SAVE="${NUM_SAVE:-200}"
PHYS_KEYS="${PHYS_KEYS:-}"                   # empty = all four maps (must match the checkpoint)
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$REPO_ROOT/ckpts/wan}"
export TOKENIZERS_PARALLELISM=false

DATA_FILE_KEYS="reference_image,video,force_video,stress_video,flow_video,damage_video,strain_video,young_modulus,poisson_ratio,lambda_onset,lambda_crit,category"
EXTRA=(); [[ -n "$PHYS_KEYS" ]] && EXTRA+=(--phys_keys "$PHYS_KEYS")
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES="$GPU" python src/fracgen/infer.py \
  --ckpt "$CKPT" --ckpt_type auto \
  --model_id "$MODEL_ID" \
  --dataset_base_path "$INPUT" --dataset_metadata_path "$INPUT/metadata.csv" \
  --data_file_keys "$DATA_FILE_KEYS" \
  --height "$HEIGHT" --width "$WIDTH" --num_frames "$NUM_FRAMES" \
  --num_inference_steps "$STEPS" --num_save "$NUM_SAVE" \
  --output_dir "$OUT" \
  --combined_latent_tiling "${EXTRA[@]}"
echo "Results in ${OUT}_$(basename "$(cd "$INPUT" && pwd)")"
