#!/bin/bash
# FracGen step 1: encode a processed FracSim dataset (metadata.csv + mp4 folders) into the
# training cache (VAE latents, text / CLIP embeddings). Run once per dataset.
#
#   DATASET=data/fracgen/sample_10/train CACHE=data/fracgen/cache/sample_10 GPUS=0 \
#   bash scripts/fracgen/preprocess_data.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_ROOT"

DATASET="${DATASET:?folder with metadata.csv and videos/ force_videos/ ...}"
CACHE="${CACHE:?output cache folder}"
GPUS="${GPUS:-0}"
NUM_FRAMES="${NUM_FRAMES:-81}"
HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-480}"
MODEL_ID="${MODEL_ID:-alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$REPO_ROOT/ckpts/wan}"
export TOKENIZERS_PARALLELISM=false

DATA_FILE_KEYS="reference_image,video,force_video,stress_video,flow_video,damage_video,strain_video,young_modulus,poisson_ratio,lambda_onset,lambda_crit,category"
NUM_PROCS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
mkdir -p "$CACHE"

CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --num_processes "$NUM_PROCS" --gpu_ids "$(seq -s, 0 $((NUM_PROCS-1)))" \
  src/fracgen/train.py \
  --task "sft:data_process" \
  --exp_id "data_process" \
  --dataset_base_path "$DATASET" \
  --dataset_metadata_path "$DATASET/metadata.csv" \
  --output_path "$CACHE" \
  --height "$HEIGHT" --width "$WIDTH" --num_frames "$NUM_FRAMES" \
  --dataset_repeat 1 --num_epochs 1 --dataset_num_workers 4 \
  --model_id_with_origin_paths "${MODEL_ID}:diffusion_pytorch_model*.safetensors,${MODEL_ID}:models_t5_umt5-xxl-enc-bf16.pth,${MODEL_ID}:Wan2.1_VAE.pth,${MODEL_ID}:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --trainable_models "dit" \
  --extra_inputs "control_video,reference_image" \
  --use_gradient_checkpointing \
  --data_file_keys "$DATA_FILE_KEYS"
echo "Cache written to $CACHE"
