#!/bin/bash
# FracGen training, three stages on the same cache (see README section 2):
#   STAGE=1  RGB-only force-conditioned LoRA
#   STAGE=2  joint RGB + physics maps (latent tiling), initialised from the stage-1 LoRA
#   STAGE=3  stage 2 + constitutive consistency loss, initialised from the stage-2 LoRA
#
#   STAGE=1 CACHE=data/fracgen/cache/sample_10 EXP_ID=stage1 GPUS=0 bash scripts/fracgen/train.sh
#   STAGE=2 CACHE=... EXP_ID=stage2 PRETRAIN_CKPT=ckpts/fracgen/stage1/step-N.safetensors bash scripts/fracgen/train.sh
#   STAGE=3 CACHE=... EXP_ID=stage3 PRETRAIN_CKPT=ckpts/fracgen/stage2/step-N.safetensors bash scripts/fracgen/train.sh
# All uppercase variables can be overridden from the environment.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO_ROOT"

STAGE="${STAGE:?1, 2 or 3}"
CACHE="${CACHE:?training cache made by scripts/fracgen/preprocess_data.sh}"
EXP_ID="${EXP_ID:-stage${STAGE}}"
OUTPUT_PATH="${OUTPUT_PATH:-ckpts/fracgen/$EXP_ID}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"            # LoRA .safetensors of the previous stage (stages 2 and 3)
GPUS="${GPUS:-0,1,2,3}"
MODEL_ID="${MODEL_ID:-alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
LR="${LR:-1e-4}"
LORA_RANK="${LORA_RANK:-64}"
GRADIENT_ACCUM="${GRADIENT_ACCUM:-4}"
SAVE_STEPS="${SAVE_STEPS:-500}"                # empty = save at the end of every epoch
NUM_FRAMES="${NUM_FRAMES:-81}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-40}"
PHYS_KEYS="${PHYS_KEYS:-stress_video,strain_video,damage_video,flow_video}"
LAMBDA_CONST="${LAMBDA_CONST:-0.1}"; GAMMA_CONST="${GAMMA_CONST:-2.0}"; CONST_WARMUP_STEPS="${CONST_WARMUP_STEPS:-0}"
WANDB="${WANDB:-0}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$REPO_ROOT/ckpts/wan}"
export TOKENIZERS_PARALLELISM=false

STAGE_ARGS=()
case "$STAGE" in
  1) ;;
  2) STAGE_ARGS=(--combined_latent_tiling --phys_keys "$PHYS_KEYS") ;;
  3) STAGE_ARGS=(--combined_latent_tiling --phys_keys "$PHYS_KEYS"
                 --lambda_const "$LAMBDA_CONST" --gamma_const "$GAMMA_CONST" --const_warmup_steps "$CONST_WARMUP_STEPS") ;;
  *) echo "STAGE must be 1, 2 or 3"; exit 1 ;;
esac
[[ -n "$PRETRAIN_CKPT" ]] && STAGE_ARGS+=(--lora_checkpoint "$PRETRAIN_CKPT")
[[ "$WANDB" == "1" ]] && STAGE_ARGS+=(--wandb)
[[ -n "$SAVE_STEPS" ]] && STAGE_ARGS+=(--save_steps "$SAVE_STEPS")

DATA_FILE_KEYS="reference_image,video,force_video,stress_video,flow_video,damage_video,strain_video,young_modulus,poisson_ratio,lambda_onset,lambda_crit,category"
NUM_PROCS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
mkdir -p "$OUTPUT_PATH"
echo "stage=$STAGE exp=$EXP_ID cache=$CACHE out=$OUTPUT_PATH gpus=$GPUS pretrain=${PRETRAIN_CKPT:-none}"

CUDA_VISIBLE_DEVICES="$GPUS" accelerate launch --num_processes "$NUM_PROCS" --gpu_ids "$(seq -s, 0 $((NUM_PROCS-1)))" \
  src/fracgen/train.py \
  --task "sft:train" \
  --exp_id "$EXP_ID" \
  --dataset_base_path "$CACHE" \
  --output_path "$OUTPUT_PATH" \
  --model_id_with_origin_paths "${MODEL_ID}:diffusion_pytorch_model*.safetensors,${MODEL_ID}:Wan2.1_VAE.pth" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --lora_base_model "dit" --lora_target_modules "q,k,v,o,ffn.0,ffn.2" --lora_rank "$LORA_RANK" \
  --learning_rate "$LR" --lr_warmup_steps "$LR_WARMUP_STEPS" \
  --num_epochs "$NUM_EPOCHS" --dataset_repeat 1 --gradient_accumulation_steps "$GRADIENT_ACCUM" \
  --num_frames "$NUM_FRAMES" \
  --dataset_num_workers 6 --use_gradient_checkpointing --find_unused_parameters --no_resume_strict \
  --extra_inputs "control_video,reference_image" \
  --data_file_keys "$DATA_FILE_KEYS" \
  "${STAGE_ARGS[@]}"
echo "Training finished: $OUTPUT_PATH"
