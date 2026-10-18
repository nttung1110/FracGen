#!/usr/bin/env bash
# Build the per-frame vertical method-comparison figure (force control video /
# force-prompting / PhyCo / CogVideoX / wan_baseline / ours) for every deduped output_random_obj
# sample, via stack_methods_compare.py, into vis_fig/main_qual/<sample>/
# frame_XXXX.png plus a 15fps <sample>_stack.mp4 next to it.
#
# Usage: [OURS_FRAC=2] ./stack_random_obj_all.sh [ours_fracture_dir] [samples...]
#   ours_fracture_dir defaults to the stage3 step-21000 random_obj run.
#   samples default to every seed42_*.mp4 present in PhyCo's random_obj output
#   (the frac0 representatives -- baselines never see the fracture params, so
#   frac0/1/2 share identical baseline outputs).
#   OURS_FRAC (default 0) picks which fracture variant of OUR output to show:
#   e.g. OURS_FRAC=2 uses <sample with frac2>.mp4 when it exists and falls back
#   to frac0 (the food objects only have frac0), reporting each fallback.
#   EXCLUDE_ROWS (default none) drops rows by label, comma-separated, e.g.
#   EXCLUDE_ROWS=wan_baseline. OUT_SUBDIR (default main_qual) selects the
#   output folder, so a variant layout doesn't overwrite the full one.
# Samples missing any of the six inputs are skipped and reported.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TP=$ROOT/eval/third_parties
OURS="${1:-$ROOT/models/train_extend_lora_forceControl/08.28_lora_1.3B_rank64_stage3_with_phys_loss_split/infer_step-21000_output_random_obj/fracture}"
shift || true
PY=python
FF=${FFMPEG:-ffmpeg}
OUT_ROOT=$ROOT/vis_fig/${OUT_SUBDIR:-main_qual}
OURS_FRAC="${OURS_FRAC:-0}"
EXCLUDE_ROWS="${EXCLUDE_ROWS:-}"

if [ $# -gt 0 ]; then
  SAMPLES=("$@")
else
  SAMPLES=($(ls $TP/phyco/output/compare_random_obj/seed42_*.mp4 | xargs -n1 basename | sed 's/^seed42_//; s/\.mp4$//'))
fi

# Build the row list once: label=<path placeholder key>, minus excluded labels.
ALL_ROWS="force force_prompt phyco cogvideox wan_baseline ours"
ROWS=""
for r in $ALL_ROWS; do
  case ",$EXCLUDE_ROWS," in *",$r,"*) continue;; esac
  ROWS="$ROWS $r"
done

done_n=0; skipped=()
echo "ours: $OURS (frac$OURS_FRAC preferred)"
echo "rows:$ROWS -> $OUT_ROOT"
for S in "${SAMPLES[@]}"; do
  FC=$ROOT/benchmarks/output_random_obj/control_video/$S.mp4
  FP=$TP/force-prompting/output/compare_random_obj/seed42_$S.mp4
  PH=$TP/phyco/output/compare_random_obj/seed42_$S.mp4
  CV=$TP/cogvideox-5b-i2v/output/compare_random_obj/seed42_$S.mp4
  WB=$TP/wan_baseline/output/compare_random_obj/seed42_$S.mp4
  OU=$OURS/${S%frac0}frac$OURS_FRAC.mp4
  if [ ! -f "$OU" ]; then
    [ "$OURS_FRAC" != "0" ] && echo "note: $S has no frac$OURS_FRAC in ours, using frac0"
    OU=$OURS/$S.mp4
  fi
  args=(); missing=""
  for r in $ROWS; do
    case $r in
      force) f=$FC;; force_prompt) f=$FP;; phyco) f=$PH;;
      cogvideox) f=$CV;; wan_baseline) f=$WB;; ours) f=$OU;;
    esac
    [ -f "$f" ] || missing="$missing $f"
    args+=("$r=$f")
  done
  if [ -n "$missing" ]; then skipped+=("$S:$missing"); continue; fi

  OUT=$OUT_ROOT/$S
  mkdir -p "$OUT"
  $PY $ROOT/vis_fig/vis_fig_script/stack_methods_compare.py --row-size 480x480 "$OUT" 81 4 \
    "${args[@]}" || { skipped+=("$S: stack failed"); continue; }
  $FF -y -v error -framerate 15 -i "$OUT/frame_%04d.png" -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart "$OUT_ROOT/${S}_stack.mp4" \
    && echo "Saved: $OUT_ROOT/${S}_stack.mp4"
  done_n=$((done_n+1))
done

echo "stacked $done_n sample(s)"
if [ ${#skipped[@]} -gt 0 ]; then
  echo "skipped ${#skipped[@]}:"; printf '  %s\n' "${skipped[@]}"
fi
