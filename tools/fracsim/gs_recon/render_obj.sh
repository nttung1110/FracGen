#!/bin/bash
# Render a multi-view dataset from a .obj with Blender + BlenderNeRF (see render_glb.sh for .glb).
# Usage: BLENDER_BIN=... BLENDER_NERF_ADDON=... bash render_obj.sh <object.obj> [name]
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OBJ_PARAMS="--obj_path ${1:?path to .obj}"
OBJ_NAME="${2:-$(basename "$1" .obj)}"

NUM_IMAGES=200
FORMAT="NERF"
CAMERA_DIST_MIN=3.0
CAMERA_DIST_MAX=4.0
OUTPUT_DIR="${REPO_ROOT}/data/fracsim/renders/${OBJ_NAME}"
SCENE_SCALE=0.7
BLENDER_NERF_ADDON_PATH="${BLENDER_NERF_ADDON:?path to blender_nerf.zip}"

"${BLENDER_BIN:-blender}" \
  --background \
  --python "${REPO_ROOT}/src/fracsim/blender/render_with_blender.py" \
  -- \
  ${OBJ_PARAMS} \
  --num_images ${NUM_IMAGES} \
  --format ${FORMAT} \
  --camera_dist_min ${CAMERA_DIST_MIN} \
  --camera_dist_max ${CAMERA_DIST_MAX} \
  --output_dir ${OUTPUT_DIR} \
  --scene_scale ${SCENE_SCALE} \
  --blender_nerf_addon_path ${BLENDER_NERF_ADDON_PATH} \
  --transparent_bg

