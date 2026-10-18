#!/bin/bash
# Render a multi-view NeRF-style dataset from a .glb asset with Blender + BlenderNeRF.
# Usage: BLENDER_BIN=/path/to/blender BLENDER_NERF_ADDON=/path/to/blender_nerf.zip \
#        bash tools/fracsim/gs_recon/render_glb.sh <object.glb> [name] [out_root]
# Output: <out_root>/<name>/{train/*.png, transforms_train.json, transforms_test.json}
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OBJ_PATH="${1:?path to .glb}"
OBJ_NAME="${2:-$(basename "${OBJ_PATH}" .glb)}"
OUT_ROOT="${3:-${REPO_ROOT}/data/fracsim/renders}"
BLENDER_BIN="${BLENDER_BIN:-blender}"                      # Blender 4.3 tested
BLENDER_NERF_ADDON="${BLENDER_NERF_ADDON:?path to blender_nerf.zip (https://github.com/maximeraafat/BlenderNeRF)}"
NUM_IMAGES="${NUM_IMAGES:-200}"
RESOLUTION="${RESOLUTION:-512}"
CAMERA_DIST_MIN="${CAMERA_DIST_MIN:-2}"
CAMERA_DIST_MAX="${CAMERA_DIST_MAX:-2.5}"
SCENE_SCALE="${SCENE_SCALE:-1.0}"

"${BLENDER_BIN}" --background \
  --python "${REPO_ROOT}/tools/fracsim/gs_recon/render_glb.py" -- \
  --obj_path "${OBJ_PATH}" \
  --output_dir "${OUT_ROOT}/${OBJ_NAME}" \
  --num_images "${NUM_IMAGES}" \
  --resolution "${RESOLUTION}" \
  --camera_dist_min "${CAMERA_DIST_MIN}" \
  --camera_dist_max "${CAMERA_DIST_MAX}" \
  --scene_scale "${SCENE_SCALE}" \
  --blender_nerf_addon_path "${BLENDER_NERF_ADDON}"
