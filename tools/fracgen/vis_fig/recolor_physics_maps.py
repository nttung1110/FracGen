"""
Recolor predicted strain/stress/damage/flow videos to match the white-background
"lava"-style icon look produced by PhysGaussian's vis_phys_map.py, WITHOUT needing
the original .npz simulation arrays. Also extracts+upscales the fracture (RGB)
video alongside them for a matching-resolution frame set (fracture is real RGB
already, so it's passed through unchanged aside from the same Lanczos upscale).

Why this works (see tool_scripts/preprocess_dataset/process_forceVideo_all_multi_object.py):
  - strain/stress/damage are single-channel scalar maps encoded as (x,x,x) grayscale,
    where x is the scalar's per-video percentile-normalized value in [0,1]
    (normalize=percentile_per_video, transform=none for this dataset -- no gamma/log
    was applied). So the grayscale pixel intensity you see in the predicted video IS
    (an approximation of, post model-prediction/compression) the normalized scalar
    vis_phys_map.py itself computes from raw .npz files.
  - flow is already encoded with the exact same 55-segment Middlebury colorwheel that
    vis_phys_map.py's export_velocity_map_frames uses (hue = direction, brightness =
    magnitude / rad_max, black = zero motion) -- the preprocessing script's docstring
    says so explicitly ("matching PhysGaussian's vis_phys_map.py").

So: decode grayscale -> s_norm, then re-render with vis_phys_map.py's actual
white-background recipe (export_scalar_hot_frames, overlay_on_rgb=False):
  - color   = Reds colormap (white -> pink -> red -> dark red) applied to s_norm
  - alpha   = a *presence mask* (s_norm > threshold, Gaussian-blurred for a soft
              edge), NOT s_norm itself -- this is the key difference from a naive
              "value = alpha" approach: it gives a solid-ish colored silhouette
              (like the reference icons) rather than a soft magnitude-only glow.

Flow uses a *different* real vis_phys_map.py mechanism than the scalar maps --
its actual flow_uv_to_colors() in PhysGaussian/my_utils.py is NOT "black at
zero, blend onto white via a presence mask": it's a direct radial blend,
`col = 1 - rad*(1 - wheel_color)`, i.e. **zero motion -> white, full motion ->
saturated hue**, with everything in between a smooth pastel blend -- no
presence mask, no thresholding, no black background at all in the reference.

Our training-data flow encoding is `RGB = rad * wheel_color` (rad = brightness
= max(R,G,B), black background). Since vis_phys_map.py's own formula reduces
to `col = 1 - rad + rad*wheel_color = 1 - rad + RGB` once you substitute our
encoding in, decoding needs no division at all (so no risk of amplifying
near-zero-magnitude compression noise, unlike an earlier "un-dim by dividing"
attempt that produced rainbow static in low-motion frames):
    output = clip(1 - rad + RGB, 0, 1),  rad = max(R,G,B)

Resolution note: the model itself renders at a fixed 480x480 (hardcoded
--height/--width in tool_scripts/scripts_fun_control_extend/infer_stage3.sh) --
there's no higher-res source to decode without re-running inference (a GPU
job, out of scope here). What this script *can* do is avoid making it look
worse than it has to: it upscales the raw frame with Lanczos resampling
*before* decoding/recoloring (default 3x -> 1440x1440), not after -- coloring
a blocky 480px field and then stretching the PNG would just make the
blockiness bigger. Doing it in the other order gives smooth gradients and
edges from true interpolation. This adds no real detail beyond what the model
predicted; it just presents the same information without pixelation/staircase
artifacts. Set upscale=1 to disable.

Usage:
    python recolor_physics_maps.py <predicted_video_dir> <sample_name> <out_dir> [upscale]

<predicted_video_dir> must contain strain/, stress/, damage/, flow/, fracture/
subfolders with "<sample_name>.mp4" (standard infer_step-* output layout).
Writes PNG sequences to <out_dir>/{strain,stress,damage,flow,fracture}/frame_%04d.png
-- the first four recolored, fracture upscaled-only (already real RGB).

Extracting frames from mp4 first (ffmpeg) is left to the caller / a wrapper script,
so this file has no video I/O dependency beyond PIL -- see extract_and_recolor()
below for the end-to-end version that does call ffmpeg.
"""

import os
import subprocess
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
REDS = plt.get_cmap("Reds")

SCALAR_MODS = ["strain", "stress", "damage"]  # Reds colormap + presence-mask alpha
FLOW_MOD = "flow"                              # preserve hue, magnitude-as-alpha
FRACTURE_MOD = "fracture"                      # already real RGB -- upscale only, no recolor


def load_frame(png_path, upscale=1):
    """Load a PNG as a float32 [0,1] array, optionally Lanczos-upscaled first."""
    im = Image.open(png_path).convert("RGB")
    if upscale != 1:
        w, h = im.size
        im = im.resize((w * upscale, h * upscale), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


def decode_scalar(arr):
    """(x,x,x)-encoded grayscale array -> normalized scalar field in [0,1]."""
    return arr.mean(axis=-1)  # mean over channels: robust to compression noise


def render_scalar_reds(s_norm, blur_sigma=1.5, presence_thresh=0.03):
    """s_norm [0,1] -> white-bg Reds-colormap RGB uint8, matching vis_phys_map.py."""
    s_color = REDS(s_norm)[..., :3].astype(np.float32)
    particle_mask = gaussian_filter((s_norm > presence_thresh).astype(np.float32), sigma=blur_sigma)
    bg = np.ones_like(s_color)
    frame = particle_mask[..., None] * s_color + (1 - particle_mask[..., None]) * bg
    return (np.clip(frame, 0, 1) * 255).astype(np.uint8)


def render_flow(arr):
    """
    Exact PhysGaussian/my_utils.py flow_uv_to_colors() convention: zero motion ->
    white, full motion -> saturated hue, via col = 1 - rad*(1 - wheel_color).
    Our encoding gives RGB = rad*wheel_color (rad = max(R,G,B)), so this reduces
    to `1 - rad + RGB` -- no division, so no noise amplification at low magnitude.
    """
    rad = arr.max(axis=-1, keepdims=True)
    frame = 1.0 - rad + arr
    return (np.clip(frame, 0, 1) * 255).astype(np.uint8)


def extract_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-i", video_path, "-vsync", "0",
         "-start_number", "0", os.path.join(out_dir, "frame_%04d.png")],
        check=True,
    )


def extract_and_recolor(predicted_video_dir, sample_name, out_dir,
                         upscale=3, blur_sigma=1.5, presence_thresh=0.03):
    """
    End-to-end: extract raw frames from <predicted_video_dir>/{mod}/<sample_name>.mp4
    for each physics-map modality, recolor each frame independently, and save
    PNG sequences under <out_dir>/{mod}/frame_%04d.png.

    upscale: Lanczos-resize the raw frame by this factor *before* decoding/
    recoloring (default 3x, e.g. 480 -> 1440). Presents the model's native
    480x480 output without pixelation; doesn't add real detail. Set to 1 to
    keep native resolution. blur_sigma is auto-scaled by `upscale` so the
    presence-mask edge softness stays visually consistent at any scale.
    """
    raw_dir = os.path.join(out_dir, "_raw_frames")
    scaled_blur_sigma = blur_sigma * upscale

    for mod in SCALAR_MODS + [FLOW_MOD, FRACTURE_MOD]:
        src = os.path.join(predicted_video_dir, mod, sample_name + ".mp4")
        raw_mod_dir = os.path.join(raw_dir, mod)
        extract_frames(src, raw_mod_dir)

    for mod in SCALAR_MODS:
        in_dir = os.path.join(raw_dir, mod)
        out_mod_dir = os.path.join(out_dir, mod)
        os.makedirs(out_mod_dir, exist_ok=True)
        for fname in sorted(os.listdir(in_dir)):
            arr = load_frame(os.path.join(in_dir, fname), upscale=upscale)
            s = decode_scalar(arr)
            out = render_scalar_reds(s, blur_sigma=scaled_blur_sigma, presence_thresh=presence_thresh)
            Image.fromarray(out).save(os.path.join(out_mod_dir, fname))
        print(f"{mod}: {len(os.listdir(in_dir))} frames -> {out_mod_dir}")

    in_dir = os.path.join(raw_dir, FLOW_MOD)
    out_mod_dir = os.path.join(out_dir, FLOW_MOD)
    os.makedirs(out_mod_dir, exist_ok=True)
    for fname in sorted(os.listdir(in_dir)):
        arr = load_frame(os.path.join(in_dir, fname), upscale=upscale)
        out = render_flow(arr)
        Image.fromarray(out).save(os.path.join(out_mod_dir, fname))
    print(f"flow: {len(os.listdir(in_dir))} frames -> {out_mod_dir}")

    in_dir = os.path.join(raw_dir, FRACTURE_MOD)
    out_mod_dir = os.path.join(out_dir, FRACTURE_MOD)
    os.makedirs(out_mod_dir, exist_ok=True)
    for fname in sorted(os.listdir(in_dir)):
        arr = load_frame(os.path.join(in_dir, fname), upscale=upscale)
        Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(os.path.join(out_mod_dir, fname))
    print(f"fracture: {len(os.listdir(in_dir))} frames -> {out_mod_dir} (upscaled only, no recolor)")

    import shutil
    shutil.rmtree(raw_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) not in (4, 5):
        print(__doc__)
        sys.exit(1)
    upscale_arg = int(sys.argv[4]) if len(sys.argv) == 5 else 3
    extract_and_recolor(sys.argv[1], sys.argv[2], sys.argv[3], upscale=upscale_arg)
