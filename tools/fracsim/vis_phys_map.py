"""
Unified visualization for all PhysGaussian physical maps.
Supports both .npy and .npz (compressed) map files transparently.

- RGB         : rendered frames compiled to video
- Force map   : Gaussian blob overlay on RGB
- Velocity map: optical-flow color wheel (Middlebury convention, Baker et al. ICCV 2007)
- Stress map  : hot colormap, white background or overlaid on RGB
- Strain map  : hot colormap, white background or overlaid on RGB
- Damage map  : hot colormap, white background or overlaid on RGB
"""

import os
import subprocess
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm

import sys
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src", "fracsim"))
from my_utils import (
    ffmpeg_bin,
    ensure_dir, read_rgb, normalize01,
    overlay_scalar_on_rgb, flow_to_color_img,
)


# -------------------------------------------------
# Unified load: supports .npy and .npz
# -------------------------------------------------

def load_map(path_no_ext):
    """
    Load a physics map from either .npz or .npy, whichever exists.
    Preference: .npz first (smaller), fallback to .npy.
    """
    npz_path = path_no_ext + ".npz"
    npy_path = path_no_ext + ".npy"

    if os.path.exists(npz_path):
        return np.load(npz_path)["data"]
    elif os.path.exists(npy_path):
        return np.load(npy_path)
    else:
        raise FileNotFoundError(f"No .npz or .npy found at: {path_no_ext}")


def sorted_stems(directory):
    """
    Return sorted unique stems (filenames without extension) for all
    .npy and .npz files in directory.  Deduplicates so a stem present
    in both formats appears only once.
    """
    stems = set()
    for fname in os.listdir(directory):
        if fname.endswith(".npz") or fname.endswith(".npy"):
            stems.add(os.path.splitext(fname)[0])
    return sorted(stems)


# -------------------------------------------------
# Force map: blob overlay on RGB
# -------------------------------------------------

def export_force_map_frames(
    force_dir,
    images_dir,
    out_dir,
    alpha=0.6,
    limit=None,
):
    """Force map stored as [1, H, W]. Visualize channel-0 blob overlaid on RGB."""
    ensure_dir(out_dir)
    stems = sorted_stems(force_dir)
    if limit:
        stems = stems[:limit]

    for stem in tqdm(stems, desc="force->png"):
        m = load_map(os.path.join(force_dir, stem))
        if m.ndim == 3:
            m = m[0]                              # [1,H,W] -> [H,W]

        rgb_path = os.path.join(images_dir, stem + ".png")
        rgb = read_rgb(rgb_path) if os.path.exists(rgb_path) \
              else np.zeros((*m.shape, 3), dtype=np.float32)

        if m.shape != rgb.shape[:2]:
            m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        frame = overlay_scalar_on_rgb(rgb, m, alpha=alpha, mode="gray")
        imageio.imwrite(
            os.path.join(out_dir, stem + ".png"),
            (frame * 255).astype(np.uint8),
        )


# -------------------------------------------------
# Velocity map: flow color wheel
# -------------------------------------------------

def export_velocity_map_frames(
    vel_dir,
    out_dir,
    limit=None,
    percentile_high=99,
    smooth_sigma=0.0,
):
    """
    Velocity stored as [H, W, 2] optical flow. Visualize with Middlebury color wheel.

    Normalizes against a single magnitude computed once across all frames so
    color intensity stays stable over time.
    """
    ensure_dir(out_dir)
    stems = sorted_stems(vel_dir)
    if limit:
        stems = stems[:limit]

    mags = []
    for stem in stems:
        flow = load_map(os.path.join(vel_dir, stem))
        mags.append(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).ravel())
    mags = np.concatenate(mags)
    global_max = float(np.percentile(mags[mags > 1e-8], percentile_high)) if (mags > 1e-8).any() else 1.0
    print(f"  [velocity] global_max (p{percentile_high} of nonzero magnitudes) = {global_max:.4f}")

    for stem in tqdm(stems, desc="velocity->png"):
        flow = load_map(os.path.join(vel_dir, stem))   # [H,W,2]
        if smooth_sigma > 0:
            valid = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2) > 1e-8
            flow = np.stack(
                [smooth_sparse_field(flow[..., 0], smooth_sigma, mask=valid),
                 smooth_sparse_field(flow[..., 1], smooth_sigma, mask=valid)],
                axis=-1,
            )
        img = flow_to_color_img(flow, global_max=global_max)  # uint8 RGB
        Image.fromarray(img).save(os.path.join(out_dir, stem + ".png"))


# -------------------------------------------------
# Scalar physics maps: stress / strain / damage
# -------------------------------------------------

def compute_global_range(scalar_dir, stems, percentile_low=1, percentile_high=99):
    """
    Single pass over all frames to get a globally consistent scale.
    Uses percentiles to be robust against outliers.
    """
    all_vals = []
    for stem in stems:
        s = np.squeeze(load_map(os.path.join(scalar_dir, stem)))
        all_vals.append(s.ravel())
    all_vals = np.concatenate(all_vals)

    nonzero = all_vals[all_vals > 1e-8]
    if len(nonzero) == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(nonzero, percentile_low))
    vmax = float(np.percentile(nonzero, percentile_high))
    print(f"  [{os.path.basename(scalar_dir)}] "
          f"global vmin={vmin:.4f}, vmax={vmax:.4f} "
          f"(p{percentile_low}/p{percentile_high} of nonzero pixels)")
    return vmin, vmax


def smooth_sparse_field(s, sigma, mask=None):
    """
    Gaussian-smooth a sparse field without letting the zero background dilute
    real values. Blurs value and validity mask separately then divides
    (normalized convolution).

    mask: optional precomputed validity mask (use for signed fields where
    `s > eps` would wrongly treat negative values as empty).
    """
    if mask is None:
        mask = (s > 1e-8).astype(np.float32)
    else:
        mask = mask.astype(np.float32)
    s_blur    = cv2.GaussianBlur(s.astype(np.float32), (0, 0), sigma)
    mask_blur = cv2.GaussianBlur(mask, (0, 0), sigma)
    out = np.zeros_like(s_blur)
    valid = mask_blur > 1e-6
    out[valid] = s_blur[valid] / mask_blur[valid]
    return out


def export_scalar_hot_frames(
    scalar_dir,
    images_dir,
    out_dir,
    alpha=0.55,
    overlay_on_rgb=False,
    limit=None,
    vmin=None,
    vmax=None,
    percentile_low=1,
    percentile_high=99,
    smooth_sigma=0.0,
):
    """
    Render a scalar physics map (stress / strain / damage) as a hot
    colormap image.

    overlay_on_rgb=False (default):
        White background. The hot colormap is shown where particles are present;
        regions with no particles remain white. Particle boundaries fade smoothly
        to white via a blurred presence mask.

    overlay_on_rgb=True:
        Alpha-blend the hot colormap over the rendered RGB frame (original
        behavior). Requires img_frames to be populated.
    """
    ensure_dir(out_dir)
    stems = sorted_stems(scalar_dir)
    if limit:
        stems = stems[:limit]

    if vmin is None or vmax is None:
        vmin, vmax = compute_global_range(
            scalar_dir, stems, percentile_low, percentile_high
        )

    # "hot" starts at black — fine on dark/RGB backgrounds but produces a black
    # object silhouette on white backgrounds.  For white-bg mode, use a colormap
    # that starts near white so low-activity particle regions fade into the bg.
    if overlay_on_rgb:
        cmap = plt.get_cmap("hot")
    else:
        cmap = plt.get_cmap("Reds")   # white → pink → red → dark red

    for stem in tqdm(stems, desc=f"{os.path.basename(scalar_dir)}->png"):
        s = np.squeeze(load_map(os.path.join(scalar_dir, stem)))

        if smooth_sigma > 0:
            s = smooth_sparse_field(s, smooth_sigma)

        if overlay_on_rgb:
            rgb_path = os.path.join(images_dir, stem + ".png")
            rgb = read_rgb(rgb_path) if os.path.exists(rgb_path) \
                  else np.zeros((*s.shape, 3), dtype=np.float32)
            if s.shape != rgb.shape[:2]:
                s = cv2.resize(s, (rgb.shape[1], rgb.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

        s_norm  = np.clip((s - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
        s_color = cmap(s_norm)[..., :3].astype(np.float32)

        if overlay_on_rgb:
            frame = (1.0 - alpha) * rgb + alpha * s_color
        else:
            # White background: heatmap where particles are present, white elsewhere.
            # Blur the binary presence map to give a smooth soft edge at boundaries.
            # With "Reds", s_norm=0 → near-white so low-activity particles naturally
            # fade into the white background without a harsh black silhouette.
            bg = np.ones((*s.shape, 3), dtype=np.float32)
            particle_mask = cv2.GaussianBlur(
                (s > 1e-8).astype(np.float32), (0, 0), 1.5
            )[..., np.newaxis]
            frame = particle_mask * s_color + (1.0 - particle_mask) * bg

        frame = np.clip(frame, 0.0, 1.0)
        imageio.imwrite(
            os.path.join(out_dir, stem + ".png"),
            (frame * 255).astype(np.uint8),
        )


# -------------------------------------------------
# Compile PNG sequence -> MP4
# -------------------------------------------------

def compile_video(frames_dir, out_mp4, fps=30, width=None, height=None):
    """Compile a folder of %04d.png frames into an mp4 using ffmpeg."""
    ensure_dir(os.path.dirname(out_mp4) or ".")

    size_flag = []
    if width is not None and height is not None:
        size_flag = ["-s", f"{width}x{height}"]

    cmd = [
        ffmpeg_bin(), "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "%04d.png"),
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        *size_flag,
        out_mp4,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Saved: {out_mp4}")
    except FileNotFoundError:
        print("ffmpeg not found — frames saved but video not compiled.")
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed:", e.stderr.decode())


# -------------------------------------------------
# One-shot: export all maps + compile videos
# -------------------------------------------------

def visualize_all_maps(
    output_path,
    vis_dir,
    fps=30,
    alpha_force=0.6,
    alpha_scalar=0.55,
    overlay_on_rgb=False,
    limit=None,
    smooth_sigma=0.0,
):
    """
    Expects the standard PhysGaussian output layout under output_path:
        output_path/
            img_frames/
            2d_force_map/
            2d_vel_map/
            2d_stress_map/
            2d_strain_map/
            2d_damage_map/

    Writes into vis_dir:
        vis_rgb.mp4           — rendered RGB video
        vis_force/  + .mp4
        vis_velocity/ + .mp4
        vis_stress/  + .mp4
        vis_strain/  + .mp4
        vis_damage/  + .mp4

    overlay_on_rgb : if True, scalar heatmaps are alpha-blended over the RGB
                     frames; if False (default), they are rendered on a white
                     background with the hot colormap.
    """
    images_dir = os.path.join(output_path, "img_frames")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Image frames folder not found: {images_dir}")

    ensure_dir(vis_dir)

    # ── RGB video ────────────────────────────────────────────────────
    print("\n── img_frames (RGB) ──")
    compile_video(images_dir, os.path.join(vis_dir, "vis_rgb.mp4"), fps=fps)

    # ── Physics maps ─────────────────────────────────────────────────
    map_configs = [
        ("2d_force_map",  "vis_force",    "force",    alpha_force),
        ("2d_vel_map",    "vis_velocity", "velocity", None),
        ("2d_stress_map", "vis_stress",   "scalar",   alpha_scalar),
        ("2d_strain_map", "vis_strain",   "scalar",   alpha_scalar),
        ("2d_damage_map", "vis_damage",   "scalar",   alpha_scalar),
    ]

    for subdir, vis_sub, kind, alpha in map_configs:
        map_dir = os.path.join(output_path, subdir)
        out_dir = os.path.join(vis_dir, vis_sub)
        out_mp4 = os.path.join(vis_dir, f"{vis_sub}.mp4")

        if not os.path.isdir(map_dir):
            print(f"Skipping {subdir} (not found)")
            continue

        print(f"\n── {subdir} ──")
        if kind == "force":
            export_force_map_frames(map_dir, images_dir, out_dir,
                                    alpha=alpha, limit=limit)
        elif kind == "velocity":
            export_velocity_map_frames(map_dir, out_dir,
                                       limit=limit, smooth_sigma=smooth_sigma)
        elif kind == "scalar":
            export_scalar_hot_frames(map_dir, images_dir, out_dir,
                                     alpha=alpha,
                                     overlay_on_rgb=overlay_on_rgb,
                                     limit=limit,
                                     smooth_sigma=smooth_sigma)

        compile_video(out_dir, out_mp4, fps=fps)

    # ── Side-by-side video: RGB | force | flow | stress | strain | damage ──
    parts = [os.path.join(vis_dir, f"{n}.mp4")
             for n in ["vis_rgb", "vis_force", "vis_velocity", "vis_stress", "vis_strain", "vis_damage"]]
    parts = [p for p in parts if os.path.exists(p)]
    if len(parts) > 1:
        out_all = os.path.join(vis_dir, "vis_all.mp4")
        cmd = [ffmpeg_bin(), "-y"]
        for p in parts:
            cmd += ["-i", p]
        cmd += ["-filter_complex", f"hstack=inputs={len(parts)}", "-c:v", "libx264", "-crf", "18",
                "-pix_fmt", "yuv420p", out_all]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Saved side-by-side video: {out_all}")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print("could not build side-by-side video:", getattr(e, "stderr", b"").decode(errors="ignore")[-400:])


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Render FracSim physics maps (.npz/.npy) into colour frames and videos.")
    ap.add_argument("--output_path", required=True, help="FracSim output folder (contains img_frames/ and 2d_*_map/)")
    ap.add_argument("--vis_dir", default=None, help="where to write vis_*.mp4 and frames (default: <output_path>/vis)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--alpha_force", type=float, default=0.6)
    ap.add_argument("--alpha_scalar", type=float, default=0.55)
    ap.add_argument("--overlay_on_rgb", action="store_true", help="blend heatmaps over RGB instead of white background")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N frames")
    ap.add_argument("--smooth_sigma", type=float, default=0.0, help="Gaussian blur sigma in pixels (0 = off)")
    args = ap.parse_args()
    visualize_all_maps(
        output_path    = args.output_path,
        vis_dir        = args.vis_dir or os.path.join(args.output_path, "vis"),
        fps            = args.fps,
        alpha_force    = args.alpha_force,
        alpha_scalar   = args.alpha_scalar,
        overlay_on_rgb = args.overlay_on_rgb,
        limit          = args.limit,
        smooth_sigma   = args.smooth_sigma,
    )
