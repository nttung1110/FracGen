"""
Clean visualization utilities for PhysGaussian debug.
"""

import os
import glob
import torch
import subprocess

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from PIL import Image
from tqdm import tqdm


def ffmpeg_bin():
    """Path to an ffmpeg executable: system ffmpeg if on PATH, else the one bundled with imageio-ffmpeg."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


# -------------------------------------------------
# Basic utilities
# -------------------------------------------------

def load_npy(path):
    return np.asarray(np.load(path))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sorted_stems(dir_path, ext=".npy"):
    files = glob.glob(os.path.join(dir_path, f"*{ext}"))
    stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return sorted(stems)


def read_rgb(path, normalize=True):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if normalize:
        img = img.astype(np.float32) / 255.0
    return img

def normalize01(x):
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


# -------------------------------------------------
# Visualization primitives
# -------------------------------------------------

def overlay_scalar_on_rgb(rgb, scalar, alpha=0.5, mode="gray"):
    scalar = np.squeeze(scalar)
    s_norm = normalize01(scalar)

    if mode == "gray":
        s_rgb = np.repeat(s_norm[..., None], 3, axis=-1)
    elif mode == "hot":
        cmap = plt.get_cmap("hot")
        s_rgb = cmap(s_norm)[..., :3]
    else:
        raise ValueError("mode must be 'gray' or 'hot'")

    out = (1 - alpha) * rgb + alpha * s_rgb
    return np.clip(out, 0.0, 1.0)


def make_colorwheel():
    """
    Generates the Middlebury optical-flow color wheel.
    Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV 2007).
    """
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col += RY
    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col += YG
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col += GC
    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col += CB
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col += BM
    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Maps pre-normalized flow components u, v (expected range [-1, 1]) to RGB
    using the Middlebury color wheel: zero motion → white, full motion → saturated.

    Args:
        u, v (np.ndarray): Normalized flow components [H, W], values in [-1, 1].
        convert_to_bgr (bool): Return BGR instead of RGB.

    Returns:
        np.ndarray: uint8 [H, W, 3] color image.
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1 - col[idx])   # brighter toward center
        col[~idx] = col[~idx] * 0.75                 # out-of-range slightly dimmed
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_color_img(flow_uv, global_max=None, convert_to_bgr=False):
    """
    Render a [H, W, 2] flow field as a Middlebury color-wheel image.

    global_max: normalize all frames by this fixed magnitude so color
    intensity stays stable across a video (per-frame normalization causes
    brightness flicker even when motion is smooth).  When None, falls back
    to the per-frame max (useful for single-frame inspection).
    """
    assert flow_uv.ndim == 3 and flow_uv.shape[2] == 2
    u = flow_uv[..., 0]
    v = flow_uv[..., 1]
    if global_max is None:
        rad_max = np.sqrt(np.square(u) + np.square(v)).max()
        global_max = rad_max
    u = u / (global_max + 1e-5)
    v = v / (global_max + 1e-5)
    return flow_uv_to_colors(u, v, convert_to_bgr=convert_to_bgr)


def save_quiver_vel(vel, save_path, step=16, arrow_scale=1.0, background=None):
    vx = vel[..., 0] * arrow_scale
    vy = vel[..., 1] * arrow_scale
    H, W = vx.shape
    mag = np.sqrt(vx * vx + vy * vy)

    yy, xx = np.mgrid[0:H:step, 0:W:step]
    u = vx[0:H:step, 0:W:step]
    v = vy[0:H:step, 0:W:step]

    plt.figure(figsize=(8, 8))
    if background is None:
        plt.imshow(np.zeros((H, W)), cmap="gray")
    else:
        plt.imshow(background)

    plt.quiver(xx, yy, u, -v, mag[0:H:step, 0:W:step],
               angles="xy", scale_units="xy", scale=1.0)
    plt.axis("off")

    ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close()


# -------------------------------------------------
# Batch runners
# -------------------------------------------------

def make_overlay_video(
    maps_dir,
    images_dir,
    out_mp4,
    map_index=None,
    alpha=0.5,
    mode="gray",
    fps=30,
    limit=None,
):
    stems = sorted_stems(maps_dir, ext=".npy")
    if limit:
        stems = stems[:limit]

    frames = []

    for stem in tqdm(stems, desc="overlay"):
        m = load_npy(os.path.join(maps_dir, stem + ".npy"))
        if map_index is not None and m.ndim == 3:
            m = m[map_index]

        rgb = read_rgb(os.path.join(images_dir, stem + ".png"))
        frame = overlay_scalar_on_rgb(rgb, m, alpha=alpha, mode=mode)
        frames.append(frame)

    vid = np.stack(frames, axis=0)
    ensure_dir(os.path.dirname(out_mp4) or ".")
    imageio.mimwrite(out_mp4, [(np.asarray(f) * 255).astype(np.uint8) if np.asarray(f).dtype != np.uint8 else np.asarray(f) for f in vid], fps=fps)


def export_flow_color_frames(flow_dir, out_dir, limit=None):
    ensure_dir(out_dir)
    stems = sorted_stems(flow_dir, ext=".npy")
    if limit:
        stems = stems[:limit]

    for stem in tqdm(stems, desc="flow->png"):
        flow = load_npy(os.path.join(flow_dir, stem + ".npy"))
        img = flow_to_color_img(flow)
        Image.fromarray(img).save(os.path.join(out_dir, stem + ".png"))


def export_scalar_colormap_frames(
    scalar_dir,
    out_dir,
    name_of_map="scalar",
    cmap_name="hot",
    limit=None,
    fps=30
):
    ensure_dir(out_dir)
    stems = sorted_stems(scalar_dir, ext=".npy")
    if limit:
        stems = stems[:limit]

    cmap = plt.get_cmap(cmap_name)
    for stem in tqdm(stems, desc="scalar->png"):
        s = load_npy(os.path.join(scalar_dir, stem + ".npy"))
        s = np.squeeze(s)

        s_norm = normalize01(s)
        rgb = cmap(s_norm)[..., :3]

        imageio.imwrite(
            os.path.join(out_dir, stem + ".png"),
            (rgb * 255).astype(np.uint8),
        )

    output_video_path = os.path.join(out_dir, f"{name_of_map}.mp4")
    print(f"Compiling video at {fps} FPS...")
    ffmpeg_cmd = [
        ffmpeg_bin(),
        "-y",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(out_dir, "%04d.png"),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{rgb.shape[0]}x{rgb.shape[1]}",
        output_video_path,
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Video successfully saved to {output_video_path}")

# -------------------------------------------------
# Example usage
# -------------------------------------------------

def get_gaussian_blob(point, radius=10, amplitude=1.0, shape=(3, 480, 720), device=None):
    x, y = point
    C, H, W = shape

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )

    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    gaussian = amplitude * torch.exp(-dist2 / (2.0 * radius ** 2))

    # expand to all channels without loop
    return gaussian.expand(C, H, W)
