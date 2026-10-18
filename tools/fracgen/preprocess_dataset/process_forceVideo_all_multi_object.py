"""
Process PhysGaussian multi-modal maps into DiffSynth UnifiedDataset format.

This script converts a PhysGaussian output folder (e.g.):
  /weka/scratch/bhattad/tnguy303/research/PhysGaussian/output_exp_update_gs

...into a dataset folder under this repo (default):
  data/physgaussian_multimodal_update_gs/

Output schema:
  metadata.csv columns:
    video,prompt,force_video,stress_video,flow_video,damage_video,strain_video,young_modulus,poisson_ratio,lambda_onset,lambda_crit,category

Folders created under dst_root:
  videos/         -> RGB mp4 (from numbered PNGs)
  force_videos/   -> mp4 encoded from 2d_force_map/*.npz (optional if src has it)
  stress_videos/  -> mp4 encoded from 2d_stress_map/*.npz
  flow_videos/    -> mp4 encoded from 2d_vel_map/*.npz
  damage_videos/  -> mp4 encoded from 2d_damage_map/*.npz
  strain_videos/  -> mp4 encoded from 2d_strain_map/*.npz

Notes:
- Frame alignment is by index (0000, 0001, ...). We enforce that all requested modalities
  have at least N frames, then encode the first N frames for every modality.
- Map-to-video conversion writes grayscale RGB (uint8) videos for compatibility with the
  existing video loader operator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
from tqdm import tqdm  # type: ignore
import numpy as np

_TQDM_INVISIBLE = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _natural_key(p: str) -> tuple:
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p))


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _tqdm_wrap(iterable: Iterable, **kwargs):  # type: ignore[no-untyped-def]
    global _TQDM_INVISIBLE
    if _TQDM_INVISIBLE or kwargs.pop("invisible", False):
        return iterable
    if "disable" not in kwargs:
        kwargs["disable"] = not sys.stderr.isatty()
    return tqdm(iterable, **kwargs)


def _relpath(path: Path, start: Path) -> str:
    return os.path.relpath(str(path), start=str(start))


def _list_numbered_files(dir_path: Path, suffix: str) -> list[str]:
    out: list[str] = []
    for name in os.listdir(dir_path):
        if not name.lower().endswith(suffix.lower()):
            continue
        stem = name[: -len(suffix)]
        if stem.isdigit():
            out.append(name)
    return sorted(out, key=_natural_key)


def _load_npz_array(path: Path) -> np.ndarray:
    data = np.load(str(path))
    if isinstance(data, np.ndarray):
        return data

    try:
        keys = list(data.files)
        if not keys:
            raise ValueError(f"No arrays found in npz file: {path}")
        if len(keys) == 1:
            return np.asarray(data[keys[0]])
        if "arr_0" in data:
            return np.asarray(data["arr_0"])
        raise ValueError(f"Expected a single array in npz file {path}, found keys: {keys}")
    finally:
        close = getattr(data, "close", None)
        if callable(close):
            close()


def _map_to_hwc(arr: np.ndarray) -> np.ndarray:
    """
    Convert common PhysGaussian map shapes into HxWxC float32 (C in {1,2,3}).
    Supports:
    - (H,W) -> (H,W,1)
    - (C,H,W) where C in {1,2,3} -> transpose to (H,W,C)
    - (H,W,C) where C in {1,2,3} -> as-is

    If channel count > 3, we raise an error instead of dropping information.
    """
    x = np.asarray(arr)
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if x.ndim == 2:
        return x[..., None]

    if x.ndim == 3:
        # channel-first: (C,H,W) with small C
        if x.shape[0] in (1, 2, 3) and x.shape[1] > 1 and x.shape[2] > 1:
            return np.transpose(x, (1, 2, 0))  # (H,W,C)

        # channel-last: (H,W,C) with small C
        if x.shape[-1] in (1, 2, 3) and x.shape[0] > 1 and x.shape[1] > 1:
            return x

        # Explicitly error out for >3 channels (either layout)
        if x.shape[0] > 3 and x.shape[-1] > 3:
            raise ValueError(f"Unsupported map shape / channel count: {x.shape}")
        if x.shape[0] > 3 and x.shape[-1] <= 3:
            # looks like HWC but C<=3 would have matched above; fallthrough -> unsupported layout
            raise ValueError(f"Unsupported map layout (expected HWC/CHW): {x.shape}")
        if x.shape[-1] > 3 and x.shape[0] <= 3:
            raise ValueError(f"Unsupported channel count (C>3) for HWC map: {x.shape}")

    raise ValueError(f"Unsupported map shape: {x.shape}")


def _hwc_to_rgb3(x_hwc: np.ndarray) -> np.ndarray:
    """Map HxWxC (C in {1,2,3}) to HxWx3 float32 without losing channel meaning."""
    if x_hwc.ndim != 3:
        raise ValueError(f"Expected HxWxC, got {x_hwc.shape}")
    c = int(x_hwc.shape[-1])
    if c == 1:
        return np.repeat(x_hwc, 3, axis=-1)
    if c == 2:
        z = np.zeros((*x_hwc.shape[:2], 1), dtype=x_hwc.dtype)
        return np.concatenate([x_hwc, z], axis=-1)  # (R,G,0)
    if c == 3:
        return x_hwc
    raise ValueError(f"Unsupported channel count (C>3): {x_hwc.shape}")


def _apply_normalize(
    x: np.ndarray,
    normalize: str,
    vmin: Optional[np.ndarray | float],
    vmax: Optional[np.ndarray | float],
) -> np.ndarray:
    if normalize == "clip01":
        return np.clip(x, 0.0, 1.0)

    if normalize == "minmax_per_frame":
        if x.ndim == 2:
            a = float(np.min(x))
            b = float(np.max(x))
            if not np.isfinite(a) or not np.isfinite(b) or b <= a:
                return np.zeros_like(x, dtype=np.float32)
            return np.clip((x - a) / (b - a), 0.0, 1.0)

        if x.ndim == 3:
            a = np.min(x, axis=(0, 1))
            b = np.max(x, axis=(0, 1))
            den = b - a
            ok = np.isfinite(a) & np.isfinite(b) & (den > 0)
            a2 = np.where(ok, a, 0.0).astype(np.float32)
            den2 = np.where(ok, den, 1.0).astype(np.float32)
            y = (x - a2) / den2
            y = np.where(ok, y, 0.0)
            return np.clip(y, 0.0, 1.0).astype(np.float32)

        raise ValueError(f"Unsupported ndim for minmax_per_frame: {x.ndim}")

    if normalize in ("minmax_per_video", "percentile_per_video"):
        # Same linear-scale-and-clip formula for both -- they only differ in how vmin/vmax
        # were computed upstream (literal extrema vs. percentile-of-nonzero; see
        # _compute_video_minmax_rgb / _compute_video_percentile_range_rgb).
        if vmin is None or vmax is None:
            raise ValueError(f"vmin/vmax required for {normalize}")
        a = np.asarray(vmin, dtype=np.float32)
        b = np.asarray(vmax, dtype=np.float32)
        den = b - a
        ok = np.isfinite(a) & np.isfinite(b) & (den > 0)
        a2 = np.where(ok, a, 0.0).astype(np.float32)
        den2 = np.where(ok, den, 1.0).astype(np.float32)
        y = (x - a2) / den2
        y = np.where(ok, y, 0.0)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    raise ValueError(f"Unknown normalize mode: {normalize}")


def _rgb_to_rgb_u8(
    rgb_f32: np.ndarray,
    gain: float,
    normalize: str,
    vmin: Optional[np.ndarray | float],
    vmax: Optional[np.ndarray | float],
    transform: str,
    gamma: float,
    log_alpha: float,
) -> np.ndarray:
    x = np.nan_to_num(rgb_f32.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x = x * float(gain)
    x = _apply_normalize(x, normalize=normalize, vmin=vmin, vmax=vmax)

    if transform == "none":
        y = x
    elif transform == "gamma":
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        y = np.power(x, float(gamma))
    elif transform == "log":
        a = float(log_alpha)
        if a <= 0:
            raise ValueError(f"log_alpha must be > 0, got {a}")
        y = np.log1p(a * x) / np.log1p(a)
    else:
        raise ValueError(f"Unknown transform: {transform}")

    u8 = (y * 255.0 + 0.5).astype(np.uint8)
    if u8.ndim != 3 or u8.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 after RGB mapping, got {u8.shape}")
    return u8


def _write_video_streaming(frames: Iterable[np.ndarray], dst_path: Path, fps: int) -> None:
    try:
        import imageio  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency `imageio`. Install DiffSynth-Studio deps (e.g. `pip install -e .`) "
            "or at least `pip install imageio imageio[ffmpeg]` in the environment you run this script."
        ) from e

    _safe_mkdir(dst_path.parent)
    if dst_path.suffix.lower() != ".mp4":
        raise ValueError(f"Only .mp4 output is supported: {dst_path}")

    writer = imageio.get_writer(
        str(dst_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=None,
    )
    try:
        for frame in frames:
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"Frame must be uint8 HxWx3, got {frame.dtype} {frame.shape}")
            writer.append_data(frame)
    finally:
        writer.close()


def _build_rgb_video(
    rgb_frames_dir: Path,
    dst_rgb_video_path: Path,
    num_frames: int,
    fps: int,
    overwrite: bool,
) -> None:
    if dst_rgb_video_path.exists() and not overwrite:
        return

    png_files = _list_numbered_files(rgb_frames_dir, ".png")
    if not png_files:
        raise FileNotFoundError(f"No numbered .png frames found in {rgb_frames_dir} (expected 0000.png, 0001.png, ...)")

    png_files = png_files[: min(num_frames, len(png_files))]
    from PIL import Image

    def _frames():
        for name in _tqdm_wrap(
            png_files,
            total=len(png_files),
            desc=f"RGB {dst_rgb_video_path.name}",
            unit="frame",
            invisible=True,
        ):
            im = Image.open(rgb_frames_dir / name).convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"Expected RGB image -> HxWx3, got {arr.shape} from {name}")
            yield arr

    _write_video_streaming(_frames(), dst_rgb_video_path, fps=fps)


def _compute_video_minmax_rgb(map_dir: Path, npz_files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    vmin = np.full((3,), np.inf, dtype=np.float32)
    vmax = np.full((3,), -np.inf, dtype=np.float32)
    for name in npz_files:
        arr = _load_npz_array(map_dir / name)
        hwc = _map_to_hwc(arr)
        rgb = _hwc_to_rgb3(hwc)
        a = np.min(rgb, axis=(0, 1))
        b = np.max(rgb, axis=(0, 1))
        vmin = np.minimum(vmin, a.astype(np.float32))
        vmax = np.maximum(vmax, b.astype(np.float32))
    if not np.all(np.isfinite(vmin)) or not np.all(np.isfinite(vmax)):
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return vmin.astype(np.float32), vmax.astype(np.float32)


def _compute_video_percentile_range_rgb(
    map_dir: Path,
    npz_files: list[str],
    percentile_low: float,
    percentile_high: float,
    sample_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Robust per-channel normalization range: percentile of NONZERO pixel values across
    the whole video, excluding background. Matches PhysGaussian's vis_phys_map.py
    (compute_global_range) -- more robust than _compute_video_minmax_rgb's literal
    min/max, which lets a single outlier pixel anywhere in the video set the scale for
    every frame, and isn't diluted by the (typically large) zero/background region the
    way literal min/max is.

    Assumes non-negative scalar fields (stress/strain/damage/energy magnitudes), same
    as vis_phys_map.py -- background/empty pixels are exactly 0, not negative, so a
    plain `> 1e-8` threshold (not `abs(x) > 1e-8`) is the correct "is this real signal"
    test here.
    """
    samples = [[] for _ in range(3)]
    for name in npz_files:
        arr = _load_npz_array(map_dir / name)
        hwc = _map_to_hwc(arr)
        rgb = _hwc_to_rgb3(hwc)
        strided = rgb[::sample_stride, ::sample_stride, :] if sample_stride > 1 else rgb
        for c in range(3):
            vals = strided[..., c].reshape(-1)
            samples[c].append(vals[vals > 1e-8])

    vmin = np.zeros((3,), dtype=np.float32)
    vmax = np.ones((3,), dtype=np.float32)
    for c in range(3):
        allvals = np.concatenate(samples[c]) if samples[c] else np.array([], dtype=np.float32)
        allvals = allvals[np.isfinite(allvals)]
        if allvals.size == 0:
            continue  # an all-zero channel (e.g. the unused 3rd channel of a 2-ch map): leave default [0,1]
        vmin[c] = float(np.percentile(allvals, percentile_low))
        vmax[c] = float(np.percentile(allvals, percentile_high))
    return vmin, vmax


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """
    hsv: (..., 3) float32 in [0,1]
    returns rgb: (..., 3) float32 in [0,1]
    """
    if hsv.shape[-1] != 3:
        raise ValueError(f"Expected HSV last dim=3, got {hsv.shape}")
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6
    r = np.empty_like(v, dtype=np.float32)
    g = np.empty_like(v, dtype=np.float32)
    b = np.empty_like(v, dtype=np.float32)

    m0 = i_mod == 0
    r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
    m1 = i_mod == 1
    r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
    m2 = i_mod == 2
    r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
    m3 = i_mod == 3
    r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
    m4 = i_mod == 4
    r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
    m5 = i_mod == 5
    r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]

    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _float_rgb_to_u8(rgb: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)
    u8 = (x * 255.0 + 0.5).astype(np.uint8)
    if u8.ndim != 3 or u8.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 uint8, got {u8.shape}")
    return u8


def _compute_flow_rad_max(
    flow_dir: Path,
    npz_files: list[str],
    method: str = "percentile",
    percentile: float = 99.0,
    sample_stride: int = 8,
    gain: float = 1.0,
) -> float:
    """
    Compute a robust magnitude scale for optical-flow visualization.
    We use a strided pixel sample to avoid huge memory.

    method="percentile" excludes near-zero (background/no-motion) magnitudes before
    taking the percentile, matching PhysGaussian's vis_phys_map.py
    (export_velocity_map_frames). Without this, "Nth percentile of all pixels" silently
    becomes a sparsity-dependent, more conservative percentile of the nonzero values
    whenever most of the frame has no motion (e.g. 99th percentile of all pixels is only
    the 80th percentile of nonzero values when 95% of pixels are exactly zero) --
    excluding background first gives a consistent "Nth percentile of the pixels that
    actually have signal" regardless of how sparse the motion is.
    """
    mags: list[np.ndarray] = []
    rad_max = 0.0
    for name in npz_files:
        arr = _load_npz_array(flow_dir / name)
        hwc = _map_to_hwc(arr)
        if hwc.shape[-1] != 2:
            raise ValueError(f"flow_video expects 2 channels (u,v); got {hwc.shape} from {flow_dir / name}")
        u = hwc[..., 0] * float(gain)
        v = hwc[..., 1] * float(gain)
        mag = np.sqrt(u * u + v * v, dtype=np.float32)
        if method == "max":
            m = float(np.max(mag))
            if np.isfinite(m):
                rad_max = max(rad_max, m)
        elif method == "percentile":
            if sample_stride <= 0:
                raise ValueError(f"sample_stride must be > 0, got {sample_stride}")
            mags.append(mag[::sample_stride, ::sample_stride].reshape(-1).astype(np.float32))
        else:
            raise ValueError(f"Unknown flow_norm: {method}")

    if method == "percentile":
        if not mags:
            return 1.0
        allm = np.concatenate(mags, axis=0)
        allm = allm[np.isfinite(allm)]
        allm = allm[allm > 1e-8]  # exclude background/no-motion pixels, see docstring
        if allm.size == 0:
            return 1.0
        q = float(np.quantile(allm, np.clip(percentile, 0.0, 100.0) / 100.0))
        rad_max = q

    if not np.isfinite(rad_max) or rad_max <= 1e-8:
        rad_max = 1.0
    return float(rad_max)


def _make_flow_colorwheel() -> np.ndarray:
    """
    Standard Middlebury optical-flow colorwheel (55 segments).
    Sequence: Red→Yellow(15), Yellow→Green(6), Green→Cyan(4),
              Cyan→Blue(11), Blue→Magenta(13), Magenta→Red(6).
    Returns float32 array of shape (55, 3) in [0, 1].
    """
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    cw = np.zeros((ncols, 3), dtype=np.float32)
    col = 0
    for cname, ch, key in [(RY,0,0),(YG,0,1),(GC,1,0),(CB,1,1),(BM,2,0),(MR,2,1)]:
        if key == 0:
            cw[col:col+cname, ch] = 1.0
            cw[col:col+cname, (ch+1)%3] = np.arange(cname, dtype=np.float32) / cname
        else:
            cw[col:col+cname, ch] = 1.0 - np.arange(cname, dtype=np.float32) / cname
            cw[col:col+cname, (ch+1)%3] = 1.0
        col += cname
    return cw

_FLOW_COLORWHEEL = _make_flow_colorwheel()  # (55, 3) built once at import


def _flow_to_rgb01(flow_hwc2: np.ndarray, *, rad_max: float, gain: float = 1.0) -> np.ndarray:
    """
    Standard Middlebury color-wheel flow visualization:
      - Direction encoded by hue: Right=Red, Down=Yellow, Left=Cyan, Up=Blue.
      - Magnitude encoded by brightness: zero flow → black, max flow → full color.
    Matches the reference gen_colorwheel() convention exactly (non-uniform segment
    allocation: RY=15, YG=6, GC=4, CB=11, BM=13, MR=6 out of 55 total).
    Zero flow → black, consistent with all other physics maps (0 = no effect = black).
    """
    if flow_hwc2.ndim != 3 or flow_hwc2.shape[-1] != 2:
        raise ValueError(f"Expected flow HxWx2, got {flow_hwc2.shape}")
    u = flow_hwc2[..., 0] * float(gain)
    v = flow_hwc2[..., 1] * float(gain)
    mag = np.sqrt(u * u + v * v).astype(np.float32)
    # Angle in [0,1): clockwise from right, matching the colorwheel's visual layout.
    ang = np.arctan2(v, u).astype(np.float32)
    fk = (ang / (2.0 * np.pi)) % 1.0  # [0, 1)

    # Bilinear interpolation into the 55-segment colorwheel
    N = _FLOW_COLORWHEEL.shape[0]
    idx_f = fk * N
    i0 = np.floor(idx_f).astype(np.int32) % N
    i1 = (i0 + 1) % N
    frac = (idx_f - np.floor(idx_f)).astype(np.float32)[..., None]  # (..., 1) for broadcast
    col = (1.0 - frac) * _FLOW_COLORWHEEL[i0] + frac * _FLOW_COLORWHEEL[i1]  # (..., 3)

    # Scale brightness by normalised magnitude (zero → black)
    brightness = np.clip(mag / float(rad_max), 0.0, 1.0)[..., None]
    return (col * brightness).astype(np.float32)


def _build_flow_video(
    flow_dir: Path,
    dst_path: Path,
    num_frames: int,
    fps: int,
    overwrite: bool,
    flow_norm: str,
    flow_percentile: float,
    flow_sample_stride: int,
    flow_gain: float,
) -> None:
    if dst_path.exists() and not overwrite:
        return
    npz_files = _list_numbered_files(flow_dir, ".npz")
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {flow_dir}")
    npz_files = npz_files[: min(num_frames, len(npz_files))]

    rad_max = _compute_flow_rad_max(
        flow_dir,
        npz_files,
        method=flow_norm,
        percentile=flow_percentile,
        sample_stride=flow_sample_stride,
        gain=flow_gain,
    )

    def _frames():
        for name in _tqdm_wrap(
            npz_files,
            total=len(npz_files),
            desc=f"Flow {dst_path.name}",
            unit="frame",
            invisible=True,
        ):
            arr = _load_npz_array(flow_dir / name)
            hwc = _map_to_hwc(arr)
            if hwc.shape[-1] != 2:
                raise ValueError(f"flow_video expects 2 channels (u,v); got {hwc.shape} from {flow_dir / name}")
            rgb01 = _flow_to_rgb01(hwc, rad_max=rad_max, gain=flow_gain)
            yield _float_rgb_to_u8(rgb01)

    _write_video_streaming(_frames(), dst_path, fps=fps)


def _build_map_video(
    map_dir: Path,
    dst_path: Path,
    num_frames: int,
    fps: int,
    overwrite: bool,
    gain: float,
    normalize: str,
    vmin: Optional[np.ndarray | float],
    vmax: Optional[np.ndarray | float],
    transform: str,
    gamma: float,
    log_alpha: float,
) -> None:
    if dst_path.exists() and not overwrite:
        return

    npz_files = _list_numbered_files(map_dir, ".npz")
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {map_dir}")
    npz_files = npz_files[: min(num_frames, len(npz_files))]

    def _frames():
        for name in _tqdm_wrap(
            npz_files,
            total=len(npz_files),
            desc=f"Map {dst_path.name}",
            unit="frame",
            invisible=True,
        ):
            arr = _load_npz_array(map_dir / name)
            hwc = _map_to_hwc(arr)
            rgb = _hwc_to_rgb3(hwc)
            local_min: Optional[np.ndarray | float] = vmin
            local_max: Optional[np.ndarray | float] = vmax
            if normalize == "minmax_per_frame":
                local_min = None
                local_max = None
            yield _rgb_to_rgb_u8(
                rgb,
                gain=gain,
                normalize=normalize,
                vmin=local_min,
                vmax=local_max,
                transform=transform,
                gamma=gamma,
                log_alpha=log_alpha,
            )

    _write_video_streaming(_frames(), dst_path, fps=fps)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    sample_dir: Path
    rgb_frames_dir: Path
    modality_dirs: dict[str, Path]


def _discover_samples(src_root: Path) -> list[Sample]:
    MODALITY_DIRS = {
        "strain_video": "2d_strain_map",
        "stress_video": "2d_stress_map",
        "flow_video":   "2d_vel_map",
        "damage_video": "2d_damage_map",
        "force_video":  "2d_force_map",
    }
    REQUIRED_ANY = list(MODALITY_DIRS.values())
    RGB_CANDIDATES = ["img_frames"]  # checked in order

    def looks_like_sample(p: Path) -> bool:
        if not p.is_dir():
            return False
        has_modality = any((p / d).is_dir() for d in REQUIRED_ANY)
        has_rgb = any((p / d).is_dir() for d in RGB_CANDIDATES)
        return has_modality and has_rgb

    candidates: list[Path] = []
    if looks_like_sample(src_root):
        candidates = [src_root]
    else:
        for child in sorted(src_root.iterdir()):
            if looks_like_sample(child):
                candidates.append(child)
    samples: list[Sample] = []
    for idx, sample_dir in enumerate(candidates):
        sample_id = sample_dir.name
        rgb_frames_dir = sample_dir / "img_frames" if (sample_dir / "img_frames").is_dir() else sample_dir

        modality_dirs = {
            "strain_video": sample_dir / "2d_strain_map",
            "stress_video": sample_dir / "2d_stress_map",
            "flow_video": sample_dir / "2d_vel_map",
            "damage_video": sample_dir / "2d_damage_map",
        }
        # force is optional but commonly present
        if (sample_dir / "2d_force_map").is_dir():
            modality_dirs["force_video"] = sample_dir / "2d_force_map"

        samples.append(
            Sample(
                sample_id=sample_id,
                sample_dir=sample_dir,
                rgb_frames_dir=rgb_frames_dir,
                modality_dirs=modality_dirs,
            )
        )
    return samples


def _set_tqdm_invisible(invisible: bool) -> None:
    global _TQDM_INVISIBLE
    _TQDM_INVISIBLE = bool(invisible)


def _resolve_sample_config_path(src_config_root: Path, sample_id: str) -> Path:
    candidates = [
        src_config_root / f"{sample_id}.json",
        src_config_root / sample_id / "config.json",
        src_config_root / sample_id / f"{sample_id}.json",
    ]
    for path in candidates:
        if path.is_file():
            return path

    glob_candidates = sorted(src_config_root.glob(f"*_{sample_id}.json"))
    if len(glob_candidates) == 1:
        return glob_candidates[0]
    if len(glob_candidates) > 1:
        raise FileNotFoundError(
            f"Found multiple config files for sample '{sample_id}' under {src_config_root}: "
            f"{', '.join(str(p) for p in glob_candidates)}"
        )

    raise FileNotFoundError(
        f"Could not find config for sample '{sample_id}' under {src_config_root}. "
        f"Tried: {', '.join(str(p) for p in candidates)} and glob '*_{sample_id}.json'"
    )


def _split_list_argument(values: Optional[list[str]]) -> list[str]:
    if not values:
        return []

    out: list[str] = []
    for value in values:
        for token in str(value).split(","):
            item = token.strip()
            if item:
                out.append(item)
    return out


def _resolve_requested_sample_ids(values: Optional[list[str]], available_ids: set[str]) -> tuple[set[str], list[str]]:
    requested_ids: set[str] = set()
    unknown_tokens: list[str] = []

    for token in _split_list_argument(values):
        normalized = token[:-5] if token.endswith(".json") else token
        candidates = [normalized]
        if normalized.isdigit():
            candidates.append(f"config_{normalized}")

        matches = [candidate for candidate in candidates if candidate in available_ids]
        if matches:
            requested_ids.update(matches)
        else:
            unknown_tokens.append(token)

    return requested_ids, unknown_tokens


def _category_id_from_config_path(config_path: Path) -> int:
    """
    Binary category inferred from the config filename: any object whose name contains
    "rubber" is category 1 (rubber-like), everything else is category 0 (bread-like).
    Accepts any object name of the form <name>_config_xxxx.json -- not limited to a
    fixed list of known object names, since the dataset now spans many more objects.
    """
    return 1 if "rubber" in config_path.stem.lower() else 0


def _load_material_properties(src_config_root: Optional[Path], sample_id: str) -> dict[str, str]:
    if src_config_root is None:
        return {}

    config_path = _resolve_sample_config_path(src_config_root, sample_id)
    with open(config_path, "r") as f:
        config = json.load(f)

    if "E" not in config:
        raise KeyError(f"Missing 'E' in config: {config_path}")
    if "nu" not in config:
        raise KeyError(f"Missing 'nu' in config: {config_path}")
    if "damage" not in config or not isinstance(config["damage"], dict):
        raise KeyError(f"Missing damage object in config: {config_path}")
    damage_config = config["damage"]
    if "lambda_onset" not in damage_config:
        raise KeyError(f"Missing damage.lambda_onset in config: {config_path}")
    if "lambda_crit" not in damage_config:
        raise KeyError(f"Missing damage.lambda_crit in config: {config_path}")

    return {
        "young_modulus": str(config["E"]),
        "poisson_ratio": str(config["nu"]),
        "lambda_onset": str(damage_config["lambda_onset"]),
        "lambda_crit": str(damage_config["lambda_crit"]),
        "category": str(_category_id_from_config_path(config_path)),
    }


def _convert_one_sample(
    sample_id: str,
    rgb_frames_dir: Path,
    modality_dirs: dict[str, Path],
    src_config_root: Optional[Path],
    dst_root: Path,
    videos_dir: Path,
    out_dirs: dict[str, Path],
    num_frames_max: int,
    fps: int,
    prompt: str,
    overwrite: bool,
    map_gain: float,
    normalize: str,
    transform: str,
    gamma: float,
    log_alpha: float,
    flow_norm: str,
    flow_percentile: float,
    flow_sample_stride: int,
    flow_gain: float,
    normalize_percentile_low: float = 1.0,
    normalize_percentile_high: float = 99.0,
    normalize_sample_stride: int = 1,
    overwrite_modalities: frozenset = frozenset(),
) -> dict[str, str]:
    png_files = _list_numbered_files(rgb_frames_dir, ".png")
    if not png_files:
        raise FileNotFoundError(f"No RGB frames found in {rgb_frames_dir}")

    modality_files: dict[str, list[str]] = {}
    for key, mdir in modality_dirs.items():
        if not mdir.is_dir():
            raise FileNotFoundError(f"Missing modality directory for {key}: {mdir}")
        mf = _list_numbered_files(mdir, ".npz")
        if not mf:
            raise FileNotFoundError(f"No .npz frames for {key} in {mdir}")
        modality_files[key] = mf

    n = int(num_frames_max)
    n = min(n, len(png_files), *(len(v) for v in modality_files.values()))
    if n <= 0:
        raise RuntimeError(f"No aligned frames available for sample {sample_id}")

    dst_rgb = videos_dir / f"{sample_id}.mp4"
    _build_rgb_video(rgb_frames_dir, dst_rgb, num_frames=n, fps=int(fps), overwrite=bool(overwrite))

    video_minmax: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if str(normalize) == "minmax_per_video":
        for key, mdir in modality_dirs.items():
            if key == "flow_video":
                continue
            files = modality_files[key][:n]
            video_minmax[key] = _compute_video_minmax_rgb(mdir, files)
    elif str(normalize) == "percentile_per_video":
        for key, mdir in modality_dirs.items():
            if key == "flow_video":
                continue
            files = modality_files[key][:n]
            video_minmax[key] = _compute_video_percentile_range_rgb(
                mdir, files,
                percentile_low=float(normalize_percentile_low),
                percentile_high=float(normalize_percentile_high),
                sample_stride=int(normalize_sample_stride),
            )

    row: dict[str, str] = {"video": _relpath(dst_rgb, dst_root), "prompt": str(prompt)}
    row.update(_load_material_properties(src_config_root, sample_id))

    for key, mdir in modality_dirs.items():
        out_subdir = out_dirs[key]
        suffix = key.replace("_video", "")
        dst_map_video = out_subdir / f"{sample_id}_{suffix}.mp4"
        key_overwrite = bool(overwrite) or (key in overwrite_modalities)
        if key == "flow_video":
            _build_flow_video(
                mdir,
                dst_map_video,
                num_frames=n,
                fps=int(fps),
                overwrite=key_overwrite,
                flow_norm=str(flow_norm),
                flow_percentile=float(flow_percentile),
                flow_sample_stride=int(flow_sample_stride),
                flow_gain=float(flow_gain),
            )
        else:
            vmin: Optional[np.ndarray | float] = None
            vmax: Optional[np.ndarray | float] = None
            if str(normalize) in ("minmax_per_video", "percentile_per_video"):
                vmin, vmax = video_minmax[key]

            _build_map_video(
                mdir,
                dst_map_video,
                num_frames=n,
                fps=int(fps),
                overwrite=key_overwrite,
                gain=float(map_gain),
                normalize=str(normalize),
                vmin=vmin,
                vmax=vmax,
                transform=str(transform),
                gamma=float(gamma),
                log_alpha=float(log_alpha),
            )
        row[key] = _relpath(dst_map_video, dst_root)

    return row


def _build_output_row(
    sample_id: str,
    modality_keys: Iterable[str],
    src_config_root: Optional[Path],
    dst_root: Path,
    videos_dir: Path,
    out_dirs: dict[str, Path],
    prompt: str,
) -> dict[str, str]:
    row: dict[str, str] = {
        "video": _relpath(videos_dir / f"{sample_id}.mp4", dst_root),
        "prompt": str(prompt),
    }
    row.update(_load_material_properties(src_config_root, sample_id))

    for key in modality_keys:
        suffix = key.replace("_video", "")
        row[key] = _relpath(out_dirs[key] / f"{sample_id}_{suffix}.mp4", dst_root)

    return row


def _sample_outputs_exist(
    sample_id: str,
    modality_keys: Iterable[str],
    videos_dir: Path,
    out_dirs: dict[str, Path],
) -> bool:
    if not (videos_dir / f"{sample_id}.mp4").is_file():
        return False

    for key in modality_keys:
        suffix = key.replace("_video", "")
        if not (out_dirs[key] / f"{sample_id}_{suffix}.mp4").is_file():
            return False
    return True


def _convert_one_sample_task(task: dict) -> dict[str, str]:  # type: ignore[no-untyped-def]
    modality_dirs = {str(k): Path(str(v)) for k, v in task["modality_dirs"].items()}
    out_dirs = {str(k): Path(str(v)) for k, v in task["out_dirs"].items()}
    src_config_root = task.get("src_config_root")
    row = _convert_one_sample(
        sample_id=str(task["sample_id"]),
        rgb_frames_dir=Path(str(task["rgb_frames_dir"])),
        modality_dirs=modality_dirs,
        src_config_root=None if src_config_root is None else Path(str(src_config_root)),
        dst_root=Path(str(task["dst_root"])),
        videos_dir=Path(str(task["videos_dir"])),
        out_dirs=out_dirs,
        num_frames_max=int(task["num_frames_max"]),
        fps=int(task["fps"]),
        prompt=str(task["prompt"]),
        overwrite=bool(task["overwrite"]),
        overwrite_modalities=frozenset(task.get("overwrite_modalities") or []),
        map_gain=float(task["map_gain"]),
        normalize=str(task["normalize"]),
        transform=str(task["transform"]),
        gamma=float(task["gamma"]),
        log_alpha=float(task["log_alpha"]),
        flow_norm=str(task["flow_norm"]),
        flow_percentile=float(task["flow_percentile"]),
        flow_sample_stride=int(task["flow_sample_stride"]),
        flow_gain=float(task["flow_gain"]),
        normalize_percentile_low=float(task.get("normalize_percentile_low", 1.0)),
        normalize_percentile_high=float(task.get("normalize_percentile_high", 99.0)),
        normalize_sample_stride=int(task.get("normalize_sample_stride", 1)),
    )
    split_name = task.get("split_name")
    if split_name is not None:
        row["__split_name"] = str(split_name)
    return row


def _prepare_dataset_dirs(dataset_root: Path) -> tuple[Path, dict[str, Path]]:
    _safe_mkdir(dataset_root)
    videos_dir = dataset_root / "videos"
    out_dirs = {
        "strain_video": dataset_root / "strain_videos",
        "stress_video": dataset_root / "stress_videos",
        "flow_video": dataset_root / "flow_videos",
        "damage_video": dataset_root / "damage_videos",
        "force_video": dataset_root / "force_videos",
    }
    _safe_mkdir(videos_dir)
    for p in out_dirs.values():
        _safe_mkdir(p)
    return videos_dir, out_dirs


def _build_sample_task(
    sample: Sample,
    dataset_root: Path,
    videos_dir: Path,
    out_dirs: dict[str, Path],
    src_config_root: Optional[Path],
    args: argparse.Namespace,
    overwrite: bool,
    split_name: Optional[str] = None,
) -> dict:
    task = {
        "sample_id": sample.sample_id,
        "rgb_frames_dir": str(sample.rgb_frames_dir),
        "modality_dirs": {k: str(v) for k, v in sample.modality_dirs.items()},
        "src_config_root": None if src_config_root is None else str(src_config_root),
        "dst_root": str(dataset_root),
        "videos_dir": str(videos_dir),
        "out_dirs": {k: str(v) for k, v in out_dirs.items()},
        "num_frames_max": int(args.num_frames),
        "fps": int(args.fps),
        "prompt": str(args.prompt),
        "overwrite": bool(overwrite),
        "overwrite_modalities": list(getattr(args, "overwrite_modalities", None) or []),
        "map_gain": float(args.map_gain),
        "normalize": str(args.normalize),
        "transform": str(args.transform),
        "gamma": float(args.gamma),
        "log_alpha": float(args.log_alpha),
        "flow_norm": str(args.flow_norm),
        "flow_percentile": float(args.flow_percentile),
        "flow_sample_stride": int(args.flow_sample_stride),
        "flow_gain": float(args.flow_gain),
        "normalize_percentile_low": float(args.normalize_percentile_low),
        "normalize_percentile_high": float(args.normalize_percentile_high),
        "normalize_sample_stride": int(args.normalize_sample_stride),
    }
    if split_name is not None:
        task["split_name"] = split_name
    return task


def _prepare_sample_row_or_task(
    sample: Sample,
    dataset_root: Path,
    videos_dir: Path,
    out_dirs: dict[str, Path],
    src_config_root: Optional[Path],
    args: argparse.Namespace,
    requested_sample_ids: Optional[set[str]],
    split_name: str,
) -> tuple[Optional[dict], Optional[dict[str, str]]]:
    modality_keys = list(sample.modality_dirs.keys())
    outputs_exist = _sample_outputs_exist(
        sample_id=sample.sample_id,
        modality_keys=modality_keys,
        videos_dir=videos_dir,
        out_dirs=out_dirs,
    )

    force_overwrite = bool(args.overwrite)
    if requested_sample_ids is not None:
        force_overwrite = force_overwrite or sample.sample_id in requested_sample_ids
        if sample.sample_id not in requested_sample_ids:
            if not outputs_exist:
                raise FileNotFoundError(
                    f"Sample {sample.sample_id} is missing generated outputs under {dataset_root}, "
                    "but it was not included in --corrupted_config_ids. Add it to the rerun list "
                    "or rerun without --corrupted_config_ids to rebuild everything needed."
                )
            row = _build_output_row(
                sample_id=sample.sample_id,
                modality_keys=modality_keys,
                src_config_root=src_config_root,
                dst_root=dataset_root,
                videos_dir=videos_dir,
                out_dirs=out_dirs,
                prompt=str(args.prompt),
            )
            row["__split_name"] = split_name
            return None, row

    overwrite_modalities = set(getattr(args, "overwrite_modalities", None) or [])
    modality_needs_overwrite = bool(overwrite_modalities & set(modality_keys))
    if not force_overwrite and not modality_needs_overwrite and outputs_exist:
        row = _build_output_row(
            sample_id=sample.sample_id,
            modality_keys=modality_keys,
            src_config_root=src_config_root,
            dst_root=dataset_root,
            videos_dir=videos_dir,
            out_dirs=out_dirs,
            prompt=str(args.prompt),
        )
        row["__split_name"] = split_name
        return None, row

    task = _build_sample_task(
        sample=sample,
        dataset_root=dataset_root,
        videos_dir=videos_dir,
        out_dirs=out_dirs,
        src_config_root=src_config_root,
        args=args,
        overwrite=force_overwrite,
        split_name=split_name,
    )
    return task, None


def _run_conversion_tasks(tasks: list[dict], workers: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not tasks:
        return rows

    if workers <= 1:
        for task in _tqdm_wrap(tasks, total=len(tasks), desc="Converting samples", unit="sample"):
            rows.append(_convert_one_sample_task(task))
    else:
        from concurrent.futures import ProcessPoolExecutor

        chunksize = max(1, len(tasks) // max(1, workers * 4))
        with ProcessPoolExecutor(max_workers=workers, initializer=_set_tqdm_invisible, initargs=(True,)) as ex:
            it = ex.map(_convert_one_sample_task, tasks, chunksize=chunksize)
            for row in _tqdm_wrap(it, total=len(tasks), desc=f"Converting samples ({workers} workers)", unit="sample"):
                rows.append(row)
    return rows


def _write_metadata(rows: list[dict[str, str]], metadata_path: Path, fieldnames: list[str]) -> None:
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_root",
        type=str,
        default="/weka/scratch/bhattad/tnguy303/research/PhysGaussian/output_exp_update_gs",
        help="Path to a PhysGaussian output folder (or a directory containing many such folders).",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        default=str(_repo_root() / "data" / "physgaussian_multimodal_update_gs"),
        help="Output dataset folder (will be created).",
    )
    parser.add_argument(
        "--src_config",
        type=str,
        default=None,
        help=(
            "Optional directory containing per-sample config JSON files. "
            "Supports either <src_config>/<sample_id>.json or <src_config>/<sample_id>/config.json."
        ),
    )
    parser.add_argument("--num_frames", type=int, default=81, help="Max frames to encode for every modality.")
    parser.add_argument(
        "--test_k",
        type=int,
        default=None,
        help="Optionally process only the first k discovered samples for quick testing.",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.95,
        help="Train split ratio in [0, 1]. Samples are deterministically split in discovery order.",
    )
    parser.add_argument("--fps", type=int, default=25, help="FPS for ALL generated videos.")
    parser.add_argument("--prompt", type=str, default="physical simulation deformation (multimodal-conditioned)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated videos.")
    parser.add_argument(
        "--overwrite_modalities",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=None,
        help=(
            "Comma-separated modality keys to force-overwrite even when --overwrite is not set. "
            "Other modalities are skipped if their output already exists. "
            "Example: --overwrite_modalities flow_video"
        ),
    )
    parser.add_argument(
        "--corrupted_config_ids",
        nargs="*",
        default=None,
        help=(
            "Optional list of sample/config ids to reprocess. Accepts whitespace- or comma-separated "
            "values like `config_0001 config_0002` or `config_0001,config_0002`. These samples are "
            "force-overwritten even when --overwrite is not set."
        ),
    )

    parser.add_argument("--map_gain", type=float, default=1.0, help="Multiply maps by this gain before normalization.")
    parser.add_argument(
        "--normalize",
        type=str,
        default="minmax_per_video",
        choices=["clip01", "minmax_per_frame", "minmax_per_video", "percentile_per_video"],
        help="How to scale map values into [0,1] before encoding as video. percentile_per_video "
             "(percentile of nonzero pixels, excluding background -- matches PhysGaussian's "
             "vis_phys_map.py) is more robust than minmax_per_video's literal min/max, which a "
             "single outlier pixel anywhere in the video can distort for every frame.",
    )
    parser.add_argument(
        "--normalize_percentile_low", type=float, default=1.0,
        help="Used when --normalize percentile_per_video.",
    )
    parser.add_argument(
        "--normalize_percentile_high", type=float, default=99.0,
        help="Used when --normalize percentile_per_video.",
    )
    parser.add_argument(
        "--normalize_sample_stride", type=int, default=1,
        help="Used when --normalize percentile_per_video. Stride for sampling pixels when "
             "estimating the percentile range (memory/speed tradeoff, matches --flow_sample_stride).",
    )
    parser.add_argument("--transform", type=str, default="none", choices=["none", "gamma", "log"])
    parser.add_argument("--gamma", type=float, default=0.5, help="Used when --transform gamma.")
    parser.add_argument("--log_alpha", type=float, default=50.0, help="Used when --transform log.")

    # Flow (2d_vel_map) visualization options (HSV optical-flow style).
    parser.add_argument(
        "--flow_norm",
        type=str,
        default="percentile",
        choices=["percentile", "max"],
        help="How to choose magnitude scale for flow visualization.",
    )
    parser.add_argument("--flow_percentile", type=float, default=99.0, help="Used when --flow_norm percentile.")
    parser.add_argument("--flow_sample_stride", type=int, default=8, help="Stride for sampling pixels when estimating flow scale.")
    parser.add_argument("--flow_gain", type=float, default=1.0, help="Multiply (u,v) by this before visualizing flow.")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes for per-sample conversion (1 disables multiprocessing, 0 uses all CPUs).",
    )
    args = parser.parse_args(argv)

    src_root = Path(args.src_root).resolve()
    src_config_root = None if args.src_config is None else Path(args.src_config).resolve()
    dst_root_input = Path(args.dst_root)
    dst_root = dst_root_input if dst_root_input.is_absolute() else (_repo_root() / dst_root_input)
    _safe_mkdir(dst_root)

    samples = _discover_samples(src_root)
    if not samples:
        raise FileNotFoundError(f"No samples found under: {src_root}")
    if args.test_k is not None:
        if int(args.test_k) <= 0:
            raise ValueError(f"--test_k must be > 0, got {args.test_k}")
        samples = samples[: int(args.test_k)]
    if not 0.0 <= float(args.split_ratio) <= 1.0:
        raise ValueError(f"--split_ratio must be in [0, 1], got {args.split_ratio}")

    available_sample_ids = {sample.sample_id for sample in samples}
    requested_sample_ids, unknown_tokens = _resolve_requested_sample_ids(
        args.corrupted_config_ids,
        available_sample_ids,
    )
    if unknown_tokens:
        raise ValueError(
            "Unknown values passed to --corrupted_config_ids: "
            f"{', '.join(unknown_tokens)}"
        )
    targeted_rerun_ids = requested_sample_ids if requested_sample_ids else None

    # Stable column order; force_video is optional.
    fieldnames = [
        "video",
        "prompt",
        "force_video",
        "stress_video",
        "flow_video",
        "damage_video",
        "strain_video",
        "young_modulus",
        "poisson_ratio",
        "lambda_onset",
        "lambda_crit",
        "category",
    ]
    num_train = int(len(samples) * float(args.split_ratio))
    train_samples = samples[:num_train]
    test_samples = samples[num_train:]

    train_root = dst_root / "train"
    test_root = dst_root / "test"
    split_jobs = {
        "train": (train_samples, train_root),
        "test": (test_samples, test_root),
    }
    split_rows: dict[str, list[dict[str, str]]] = {"train": [], "test": []}
    all_tasks: list[dict] = []
    skipped_existing = 0

    for split_name, (split_samples, split_root) in _tqdm_wrap(
        list(split_jobs.items()),
        total=len(split_jobs),
        desc="Preparing dataset splits",
        unit="split",
    ):
        videos_dir, out_dirs = _prepare_dataset_dirs(split_root)
        for sample in split_samples:
            task, existing_row = _prepare_sample_row_or_task(
                sample=sample,
                dataset_root=split_root,
                videos_dir=videos_dir,
                out_dirs=out_dirs,
                src_config_root=src_config_root,
                args=args,
                requested_sample_ids=targeted_rerun_ids,
                split_name=split_name,
            )
            if task is not None:
                all_tasks.append(task)
            if existing_row is not None:
                skipped_existing += 1
                split_rows[split_name].append(existing_row)

    workers = int(args.workers)
    if workers <= 0:
        workers = os.cpu_count() or 1
    converted_rows = _run_conversion_tasks(all_tasks, workers=workers)
    for row in converted_rows:
        split_name = row.pop("__split_name")
        split_rows[split_name].append(row)

    train_metadata_path = train_root / "metadata.csv"
    test_metadata_path = test_root / "metadata.csv"
    _write_metadata(split_rows["train"], train_metadata_path, fieldnames)
    _write_metadata(split_rows["test"], test_metadata_path, fieldnames)

    print("Done.")
    print(f"Dataset written to: {dst_root}")
    print(f"Train metadata: {train_metadata_path}")
    print(f"Test metadata: {test_metadata_path}")
    print(f"Skipped existing samples: {skipped_existing}")
    print(f"Converted samples this run: {len(converted_rows)}")
    if targeted_rerun_ids is not None:
        print(f"Targeted corrupted configs: {len(targeted_rerun_ids)}")
    print()
    print("Training usage (example):")
    print(
        "  accelerate launch examples/wanvideo/model_training/train_forceCond.py \\\n"
        f"    --dataset_base_path \"{train_root}\" \\\n"
        f"    --dataset_metadata_path \"{train_metadata_path}\" \\\n"
        "    --data_file_keys \"video,force_video,stress_video,flow_video,damage_video,young_modulus,poisson_ratio,lambda_onset,lambda_crit,category\" \\\n"
        "    --extra_inputs \"input_image\" \\\n"
        "    ..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
