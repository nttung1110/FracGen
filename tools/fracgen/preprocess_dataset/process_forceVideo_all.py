"""
Process PhysGaussian multi-modal maps into DiffSynth UnifiedDataset format.

This script converts a PhysGaussian output folder (e.g.):
  /weka/scratch/bhattad/tnguy303/research/PhysGaussian/output_exp_update_gs

...into a dataset folder under this repo (default):
  data/physgaussian_multimodal_update_gs/

Output schema:
  metadata.csv columns:
    video,prompt,force_video,stress_video,flow_video,damage_video,strain_video

Folders created under dst_root:
  videos/         -> RGB mp4 (from numbered PNGs)
  force_videos/   -> mp4 encoded from 2d_force_map/*.npy (optional if src has it)
  stress_videos/  -> mp4 encoded from 2d_stress_map/*.npy
  flow_videos/    -> mp4 encoded from 2d_vel_map/*.npy
  damage_videos/  -> mp4 encoded from 2d_damage_map/*.npy
  strain_videos/  -> mp4 encoded from 2d_strain_map/*.npy

Notes:
- Frame alignment is by index (0000, 0001, ...). We enforce that all requested modalities
  have at least N frames, then encode the first N frames for every modality.
- Map-to-video conversion writes grayscale RGB (uint8) videos for compatibility with the
  existing video loader operator.
"""

from __future__ import annotations

import argparse
import csv
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

    if normalize == "minmax_per_video":
        if vmin is None or vmax is None:
            raise ValueError("vmin/vmax required for minmax_per_video")
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


def _compute_video_minmax_rgb(map_dir: Path, npy_files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    vmin = np.full((3,), np.inf, dtype=np.float32)
    vmax = np.full((3,), -np.inf, dtype=np.float32)
    for name in npy_files:
        arr = np.load(str(map_dir / name))
        hwc = _map_to_hwc(arr)
        rgb = _hwc_to_rgb3(hwc)
        a = np.min(rgb, axis=(0, 1))
        b = np.max(rgb, axis=(0, 1))
        vmin = np.minimum(vmin, a.astype(np.float32))
        vmax = np.maximum(vmax, b.astype(np.float32))
    if not np.all(np.isfinite(vmin)) or not np.all(np.isfinite(vmax)):
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return vmin.astype(np.float32), vmax.astype(np.float32)


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
    npy_files: list[str],
    method: str = "percentile",
    percentile: float = 99.0,
    sample_stride: int = 8,
    gain: float = 1.0,
) -> float:
    """
    Compute a robust magnitude scale for optical-flow visualization.
    We use a strided pixel sample to avoid huge memory.
    """
    mags: list[np.ndarray] = []
    rad_max = 0.0
    for name in npy_files:
        arr = np.load(str(flow_dir / name))
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
        if allm.size == 0:
            return 1.0
        q = float(np.quantile(allm, np.clip(percentile, 0.0, 100.0) / 100.0))
        rad_max = q

    if not np.isfinite(rad_max) or rad_max <= 1e-8:
        rad_max = 1.0
    return float(rad_max)


def _flow_to_rgb01(flow_hwc2: np.ndarray, *, rad_max: float, gain: float = 1.0) -> np.ndarray:
    """
    Standard flow visualization:
    - Hue encodes direction (angle)
    - Saturation encodes magnitude (normalized by rad_max)
    - Value fixed to 1
    """
    if flow_hwc2.ndim != 3 or flow_hwc2.shape[-1] != 2:
        raise ValueError(f"Expected flow HxWx2, got {flow_hwc2.shape}")
    u = flow_hwc2[..., 0] * float(gain)
    v = flow_hwc2[..., 1] * float(gain)
    ang = np.arctan2(v, u).astype(np.float32)  # [-pi, pi]
    hue = (ang + np.pi) / (2.0 * np.pi)  # [0,1]
    mag = np.sqrt(u * u + v * v, dtype=np.float32)
    sat = np.clip(mag / float(rad_max), 0.0, 1.0)
    val = np.ones_like(sat, dtype=np.float32)
    hsv = np.stack([hue, sat, val], axis=-1).astype(np.float32)
    return _hsv_to_rgb(hsv)


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
    npy_files = _list_numbered_files(flow_dir, ".npy")
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {flow_dir}")
    npy_files = npy_files[: min(num_frames, len(npy_files))]

    rad_max = _compute_flow_rad_max(
        flow_dir,
        npy_files,
        method=flow_norm,
        percentile=flow_percentile,
        sample_stride=flow_sample_stride,
        gain=flow_gain,
    )

    def _frames():
        for name in _tqdm_wrap(
            npy_files,
            total=len(npy_files),
            desc=f"Flow {dst_path.name}",
            unit="frame",
            invisible=True,
        ):
            arr = np.load(str(flow_dir / name))
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

    npy_files = _list_numbered_files(map_dir, ".npy")
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {map_dir}")
    npy_files = npy_files[: min(num_frames, len(npy_files))]

    def _frames():
        for name in _tqdm_wrap(
            npy_files,
            total=len(npy_files),
            desc=f"Map {dst_path.name}",
            unit="frame",
            invisible=True,
        ):
            arr = np.load(str(map_dir / name))
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


def _convert_one_sample(
    sample_id: str,
    rgb_frames_dir: Path,
    modality_dirs: dict[str, Path],
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
) -> dict[str, str]:
    png_files = _list_numbered_files(rgb_frames_dir, ".png")
    if not png_files:
        raise FileNotFoundError(f"No RGB frames found in {rgb_frames_dir}")

    modality_files: dict[str, list[str]] = {}
    for key, mdir in modality_dirs.items():
        if not mdir.is_dir():
            raise FileNotFoundError(f"Missing modality directory for {key}: {mdir}")
        mf = _list_numbered_files(mdir, ".npy")
        if not mf:
            raise FileNotFoundError(f"No .npy frames for {key} in {mdir}")
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

    row: dict[str, str] = {"video": _relpath(dst_rgb, dst_root), "prompt": str(prompt)}

    for key, mdir in modality_dirs.items():
        out_subdir = out_dirs[key]
        suffix = key.replace("_video", "")
        dst_map_video = out_subdir / f"{sample_id}_{suffix}.mp4"
        if key == "flow_video":
            _build_flow_video(
                mdir,
                dst_map_video,
                num_frames=n,
                fps=int(fps),
                overwrite=bool(overwrite),
                flow_norm=str(flow_norm),
                flow_percentile=float(flow_percentile),
                flow_sample_stride=int(flow_sample_stride),
                flow_gain=float(flow_gain),
            )
        else:
            vmin: Optional[np.ndarray | float] = None
            vmax: Optional[np.ndarray | float] = None
            if str(normalize) == "minmax_per_video":
                vmin, vmax = video_minmax[key]

            _build_map_video(
                mdir,
                dst_map_video,
                num_frames=n,
                fps=int(fps),
                overwrite=bool(overwrite),
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


def _convert_one_sample_task(task: dict) -> dict[str, str]:  # type: ignore[no-untyped-def]
    modality_dirs = {str(k): Path(str(v)) for k, v in task["modality_dirs"].items()}
    out_dirs = {str(k): Path(str(v)) for k, v in task["out_dirs"].items()}
    return _convert_one_sample(
        sample_id=str(task["sample_id"]),
        rgb_frames_dir=Path(str(task["rgb_frames_dir"])),
        modality_dirs=modality_dirs,
        dst_root=Path(str(task["dst_root"])),
        videos_dir=Path(str(task["videos_dir"])),
        out_dirs=out_dirs,
        num_frames_max=int(task["num_frames_max"]),
        fps=int(task["fps"]),
        prompt=str(task["prompt"]),
        overwrite=bool(task["overwrite"]),
        map_gain=float(task["map_gain"]),
        normalize=str(task["normalize"]),
        transform=str(task["transform"]),
        gamma=float(task["gamma"]),
        log_alpha=float(task["log_alpha"]),
        flow_norm=str(task["flow_norm"]),
        flow_percentile=float(task["flow_percentile"]),
        flow_sample_stride=int(task["flow_sample_stride"]),
        flow_gain=float(task["flow_gain"]),
    )


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
    parser.add_argument("--num_frames", type=int, default=81, help="Max frames to encode for every modality.")
    parser.add_argument("--fps", type=int, default=25, help="FPS for ALL generated videos.")
    parser.add_argument("--prompt", type=str, default="physical simulation deformation (multimodal-conditioned)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated videos.")

    parser.add_argument("--map_gain", type=float, default=1.0, help="Multiply maps by this gain before normalization.")
    parser.add_argument(
        "--normalize",
        type=str,
        default="minmax_per_video",
        choices=["clip01", "minmax_per_frame", "minmax_per_video"],
        help="How to scale map values into [0,1] before encoding as video.",
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
        help="Number of worker processes for per-sample conversion (1 disables multiprocessing).",
    )
    args = parser.parse_args(argv)

    src_root = Path(args.src_root).resolve()
    dst_root_input = Path(args.dst_root)
    dst_root = dst_root_input if dst_root_input.is_absolute() else (_repo_root() / dst_root_input)
    _safe_mkdir(dst_root)

    samples = _discover_samples(src_root)
    if not samples:
        raise FileNotFoundError(f"No samples found under: {src_root}")

    videos_dir = dst_root / "videos"
    out_dirs = {
        "strain_video": dst_root / "strain_videos",
        "stress_video": dst_root / "stress_videos",
        "flow_video": dst_root / "flow_videos",
        "damage_video": dst_root / "damage_videos",
        "force_video": dst_root / "force_videos",
    }
    _safe_mkdir(videos_dir)
    for p in out_dirs.values():
        _safe_mkdir(p)

    rows: list[dict[str, str]] = []
    workers = int(args.workers)
    if workers <= 1:
        for s in _tqdm_wrap(samples, total=len(samples), desc="Converting samples", unit="sample"):
            rows.append(
                _convert_one_sample(
                    sample_id=s.sample_id,
                    rgb_frames_dir=s.rgb_frames_dir,
                    modality_dirs=s.modality_dirs,
                    dst_root=dst_root,
                    videos_dir=videos_dir,
                    out_dirs=out_dirs,
                    num_frames_max=int(args.num_frames),
                    fps=int(args.fps),
                    prompt=str(args.prompt),
                    overwrite=bool(args.overwrite),
                    map_gain=float(args.map_gain),
                    normalize=str(args.normalize),
                    transform=str(args.transform),
                    gamma=float(args.gamma),
                    log_alpha=float(args.log_alpha),
                    flow_norm=str(args.flow_norm),
                    flow_percentile=float(args.flow_percentile),
                    flow_sample_stride=int(args.flow_sample_stride),
                    flow_gain=float(args.flow_gain),
                )
            )
    else:
        from concurrent.futures import ProcessPoolExecutor

        tasks: list[dict] = []
        for s in samples:
            tasks.append(
                {
                    "sample_id": s.sample_id,
                    "rgb_frames_dir": str(s.rgb_frames_dir),
                    "modality_dirs": {k: str(v) for k, v in s.modality_dirs.items()},
                    "dst_root": str(dst_root),
                    "videos_dir": str(videos_dir),
                    "out_dirs": {k: str(v) for k, v in out_dirs.items()},
                    "num_frames_max": int(args.num_frames),
                    "fps": int(args.fps),
                    "prompt": str(args.prompt),
                    "overwrite": bool(args.overwrite),
                    "map_gain": float(args.map_gain),
                    "normalize": str(args.normalize),
                    "transform": str(args.transform),
                    "gamma": float(args.gamma),
                    "log_alpha": float(args.log_alpha),
                    "flow_norm": str(args.flow_norm),
                    "flow_percentile": float(args.flow_percentile),
                    "flow_sample_stride": int(args.flow_sample_stride),
                    "flow_gain": float(args.flow_gain),
                }
            )

        with ProcessPoolExecutor(max_workers=workers, initializer=_set_tqdm_invisible, initargs=(True,)) as ex:
            it = ex.map(_convert_one_sample_task, tasks)
            for row in _tqdm_wrap(it, total=len(tasks), desc=f"Converting samples ({workers} workers)", unit="sample"):
                rows.append(row)

    # Stable column order; force_video is optional.
    fieldnames = ["video", "prompt", "force_video", "stress_video", "flow_video", "damage_video", "strain_video"]
    metadata_path = dst_root / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Done.")
    print(f"Dataset written to: {dst_root}")
    print(f"Metadata: {metadata_path}")
    print()
    print("Training usage (example):")
    print(
        "  accelerate launch examples/wanvideo/model_training/train_forceCond.py \\\n"
        f"    --dataset_base_path \"{dst_root}\" \\\n"
        f"    --dataset_metadata_path \"{metadata_path}\" \\\n"
        "    --data_file_keys \"video,force_video,stress_video,flow_video,damage_video\" \\\n"
        "    --extra_inputs \"input_image\" \\\n"
        "    ..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
