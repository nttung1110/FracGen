"""
Process PhysGaussian force-video dataset into DiffSynth UnifiedDataset format.

This script converts the dataset at:
  /weka/scratch/bhattad/tnguy303/research/PhysGaussian/all_bread_data/

...into a folder under this repo:
  data/force_dataset_20/

Compatible with:
  examples/wanvideo/model_training/train_forceCond.py

Expected output schema (same as data/example_force_dataset/metadata.csv):
  metadata.csv columns: video,prompt,force_video

Each sample provides:
- video: an RGB video (mp4) encoded from frame PNGs (0000.png, 0001.png, ...)
- force_video: an RGB video (mp4) encoded from 2d_force_map/*.npy
- prompt: a text prompt derived from the config json (or a fallback prompt)

Important:
- This script enforces that `video` and `force_video` have the SAME frame count and FPS.
- Alignment is by frame index: RGB frame i uses `src_dir/{i:04d}.png`,
  force frame i uses `src_dir/2d_force_map/{i:04d}.npy`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import imageio  # type: ignore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _natural_key(p: str) -> tuple:
    # "0010.npy" sorts after "0009.npy"
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p))


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _relpath(path: Path, start: Path) -> str:
    return os.path.relpath(str(path), start=str(start))


def _list_numbered_files(dir_path: Path, suffix: str) -> list[str]:
    # Match files like 0000.png, 0123.npy
    out = []
    for name in os.listdir(dir_path):
        if not name.lower().endswith(suffix.lower()):
            continue
        stem = name[: -len(suffix)]
        if stem.isdigit():
            out.append(name)
    return sorted(out, key=_natural_key)


def _load_force_map(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    # Observed shape: (1, 1000, 1000), dtype float32, range [0,1]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported force map shape {arr.shape} at {path}")
    return arr


def _force_to_rgb_u8(
    force_hw: np.ndarray,
    *,
    gain: float = 1.0,
    transform: str = "none",
    gamma: float = 0.5,
    log_alpha: float = 50.0,
) -> np.ndarray:
    x = np.nan_to_num(force_hw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x * float(gain), 0.0, 1.0)

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
    rgb = np.stack([u8, u8, u8], axis=-1)  # (H,W,3)
    return rgb


def _write_video_streaming(
    frames: Iterable[np.ndarray],
    dst_path: Path,
    fps: int,
) -> None:
    """Write uint8 RGB frames (H,W,3) to .mp4 via imageio."""
    _safe_mkdir(dst_path.parent)
    suffix = dst_path.suffix.lower()
    if suffix != ".mp4":
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


def _generate_prompt_from_config(config_json: dict, *, config_id: str) -> str:
    material = config_json.get("material", "material")
    return f"{material} object deformation (force-conditioned)"


@dataclass(frozen=True)
class SamplePaths:
    config_id: str
    src_dir: Path
    src_rgb_mp4: Path
    src_force_dir: Path
    src_config_json: Path


def _iter_samples(src_root: Path, *, max_sequences: int) -> list[SamplePaths]:
    samples: list[SamplePaths] = []
    for i in range(max_sequences):
        config_id = f"config_{i:04d}"
        src_dir = src_root / config_id
        rgb_mp4 = src_dir / "output.mp4"
        force_dir = src_dir / "2d_force_map"
        cfg_json = src_root / "all_config_files" / f"config_{i:04d}.json"

        if not src_dir.is_dir():
            raise FileNotFoundError(f"Missing source folder: {src_dir}")
        if not rgb_mp4.is_file():
            raise FileNotFoundError(f"Missing RGB video: {rgb_mp4}")
        if not force_dir.is_dir():
            raise FileNotFoundError(f"Missing force map dir: {force_dir}")
        if not cfg_json.is_file():
            raise FileNotFoundError(f"Missing config json: {cfg_json}")

        samples.append(
            SamplePaths(
                config_id=config_id,
                src_dir=src_dir,
                src_rgb_mp4=rgb_mp4,
                src_force_dir=force_dir,
                src_config_json=cfg_json,
            )
        )
    return samples


def _build_force_video(
    sample: SamplePaths,
    dst_force_video_path: Path,
    num_frames: int,
    fps: int,
    gain: float,
    transform: str,
    gamma: float,
    log_alpha: float,
    overwrite: bool,
) -> None:
    if dst_force_video_path.exists() and not overwrite:
        return

    npy_files = _list_numbered_files(sample.src_force_dir, ".npy")
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {sample.src_force_dir}")

    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")
    npy_files = npy_files[: min(num_frames, len(npy_files))]

    def _frames():
        for name in npy_files:
            force_hw = _load_force_map(sample.src_force_dir / name)
            yield _force_to_rgb_u8(
                force_hw,
                gain=gain,
                transform=transform,
                gamma=gamma,
                log_alpha=log_alpha,
            )

    _write_video_streaming(_frames(), dst_force_video_path, fps=fps)


def _build_rgb_video(
    sample: SamplePaths,
    dst_rgb_video_path: Path,
    num_frames: int,
    fps: int,
    overwrite: bool,
) -> None:
    if dst_rgb_video_path.exists() and not overwrite:
        return

    png_files = _list_numbered_files(sample.src_dir, ".png")
    if not png_files:
        raise FileNotFoundError(f"No numbered .png frames found in {sample.src_dir} (expected 0000.png, 0001.png, ...)")

    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")
    png_files = png_files[: min(num_frames, len(png_files))]

    from PIL import Image

    def _frames():
        for name in png_files:
            im = Image.open(sample.src_dir / name).convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"Expected RGB image -> HxWx3, got {arr.shape} from {name}")
            yield arr

    _write_video_streaming(_frames(), dst_rgb_video_path, fps=fps)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_root",
        type=str,
        default="/weka/scratch/bhattad/tnguy303/research/PhysGaussian/all_bread_data",
        help="Path to PhysGaussian all_bread_data/",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        default=str(_repo_root() / "data" / "force_dataset_20"),
        help="Output dataset folder (will be created).",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=20,
        help="How many config_XXXX folders to process (starting from 0000).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="How many frames to encode into force_video (training reads the prefix).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="FPS for BOTH generated video and force_video.",
    )
    parser.add_argument("--force_gain", type=float, default=1.0, help="Multiply force map by this gain then clamp to [0,1].")
    parser.add_argument("--force_transform", type=str, default="none", choices=["none", "gamma", "log"])
    parser.add_argument("--force_gamma", type=float, default=0.5, help="Used when --force_transform gamma.")
    parser.add_argument("--force_log_alpha", type=float, default=50.0, help="Used when --force_transform log.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated force videos.")
    args = parser.parse_args(argv)

    src_root = Path(args.src_root).resolve()
    dst_root_input = Path(args.dst_root)
    dst_root = dst_root_input if dst_root_input.is_absolute() else (_repo_root() / dst_root_input)
    videos_dir = dst_root / "videos"
    force_dir = dst_root / "force_videos"
    _safe_mkdir(videos_dir)
    _safe_mkdir(force_dir)

    samples = _iter_samples(src_root, max_sequences=int(args.max_sequences))

    rows = []
    for s in samples:
        # Determine aligned frame count N.
        png_files = _list_numbered_files(s.src_dir, ".png")
        npy_files = _list_numbered_files(s.src_force_dir, ".npy")
        if not png_files:
            raise FileNotFoundError(f"No numbered .png frames found in {s.src_dir}")
        if not npy_files:
            raise FileNotFoundError(f"No .npy force frames found in {s.src_force_dir}")

        n = min(int(args.num_frames), len(png_files), len(npy_files))
        if n <= 0:
            raise RuntimeError(f"No aligned frames available for {s.config_id}")

        dst_rgb = videos_dir / f"{s.config_id}.mp4"
        dst_force = force_dir / f"{s.config_id}_force.mp4"

        _build_rgb_video(
            s,
            dst_rgb,
            num_frames=n,
            fps=int(args.fps),
            overwrite=bool(args.overwrite),
        )

        _build_force_video(
            s,
            dst_force,
            num_frames=n,
            fps=int(args.fps),
            gain=float(args.force_gain),
            transform=str(args.force_transform),
            gamma=float(args.force_gamma),
            log_alpha=float(args.force_log_alpha),
            overwrite=bool(args.overwrite),
        )

        with open(s.src_config_json, "r") as f:
            cfg = json.load(f)
        prompt = _generate_prompt_from_config(cfg, config_id=s.config_id)

        rows.append({
            "video": _relpath(dst_rgb, dst_root),
            "prompt": prompt,
            "force_video": _relpath(dst_force, dst_root),
        })

    metadata_path = dst_root / "metadata.csv"
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "prompt", "force_video"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("Done.")
    print(f"Dataset written to: {dst_root}")
    print(f"Metadata: {metadata_path}")
    print()
    print("Training usage:")
    print(
        "  accelerate launch examples/wanvideo/model_training/train_forceCond.py \\\n"
        f"    --dataset_base_path \"{dst_root}\" \\\n"
        f"    --dataset_metadata_path \"{metadata_path}\" \\\n"
        "    --data_file_keys \"video,force_video\" \\\n"
        "    --extra_inputs \"input_image\" \\\n"
        "    ..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

