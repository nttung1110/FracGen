"""
generate_control_video.py

Creates four outputs from a single input image, or processes a batch from a JSON file.

Directory layout produced under --output-dir (default: ./output):

  Single-image mode:
    <output_dir>/
    ├── visualization_video/
    │   └── <image_stem>.mp4
    ├── control_video/
    │   └── <image_stem>.mp4
    ├── reference_video/
    │   └── <image_stem>.mp4
    └── metadata.csv

  Batch JSON mode (--samples-json):
    <output_dir>/
    ├── visualization_video/
    │   ├── image_00000.mp4
    │   ├── image_00001.mp4
    │   └── ...
    ├── control_video/
    │   ├── image_00000.mp4
    │   └── ...
    ├── reference_video/
    │   ├── image_00000.mp4
    │   └── ...
    └── metadata.csv          ← all samples, one row each

JSON format (list of sample objects):
    [
      {
        "image":     "data/random_test/images/image_00000.png",
        "instance1": "x=370,y=350,vx=-4.0,vy=-4.0",
        "instance2": "x=570,y=490,vx=4.0,vy=4.0"   # optional
      },
      ...
    ]

Usage:
    # Single image:
    python generate_control_video.py \
        --input image.png \
        --output-dir ./run \
        --instance1 x=320,y=240,vx=2.5,vy=-1.0 \
        --instance2 x=100,y=100,vx=-1.5,vy=3.0 \
        --frames 60 --fps 24

    # Batch mode:
    python generate_control_video.py \
        --samples-json samples.json \
        --output-dir ./run \
        --frames 60 --fps 24

Dependencies:
    pip install numpy opencv-python Pillow
"""

import argparse
import copy
import csv
import json
import os
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


# ─────────────────────────────────────────────
# Gaussian blob rendering
# ─────────────────────────────────────────────

def gaussian_blob(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    sigma: float,
    color: tuple[int, int, int],
    alpha: float = 1.0,
) -> np.ndarray:
    H, W = canvas.shape[:2]
    r = int(np.ceil(3 * sigma))
    x0, x1 = max(0, int(cx) - r), min(W, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(H, int(cy) + r + 1)

    if x0 >= x1 or y0 >= y1:
        return canvas

    xs = np.arange(x0, x1, dtype=np.float32) - cx
    ys = np.arange(y0, y1, dtype=np.float32) - cy
    xx, yy = np.meshgrid(xs, ys)
    g = alpha * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    for c, val in enumerate(color):
        canvas[y0:y1, x0:x1, c] = np.clip(
            canvas[y0:y1, x0:x1, c] + g * val, 0, 255
        )
    return canvas


# ─────────────────────────────────────────────
# Control blob (white-on-black)
# ─────────────────────────────────────────────

def get_gaussian_blob(
    point: tuple[float, float],
    radius: float,
    H: int,
    W: int,
    amplitude: float = 1.0,
) -> np.ndarray:
    x, y = point
    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    return amplitude * np.exp(-dist2 / (2.0 * radius ** 2))


def generate_control_frames(
    H: int,
    W: int,
    instances: list[dict],
    n_frames: int,
    radius: float,
) -> list[np.ndarray]:
    frames = []
    positions = [{"x": inst["x"], "y": inst["y"]} for inst in instances]

    for _ in range(n_frames):
        canvas = np.zeros((H, W), dtype=np.float32)

        for i, inst in enumerate(instances):
            blob = get_gaussian_blob(
                point=(positions[i]["x"], positions[i]["y"]),
                radius=radius,
                H=H,
                W=W,
                amplitude=1.0,
            )
            canvas = np.maximum(canvas, blob)

            nx = positions[i]["x"] + inst["vx"]
            ny = positions[i]["y"] + inst["vy"]
            if not (0 <= nx < W):
                inst["vx"] *= -1
                nx = float(np.clip(nx, 0, W - 1))
            if not (0 <= ny < H):
                inst["vy"] *= -1
                ny = float(np.clip(ny, 0, H - 1))
            positions[i]["x"] = nx
            positions[i]["y"] = ny

        gray = (canvas * 255).clip(0, 255).astype(np.uint8)
        bgr  = np.stack([gray, gray, gray], axis=-1)
        frames.append(bgr)

    return frames


def generate_frames(
    background: np.ndarray,
    instances: list[dict],
    n_frames: int,
    sigma: float,
    blend_alpha: float,
) -> list[np.ndarray]:
    H, W = background.shape[:2]
    frames = []
    positions = [{"x": inst["x"], "y": inst["y"]} for inst in instances]

    for _ in range(n_frames):
        canvas = background.astype(np.float32).copy()

        for i, inst in enumerate(instances):
            cx = positions[i]["x"]
            cy = positions[i]["y"]
            gaussian_blob(canvas, cx, cy, sigma, inst["color"], alpha=blend_alpha)

            nx = positions[i]["x"] + inst["vx"]
            ny = positions[i]["y"] + inst["vy"]
            if not (0 <= nx < W):
                inst["vx"] *= -1
                nx = np.clip(nx, 0, W - 1)
            if not (0 <= ny < H):
                inst["vy"] *= -1
                ny = np.clip(ny, 0, H - 1)
            positions[i]["x"] = nx
            positions[i]["y"] = ny

        frames.append(np.clip(canvas, 0, 255).astype(np.uint8))

    return frames


# ─────────────────────────────────────────────
# Video writer
# ─────────────────────────────────────────────

def write_video(frames: list[np.ndarray], output_path: str, fps: int) -> None:
    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"  ✓ {output_path}")


# ─────────────────────────────────────────────
# Reference video
# ─────────────────────────────────────────────

def create_reference_video(
    input_image_path: str,
    output_video_path: str,
    n_frames: int,
    fps: int,
) -> None:
    img = Image.open(input_image_path).convert("RGB")
    frame_bgr = np.array(img, dtype=np.uint8)[..., ::-1]
    H, W = frame_bgr.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
    for _ in range(n_frames):
        writer.write(frame_bgr)
    writer.release()
    print(f"  ✓ {output_video_path}")


# ─────────────────────────────────────────────
# Metadata CSV
# ─────────────────────────────────────────────

METADATA_COLUMNS = [
    "video",
    "prompt",
    "force_video",
    "stress_video",
    "flow_video",
    "damage_video",
    "strain_video",
]

PROMPT = "physical simulation deformation (multimodal-conditioned)"


def write_metadata(
    csv_path: str,
    ref_video_path: str,
    control_video_path: str,
    append: bool = False,
) -> None:
    file_exists = os.path.isfile(csv_path)
    mode = "a" if (append and file_exists) else "w"

    with open(csv_path, mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METADATA_COLUMNS)
        if mode == "w":
            writer.writeheader()
        writer.writerow({
            "video":        ref_video_path,
            "prompt":       PROMPT,
            "force_video":  control_video_path,
            "stress_video": control_video_path,
            "flow_video":   control_video_path,
            "damage_video": control_video_path,
            "strain_video": control_video_path,
        })


# ─────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────

INSTANCE_COLORS = [
    (0, 220, 255),
    (255, 80, 60),
]


def parse_instance(raw: str, idx: int) -> dict:
    """Parse 'x=320,y=240,vx=2.5,vy=-1.0' into a dict."""
    parts = {}
    for token in raw.split(","):
        k, v = token.strip().split("=")
        parts[k.strip()] = float(v.strip())

    required = {"x", "y", "vx", "vy"}
    missing = required - parts.keys()
    if missing:
        raise ValueError(f"Instance {idx + 1} is missing keys: {missing}")

    return {
        "x":     parts["x"],
        "y":     parts["y"],
        "vx":    parts["vx"],
        "vy":    parts["vy"],
        "color": INSTANCE_COLORS[idx],
    }


# ─────────────────────────────────────────────
# Output path helpers
# ─────────────────────────────────────────────

def make_output_dirs(output_dir: str) -> dict[str, str]:
    """
    Create the three video subfolders and return their paths.

        <output_dir>/
        ├── control_video/
        ├── visualization_video/
        └── reference_video/
    """
    dirs = {
        "control":       os.path.join(output_dir, "control_video"),
        "visualization": os.path.join(output_dir, "visualization_video"),
        "reference":     os.path.join(output_dir, "reference_video"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def video_paths_for_stem(dirs: dict[str, str], stem: str) -> dict[str, str]:
    """Return the three output .mp4 paths for a given image stem."""
    return {
        "control":       os.path.join(dirs["control"],       f"{stem}.mp4"),
        "visualization": os.path.join(dirs["visualization"], f"{stem}.mp4"),
        "reference":     os.path.join(dirs["reference"],     f"{stem}.mp4"),
    }


# ─────────────────────────────────────────────
# Core per-image processor
# ─────────────────────────────────────────────

def create_control_video(
    input_image_path: str,
    instances: list[dict],
    output_dir: str = "output",
    name: str | None = None,          # ← new: overrides image stem if provided
    n_frames: int = 60,
    fps: int = 24,
    sigma: float = 20.0,
    blob_alpha: float = 0.85,
    dim_background: float = 0.45,
    skip_ref: bool = False,
    metadata_csv_path: str | None = None,
    append_metadata: bool = False,
) -> None:
    if not (1 <= len(instances) <= 2):
        raise ValueError("Provide 1 or 2 instances.")

    stem  = name or Path(input_image_path).stem   # ← use name if given
    dirs  = make_output_dirs(output_dir)
    paths = video_paths_for_stem(dirs, stem)
    csv_path = metadata_csv_path or os.path.join(output_dir, "metadata.csv")

    img    = Image.open(input_image_path).convert("RGB")
    bg_rgb = np.array(img, dtype=np.float32)
    H, W   = bg_rgb.shape[:2]
    bg_bgr = bg_rgb[..., ::-1] * dim_background

    for i, inst in enumerate(instances):
        if "color" not in inst:
            inst["color"] = INSTANCE_COLORS[i]

    viz_instances  = copy.deepcopy(instances)
    ctrl_instances = copy.deepcopy(instances)

    viz_frames = generate_frames(bg_bgr, viz_instances, n_frames, sigma, blob_alpha)
    write_video(viz_frames, paths["visualization"], fps)

    ctrl_frames = generate_control_frames(H, W, ctrl_instances, n_frames, radius=sigma)
    write_video(ctrl_frames, paths["control"], fps)

    if not skip_ref:
        create_reference_video(input_image_path, paths["reference"], n_frames, fps)

    if metadata_csv_path is not None:
        ref_col = paths["reference"] if not skip_ref else input_image_path
        write_metadata(
            csv_path=csv_path,
            ref_video_path=ref_col,
            control_video_path=paths["control"],
            append=append_metadata,
        )


# ── Batch processing from JSON ────────────────────────────────────────────────

def process_samples_json(
    json_path: str,
    output_dir: str,
    n_frames: int,
    fps: int,
    sigma: float,
    blob_alpha: float,
    dim_background: float,
    skip_ref: bool,
    skip_metadata: bool,
) -> None:
    with open(json_path) as fh:
        samples = json.load(fh)

    if not isinstance(samples, list):
        raise ValueError("JSON file must contain a top-level list of sample objects.")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = None if skip_metadata else os.path.join(output_dir, "metadata.csv")

    print(f"Processing {len(samples)} sample(s) from {json_path}")
    print(f"Output directory : {os.path.abspath(output_dir)}\n")

    for idx, sample in enumerate(samples):
        missing_keys = [k for k in ("image", "instance1") if k not in sample]
        if missing_keys:
            raise ValueError(f"Sample {idx} is missing required key(s): {missing_keys}")

        image_path = sample["image"]
        name       = sample.get("name")          # ← None if absent → falls back to stem
        label      = name or Path(image_path).stem
        print(f"[{idx + 1}/{len(samples)}] {label}  ({image_path})")

        instances = [parse_instance(sample["instance1"], 0)]
        if sample.get("instance2"):
            instances.append(parse_instance(sample["instance2"], 1))

        create_control_video(
            input_image_path=image_path,
            instances=instances,
            output_dir=output_dir,
            name=name,                           # ← passed through
            n_frames=n_frames,
            fps=fps,
            sigma=sigma,
            blob_alpha=blob_alpha,
            dim_background=dim_background,
            skip_ref=skip_ref,
            metadata_csv_path=csv_path,
            append_metadata=(idx > 0),
        )
        print()

    print("Done.")
    if csv_path:
        print(f"Metadata CSV : {csv_path}")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate visualization + control + reference videos and metadata.csv.\n"
            "Run in single-image mode (--input / --instance1) or batch mode (--samples-json)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── mutually exclusive input modes ────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        help="Single input image path (use with --instance1 / --instance2).",
    )
    input_group.add_argument(
        "--samples-json",
        metavar="FILE",
        help='Path to a JSON file: list of {"image", "instance1", "instance2"} dicts.',
    )

    # ── single-image options ──────────────────────────────────────────────────
    parser.add_argument("--instance1", default=None, help="Instance 1: x=<f>,y=<f>,vx=<f>,vy=<f>")
    parser.add_argument("--instance2", default=None, help="Instance 2 (optional): x=<f>,y=<f>,vx=<f>,vy=<f>")
    parser.add_argument("--append-metadata", action="store_true", help="(single-image) Append to existing metadata.csv")

    # ── shared options ────────────────────────────────────────────────────────
    parser.add_argument("--output-dir",  default="output",  help="Root output directory (default: ./output)")
    parser.add_argument("--frames",      type=int,   default=60,   help="Number of frames (default: 60)")
    parser.add_argument("--fps",         type=int,   default=24,   help="Frames per second (default: 24)")
    parser.add_argument("--sigma",       type=float, default=20.0, help="Blob radius in pixels (default: 20)")
    parser.add_argument("--alpha",       type=float, default=0.85, help="Blob peak intensity 0-1 (default: 0.85)")
    parser.add_argument("--dim-bg",      type=float, default=0.45, help="Background dimming factor 0-1 (default: 0.45)")
    parser.add_argument("--no-ref",      action="store_true",      help="Skip writing reference videos")
    parser.add_argument("--no-metadata", action="store_true",      help="Skip writing metadata.csv")

    args = parser.parse_args()

    # ── dispatch ──────────────────────────────────────────────────────────────
    if args.samples_json:
        if args.instance1 or args.instance2:
            parser.error("--instance1/--instance2 cannot be used with --samples-json.")

        process_samples_json(
            json_path=args.samples_json,
            output_dir=args.output_dir,
            n_frames=args.frames,
            fps=args.fps,
            sigma=args.sigma,
            blob_alpha=args.alpha,
            dim_background=args.dim_bg,
            skip_ref=args.no_ref,
            skip_metadata=args.no_metadata,
        )

    else:
        # Single-image mode
        if not args.instance1:
            parser.error("--instance1 is required when using --input.")

        instances = [parse_instance(args.instance1, 0)]
        if args.instance2:
            instances.append(parse_instance(args.instance2, 1))

        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = None if args.no_metadata else os.path.join(args.output_dir, "metadata.csv")

        print(f"Output directory : {os.path.abspath(args.output_dir)}")

        create_control_video(
            input_image_path=args.input,
            instances=instances,
            output_dir=args.output_dir,
            n_frames=args.frames,
            fps=args.fps,
            sigma=args.sigma,
            blob_alpha=args.alpha,
            dim_background=args.dim_bg,
            skip_ref=args.no_ref,
            metadata_csv_path=csv_path,
            append_metadata=args.append_metadata,
        )


if __name__ == "__main__":
    main()