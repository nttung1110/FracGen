"""
video_gen.py

Force-point video rendering used by benchmarks/build_benchmark.py.

Given a static "intact object" image and one or two force-application points
(each with a pixel position and a per-frame velocity), `create_control_video`
renders three clips:
  - control video:       white Gaussian blobs on black, tracking each force
                          point (the force-conditioning signal fed to the model)
  - visualization video:  the object image with colored blobs overlaid at each
                          force point (for human-readable preview)
  - reference video:      the static intact image repeated for every frame
                          (the "video" / start-frame conditioning column)

All videos are encoded as H.264/yuv420p mp4 via imageio's ffmpeg backend so
they play back in browsers and editor video previews (e.g. Cursor/VS Code) --
OpenCV's default `mp4v` fourcc is not reliably playable there.
"""

import copy
import os
from pathlib import Path

import imageio
import numpy as np
from PIL import Image

INSTANCE_COLORS = [
    (0, 220, 255),
    (255, 80, 60),
]

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
                inst["vx"] = 0.0
                nx = float(np.clip(nx, 0, W - 1))
            if not (0 <= ny < H):
                inst["vy"] = 0.0
                ny = float(np.clip(ny, 0, H - 1))
            positions[i]["x"] = nx
            positions[i]["y"] = ny

        gray = (canvas * 255).clip(0, 255).astype(np.uint8)
        bgr = np.stack([gray, gray, gray], axis=-1)
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
                inst["vx"] = 0.0
                nx = np.clip(nx, 0, W - 1)
            if not (0 <= ny < H):
                inst["vy"] = 0.0
                ny = np.clip(ny, 0, H - 1)
            positions[i]["x"] = nx
            positions[i]["y"] = ny

        frames.append(np.clip(canvas, 0, 255).astype(np.uint8))

    return frames


# ─────────────────────────────────────────────
# Video writer (H.264/yuv420p via ffmpeg)
# ─────────────────────────────────────────────

def write_video(frames: list[np.ndarray], output_path: str, fps: int) -> None:
    """Encode BGR uint8 frames as H.264/yuv420p mp4."""
    with imageio.get_writer(
        output_path, fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    ) as writer:
        for frame in frames:
            writer.append_data(frame[..., ::-1])  # BGR -> RGB
    print(f"  ✓ {output_path}")


def create_reference_video(
    input_image_path: str,
    output_video_path: str,
    n_frames: int,
    fps: int,
) -> None:
    img = Image.open(input_image_path).convert("RGB")
    frame_rgb = np.array(img, dtype=np.uint8)
    with imageio.get_writer(
        output_video_path, fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    ) as writer:
        for _ in range(n_frames):
            writer.append_data(frame_rgb)
    print(f"  ✓ {output_video_path}")


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
    """Return the three output .mp4 paths for a given sample stem."""
    return {
        "control":       os.path.join(dirs["control"],       f"{stem}.mp4"),
        "visualization": os.path.join(dirs["visualization"], f"{stem}.mp4"),
        "reference":     os.path.join(dirs["reference"],     f"{stem}.mp4"),
    }


# ─────────────────────────────────────────────
# Core per-sample video generator
# ─────────────────────────────────────────────

def create_control_video(
    input_image_path: str,
    instances: list[dict],
    output_dir: str = "output",
    name: str | None = None,
    n_frames: int = 60,
    fps: int = 24,
    sigma: float = 20.0,
    blob_alpha: float = 0.85,
    dim_background: float = 0.45,
    skip_ref: bool = False,
) -> dict[str, str]:
    """Render control/visualization/reference videos for one sample.

    Returns the dict of output paths (see `video_paths_for_stem`); the
    "reference" entry is omitted if `skip_ref` is set.
    """
    if not (1 <= len(instances) <= 2):
        raise ValueError("Provide 1 or 2 instances.")

    stem  = name or Path(input_image_path).stem
    dirs  = make_output_dirs(output_dir)
    paths = video_paths_for_stem(dirs, stem)

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

    return paths
