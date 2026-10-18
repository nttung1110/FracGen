"""Render static multi-view images and an azimuth-orbit video from a 3DGS checkpoint.

No simulation is run: the raw Gaussians are rendered as-is. The config JSON is only
used for its preprocessing (rotation/scale) and camera parameters so that the
view center / up-axis / azimuth-elevation-radius convention matches the
simulation scripts.

Example:
  python render_gs_views.py \
    --model_path data/fracsim/gaussians/bread --config src/fracsim/configs/tear_bread_config_exp.json \
    --output_path output/bread_views --white_bg --num_views 10 --orbit_frames 120
"""
import sys
import os
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src", "fracsim"))
sys.path.append(os.path.join(_REPO, "src", "fracsim", "gaussian-splatting"))

import argparse
import os
import json
import numpy as np
import torch
import imageio
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.system_utils import searchForMaxIteration
from utils.decode_param import decode_param_json
from utils.transformation_utils import (
    generate_rotation_matrices,
    apply_rotations,
    transform2origin,
    get_center_view_worldspace_and_observant_coordinate,
)
from utils.camera_view_utils import get_camera_view


class PipelineParamsNoparse:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(checkpt_dir, f"iteration_{iteration}", "point_cloud.ply")
    print("Loading", checkpt_path)
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


def to_uint8(img):
    return (torch.clamp(img, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--num_views", type=int, default=10)
    parser.add_argument("--orbit_frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--elevation", type=float, default=None, help="override config init_elevation")
    parser.add_argument("--radius", type=float, default=None, help="override config init_radius")
    parser.add_argument("--azimuth", type=float, default=None, help="override config init_azimuthm (orbit start)")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    views_dir = os.path.join(args.output_path, "views")
    orbit_dir = os.path.join(args.output_path, "orbit_frames")
    os.makedirs(views_dir, exist_ok=True)
    os.makedirs(orbit_dir, exist_ok=True)

    (_, _, _, preprocessing_params, camera_params) = decode_param_json(args.config)

    gaussians = load_checkpoint(args.model_path, iteration=args.iteration)
    pipeline = PipelineParamsNoparse()
    background = torch.tensor(
        [1, 1, 1] if args.white_bg else [0, 0, 0], dtype=torch.float32, device="cuda"
    )

    # Replicate the simulation scripts' preprocessing *only* to derive the view center
    # and the up axis in world space. The rendered Gaussians themselves are untouched.
    with torch.no_grad():
        pos = gaussians.get_xyz
        opacity = gaussians.get_opacity
        mask = opacity[:, 0] > preprocessing_params["opacity_threshold"]
        pos = pos[mask]
        rotation_matrices = generate_rotation_matrices(
            torch.tensor(preprocessing_params["rotation_degree"]),
            preprocessing_params["rotation_axis"],
        )
        rotated_pos = apply_rotations(pos, rotation_matrices)
        if preprocessing_params["sim_area"] is not None:
            b = preprocessing_params["sim_area"]
            m = torch.ones(rotated_pos.shape[0], dtype=torch.bool, device="cuda")
            for i in range(3):
                m &= (rotated_pos[:, i] > b[2 * i]) & (rotated_pos[:, i] < b[2 * i + 1])
            rotated_pos = rotated_pos[m]
        _, scale_origin, original_mean_pos = transform2origin(rotated_pos, preprocessing_params["scale"])

    mpm_center = torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    mpm_up = torch.tensor(camera_params["mpm_space_vertical_upward_axis"]).reshape((1, 3)).cuda()
    view_center, observant_coordinates = get_center_view_worldspace_and_observant_coordinate(
        mpm_center, mpm_up, rotation_matrices, scale_origin, original_mean_pos
    )

    az0 = args.azimuth if args.azimuth is not None else camera_params["init_azimuthm"]
    el0 = args.elevation if args.elevation is not None else camera_params["init_elevation"]
    r0 = args.radius if args.radius is not None else camera_params["init_radius"]
    print(f"view center (world): {view_center}, azimuth={az0}, elevation={el0}, radius={r0}")

    def render_view(az, el, r):
        cam = get_camera_view(
            args.model_path,
            default_camera_index=-1,
            center_view_world_space=view_center,
            observant_coordinates=observant_coordinates,
            init_azimuthm=az,
            init_elevation=el,
            init_radius=r,
            move_camera=False,
        )
        with torch.no_grad():
            out = render(cam, gaussians, pipeline, background)
        return to_uint8(out["render"])

    # ---- 10 (or N) distinct views: spread over azimuth, alternate elevations ----
    view_log = []
    elevations = [el0, el0 + 20, el0 - 10, el0 + 40, el0 + 5]
    for i in range(args.num_views):
        az = az0 + 360.0 * i / args.num_views
        el = elevations[i % len(elevations)]
        img = render_view(az, el, r0)
        name = f"view_{i:02d}_az{az:.0f}_el{el:.0f}_r{r0:.2f}.png"
        imageio.imwrite(os.path.join(views_dir, name), img)
        view_log.append({"file": name, "azimuth": az, "elevation": el, "radius": r0})
        print("saved", name)
    with open(os.path.join(views_dir, "views.json"), "w") as f:
        json.dump(view_log, f, indent=2)

    # ---- orbit video: fixed elevation + radius, azimuth sweeps 360 degrees ----
    frames = []
    for k in tqdm(range(args.orbit_frames), desc="orbit"):
        az = az0 + 360.0 * k / args.orbit_frames
        img = render_view(az, el0, r0)
        imageio.imwrite(os.path.join(orbit_dir, f"{k:04d}.png"), img)
        frames.append(img)
    video_path = os.path.join(args.output_path, f"orbit_el{el0:.0f}_r{r0:.2f}.mp4")
    imageio.mimwrite(video_path, frames, fps=args.fps, codec="libx264", quality=8, pixelformat="yuv420p")
    print("saved", video_path)


if __name__ == "__main__":
    main()
