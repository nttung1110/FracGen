#!/usr/bin/env python3
import argparse
import json
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GS_ROOT = os.path.join(REPO_ROOT, "gaussian-splatting")
if GS_ROOT not in sys.path:
    sys.path.insert(0, GS_ROOT)

import numpy as np
import torch
from gaussian_renderer import GaussianModel, render
from scene.cameras import Camera as GSCamera
from utils.graphics_utils import focal2fov

try:
    from torchvision.utils import save_image
except Exception:
    save_image = None


class PipelineStub:
    debug = False
    compute_cov3D_python = False
    convert_SHs_python = False


def load_camera(raw_camera, uid):
    tmp = np.zeros((4, 4), dtype=np.float32)
    tmp[:3, :3] = np.asarray(raw_camera["rotation"], dtype=np.float32)
    tmp[:3, 3] = np.asarray(raw_camera["position"], dtype=np.float32)
    tmp[3, 3] = 1.0
    c2w = np.linalg.inv(tmp)
    r = c2w[:3, :3].transpose()
    t = c2w[:3, 3]

    width = int(raw_camera["width"])
    height = int(raw_camera["height"])
    image = torch.zeros((3, height, width), dtype=torch.float32)
    return GSCamera(
        colmap_id=uid,
        R=r,
        T=t,
        FoVx=focal2fov(raw_camera["fx"], width),
        FoVy=focal2fov(raw_camera["fy"], height),
        image=image,
        gt_alpha_mask=None,
        image_name=raw_camera["img_name"],
        uid=uid,
    )


def main():
    parser = argparse.ArgumentParser(description="Render Gaussian splats from cameras.json.")
    parser.add_argument("--ply", required=True, help="Path to the Gaussian point cloud .ply")
    parser.add_argument("--cameras", required=True, help="Path to cameras.json")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered PNGs")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--white-background", action="store_true")
    args = parser.parse_args()

    if save_image is None:
        raise RuntimeError("torchvision is required to save rendered images.")

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.cameras) as f:
        raw_cameras = json.load(f)

    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(args.ply)

    bg_color = [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipeline = PipelineStub()

    with torch.no_grad():
        for idx, raw_camera in enumerate(raw_cameras):
            camera = load_camera(raw_camera, idx)
            rendering = render(camera, gaussians, pipeline, background)["render"]
            out_path = os.path.join(args.output_dir, raw_camera["img_name"] + ".png")
            save_image(rendering, out_path)
            print(f"[{idx + 1}/{len(raw_cameras)}] {out_path}")


if __name__ == "__main__":
    main()
