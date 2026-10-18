import json
import numpy as np
from plyfile import PlyData
import argparse


def load_ply_xyz(ply_path):
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data

    x = v["x"]
    y = v["y"]
    z = v["z"]

    pts = np.stack([x, y, z], axis=1)
    return pts


def compute_scene_stats(points):
    center = points.mean(axis=0)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    bbox_size = maxs - mins

    scale = np.linalg.norm(bbox_size)
    return center, scale


def look_at_rotation(camera_pos, target, up=np.array([0, 0, 1])):
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)

    R = np.stack([right, true_up, -forward], axis=1)
    return R


def create_cameras_from_ply(
    ply_path,
    output,
    num_views=100,
    radius_scale=2.0,
    elevation_deg=20,
    width=768,
    height=768,
    fx=823.49,
    fy=823.49,
):
    pts = load_ply_xyz(ply_path)
    center, scale = compute_scene_stats(pts)

    radius = scale * radius_scale
    elevation = np.deg2rad(elevation_deg)

    cameras = []

    for i in range(num_views):
        theta = 2 * np.pi * i / num_views

        pos = np.array([
            radius * np.cos(theta),
            radius * np.sin(theta),
            radius * np.sin(elevation),
        ]) + center

        R = look_at_rotation(pos, center)

        cameras.append({
            "id": i,
            "img_name": f"{i:04d}",
            "width": width,
            "height": height,
            "position": pos.tolist(),
            "rotation": R.tolist(),
            "fy": fy,
            "fx": fx,
        })

    with open(output, "w") as f:
        json.dump(cameras, f, indent=2)

    print("Saved cameras to:", output)
    print("Scene center:", center)
    print("Scene scale:", scale)
    print("Camera radius:", radius)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--output", default="cameras.json")
    parser.add_argument("--num_views", type=int, default=100)
    parser.add_argument("--radius_scale", type=float, default=2.0)
    parser.add_argument("--elevation", type=float, default=20)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--fx", type=float, default=823.49)
    parser.add_argument("--fy", type=float, default=823.49)

    args = parser.parse_args()

    create_cameras_from_ply(
        ply_path=args.ply,
        output=args.output,
        num_views=args.num_views,
        radius_scale=args.radius_scale,
        elevation_deg=args.elevation,
        width=args.width,
        height=args.height,
        fx=args.fx,
        fy=args.fy,
    )