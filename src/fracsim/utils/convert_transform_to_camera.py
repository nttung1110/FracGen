import json
import argparse
import numpy as np
from pathlib import Path


def camera_to_transforms(camera_path, output_path, image_prefix="image", ext="png"):
    cams = json.load(open(camera_path, "r"))

    if not isinstance(cams, list):
        raise ValueError("camera.json should be a list of camera dictionaries.")

    first = cams[0]
    w, h = first["width"], first["height"]
    fx, fy = first["fx"], first["fy"]

    transforms = {
        "fl_x": fx,
        "fl_y": fy,
        "cx": w / 2.0,
        "cy": h / 2.0,
        "w": w,
        "h": h,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": []
    }

    for cam in cams:
        R = np.array(cam["rotation"], dtype=np.float64)
        C = np.array(cam["position"], dtype=np.float64)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = C

        img_name = cam.get("img_name", f"{cam['id']:04d}")
        frame = {
            "file_path": f"{image_prefix}/{img_name}.{ext}",
            "transform_matrix": T.tolist()
        }
        transforms["frames"].append(frame)

    with open(output_path, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"Saved transforms.json to {output_path}")


def transforms_to_camera(transforms_path, output_path):
    data = json.load(open(transforms_path, "r"))

    fx = data["fl_x"]
    fy = data["fl_y"]
    w = data["w"]
    h = data["h"]

    cams = []

    for idx, frame in enumerate(data["frames"]):
        T = np.array(frame["transform_matrix"], dtype=np.float64)

        R = T[:3, :3]
        C = T[:3, 3]

        file_path = frame.get("file_path", f"{idx:04d}")
        img_name = Path(file_path).stem

        cam = {
            "id": idx,
            "img_name": img_name,
            "width": w,
            "height": h,
            "position": C.tolist(),
            "rotation": R.tolist(),
            "fy": fy,
            "fx": fx
        }
        cams.append(cam)

    with open(output_path, "w") as f:
        json.dump(cams, f, indent=2)

    print(f"Saved camera.json to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["camera_to_transforms", "transforms_to_camera"]
    )
    parser.add_argument("--image_prefix", default="image")
    parser.add_argument("--ext", default="png")
    args = parser.parse_args()

    if args.mode == "camera_to_transforms":
        camera_to_transforms(args.input, args.output, args.image_prefix, args.ext)
    else:
        transforms_to_camera(args.input, args.output)


if __name__ == "__main__":
    main()