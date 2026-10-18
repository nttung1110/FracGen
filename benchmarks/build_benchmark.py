"""
build_benchmark.py

Build fracture-video benchmark samples from per-object annotation JSONs.

Input layout (<data-root>, e.g. benchmarks/data):
    <data-root>/
    ├── imgs/<object_id>.png                 (intact object image)
    └── annotations/<object_id>.json         (loading + material + fracture config)

Annotation JSON schema (see benchmarks/data/annotations/bee.json):
    {
      "image": {"name": "imgs/<object_id>.png", "width_px": .., "height_px": ..},
      "units": {"distance": "pixel", "time": "frame"},
      "vector_display_scale_frames": <float>,   # annotation-tool display only, unused here
      "loading": [
        {
          "name": "...", "type": "uniaxial", "mode": "tension",
          "velocity_px_per_frame": <float>,             # default speed
          "velocity_options_px_per_frame": [<float>, ...],  # speed variants to benchmark (optional)
          "point_A": {"position_px": {"x": .., "y": ..}, "direction_deg": ..},
          "point_B": {...}   # optional second force point
        }, ...
      ],
      "properties": {
        "material": {"youngs_modulus_Pa": [...], "poisson_ratio": [...]},  # parallel lists
        "fracture": [{"lambda_onset": .., "lambda_crit": .., "category": "rubber-like"}, ...]
      }
    }

For every object we expand the cross product of
  (loading config) x (velocity variant) x (material variant) x (fracture variant)
into one benchmark sample. Each loading's point_A/point_B become the two
force points rendered by benchmarks/utils/video_gen.py; velocity_px_per_frame
and direction_deg are used directly as the per-frame pixel velocity (the
annotation is already in pixel/frame units, per "units", so no unit
conversion or frame-count-dependent scaling is needed).

Loading and velocity are always fully expanded, but material/fracture variants
are limited to the first N (--num-material-variants / --num-fracture-variants,
both default 1) while the benchmark is being bootstrapped -- e.g. with 4
loadings x 3 velocity options each and the defaults, that's 12 samples per
object. Pass <=0 (or a larger count) to expand to the full annotated set later.

Video generation and folder layout follow the same conventions as
tool_scripts/gen_force_from_points.py, so the output matches training-data
conventions:

    <output_dir>/
    ├── control_video/<sample_id>.mp4
    ├── visualization_video/<sample_id>.mp4
    ├── reference_video/<sample_id>.mp4
    ├── metadata.csv             (same columns as a training metadata.csv)
    └── benchmark_metadata.json  (metadata.csv rows + benchmark bookkeeping fields)

metadata.csv columns exactly match the training/inference dataset convention
(see data/random_test/preprocessed_test_multi_object2/metadata.csv and
tool_scripts/preprocess_dataset/process_forceVideo_all_multi_object.py):
    video, prompt, force_video, stress_video, flow_video, damage_video, strain_video,
    young_modulus, poisson_ratio, lambda_onset, lambda_crit, category
Extra benchmark bookkeeping (object_id, sample_id, loading_name, force point
positions, ...) is only written to benchmark_metadata.json, not metadata.csv.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from utils import video_gen as vg  # noqa: E402


# Fixed material-category vocabulary consumed by WanVideoUnitMaterialPropEmbedder
# (diffsynth/pipelines/wan_video.py), which requires category_id in {0, 1, 2}.
# Extend this map if new fracture "category" strings are introduced.
CATEGORY_MAP = {
    "bread-like": 0,
    "rubber-like": 1,
    "metal-like": 2,
}

# Exactly the columns of a training/inference metadata.csv (see
# data/random_test/preprocessed_test_multi_object2/metadata.csv). Only these
# are written to metadata.csv; everything else is benchmark-only bookkeeping
# that goes to benchmark_metadata.json instead.
TRAIN_METADATA_COLUMNS = vg.METADATA_COLUMNS + [
    "young_modulus", "poisson_ratio", "lambda_onset", "lambda_crit", "category",
]
BENCHMARK_EXTRA_COLUMNS = [
    "object_id", "sample_id", "intact_image",
    "loading_name", "loading_type", "loading_mode",
    "velocity_px_per_frame", "velocity_variant_idx",
    "point_A_x_px", "point_A_y_px", "point_A_direction_deg",
    "point_B_x_px", "point_B_y_px", "point_B_direction_deg",
    "material_variant_idx", "fracture_variant_idx", "fracture_category",
]
JSON_METADATA_COLUMNS = TRAIN_METADATA_COLUMNS + BENCHMARK_EXTRA_COLUMNS


def category_to_id(category: str) -> int:
    try:
        return CATEGORY_MAP[category]
    except KeyError:
        raise ValueError(
            f"Unknown fracture category '{category}'. "
            f"Add it to CATEGORY_MAP in benchmarks/build_benchmark.py (known: {list(CATEGORY_MAP)})."
        )


def velocity_from_direction(speed_px_per_frame: float, direction_deg: float) -> tuple[float, float]:
    """Convert a px/frame speed + direction into a pixel-space (vx, vy) velocity.

    direction_deg follows the annotation tool's convention: standard math angles
    (0 deg = +x, counter-clockwise, y-axis pointing UP). Pixel space has the
    y-axis pointing DOWN, so the vertical component must be negated when
    converting to a pixel-space velocity.
    """
    theta = math.radians(direction_deg)
    vx = speed_px_per_frame * math.cos(theta)
    vy = -speed_px_per_frame * math.sin(theta)
    return vx, vy


def build_instances(loading: dict, velocity_px_per_frame: float) -> list[dict]:
    point_a = loading["point_A"]
    vx, vy = velocity_from_direction(velocity_px_per_frame, point_a["direction_deg"])
    instances = [{
        "x": float(point_a["position_px"]["x"]),
        "y": float(point_a["position_px"]["y"]),
        "vx": vx, "vy": vy,
        "color": vg.INSTANCE_COLORS[0],
    }]

    point_b = loading.get("point_B")
    if point_b:
        vx, vy = velocity_from_direction(velocity_px_per_frame, point_b["direction_deg"])
        instances.append({
            "x": float(point_b["position_px"]["x"]),
            "y": float(point_b["position_px"]["y"]),
            "vx": vx, "vy": vy,
            "color": vg.INSTANCE_COLORS[1],
        })
    return instances


def _limit(items: list, limit: int) -> list:
    """Keep only the first `limit` items, or all of them if limit <= 0 ("all")."""
    return items if limit <= 0 else items[:limit]


def expand_samples(
    object_id: str,
    cfg: dict,
    data_root: Path,
    num_material_variants: int = 1,
    num_fracture_variants: int = 1,
) -> list[dict]:
    image_path = str((data_root / cfg["image"]["name"]).resolve())

    units = cfg.get("units", {"distance": "pixel", "time": "frame"})
    if units.get("distance") != "pixel" or units.get("time") != "frame":
        raise ValueError(
            f"{object_id}: unsupported units {units}; this script assumes "
            f"velocity_px_per_frame is already in pixel/frame units."
        )

    material = cfg["properties"]["material"]
    youngs_list = material["youngs_modulus_Pa"]
    poisson_list = material["poisson_ratio"]
    if len(youngs_list) != len(poisson_list):
        raise ValueError(
            f"{object_id}: youngs_modulus_Pa ({len(youngs_list)}) and "
            f"poisson_ratio ({len(poisson_list)}) must have the same length (paired material variants)."
        )
    # Only the material/fracture axes are limited for now (kept small while the
    # benchmark is being bootstrapped); loading and velocity are always fully expanded.
    youngs_list = _limit(youngs_list, num_material_variants)
    poisson_list = _limit(poisson_list, num_material_variants)

    fracture_list = cfg["properties"]["fracture"]
    if isinstance(fracture_list, dict):
        fracture_list = [fracture_list]
    fracture_list = _limit(fracture_list, num_fracture_variants)

    samples = []
    for load_idx, loading in enumerate(cfg["loading"]):
        velocity_options = loading.get("velocity_options_px_per_frame") or [loading["velocity_px_per_frame"]]
        for vel_idx, velocity in enumerate(velocity_options):
            for mat_idx, (young_modulus, poisson_ratio) in enumerate(zip(youngs_list, poisson_list)):
                for frac_idx, frac in enumerate(fracture_list):
                    samples.append({
                        "sample_id": f"{object_id}__load{load_idx}__vel{vel_idx}__mat{mat_idx}__frac{frac_idx}",
                        "object_id": object_id,
                        "image_path": image_path,
                        "loading": loading,
                        "velocity_px_per_frame": velocity,
                        "velocity_variant_idx": vel_idx,
                        "material_variant_idx": mat_idx,
                        "young_modulus": young_modulus,
                        "poisson_ratio": poisson_ratio,
                        "fracture_variant_idx": frac_idx,
                        "lambda_onset": frac["lambda_onset"],
                        "lambda_crit": frac["lambda_crit"],
                        "fracture_category": frac["category"],
                    })
    return samples


def build_metadata_row(sample: dict, ref_video_path: str, control_video_path: str) -> dict:
    loading = sample["loading"]
    point_a = loading["point_A"]
    point_b = loading.get("point_B") or {}
    point_b_pos = point_b.get("position_px", {})

    return {
        "video": ref_video_path,
        "prompt": vg.PROMPT,
        "force_video": control_video_path,
        "stress_video": control_video_path,
        "flow_video": control_video_path,
        "damage_video": control_video_path,
        "strain_video": control_video_path,
        "young_modulus": sample["young_modulus"],
        "poisson_ratio": sample["poisson_ratio"],
        "lambda_onset": sample["lambda_onset"],
        "lambda_crit": sample["lambda_crit"],
        "category": category_to_id(sample["fracture_category"]),
        "object_id": sample["object_id"],
        "sample_id": sample["sample_id"],
        "intact_image": sample["image_path"],
        "loading_name": loading.get("name", ""),
        "loading_type": loading.get("type", ""),
        "loading_mode": loading.get("mode", ""),
        "velocity_px_per_frame": sample["velocity_px_per_frame"],
        "velocity_variant_idx": sample["velocity_variant_idx"],
        "point_A_x_px": point_a["position_px"]["x"],
        "point_A_y_px": point_a["position_px"]["y"],
        "point_A_direction_deg": point_a["direction_deg"],
        "point_B_x_px": point_b_pos.get("x", ""),
        "point_B_y_px": point_b_pos.get("y", ""),
        "point_B_direction_deg": point_b.get("direction_deg", ""),
        "material_variant_idx": sample["material_variant_idx"],
        "fracture_variant_idx": sample["fracture_variant_idx"],
        "fracture_category": sample["fracture_category"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="Folder containing imgs/ and annotations/ subfolders.")
    parser.add_argument("--output-dir", default="benchmarks/output", help="Where to write the generated benchmark.")
    parser.add_argument("--object-ids", default=None, help="Comma-separated subset of object ids (annotation json stems) to build. Default: all.")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames per video (default: 60).")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second (default: 24).")
    parser.add_argument("--sigma", type=float, default=20.0, help="Force blob radius in pixels (default: 20).")
    parser.add_argument("--alpha", type=float, default=0.85, help="Blob peak intensity 0-1 (default: 0.85).")
    parser.add_argument("--dim-bg", type=float, default=0.45, help="Background dimming factor 0-1 (default: 0.45).")
    parser.add_argument("--no-ref", action="store_true", help="Skip writing reference videos (use the intact image as 'video' instead).")
    parser.add_argument("--num-material-variants", type=int, default=1, help="Number of material variants to use per object, from the start of the annotation's list; <=0 for all (default: 1).")
    parser.add_argument("--num-fracture-variants", type=int, default=1, help="Number of fracture variants to use per object, from the start of the annotation's list; <=0 for all (default: 1).")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    annotations_dir = data_root / "annotations"
    if not annotations_dir.is_dir():
        parser.error(f"--data-root must contain an 'annotations/' subfolder: {annotations_dir} not found.")

    object_jsons = sorted(annotations_dir.glob("*.json"))
    if args.object_ids:
        wanted = set(args.object_ids.split(","))
        object_jsons = [p for p in object_jsons if p.stem in wanted]
        missing = wanted - {p.stem for p in object_jsons}
        if missing:
            parser.error(f"--object-ids not found in {annotations_dir}: {sorted(missing)}")

    if not object_jsons:
        parser.error(f"No annotation JSONs found under {annotations_dir}.")

    os.makedirs(args.output_dir, exist_ok=True)

    metadata_rows = []
    for json_path in object_jsons:
        object_id = json_path.stem
        with open(json_path) as f:
            cfg = json.load(f)

        samples = expand_samples(
            object_id, cfg, data_root,
            num_material_variants=args.num_material_variants,
            num_fracture_variants=args.num_fracture_variants,
        )
        num_fracture = len(_limit(cfg["properties"]["fracture"], args.num_fracture_variants)) if isinstance(cfg["properties"]["fracture"], list) else 1
        num_material = len(_limit(cfg["properties"]["material"]["youngs_modulus_Pa"], args.num_material_variants))
        num_velocity = sum(len(loading.get("velocity_options_px_per_frame") or [loading["velocity_px_per_frame"]]) for loading in cfg["loading"])
        print(f"[{object_id}] {len(samples)} sample(s) "
              f"({len(cfg['loading'])} loading, {num_velocity} loading x velocity combo(s) x "
              f"{num_material} material x "
              f"{num_fracture} fracture)")

        for i, sample in enumerate(samples):
            print(f"  [{i + 1}/{len(samples)}] {sample['sample_id']}")
            instances = build_instances(sample["loading"], sample["velocity_px_per_frame"])

            paths = vg.create_control_video(
                input_image_path=sample["image_path"],
                instances=instances,
                output_dir=args.output_dir,
                name=sample["sample_id"],
                n_frames=args.frames,
                fps=args.fps,
                sigma=args.sigma,
                blob_alpha=args.alpha,
                dim_background=args.dim_bg,
                skip_ref=args.no_ref,
            )

            # metadata.csv paths are relative to --output-dir (the training/inference
            # loader resolves them via ToAbsolutePath(dataset_base_path)), so the
            # benchmark folder stays portable if moved/copied elsewhere.
            ref_video_path = sample["image_path"] if args.no_ref else os.path.relpath(paths["reference"], args.output_dir)
            control_video_path = os.path.relpath(paths["control"], args.output_dir)
            metadata_rows.append(build_metadata_row(sample, ref_video_path, control_video_path))

    csv_path = os.path.join(args.output_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRAIN_METADATA_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metadata_rows)

    json_path_out = os.path.join(args.output_dir, "benchmark_metadata.json")
    with open(json_path_out, "w") as fh:
        json.dump([{k: row[k] for k in JSON_METADATA_COLUMNS} for row in metadata_rows], fh, indent=2)

    print(f"\nDone. {len(metadata_rows)} sample(s) written to {os.path.abspath(args.output_dir)}")
    print(f"  metadata.csv           : {csv_path}")
    print(f"  benchmark_metadata.json: {json_path_out}")


if __name__ == "__main__":
    main()
