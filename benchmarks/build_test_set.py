"""
build_test_set.py

Filter and copy a subset of an existing training-format dataset split into
benchmarks/output_test_set: only samples whose force_video shows clearly
HORIZONTAL or VERTICAL tearing, excluding diagonal or near-static ones.

Direction is determined purely from the force_video itself (a Gaussian-blob
control video, same format this project's own benchmarks/utils/video_gen.py
produces): each blob's centroid is located in frame 0 and frame
COMPARE_FRAME_INDEX (connected-component labeling on a brightness threshold),
matched by nearest neighbor, and its net (dx, dy) displacement classified by
the |dx| vs |dy| ratio. A sample is kept only if every blob agrees on the same
direction and moved enough to be a meaningful signal.

Uses an early frame (default: frame 10) rather than the true last frame:
blobs move at constant per-frame velocity in this format, so direction is
identical either way, but many samples (e.g. the rubber3/rubber4 series) have
large enough displacement that a blob exits the visible canvas well before
the clip ends -- comparing against the true last frame then finds 0 or
mismatched blob counts and excludes the sample outright, regardless of its
actual direction. Frame 10 is early enough that blobs are essentially always
still on-canvas, and decoding only needs to run to frame 10, not the full
clip, which is also considerably faster.

Output layout mirrors build_demo_benchmark.py's (control_video/,
reference_video/) instead of the source split's own layout, so this benchmark
is usable the same way as benchmarks/output_demo2, output_demo3:

    output_test_set/
    |-- control_video/<file>.mp4    (was force_videos/ -- the force condition)
    |-- reference_video/<file>.mp4  (NEW: intact first frame repeated, same
    |                                frame count/fps as control_video so the
    |                                dataloader doesn't hit a shape mismatch --
    |                                pixel content comes from gt_vid/videos'
    |                                first frame)
    |-- gt_vid/
    |   |-- videos/<file>.mp4          (real simulation RGB -- ground truth)
    |   |-- stress_videos/<file>.mp4
    |   |-- flow_videos/<file>.mp4
    |   |-- damage_videos/<file>.mp4
    |   `-- strain_videos/<file>.mp4
    |-- metadata.csv                (video=reference_video, force_video=
    |                                control_video, stress/flow/damage/strain
    |                                =gt_vid/...; standard training columns)
    `-- selection_info.json         (per-kept-sample direction + blob displacement)

Usage:
    conda activate wan2.2
    python benchmarks/build_test_set.py \
        --source-dir data/preprocessed_sim_output_split_08_12/test \
        --output-dir benchmarks/output_test_set
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import imageio
import numpy as np
from scipy import ndimage

DIRECTION_RATIO = 10.0  # |dominant| must exceed this multiple of |other| to call a direction
# (empirically, this dataset's blob-displacement ratios cluster sharply below
# ~5 (11-22deg off-axis -- visibly diagonal) or above ~20 (<3deg off-axis --
# truly axis-aligned), with nothing in between; 10 sits in that gap.)
COMPARE_FRAME_INDEX = 10  # compare frame 0 against this frame, not the true last frame
MIN_DISPLACEMENT_PX = 3.0  # blobs moving less than this by COMPARE_FRAME_INDEX are "static" (excluded)
MIN_BLOB_PIXELS = 3
BRIGHTNESS_THRESHOLD = 40

GT_COLUMNS = {
    "video": "videos",
    "stress_video": "stress_videos",
    "flow_video": "flow_videos",
    "damage_video": "damage_videos",
    "strain_video": "strain_videos",
}


def find_blob_centroids(frame: np.ndarray) -> list[tuple[float, float]]:
    gray = frame.max(axis=-1) if frame.ndim == 3 else frame
    mask = gray > BRIGHTNESS_THRESHOLD
    labeled, n = ndimage.label(mask)
    centroids = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labeled == i)
        if len(ys) < MIN_BLOB_PIXELS:
            continue
        centroids.append((float(xs.mean()), float(ys.mean())))
    return centroids


def match_blobs(first: list, last: list) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Greedy nearest-neighbor matching between two small blob-centroid lists."""
    if not first or not last or len(first) != len(last):
        return []
    remaining = list(range(len(last)))
    pairs = []
    for fx, fy in first:
        dists = [((last[j][0] - fx) ** 2 + (last[j][1] - fy) ** 2, j) for j in remaining]
        _, j = min(dists)
        pairs.append(((fx, fy), last[j]))
        remaining.remove(j)
    return pairs


def get_frame_pair(path: str, compare_frame_index: int = COMPARE_FRAME_INDEX) -> tuple[np.ndarray, np.ndarray]:
    """Returns (frame 0, frame `compare_frame_index`). Decoding stops as soon
    as the compare frame is read (no need to decode the rest of the clip). If
    the video has fewer frames than that, falls back to its true last frame.
    """
    reader = imageio.get_reader(path)
    first = None
    last_seen = None
    compare = None
    for i, frame in enumerate(reader):
        if i == 0:
            first = frame
        last_seen = frame
        if i == compare_frame_index:
            compare = frame
            break
    reader.close()
    if compare is None:
        compare = last_seen
    return first, compare


def classify_direction(force_video_path: str):
    """Returns (direction | None, [(dx, dy, mag), ...]) -- direction is
    "horizontal"/"vertical" only if EVERY detected blob agrees; None
    (excluded) for diagonal, static, or ambiguous (unmatched blob count) cases.
    """
    first, last = get_frame_pair(force_video_path)
    blobs0 = find_blob_centroids(first)
    blobs1 = find_blob_centroids(last)
    pairs = match_blobs(blobs0, blobs1)
    if not pairs:
        return None, []

    per_blob = []
    directions = []
    for (x0, y0), (x1, y1) in pairs:
        dx, dy = x1 - x0, y1 - y0
        mag = (dx ** 2 + dy ** 2) ** 0.5
        per_blob.append((dx, dy, mag))
        if mag < MIN_DISPLACEMENT_PX:
            directions.append("static")
        elif abs(dx) > DIRECTION_RATIO * abs(dy):
            directions.append("horizontal")
        elif abs(dy) > DIRECTION_RATIO * abs(dx):
            directions.append("vertical")
        else:
            directions.append("diagonal")

    unique = set(directions)
    if unique == {"horizontal"}:
        return "horizontal", per_blob
    if unique == {"vertical"}:
        return "vertical", per_blob
    return None, per_blob


def build_reference_video(gt_video_path: Path, control_video_path: Path, output_path: Path) -> None:
    """Write a video that repeats the GT video's first frame (the intact,
    not-yet-deformed object) for the same frame count/fps as control_video --
    matching lengths is required, or the dataloader hits a tensor shape
    mismatch when video and force_video disagree on frame count."""
    reader = imageio.get_reader(str(gt_video_path))
    first_frame = reader.get_data(0)
    reader.close()

    reader = imageio.get_reader(str(control_video_path))
    fps = reader.get_meta_data().get("fps", 25)
    n_frames = sum(1 for _ in reader)
    reader.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    ) as writer:
        for _ in range(n_frames):
            writer.append_data(first_frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", required=True, help="Existing training-format split (has metadata.csv + videos/force_videos/... subfolders).")
    parser.add_argument("--output-dir", default="benchmarks/output_test_set", help="Where to copy the filtered subset.")
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N rows of metadata.csv (for a quick test run).")
    parser.add_argument("--include-prefixes", default=None, help="Comma-separated video-filename prefixes to keep (e.g. 'bread_config,rubber_toy_config'); everything else is skipped before direction classification. Default: keep all.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    src_metadata_path = source_dir / "metadata.csv"
    if not src_metadata_path.is_file():
        parser.error(f"{src_metadata_path} not found.")

    with open(src_metadata_path) as f:
        rows = list(csv.DictReader(f))
    if args.include_prefixes:
        prefixes = tuple(args.include_prefixes.split(","))
        rows = [r for r in rows if Path(r["video"]).name.startswith(prefixes)]
    if args.limit:
        rows = rows[: args.limit]
    fieldnames = list(rows[0].keys()) if rows else []

    kept_rows = []
    selection_info = []
    counts = {"horizontal": 0, "vertical": 0, "excluded": 0}

    for i, row in enumerate(rows):
        force_video_path = source_dir / row["force_video"]
        direction, per_blob = classify_direction(str(force_video_path))
        print(f"[{i + 1}/{len(rows)}] {row['force_video']}: {direction or 'excluded'} "
              f"({[(round(dx, 1), round(dy, 1)) for dx, dy, _ in per_blob]})")

        if direction is None:
            counts["excluded"] += 1
            continue
        counts[direction] += 1

        # control_video: the force condition itself, just relocated.
        control_dst = output_dir / "control_video" / Path(row["force_video"]).name
        control_dst.parent.mkdir(parents=True, exist_ok=True)
        if not control_dst.exists():
            shutil.copy2(force_video_path, control_dst)

        # gt_vid/<modality>/: real simulation outputs, relocated verbatim.
        for col, subdir in GT_COLUMNS.items():
            src = source_dir / row[col]
            dst = output_dir / "gt_vid" / subdir / Path(row[col]).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)

        # reference_video: NEW, derived (not a straight copy).
        gt_video_dst = output_dir / "gt_vid" / "videos" / Path(row["video"]).name
        reference_dst = output_dir / "reference_video" / Path(row["video"]).name
        if not reference_dst.exists():
            build_reference_video(gt_video_dst, control_dst, reference_dst)

        new_row = dict(row)
        new_row["video"] = str(Path("reference_video") / Path(row["video"]).name)
        new_row["force_video"] = str(Path("control_video") / Path(row["force_video"]).name)
        for col, subdir in GT_COLUMNS.items():
            if col == "video":
                continue
            new_row[col] = str(Path("gt_vid") / subdir / Path(row[col]).name)

        kept_rows.append(new_row)
        selection_info.append({
            "sample": row["force_video"],
            "direction": direction,
            "blob_displacements_px": [{"dx": dx, "dy": dy, "mag": mag} for dx, dy, mag in per_blob],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    info_path = output_dir / "selection_info.json"
    with open(info_path, "w") as fh:
        json.dump(selection_info, fh, indent=2)

    print(f"\nScanned {len(rows)} sample(s): "
          f"{counts['horizontal']} horizontal, {counts['vertical']} vertical, {counts['excluded']} excluded.")
    print(f"Kept {len(kept_rows)} sample(s) -> {output_dir}")
    print(f"  metadata.csv       : {csv_path}")
    print(f"  selection_info.json: {info_path}")


if __name__ == "__main__":
    main()
