from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from collections import Counter
from pathlib import Path

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    tqdm = None


DEFAULT_DATASET_ROOT = Path("data/fracgen/sample_10/train")
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    if "disable" not in kwargs:
        kwargs["disable"] = False
    return tqdm(iterable, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a folder recursively, compare video frame lengths, and log videos "
            "whose frame counts differ from the dominant length."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Folder to scan recursively for videos.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional CSV log path. Defaults to <root>/irregular_video_lengths.csv.",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help="Allowed absolute difference from the dominant frame count.",
    )
    return parser.parse_args()


def find_videos(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def get_frame_count(video_path: Path) -> int:
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        cmd = [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        output = result.stdout.strip()
        if result.returncode == 0 and output.isdigit():
            return int(output)

    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

        if frame_count < 0:
            raise RuntimeError(f"Invalid frame count {frame_count} for video: {video_path}")
        return frame_count

    raise RuntimeError(
        "No supported video backend found. Install OpenCV (`pip install opencv-python`) "
        "or make `ffprobe` available on PATH."
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    log_path = args.log_path.resolve() if args.log_path else root / "irregular_video_lengths.csv"

    if not root.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {root}")

    videos = find_videos(root)
    if not videos:
        raise FileNotFoundError(f"No video files found under: {root}")

    rows: list[tuple[Path, int]] = []
    failures: list[tuple[Path, str]] = []

    for video_path in progress(videos, desc="Checking videos", unit="video"):
        try:
            rows.append((video_path, get_frame_count(video_path)))
        except Exception as exc:
            failures.append((video_path, str(exc)))

    if not rows:
        raise RuntimeError("No readable videos found.")

    frame_counts = [frame_count for _, frame_count in rows]
    dominant_length, dominant_count = Counter(frame_counts).most_common(1)[0]

    irregular_rows = [
        (video_path, frame_count)
        for video_path, frame_count in rows
        if abs(frame_count - dominant_length) > args.tolerance
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_count", "expected_frame_count", "delta", "status"])
        for video_path, frame_count in irregular_rows:
            writer.writerow(
                [
                    str(video_path),
                    frame_count,
                    dominant_length,
                    frame_count - dominant_length,
                    "irregular",
                ]
            )
        for video_path, error in failures:
            writer.writerow([str(video_path), "", dominant_length, "", f"error: {error}"])

    print(f"Scanned {len(videos)} videos under: {root}")
    print(f"Readable videos: {len(rows)}")
    print(f"Dominant frame length: {dominant_length} frames ({dominant_count} videos)")
    print(f"Irregular videos: {len(irregular_rows)}")
    print(f"Unreadable videos: {len(failures)}")
    print(f"Log written to: {log_path}")

    if irregular_rows:
        print("\nIrregular videos:")
        for video_path, frame_count in irregular_rows[:20]:
            print(f"  {video_path}: {frame_count} frames")
        if len(irregular_rows) > 20:
            print(f"  ... and {len(irregular_rows) - 20} more")

    if failures:
        print("\nUnreadable videos:")
        for video_path, error in failures[:20]:
            print(f"  {video_path}: {error}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
