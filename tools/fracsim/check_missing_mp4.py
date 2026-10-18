#!/usr/bin/env python3
"""Report subfolders that do not contain an MP4 file."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ROOT = Path("data/fracsim/sim_output")


def has_mp4(folder: Path, recursive: bool) -> bool:
    pattern = "**/*.mp4" if recursive else "*.mp4"
    return any(folder.glob(pattern))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check which immediate subfolders are missing MP4 files."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory to scan. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Count MP4 files inside nested folders too.",
    )
    parser.add_argument(
        "--show-present",
        action="store_true",
        help="Also print subfolders that contain an MP4 file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.is_dir():
        print(f"Error: directory does not exist: {root}")
        return 1

    subfolders = sorted(path for path in root.iterdir() if path.is_dir())
    missing = []
    present = []

    for folder in subfolders:
        if has_mp4(folder, args.recursive):
            present.append(folder)
        else:
            missing.append(folder)

    print(f"Scanned {len(subfolders)} subfolders in {root}")
    print(f"Missing MP4: {len(missing)}")

    for folder in missing:
        print(folder)

    if args.show_present:
        print(f"\nPresent MP4: {len(present)}")
        for folder in present:
            print(folder)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
