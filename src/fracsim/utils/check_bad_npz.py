#!/usr/bin/env python3
"""
Scan simulation config folders for corrupted .npz files.

Usage:
  python check_bad_npz.py <root_folder> [--delete] [--log corrupted.txt]

Expected layout:
  <root_folder>/config_0000/...
  <root_folder>/config_0001/...
  ...

For each config folder, this script recursively checks all `.npz` files and
reports the exact corrupted path(s) and their error messages.
"""

import argparse
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm  # type: ignore


def check_npz(path: Path) -> tuple[bool, str]:
    """
    Return (is_ok, error_message).

    We validate in two stages:
    1. Zip-level CRC verification.
    2. Actual array loading via np.load.
    """
    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                return False, f"Bad CRC in member '{bad}'"
    except zipfile.BadZipFile as e:
        return False, f"BadZipFile: {e}"
    except Exception as e:
        return False, f"Zip open error: {e}"

    try:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                _ = data[key]
    except zipfile.BadZipFile as e:
        return False, f"BadZipFile during np.load: {e}"
    except Exception as e:
        return False, f"np.load error: {e}"

    return True, ""


def _discover_config_dirs(root: Path) -> list[Path]:
    config_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if config_dirs:
        return config_dirs
    if root.is_dir():
        return [root]
    return []


def _extract_config_idx(config_dir: Path) -> int:
    match = re.search(r"(\d+)$", config_dir.name)
    if match is None:
        raise ValueError(f"Could not extract numeric config index from directory name: {config_dir.name}")
    return int(match.group(1))


def _filter_config_dirs(
    config_dirs: list[Path],
    start_config_idx: int | None,
    end_config_idx: int | None,
) -> list[Path]:
    if start_config_idx is None and end_config_idx is None:
        return config_dirs

    filtered: list[Path] = []
    for config_dir in config_dirs:
        config_idx = _extract_config_idx(config_dir)
        if start_config_idx is not None and config_idx < start_config_idx:
            continue
        if end_config_idx is not None and config_idx > end_config_idx:
            continue
        filtered.append(config_dir)
    return filtered


def _range_suffix(start_config_idx: int | None, end_config_idx: int | None) -> str:
    if start_config_idx is None and end_config_idx is None:
        return "all"
    start_str = "start" if start_config_idx is None else f"{start_config_idx:04d}"
    end_str = "end" if end_config_idx is None else f"{end_config_idx:04d}"
    return f"{start_str}_{end_str}"


def _resolve_log_path(
    log: str | None,
    start_config_idx: int | None,
    end_config_idx: int | None,
) -> Path:
    suffix = _range_suffix(start_config_idx, end_config_idx)

    if log is None:
        log_dir = Path.cwd() / "log_bad_npz"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"bad_npz_{suffix}.txt"

    log_path = Path(log)
    if log_path.suffix:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return log_path.with_name(f"{log_path.stem}_{suffix}{log_path.suffix}")

    log_path.mkdir(parents=True, exist_ok=True)
    return log_path / f"bad_npz_{suffix}.txt"


def _scan_one_config(config_dir: Path, root: Path, print_results: bool = True) -> list[tuple[Path, str]]:
    npz_files = sorted(config_dir.rglob("*.npz"))
    corrupted: list[tuple[Path, str]] = []

    for npz_path in npz_files:
        ok, msg = check_npz(npz_path)
        if not ok:
            corrupted.append((npz_path, msg))

    if corrupted and print_results:
        print(f"\nCONFIG CORRUPTED: {config_dir.relative_to(root)}")
        for npz_path, msg in corrupted:
            print(f"  {npz_path.relative_to(root)}  [{msg}]")

    return corrupted


def _scan_one_config_task(task: tuple[str, str]) -> tuple[str, list[tuple[str, str]], int]:
    config_dir = Path(task[0])
    root = Path(task[1])
    npz_files = sorted(config_dir.rglob("*.npz"))
    corrupted = _scan_one_config(config_dir, root, print_results=False)
    return str(config_dir), [(str(npz_path), msg) for npz_path, msg in corrupted], len(npz_files)


def scan_configs(
    folder: str,
    delete: bool = False,
    log: str | None = None,
    workers: int = 1,
    start_config_idx: int | None = None,
    end_config_idx: int | None = None,
) -> int:
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    config_dirs = _discover_config_dirs(root)
    if start_config_idx is not None and start_config_idx < 0:
        raise ValueError(f"--start_config_idx must be >= 0, got {start_config_idx}")
    if end_config_idx is not None and end_config_idx < 0:
        raise ValueError(f"--end_config_idx must be >= 0, got {end_config_idx}")
    if start_config_idx is not None and end_config_idx is not None and start_config_idx > end_config_idx:
        raise ValueError(
            f"--start_config_idx must be <= --end_config_idx, got {start_config_idx} > {end_config_idx}"
        )
    config_dirs = _filter_config_dirs(config_dirs, start_config_idx=start_config_idx, end_config_idx=end_config_idx)
    if not config_dirs:
        print("No config directories found for the requested index range.")
        return 0

    range_suffix = ""
    if start_config_idx is not None or end_config_idx is not None:
        range_suffix = (
            f" (filtered to config indices "
            f"{start_config_idx if start_config_idx is not None else '-inf'}"
            f"..{end_config_idx if end_config_idx is not None else '+inf'})"
        )
    print(f"Scanning {len(config_dirs)} config folder(s) in '{root}'{range_suffix} ...")

    all_corrupted: list[tuple[Path, Path, str]] = []
    total_npz = 0

    if workers <= 0:
        workers = os.cpu_count() or 1

    if workers == 1:
        for config_dir in tqdm(config_dirs, desc="Checking config folders", unit="config"):
            npz_files = list(config_dir.rglob("*.npz"))
            total_npz += len(npz_files)
            corrupted = _scan_one_config(config_dir, root)
            for npz_path, msg in corrupted:
                all_corrupted.append((config_dir, npz_path, msg))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_scan_one_config_task, (str(config_dir), str(root)))
                for config_dir in config_dirs
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Checking config folders ({workers} workers)",
                unit="config",
            ):
                config_dir_str, corrupted_items, npz_count = future.result()
                config_dir = Path(config_dir_str)
                total_npz += int(npz_count)
                if corrupted_items:
                    print(f"\nCONFIG CORRUPTED: {config_dir.relative_to(root)}")
                    for npz_path_str, msg in corrupted_items:
                        npz_path = Path(npz_path_str)
                        print(f"  {npz_path.relative_to(root)}  [{msg}]")
                        all_corrupted.append((config_dir, npz_path, msg))

    print(f"\n{'=' * 60}")
    print(f"Config folders : {len(config_dirs)}")
    print(f"Total npz files: {total_npz}")
    print(f"Bad npz files  : {len(all_corrupted)}")

    bad_configs = sorted({config_dir for config_dir, _, _ in all_corrupted})
    print(f"Bad configs    : {len(bad_configs)}")

    if all_corrupted:
        print("\nCorrupted configs and paths:")
        current_config: Path | None = None
        for config_dir, npz_path, msg in all_corrupted:
            if config_dir != current_config:
                current_config = config_dir
                print(f"  {config_dir.relative_to(root)}")
            print(f"  {npz_path.relative_to(root)}  [{msg}]")

    if log is not None:
        log_path = _resolve_log_path(log, start_config_idx=start_config_idx, end_config_idx=end_config_idx)
        lines = [
            f"{config_dir}\t{npz_path}\t{msg}"
            for config_dir, npz_path, msg in all_corrupted
        ]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(lines)
        if content:
            content += "\n"
        log_path.write_text(content, encoding="utf-8")
        print(f"\nLog written to: {log_path}")
        if not all_corrupted:
            print("Log is empty because no corrupted .npz files were found.")

    if all_corrupted:
        if delete:
            print("\nDeleting corrupted files...")
            for _, npz_path, _ in all_corrupted:
                npz_path.unlink()
                print(f"  Deleted: {npz_path}")

    return len(all_corrupted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check config folders for corrupted .npz files.")
    parser.add_argument("folder", help="Root folder containing config subfolders")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete corrupted .npz files after scanning",
    )
    parser.add_argument(
        "--log",
        metavar="FILE",
        default="./log_bad_npz",
        help=(
            "Optional log file or directory. The script automatically appends the "
            "config-index-range suffix. Default folder is ./log_bad_npz."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use across config folders (0 uses all CPUs).",
    )
    parser.add_argument(
        "--start_config_idx",
        type=int,
        default=None,
        help="Inclusive lower bound of config index to scan, e.g. 100 for config_0100.",
    )
    parser.add_argument(
        "--end_config_idx",
        type=int,
        default=None,
        help="Inclusive upper bound of config index to scan, e.g. 999 for config_0999.",
    )
    args = parser.parse_args()

    raise SystemExit(
        1
        if scan_configs(
            args.folder,
            delete=args.delete,
            log=args.log,
            workers=args.workers,
            start_config_idx=args.start_config_idx,
            end_config_idx=args.end_config_idx,
        )
        else 0
    )
