#!/usr/bin/env python3
"""
Plot value range history of physical maps across multiple simulation folders.
Accepts either:
  - a root folder containing multiple simulation subfolders (multi-sim mode)
  - a single simulation folder directly (single-sim mode)
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Utils
# ============================================================

def sorted_npy_files(folder):
    return sorted(glob.glob(os.path.join(folder, "*.npy")))


def load_frame_stats(fpath):
    arr = np.load(fpath).astype(np.float32).ravel()

    # exclude background zeros from stats — black background rasterizer
    nonzero = arr[arr > 1e-8]
    if len(nonzero) == 0:
        nonzero = arr   # fallback if truly all zero

    return {
        "min" : float(nonzero.min()),
        "max" : float(nonzero.max()),
        "mean": float(nonzero.mean()),
        "std" : float(nonzero.std()),
        "p25" : float(np.percentile(nonzero, 25)),
        "p50" : float(np.percentile(nonzero, 50)),
        "p75" : float(np.percentile(nonzero, 75)),
        "p95" : float(np.percentile(nonzero, 95)),
    }


def collect_sim_stats(sim_folder, map_name, folder_map):
    """Collect per-frame stats for one simulation folder and one map."""
    map_dir = os.path.join(sim_folder, folder_map[map_name])
    if not os.path.isdir(map_dir):
        return None

    files = sorted_npy_files(map_dir)
    if not files:
        return None

    return [load_frame_stats(f) for f in tqdm(files, desc=f"  frames", leave=False)]


def is_single_sim_folder(folder, folder_map):
    """
    Detect if folder is a single simulation output (contains map subdirs directly)
    vs a root folder containing multiple simulation subfolders.
    """
    return any(
        os.path.isdir(os.path.join(folder, subdir))
        for subdir in folder_map.values()
    )


def resolve_sim_folders(sim_root, pattern, folder_map):
    """
    Returns (sim_folders, mode) where mode is 'single' or 'multi'.
    
    Single mode: sim_root itself is one simulation folder
    Multi mode:  sim_root contains multiple simulation subfolders
    """
    if is_single_sim_folder(sim_root, folder_map):
        print(f"Detected single simulation folder: {sim_root}")
        return [sim_root], "single"

    # multi mode — discover subfolders
    candidates = sorted([
        d for d in glob.glob(os.path.join(sim_root, pattern))
        if os.path.isdir(d)
    ])

    # filter to only folders that actually contain simulation data
    sim_folders = [
        d for d in candidates
        if is_single_sim_folder(d, folder_map)
    ]

    if not sim_folders:
        raise FileNotFoundError(
            f"No valid simulation subfolders found under '{sim_root}'.\n"
            f"Expected either:\n"
            f"  (a) '{sim_root}' to directly contain map folders like '2d_stress_map/'\n"
            f"  (b) subfolders of '{sim_root}' to each contain map folders"
        )

    print(f"Detected multi-simulation root: {len(sim_folders)} folders found.")
    return sim_folders, "multi"


# ============================================================
# Aggregate across simulations
# ============================================================

def aggregate_across_sims(all_sim_records):
    """
    all_sim_records: list of sim records, each a list of per-frame dicts.
    Aligns by frame index (shortest truncates).
    Returns per-frame aggregated arrays of shape [n_frames, n_sims].
    """
    n_frames = min(len(r) for r in all_sim_records)
    keys = ["min", "max", "mean", "std", "p25", "p50", "p75", "p95"]
    agg  = {k: [] for k in keys}

    for fi in range(n_frames):
        for k in keys:
            agg[k].append([sim[fi][k] for sim in all_sim_records])

    return {k: np.array(v) for k, v in agg.items()}, n_frames


# ============================================================
# Plotting
# ============================================================

MAP_COLORS = {
    "damage": "#e74c3c",
    "stress": "#e67e22",
    "strain": "#2ecc71",
}

MAP_UNITS = {
    "damage": "[0–1]",
    "stress": "[Pa]",
    "strain": "[–]",
}

def plot_map_history(agg, n_frames, map_name, ax, mode, sim_labels=None):
    frames = np.arange(n_frames)
    color  = MAP_COLORS.get(map_name, "#888888")
    unit   = MAP_UNITS.get(map_name, "")

    if mode == "single":
        mean = agg["mean"][:, 0]
        ax.plot(frames, mean, color=color, lw=2.0, label="mean")

    else:
        avg_mean = agg["mean"].mean(axis=1)   # [n_frames] — average across sims
        ax.plot(frames, avg_mean, color=color, lw=2.0, label="mean across sims")

        # faint individual sim lines if small number of sims
        if sim_labels and len(sim_labels) <= 10:
            for i, label in enumerate(sim_labels):
                ax.plot(
                    frames,
                    agg["mean"][:, i],
                    color=color, lw=0.8, alpha=0.3, label=label,
                )

    ax.set_title(f"{map_name.capitalize()} {unit}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean Value")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.set_xlim(0, n_frames - 1)

def plot_all_maps_history(
    sim_folders,
    maps=("damage", "stress", "strain"),
    out_path="./map_range_history.png",
    sim_labels=None,
    log_scale_maps=("stress",),
    mode="multi",
):
    folder_map = {
        "damage": "2d_damage_map",
        "stress": "2d_stress_map",
        "strain": "2d_strain_map",
    }

    n_maps = len(maps)
    fig, axes = plt.subplots(n_maps, 1, figsize=(14, 4 * n_maps), sharex=False)
    if n_maps == 1:
        axes = [axes]

    title = (
        f"Physical Map Value Ranges — {sim_labels[0] if sim_labels else os.path.basename(sim_folders[0])}"
        if mode == "single"
        else f"Physical Map Value Ranges — {len(sim_folders)} simulations"
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for ax, map_name in zip(axes, maps):
        print(f"\nCollecting [{map_name}]...")

        all_sim_records = []
        valid_labels    = []

        for i, folder in enumerate(sim_folders):
            records = collect_sim_stats(folder, map_name, folder_map)
            if records is None:
                print(f"  [SKIP] {folder} — map not found")
                continue
            all_sim_records.append(records)
            valid_labels.append(
                sim_labels[i] if sim_labels else os.path.basename(folder)
            )

        if not all_sim_records:
            ax.set_title(f"{map_name} — no data found", color="red")
            continue

        agg, n_frames = aggregate_across_sims(all_sim_records)
        plot_map_history(agg, n_frames, map_name, ax, mode=mode, sim_labels=valid_labels)

        if map_name in log_scale_maps:
            ymin, _ = ax.get_ylim()
            if ymin >= 0:
                ax.set_yscale("symlog", linthresh=1.0)
                ax.set_ylabel("Value (symlog)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot physical map value histories.")
    parser.add_argument(
        "--sim_root",
        type=str,
        required=True,
        help=(
            "Either a single simulation output folder (containing 2d_stress_map/ etc.) "
            "or a root folder containing multiple simulation subfolders."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./vis_phys_map/stat_phys_map.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        default=["damage", "stress", "strain"],
        choices=["damage", "stress", "strain"],
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help="Glob pattern to filter subfolders in multi-sim mode, e.g. 'config_*'",
    )
    args = parser.parse_args()

    folder_map = {
        "damage": "2d_damage_map",
        "stress": "2d_stress_map",
        "strain": "2d_strain_map",
    }

    sim_folders, mode = resolve_sim_folders(args.sim_root, args.pattern, folder_map)

    print(f"Mode: {mode} | Folders: {len(sim_folders)}")
    for f in sim_folders:
        print(f"  {f}")

    plot_all_maps_history(
        sim_folders = sim_folders,
        maps        = tuple(args.maps),
        out_path    = args.out,
        mode        = mode,
    )
