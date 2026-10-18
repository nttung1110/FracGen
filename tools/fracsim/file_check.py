import os
import re
import cv2
from collections import defaultdict
from statistics import median
from tqdm import tqdm

import sys
BASE_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/fracgen/preprocessed_sim_output"  # usage: python file_check.py <preprocessed_root>

VIDEO_FOLDERS = [
    "force_videos",
    "damage_videos",
    "flow_videos",
    "strain_videos",
    "stress_videos",
    "videos"
]

def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count

def extract_config_id(fname):
    """Extract config_XXXX from any filename variant like config_0060_force.mp4 or config_0060.mp4"""
    match = re.search(r"(config_\d+)", fname)
    return match.group(1) if match else None

# Collect all video paths first
all_tasks = []
for folder in VIDEO_FOLDERS:
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        print(f"⚠️  Folder not found: {folder}")
        continue
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".mp4"):
            config_id = extract_config_id(fname)
            if config_id:
                all_tasks.append((config_id, fname, folder, os.path.join(folder_path, fname)))

# Collect frame counts keyed by config_id
# all_data[config_id][folder] = (fname, frame_count)
all_data = defaultdict(dict)

for config_id, fname, folder, fpath in tqdm(all_tasks, desc="Scanning videos", unit="video"):
    all_data[config_id][folder] = (fname, get_frame_count(fpath))

# Report outliers
print(f"\n{'Config':<20} {'Folder':<20} {'File':<40} {'Frames':>8}  Status")
print("-" * 100)

outlier_found = False

for config_id, folder_data in sorted(all_data.items()):
    counts = [count for _, count in folder_data.values() if count is not None]
    if not counts:
        continue

    expected = median(counts)

    for folder, (fname, count) in sorted(folder_data.items()):
        is_outlier = count != expected
        if is_outlier:
            outlier_found = True
            print(f"{config_id:<20} {folder:<20} {fname:<40} {str(count):>8}  ❌ DIFFERENT (expected {int(expected)})")

    # Also flag if a folder is missing entirely for this config
    missing = set(VIDEO_FOLDERS) - set(folder_data.keys())
    for folder in missing:
        print(f"{config_id:<20} {folder:<20} {'N/A':<40} {'N/A':>8}  ❌ MISSING")
        outlier_found = True

if not outlier_found:
    print("✅ All videos have consistent frame counts across folders.")