"""
Stack several methods' result videos into one vertically-concatenated figure
per frame, for qualitative comparison (e.g. final_results/output_test_set/*).

Since methods have different native frame counts / fps / durations, frames
are aligned by *normalized time* (relative progress through each clip, 0..1),
not by raw frame index or wall-clock time -- each output frame i represents
the same fraction of the way through every method's own clip. All rows are
resized to a common width/height (`row_size`, defaulting to the first method's
native size) before stacking, with a thin black separator line between rows.

Usage (as a library -- see __main__ for the CLI form):
    from stack_methods_compare import stack_methods
    stack_methods(
        [("force_prompt", ".../seed42_croissant_config_0098.mp4"),
         ("phyco", ".../seed42_croissant_config_0098.mp4"),
         ("cogvideox", ".../seed42_croissant_config_0098.mp4"),
         ("wan_baseline", ".../seed42_croissant_config_0098.mp4"),
         ("stage3", ".../fracture/croissant_config_0098.mp4")],
        out_dir="vis_fig/main_qual/croissant_config_0098",
        num_frames=81,
    )

CLI:
    python stack_methods_compare.py [--row-size WxH] <out_dir> <num_frames> <line_width> \\
        <label1>=<video1> <label2>=<video2> ...
    (--row-size is needed when the first row is not at the target size, e.g. a
    1024x1024 benchmark control_video placed above 480x480 method outputs.)
"""

import os
import shutil
import subprocess
from PIL import Image, ImageDraw

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")
LINE_COLOR = (0, 0, 0)


def _extract_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-i", video_path, "-vsync", "0",
         "-start_number", "0", os.path.join(out_dir, "frame_%04d.png")],
        check=True,
    )
    return sorted(os.listdir(out_dir))


def stack_methods(methods, out_dir, num_frames=81, line_width=4, tmp_dir=None, row_size=None):
    """
    methods: list of (label, video_path) tuples, in the order they should be
    stacked top-to-bottom.
    """
    own_tmp = tmp_dir is None
    if tmp_dir is None:
        tmp_dir = os.path.join(out_dir, "_raw_frames")
    os.makedirs(out_dir, exist_ok=True)

    raw = {}  # label -> sorted list of frame filenames
    raw_dirs = {}
    for label, video_path in methods:
        d = os.path.join(tmp_dir, label)
        raw_dirs[label] = d
        raw[label] = _extract_frames(video_path, d)
        if not raw[label]:
            raise ValueError(f"No frames extracted for '{label}' from {video_path}")

    if row_size is None:
        ref_label = methods[0][0]
        row_size = Image.open(os.path.join(raw_dirs[ref_label], raw[ref_label][0])).size
    row_w, row_h = row_size

    for i in range(num_frames):
        rows = []
        for label, _ in methods:
            frames = raw[label]
            n = len(frames)
            src_idx = round(i * (n - 1) / (num_frames - 1)) if num_frames > 1 else 0
            im = Image.open(os.path.join(raw_dirs[label], frames[src_idx])).convert("RGB")
            if im.size != (row_w, row_h):
                im = im.resize((row_w, row_h), Image.LANCZOS)
            rows.append(im)

        total_h = row_h * len(rows) + line_width * (len(rows) - 1)
        combined = Image.new("RGB", (row_w, total_h), "white")
        y = 0
        for idx, im in enumerate(rows):
            combined.paste(im, (0, y))
            y += row_h
            if idx < len(rows) - 1 and line_width > 0:
                draw = ImageDraw.Draw(combined)
                draw.rectangle([0, y, row_w, y + line_width], fill=LINE_COLOR)
                y += line_width

        combined.save(os.path.join(out_dir, f"frame_{i:04d}.png"))

    if own_tmp:
        shutil.rmtree(tmp_dir)

    print(f"stacked: {num_frames} frames ({len(methods)} rows: {[m[0] for m in methods]}) -> {out_dir}")


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    row_size_arg = None
    if argv and argv[0] == "--row-size":
        w, h = argv[1].lower().split("x")
        row_size_arg = (int(w), int(h))
        argv = argv[2:]
    if len(argv) < 4:
        print(__doc__)
        sys.exit(1)
    out_dir_arg = argv[0]
    num_frames_arg = int(argv[1])
    line_width_arg = int(argv[2])
    methods_arg = []
    for spec in argv[3:]:
        label, path = spec.split("=", 1)
        methods_arg.append((label, path))
    stack_methods(methods_arg, out_dir_arg, num_frames=num_frames_arg, line_width=line_width_arg, row_size=row_size_arg)
