"""
Combine a recolored sample's per-frame PNGs into a single figure per frame:
left = fracture (full size), right = a 2x2 grid (each cell half the fracture's
size) of flow | strain / stress | damage (reading order top-left, top-right,
bottom-left, bottom-right). Thin black separator lines are drawn between the
fracture panel and the grid, and between the 4 grid cells, so tiles read as
distinct panels rather than bleeding into each other.

Usage:
    python combine_fracture_grid.py <recolored_dir> [out_dir] [line_width]

<recolored_dir> is a <sample>_recolored/ folder as produced by
recolor_physics_maps.py (must contain fracture/, flow/, strain/, stress/,
damage/ subfolders with matching frame_%04d.png names). Defaults
out_dir to <recolored_dir>/combined, line_width to 4px (scaled for the
default 1440px recolored frames; pass a smaller value for lower upscale
factors).
"""

import os
from PIL import Image, ImageDraw

GRID_ORDER = ["flow", "strain", "stress", "damage"]  # top-left, top-right, bottom-left, bottom-right
LINE_COLOR = (0, 0, 0)


def combine_frame(recolored_dir, fname, line_width=4):
    fracture = Image.open(os.path.join(recolored_dir, "fracture", fname)).convert("RGB")
    fw, fh = fracture.size
    cw, ch = fw // 2, fh // 2

    grid = Image.new("RGB", (fw, fh), "white")
    positions = [(0, 0), (cw, 0), (0, ch), (cw, ch)]
    for mod, (x, y) in zip(GRID_ORDER, positions):
        im = Image.open(os.path.join(recolored_dir, mod, fname)).convert("RGB")
        im = im.resize((cw, ch), Image.LANCZOS)
        grid.paste(im, (x, y))

    combined = Image.new("RGB", (fw + fw, fh), "white")
    combined.paste(fracture, (0, 0))
    combined.paste(grid, (fw, 0))

    if line_width > 0:
        draw = ImageDraw.Draw(combined)
        half = line_width / 2
        # fracture | grid divider
        draw.rectangle([fw - half, 0, fw + half, fh], fill=LINE_COLOR)
        # grid interior divider (vertical + horizontal, centered on the grid)
        gx = fw + cw
        draw.rectangle([gx - half, 0, gx + half, fh], fill=LINE_COLOR)
        draw.rectangle([fw, ch - half, fw + fw, ch + half], fill=LINE_COLOR)

    return combined


def combine_all(recolored_dir, out_dir=None, line_width=4):
    if out_dir is None:
        out_dir = os.path.join(recolored_dir, "combined")
    os.makedirs(out_dir, exist_ok=True)

    frames = sorted(os.listdir(os.path.join(recolored_dir, "fracture")))
    for fname in frames:
        out = combine_frame(recolored_dir, fname, line_width=line_width)
        out.save(os.path.join(out_dir, fname))
    print(f"combined: {len(frames)} frames -> {out_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) not in (2, 3, 4):
        print(__doc__)
        sys.exit(1)
    out_dir_arg = sys.argv[2] if len(sys.argv) >= 3 else None
    line_width_arg = int(sys.argv[3]) if len(sys.argv) == 4 else 4
    combine_all(sys.argv[1], out_dir_arg, line_width=line_width_arg)
