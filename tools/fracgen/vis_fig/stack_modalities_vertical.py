"""
Stack a recolored sample's modality frames vertically, in a given order, with
a thin black separator line between rows -- like combine_fracture_grid.py's
fracture+2x2-grid layout, but as a plain N-row vertical stack instead of a
1+2x2 layout, so any subset/order of modalities can be compared frame by
frame (e.g. stress, strain, damage, flow, fracture top-to-bottom).

Usage:
    python stack_modalities_vertical.py <recolored_dir> <out_dir> [line_width] <mod1> <mod2> ...

<recolored_dir> is a <sample>_recolored/ folder as produced by
recolor_physics_maps.py (must contain a subfolder per named modality with
matching frame_%04d.png names). Modalities are stacked top-to-bottom in the
order given on the command line. line_width defaults to 4px (matching
recolor_physics_maps.py's default upscale=3 -> 1440px frames).
"""

import os
from PIL import Image, ImageDraw

LINE_COLOR = (0, 0, 0)


def stack_frame(recolored_dir, fname, order, line_width=4):
    rows = []
    for mod in order:
        im = Image.open(os.path.join(recolored_dir, mod, fname)).convert("RGB")
        rows.append(im)

    row_w, row_h = rows[0].size
    for im in rows[1:]:
        if im.size != (row_w, row_h):
            raise ValueError(f"Size mismatch: expected {(row_w, row_h)}, got {im.size}")

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
    return combined


def stack_all(recolored_dir, out_dir, order, line_width=4):
    os.makedirs(out_dir, exist_ok=True)
    frames = sorted(os.listdir(os.path.join(recolored_dir, order[0])))
    for fname in frames:
        out = stack_frame(recolored_dir, fname, order, line_width=line_width)
        out.save(os.path.join(out_dir, fname))
    print(f"stacked: {len(frames)} frames ({len(order)} rows: {order}) -> {out_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    recolored_dir_arg = sys.argv[1]
    out_dir_arg = sys.argv[2]
    rest = sys.argv[3:]
    if rest[0].isdigit():
        line_width_arg = int(rest[0])
        order_arg = rest[1:]
    else:
        line_width_arg = 4
        order_arg = rest
    stack_all(recolored_dir_arg, out_dir_arg, order_arg, line_width=line_width_arg)
