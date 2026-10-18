"""
Frame-by-frame comparison between two folders of matching-named PNG frames
(e.g. two <sample>_recolored/combined/ folders): horizontally concatenates
frame_%04d.png from folder A next to the same-named frame from folder B,
saving each result as an independent image (not a video) -- same idea as the
earlier ad-hoc A/B stage comparisons in this folder (A_stage1_.../,
B_stage3_.../, concat_A_B/), just as a reusable script.

Usage:
    python concat_frame_folders.py <folder_A> <folder_B> <out_dir>

Both folders must contain the same set of frame_%04d.png filenames (same
count, zero-padded the same way). Frames are matched by filename, not by
sorted-order-position, so mismatched sets fail loudly instead of silently
pairing the wrong frames.
"""

import os
from PIL import Image


def concat_frame_folders(dir_a, dir_b, out_dir):
    names_a = set(os.listdir(dir_a))
    names_b = set(os.listdir(dir_b))
    if names_a != names_b:
        only_a = sorted(names_a - names_b)
        only_b = sorted(names_b - names_a)
        raise ValueError(
            f"Frame sets differ: {len(only_a)} only in {dir_a} (e.g. {only_a[:3]}), "
            f"{len(only_b)} only in {dir_b} (e.g. {only_b[:3]})"
        )

    os.makedirs(out_dir, exist_ok=True)
    for fname in sorted(names_a):
        im_a = Image.open(os.path.join(dir_a, fname)).convert("RGB")
        im_b = Image.open(os.path.join(dir_b, fname)).convert("RGB")
        if im_a.size[1] != im_b.size[1]:
            # match heights (Lanczos) before hstacking, in case sources differ
            h = min(im_a.size[1], im_b.size[1])
            im_a = im_a.resize((int(im_a.size[0] * h / im_a.size[1]), h), Image.LANCZOS)
            im_b = im_b.resize((int(im_b.size[0] * h / im_b.size[1]), h), Image.LANCZOS)
        combined = Image.new("RGB", (im_a.width + im_b.width, im_a.height), "white")
        combined.paste(im_a, (0, 0))
        combined.paste(im_b, (im_a.width, 0))
        combined.save(os.path.join(out_dir, fname))

    print(f"combined: {len(names_a)} frames -> {out_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    concat_frame_folders(sys.argv[1], sys.argv[2], sys.argv[3])
