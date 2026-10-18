"""
build_demo.py -- build FracGen inference inputs that vary the force condition and the fracture
timing for one object, from an intact image and a two-blob pull trajectory.

    python benchmarks/build_demo.py \
        --image data/benchmark/demo_bread/bread_intact.png \
        --trajectory data/benchmark/demo_bread/trajectory_info.json \
        --output-dir outputs/fracgen/demo_bread

Writes <output-dir>/{metadata.csv, control_video/, reference_video/, visualization_video/}
in the layout scripts/fracgen/infer.sh expects. One row per sample:

  force condition  : horizontal pull at each --speeds factor, and a vertical pull (the trajectory
                     rotated by 90 degrees about the image centre) at the middle speed
  fracture timing  : horizontal pull at the middle speed for each --timing lambda_onset:lambda_crit
                     pair (low thresholds tear early, high thresholds tear late)
  material         : horizontal pull at the middle speed for each --materials Young's modulus
                     (soft ... stiff), e.g. --materials 1e3,1e6

The pull trajectory comes either from --trajectory (json with per-blob start position and
px/frame velocity, as produced by tracking a FracSim force video) or from a benchmark annotation
(--annotation data/benchmark/annotations/<object>.json --load "Load 2"), whose loading entry gives
the two handle positions, pull directions and velocity.  --cases selects which groups to build.

The trajectory json holds, per blob, the start position (x0, y0) and velocity (vx, vy) in
px/frame at unit speed; `benchmarks/build_bread_only.py`-style tracking of a FracSim force video
produces it, or write it by hand for a new image.
"""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCHMARKS_DIR))
from utils import video_gen as vg  # noqa: E402

COLUMNS = vg.METADATA_COLUMNS + ["young_modulus", "poisson_ratio", "lambda_onset", "lambda_crit", "category"]


def instances(blobs, speed, rotate90, W, H):
    out = []
    for i, b in enumerate(blobs):
        x, y, vx, vy = b["x0"], b["y0"], b["vx"] * speed, b["vy"] * speed
        if rotate90:  # rotate about the image centre: (x, y) -> (cx - (y - cy), cy + (x - cx))
            cx, cy = W / 2, H / 2
            x, y = cx - (y - cy), cy + (x - cx)
            vx, vy = -vy, vx
        out.append({"x": x, "y": y, "vx": vx, "vy": vy, "color": vg.INSTANCE_COLORS[i]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="intact object image (square, e.g. 512x512)")
    ap.add_argument("--trajectory", default=None, help="json with 'blobs': [{x0,y0,vx,vy}, ...]")
    ap.add_argument("--annotation", default=None, help="benchmark annotation json (alternative to --trajectory)")
    ap.add_argument("--load", default=None, help="loading entry name in the annotation, e.g. 'Load 2'")
    ap.add_argument("--cases", default="force,timing,material", help="which groups to build")
    ap.add_argument("--materials", default="", help="Young's moduli for the material case, e.g. 1e3,1e6")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--name", default=None, help="sample name prefix (default: image stem)")
    ap.add_argument("--speeds", default="1.5,2.5,4", help="speed factors for the horizontal pull")
    ap.add_argument("--timing", default="2:3,3.3:4", help="lambda_onset:lambda_crit pairs (early ... late)")
    ap.add_argument("--young-modulus", type=float, default=1e4)
    ap.add_argument("--poisson-ratio", type=float, default=0.25)
    ap.add_argument("--category", type=int, default=0, help="0 bread-like, 1 rubber-like")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--sigma", type=float, default=6.0, help="force blob radius in px")
    args = ap.parse_args()

    from PIL import Image
    W, H = Image.open(args.image).size
    if args.trajectory:
        blobs = json.load(open(args.trajectory))["blobs"]
    elif args.annotation and args.load:
        import math
        cfg = json.load(open(args.annotation))
        load = next(l for l in cfg["loading"] if l["name"] == args.load)
        v = float(load["velocity_px_per_frame"])
        blobs = []
        for key in ("point_A", "point_B"):
            if key in load:
                th = math.radians(load[key]["direction_deg"])   # math convention, y up -> negate vy
                blobs.append({"x0": float(load[key]["position_px"]["x"]), "y0": float(load[key]["position_px"]["y"]),
                              "vx": v * math.cos(th), "vy": -v * math.sin(th)})
    else:
        ap.error("give --trajectory, or --annotation with --load")
    cases = set(args.cases.split(","))
    name = args.name or Path(args.image).stem
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    speeds = [float(s) for s in args.speeds.split(",")]
    timings = [tuple(float(v) for v in t.split(":")) for t in args.timing.split(",")]
    mid = speeds[len(speeds) // 2]
    base_lam = timings[0]

    # (sample id, direction, speed, (onset, crit), tag)
    samples = []
    if "force" in cases:
        samples += [(f"{name}__horizontal__x{s:g}__on{base_lam[0]:g}_cr{base_lam[1]:g}", False, s, base_lam, args.young_modulus, "force") for s in speeds]
        samples.append((f"{name}__vertical__x{mid:g}__on{base_lam[0]:g}_cr{base_lam[1]:g}", True, mid, base_lam, args.young_modulus, "force"))
    if "timing" in cases:
        for lam in timings:
            samples.append((f"{name}__horizontal__x{mid:g}__on{lam[0]:g}_cr{lam[1]:g}", False, mid, lam, args.young_modulus, "timing"))
    if "material" in cases and args.materials:
        for E in [float(e) for e in args.materials.split(",")]:
            mant, exp = f"{E:.0e}".split("e")
            samples.append((f"{name}__horizontal__x{mid:g}__E{mant}e{int(exp)}", False, mid, base_lam, E, "material"))

    rows, made, seen = [], {}, set()
    for sid, rot, speed, (onset, crit), E, tag in samples:
        if sid in seen:
            continue
        seen.add(sid)
        ctrl_key = (rot, speed)
        if ctrl_key not in made:  # control video depends only on direction and speed
            ctrl_stem = f"{name}__{'vertical' if rot else 'horizontal'}__x{speed:g}"
            paths = vg.create_control_video(str(args.image), instances(blobs, speed, rot, W, H), str(out),
                                            name=ctrl_stem, n_frames=args.frames, fps=args.fps, sigma=args.sigma)
            made[ctrl_key] = paths
        paths = made[ctrl_key]
        ref_rel = f"reference_video/{sid}.mp4"           # inference names outputs after the `video` stem
        shutil.copy2(paths["reference"], out / ref_rel)
        ctrl_rel = str(Path(paths["control"]).relative_to(out))
        rows.append({
            "video": ref_rel, "prompt": vg.PROMPT, "force_video": ctrl_rel,
            "stress_video": ctrl_rel, "flow_video": ctrl_rel, "damage_video": ctrl_rel, "strain_video": ctrl_rel,
            "young_modulus": E, "poisson_ratio": args.poisson_ratio,
            "lambda_onset": onset, "lambda_crit": crit, "category": args.category,
        })
        print(f"{tag:8s} {sid}")
    for p in made.values():
        Path(p["reference"]).unlink(missing_ok=True)
    with open(out / "metadata.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} samples -> {out}/metadata.csv")


if __name__ == "__main__":
    main()
