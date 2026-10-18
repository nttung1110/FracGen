#!/usr/bin/env python3
"""
Generate PhysGaussian sweep configs from a template.

This version is intentionally conservative:
  - start from config/tear_bread_config_exp.json or another template
  - preserve every template field by default
  - sweep only:
      * top-level E
      * top-level nu
      * damage.lambda_onset / damage.lambda_crit
      * uniaxial stretch axis
  - use pure pull only: two enforce_particle_translation handles move in
    opposite directions along the chosen axis.

Edit the USER CONFIG block at the bottom, or override values from CLI.
"""

import argparse
import ast
import copy
import json
import math
import os
import random
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Vec3 = Tuple[float, float, float]


def _literal(value: str):
    return ast.literal_eval(value)


def _float_list(value: str) -> List[float]:
    parsed = _literal(value) if value.strip().startswith("[") else value.split(",")
    return [float(v) for v in parsed]


def _lambda_set(value: str) -> List[Tuple[float, float, int]]:
    parsed = _literal(value)
    triples = []
    for item in parsed:
        if len(item) != 3:
            raise ValueError(
                f"lambda_set entries must be (lambda_onset, lambda_crit, n_grid), got {item!r}"
            )
        onset, crit, n_grid = float(item[0]), float(item[1]), int(item[2])
        if crit <= onset:
            raise ValueError(f"lambda_crit must be > lambda_onset, got {(onset, crit)}")
        if n_grid < 1:
            raise ValueError(f"n_grid must be >= 1, got {n_grid}")
        triples.append((onset, crit, n_grid))
    return triples


def _norm_yz(v: Sequence[float]) -> float:
    return math.hypot(float(v[1]), float(v[2]))


def _axis_yz(angle_deg: float) -> Tuple[float, float]:
    """Return unit axis in the YZ plane as (ay, az).

    angle=0   -> +Z vertical pull
    angle=90  -> +Y horizontal pull, matching the bread template
    """
    r = math.radians(angle_deg)
    ay = math.sin(r)
    az = math.cos(r)
    return ay, az


def _point_from_axis(center: Vec3, radius: float, ay: float, az: float, sign: float) -> Vec3:
    cx, cy, cz = center
    return (cx, cy + sign * radius * ay, cz + sign * radius * az)


def _velocity_from_axis(speed: float, ay: float, az: float, sign: float) -> Vec3:
    return (0.0, sign * speed * ay, sign * speed * az)


def _find_pull_bcs(cfg: Dict[str, Any]) -> Tuple[int, int]:
    bcs = cfg.get("boundary_conditions", [])
    idx = [i for i, bc in enumerate(bcs) if bc.get("type") == "enforce_particle_translation"]
    if len(idx) != 2:
        raise ValueError(
            f"Expected exactly 2 enforce_particle_translation boundary conditions, got {len(idx)}"
        )
    return idx[0], idx[1]


def _infer_center_and_radius(cfg: Dict[str, Any], bc_i: int, bc_j: int) -> Tuple[Vec3, float]:
    p0 = cfg["boundary_conditions"][bc_i]["point"]
    p1 = cfg["boundary_conditions"][bc_j]["point"]
    center = (
        0.5 * (float(p0[0]) + float(p1[0])),
        0.5 * (float(p0[1]) + float(p1[1])),
        0.5 * (float(p0[2]) + float(p1[2])),
    )
    radius = 0.5 * math.hypot(float(p1[1]) - float(p0[1]), float(p1[2]) - float(p0[2]))
    if radius <= 0.0:
        raise ValueError("Could not infer a positive pull radius from the template boundary points.")
    return center, radius


def _infer_speed(cfg: Dict[str, Any], bc_i: int, bc_j: int) -> float:
    v0 = cfg["boundary_conditions"][bc_i]["velocity"]
    v1 = cfg["boundary_conditions"][bc_j]["velocity"]
    speed = 0.5 * (_norm_yz(v0) + _norm_yz(v1))
    if speed <= 0.0:
        raise ValueError("Could not infer a positive pull speed from the template velocities.")
    return speed


def _infer_grip_extents(cfg: Dict[str, Any], bc_i: int) -> Tuple[float, float, float]:
    size = cfg["boundary_conditions"][bc_i]["size"]
    sx = float(size[0])
    slab_thickness = min(float(size[1]), float(size[2]))
    wide_extent = max(float(size[1]), float(size[2]))
    return sx, slab_thickness, wide_extent


def compute_grip_size(
    angle_deg: float,
    separation: float,
    sx: float,
    slab_thickness: float,
    wide_extent: float,
    gap_factor: float,
) -> Vec3:
    """Axis-aware AABB size for the pull handle.

    The handle is thin along the chosen YZ axis and wide perpendicular to it.
    The diagonal clamp avoids the two AABBs overlapping too much for oblique axes.
    """
    r = math.radians(angle_deg)
    c = abs(math.cos(r))       # Z projection
    s = abs(math.sin(r))       # Y projection
    sin2 = abs(math.sin(2.0 * r))

    max_half = (separation / 2.0) * (1.0 - gap_factor)
    if sin2 > 1e-9:
        wide_eff = min(wide_extent, (2.0 * max_half - slab_thickness) / sin2)
        wide_eff = max(wide_eff, slab_thickness)
    else:
        wide_eff = wide_extent

    sy = s * slab_thickness + c * wide_eff
    sz = c * slab_thickness + s * wide_eff
    return (sx, sy, sz)


def make_config(
    template_cfg: Dict[str, Any],
    bc_i: int,
    bc_j: int,
    center: Vec3,
    radius: float,
    speed: float,
    axis_angle: float,
    grip_size: Vec3,
    young_modulus: float,
    poisson_ratio: float,
    lambda_triple: Tuple[float, float, int],
    frame_num: Optional[int],
    end_time: Optional[float] = None,
    behavior_name: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(template_cfg)
    ay, az = _axis_yz(axis_angle)
    lambda_onset, lambda_crit, n_grid = lambda_triple

    # Negative side moves negative along the axis; positive side moves positive.
    cfg["boundary_conditions"][bc_i]["point"] = list(_point_from_axis(center, radius, ay, az, -1.0))
    cfg["boundary_conditions"][bc_j]["point"] = list(_point_from_axis(center, radius, ay, az, +1.0))
    cfg["boundary_conditions"][bc_i]["velocity"] = list(_velocity_from_axis(speed, ay, az, -1.0))
    cfg["boundary_conditions"][bc_j]["velocity"] = list(_velocity_from_axis(speed, ay, az, +1.0))
    cfg["boundary_conditions"][bc_i]["size"] = list(grip_size)
    cfg["boundary_conditions"][bc_j]["size"] = list(grip_size)

    if end_time is not None:
        cfg["boundary_conditions"][bc_i]["end_time"] = float(end_time)
        cfg["boundary_conditions"][bc_j]["end_time"] = float(end_time)

    cfg["E"] = float(young_modulus)
    cfg["nu"] = float(poisson_ratio)
    cfg["n_grid"] = int(n_grid)
    cfg.setdefault("damage", {})
    cfg["damage"]["enabled"] = True
    cfg["damage"]["lambda_onset"] = float(lambda_onset)
    cfg["damage"]["lambda_crit"] = float(lambda_crit)

    if frame_num is not None:
        cfg["frame_num"] = int(frame_num)

    cfg["sweep_meta"] = {
        "mode": "pure_pull",
        "axis_angle_deg": float(axis_angle),
        "radius": float(radius),
        "speed": float(speed),
        "E": float(young_modulus),
        "nu": float(poisson_ratio),
        "lambda_onset": float(lambda_onset),
        "lambda_crit": float(lambda_crit),
        "n_grid": int(n_grid),
        "end_time": float(end_time) if end_time is not None else None,
        "behavior": behavior_name,
    }
    return cfg


def generate_configs(
    template_path: str,
    out_dir: str,
    young_moduli: Iterable[float],
    poisson_ratios: Iterable[float],
    axis_angles_deg: Iterable[float],
    # flat mode (bread-style): all three swept independently
    lambda_set: Optional[Iterable[Tuple[float, float, int]]] = None,
    velocities: Optional[Iterable[float]] = None,
    end_times: Optional[Iterable[float]] = None,
    # grouped mode (rubber-style): (lambda_set, velocities, end_times) tied per behavior
    behaviors: Optional[List[Dict[str, Any]]] = None,
    frame_num: Optional[int] = None,
    center: Optional[Vec3] = None,
    radius: Optional[float] = None,
    speed: Optional[float] = None,
    slab_thickness: Optional[float] = None,
    wide_extent: Optional[float] = None,
    grip_depth: Optional[float] = None,
    gap_factor: float = 0.4,
    filename_prefix: str = "bread",
    shuffle: bool = True,
    seed: int = 42,
) -> int:
    with open(template_path, "r") as f:
        template_cfg = json.load(f)

    bc_i, bc_j = _find_pull_bcs(template_cfg)
    inferred_center, inferred_radius = _infer_center_and_radius(template_cfg, bc_i, bc_j)
    inferred_speed = _infer_speed(template_cfg, bc_i, bc_j)
    inferred_sx, inferred_thin, inferred_wide = _infer_grip_extents(template_cfg, bc_i)

    center = inferred_center if center is None else center
    radius = inferred_radius if radius is None else float(radius)
    speed = inferred_speed if speed is None else float(speed)
    slab_thickness = inferred_thin if slab_thickness is None else float(slab_thickness)
    wide_extent = inferred_wide if wide_extent is None else float(wide_extent)
    grip_depth = inferred_sx if grip_depth is None else float(grip_depth)

    os.makedirs(out_dir, exist_ok=True)

    if behaviors is not None:
        # Grouped mode: expand each behavior group into (speed, lambda_triple, end_time, name)
        # tuples, then cross with axes × E × nu.
        behavior_combos: List[Tuple] = []
        for group in behaviors:
            for combo in product(group["velocities"], group["lambda_set"], group["end_times"]):
                behavior_combos.append((*combo, group["name"]))
        raw_combos = list(product(axis_angles_deg, young_moduli, poisson_ratios, behavior_combos))
        # Flatten to uniform 7-tuple: (axis_angle, speed, E, nu, lambdas, end_time, behavior_name)
        combos = [
            (ax, bc[0], E, nu, bc[1], float(bc[2]), bc[3])
            for ax, E, nu, bc in raw_combos
        ]
        behavior_names = [g["name"] for g in behaviors]
        print(f"[sweep] behaviors={behavior_names}")
    else:
        # Flat mode: independent sweep of lambda_set × velocities × end_times
        velocities_list = [speed] if velocities is None else [float(v) for v in velocities]
        end_times_list = [None] if end_times is None else [float(t) for t in end_times]
        lambda_list = list(lambda_set) if lambda_set is not None else []
        raw_combos = list(product(axis_angles_deg, velocities_list, young_moduli, poisson_ratios, lambda_list, end_times_list))
        combos = [(ax, v, E, nu, lam, et, None) for ax, v, E, nu, lam, et in raw_combos]
        print(f"[sweep] velocities={velocities_list}")
        print(f"[sweep] lambda_set={lambda_list}")
        print(f"[sweep] end_times={end_times_list}")

    if shuffle:
        random.Random(seed).shuffle(combos)

    for idx, (axis_angle, combo_speed, E, nu, lambdas, end_time, behavior_name) in enumerate(combos):
        grip_size = compute_grip_size(
            float(axis_angle),
            separation=2.0 * radius,
            sx=grip_depth,
            slab_thickness=slab_thickness,
            wide_extent=wide_extent,
            gap_factor=gap_factor,
        )
        cfg = make_config(
            template_cfg=template_cfg,
            bc_i=bc_i,
            bc_j=bc_j,
            center=center,
            radius=radius,
            speed=float(combo_speed),
            axis_angle=float(axis_angle),
            grip_size=grip_size,
            young_modulus=float(E),
            poisson_ratio=float(nu),
            lambda_triple=(float(lambdas[0]), float(lambdas[1]), int(lambdas[2])),
            frame_num=frame_num,
            end_time=end_time,
            behavior_name=behavior_name,
        )
        safe_fname = f"{filename_prefix}_config_{idx:04d}.json"
        with open(os.path.join(out_dir, safe_fname), "w") as f:
            json.dump(cfg, f, indent=2)

    print(f"[writer] Wrote {len(combos)} pure-pull configs to {out_dir}")
    print(f"[template] {template_path}")
    print(f"[sweep] axes={list(axis_angles_deg)}")
    print(f"[sweep] E={list(young_moduli)}")
    print(f"[sweep] nu={list(poisson_ratios)}")
    print(f"[inferred/default] center={center}, radius={radius}, speed={speed}")
    return len(combos)


# ──────────────────────────────────────────────────────────────────
# USER CONFIG
# ──────────────────────────────────────────────────────────────────

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CFG = os.path.join(_REPO, "src", "fracsim", "configs")

# ── Global settings (apply to every object) ───────────────────────
OUTDIR          = os.path.join(_REPO, "data", "fracsim", "sweep_configs")
FRAME_NUM       = 100
SHUFFLE         = True
SEED            = 42
GAP_FACTOR      = 0.1

# Axis is measured in the YZ plane:
#   0 deg  = vertical Z pull
#   90 deg = horizontal Y pull, matching the bread template
AXIS_ANGLES_DEG = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]

# ── Material parameter groups ──────────────────────────────────────
# Reuse these dicts inside OBJECT_CONFIGS to avoid repetition.
# Each lambda_set entry: (lambda_onset, lambda_crit, n_grid)
#   n_grid = top-level simulation grid resolution (NOT particle_filling.n_grid)

_BREAD_PARAMS = dict(
    young_moduli   = [2e3, 6e3, 1e4],
    poisson_ratios = [0.2, 0.25, 0.3],
    lambda_set     = [(2.0, 3.0, 150)],
    velocities     = [0.5, 0.6, 0.7, 0.8],
)

_RUBBER_BEHAVIORS = [
    # Force stops early — object stores energy and snaps back without fracturing.
    # High lambda_onset ensures no damage accumulates during the brief pull.
    dict(
        name="strong_snap_back",
        lambda_set=[(3.0, 3.3, 60)],
        velocities=[0.5, 0.7],
        end_times=[0.6, 0.8],
    ),
    # Milder snap-back: lower velocity and earlier stop, object barely approaches onset.
    dict(
        name="light_snap_back",
        lambda_set=[(2.6, 3.0, 80)],
        velocities=[0.3, 0.5],
        end_times=[0.4, 0.6],
    ),
    # Force runs until end of sim; object fractures late due to high lambda threshold.
    dict(
        name="late_frac",
        lambda_set=[(2.6, 3.0, 80), (3.0, 3.3, 60)],
        velocities=[1.2, 1.4],
        end_times=[3],
    ),
    # Force runs until end of sim; low lambda onset causes early fracture.
    dict(
        name="early_frac",
        lambda_set=[(1.5, 2.3, 120)],
        velocities=[0.6, 0.8],
        end_times=[3],
    ),
]

_RUBBER_PARAMS = dict(
    young_moduli   = [2e4, 6e4, 1e5],
    poisson_ratios = [0.45, 0.47, 0.49],
    behaviors      = _RUBBER_BEHAVIORS,
)

# ── Per-object configurations ──────────────────────────────────────
# Add or remove entries here; no commenting / uncommenting required.
# Geometry overrides (center, radius, slab_thickness, wide_extent,
# grip_depth) default to None, which means "infer from template".
OBJECT_CONFIGS: List[Dict[str, Any]] = [
    # ── Bread-like ────────────────────────────────────────────────
    dict(template=f"{_CFG}/tear_bread_config_exp.json",      prefix="bread",      **_BREAD_PARAMS),
    dict(template=f"{_CFG}/tear_cake_config_exp.json",       prefix="cake",       **_BREAD_PARAMS),
    dict(template=f"{_CFG}/tear_croissant_config_exp.json",  prefix="croissant",  **_BREAD_PARAMS),
    dict(template=f"{_CFG}/tear_bread_roll_config_exp.json", prefix="bread_roll", **_BREAD_PARAMS),
    dict(template=f"{_CFG}/tear_muffin_config_exp.json",     prefix="muffin",     **_BREAD_PARAMS),
    # ── Rubber-like ───────────────────────────────────────────────
    dict(template=f"{_CFG}/tear_rubber_toy_config_exp.json", prefix="rubber_toy", **_RUBBER_PARAMS),
    dict(template=f"{_CFG}/tear_rubber1_config_exp.json",    prefix="rubber1",    **_RUBBER_PARAMS),
    dict(template=f"{_CFG}/tear_rubber2_config_exp.json",    prefix="rubber2",    **_RUBBER_PARAMS),
    dict(template=f"{_CFG}/tear_rubber3_config_exp.json",    prefix="rubber3",    **_RUBBER_PARAMS),
    dict(template=f"{_CFG}/tear_rubber4_config_exp.json",    prefix="rubber4",    **_RUBBER_PARAMS),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PhysGaussian sweep configs for all objects in one run."
    )
    # Global overrides — per-object material params live in OBJECT_CONFIGS above.
    parser.add_argument("--out_dir",     default=OUTDIR,
                        help="Output directory for all generated configs.")
    parser.add_argument("--frame_num",   type=int, default=FRAME_NUM)
    parser.add_argument("--axis_angles", type=_float_list, default=AXIS_ANGLES_DEG)
    parser.add_argument("--gap_factor",  type=float, default=GAP_FACTOR)
    parser.add_argument("--seed",        type=int, default=SEED)
    parser.add_argument("--no_shuffle",  action="store_true")
    parser.add_argument(
        "--objects",
        default=None,
        help="Comma-separated list of prefixes to process (e.g. bread,muffin). "
             "Omit to run all objects defined in OBJECT_CONFIGS.",
    )
    args = parser.parse_args()

    wanted = (
        set(p.strip() for p in args.objects.split(",")) if args.objects else None
    )

    total = 0
    skipped = 0
    for obj in OBJECT_CONFIGS:
        prefix = obj["prefix"]
        if wanted is not None and prefix not in wanted:
            skipped += 1
            continue

        template = obj["template"]
        if not os.path.exists(template):
            print(f"[skip] template not found for {prefix!r}: {template}")
            skipped += 1
            continue

        print(f"\n{'═'*60}")
        print(f"  Object : {prefix}")
        print(f"  Template: {template}")
        print(f"{'═'*60}")
        count = generate_configs(
            template_path  = template,
            out_dir        = args.out_dir,
            young_moduli   = obj["young_moduli"],
            poisson_ratios = obj["poisson_ratios"],
            axis_angles_deg= args.axis_angles,
            # grouped mode when behaviors key present, flat mode otherwise
            behaviors      = obj.get("behaviors"),
            lambda_set     = obj.get("lambda_set"),
            velocities     = obj.get("velocities"),
            end_times      = obj.get("end_times"),
            frame_num      = args.frame_num,
            center         = obj.get("center"),
            radius         = obj.get("radius"),
            slab_thickness = obj.get("slab_thickness"),
            wide_extent    = obj.get("wide_extent"),
            grip_depth     = obj.get("grip_depth"),
            gap_factor     = args.gap_factor,
            filename_prefix= prefix,
            shuffle        = not args.no_shuffle,
            seed           = args.seed,
        )
        total += count

    processed = len(OBJECT_CONFIGS) - skipped
    print(f"\n{'═'*60}")
    print(f"Done. {processed} object(s) processed, {total} configs written to {args.out_dir}")
    if skipped:
        print(f"      {skipped} object(s) skipped (missing template or filtered out).")


if __name__ == "__main__":
    main()
