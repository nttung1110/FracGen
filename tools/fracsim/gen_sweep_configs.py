#!/usr/bin/env python3
"""
Axis-centric fracture config sampler for PhysGaussian.

Strategy
--------
1. Define a set of *tearing axes* through the object centre at varying angles.
2. For each axis, place two boundary-condition particles **symmetrically** on
   opposite sides of the centre, at varying offsets (radii).
3. Choose velocities that are guaranteed to produce opposing tensile / shear
   stress along that axis.
4. Compute an axis-aligned `size` box for each grip point so that the MPM
   particle selection slab is thin along the tearing axis (to avoid the two
   grips overlapping) and wide in all perpendicular directions (to capture
   enough particles for a clean fracture).

Fracture guarantee
------------------
For every combo we assert  dot(v1_unit, v2_unit) < COS_THRESHOLD  so that
the two particles are always pulling/shearing *away* from each other.

Modes
-----
  pure_pull  – both particles move directly away along the axis          (dot ≈ -1)
  shear      – both particles slide perpendicular, opposite sides        (dot =  0)
  mixed_45   – 45° blend of pull + shear                                 (dot ≈ -0.5)
  mixed_135  – 135° blend (reversed shear component)                     (dot ≈ -0.5)

MPM coordinate convention
--------------------------
PhysGaussian works in 3-D with the following axis roles:

  * X axis  — depth      (out-of-plane; fixed for 2-D tearing)
  * Y axis  — horizontal (left/right in the scene)
  * Z axis  — vertical   (up/down in the scene)

The 2-D tearing plane is therefore the **YZ plane**.
All tearing geometry is computed in a 2-D (y, z) space and then lifted
into 3-D as (x=fixed, y, z).

axis_angle is measured from the Z (vertical) axis, rotating toward Y:
  * axis_angle =  0°  → pure vertical   pull (along Z) → size thin in Z
  * axis_angle = 90°  → pure horizontal pull (along Y) → size thin in Y
                         matches reference: size=[1.0, 0.05, 1.0], v=[0,±v,0]
  * axis_angle = 135° → diagonal

Grip-size logic
---------------
The `size` parameter in `enforce_particle_translation` is an axis-aligned
full-extent box in 3-D world space.  In the 2-D YZ tearing plane:

  * THIN dimension  – along the tearing axis direction.  Kept small
                      (= slab_thickness) so each grip selects only particles
                      on its own side of the fracture plane.
  * WIDE dimensions – perpendicular to the axis.  Kept large (= wide_extent)
                      to capture the full cross-section of the object.

For axis angle θ (measured from Z / vertical):

    size_x = wide                               (depth — always wide)
    size_y = |sin θ| * thin + |cos θ| * wide   (horizontal)
    size_z = |cos θ| * thin + |sin θ| * wide   (vertical)

Sanity checks:
  θ =  0°  (vertical pull)   → size_y = wide, size_z = thin  ✓
  θ = 90°  (horizontal pull) → size_y = thin, size_z = wide  ✓
                                matches reference: [1.0, 0.05, 1.0]
  θ = 45°                    → size_y = size_z = (thin+wide)/√2

Usage
-----
  python gen_fracture_configs.py              # uses defaults below
  python gen_fracture_configs.py --help       # not implemented; edit CONFIG block
"""

import os, json, copy, math, random
from itertools import product as iproduct
from typing import List, Tuple, Optional, Dict, Any

# ──────────────────────────────────────────────────────────────────
# Tiny math helpers
# ──────────────────────────────────────────────────────────────────

def _norm2(v: Tuple[float, float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2)

def _unit2(v: Tuple[float, float], eps: float = 1e-12) -> Optional[Tuple[float, float]]:
    n = _norm2(v)
    return None if n < eps else (v[0] / n, v[1] / n)

def _dot2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]

def _rotate2(v: Tuple[float, float], deg: float) -> Tuple[float, float]:
    """Rotate 2-D vector *v* by *deg* degrees counter-clockwise."""
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return (c * v[0] - s * v[1],
            s * v[0] + c * v[1])

def _frange(vmin: float, vmax: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    n = round((vmax - vmin) / step)
    return [vmin + i * step for i in range(n + 1)
            if vmin - 1e-9 <= vmin + i * step <= vmax + 1e-9]

# ──────────────────────────────────────────────────────────────────
# 2-D ↔ 3-D coordinate helpers  (YZ tearing plane)
# ──────────────────────────────────────────────────────────────────
# Internal 2-D representation: (a, b) where
#   a  ≡  Z component (vertical)
#   b  ≡  Y component (horizontal)
# axis_angle is measured from Z toward Y (standard math angle in ZY plane).
#
# Lifting to 3-D:  point_3d  = (cx,  cy + b,  cz + a)
#                  velocity_3d = (0,   vy=b,    vz=a )
# ──────────────────────────────────────────────────────────────────

def _2d_axis(angle_deg: float) -> Tuple[float, float]:
    """Unit 2-D axis vector (az, ay) for *angle_deg* measured from Z toward Y."""
    r = math.radians(angle_deg)
    az = math.cos(r)   # Z (vertical) component
    ay = math.sin(r)   # Y (horizontal) component
    return az, ay

def _lift_point(centre: Tuple[float, float, float],
                da: float, db: float) -> Tuple[float, float, float]:
    """Add 2-D offset (da=Δz, db=Δy) to 3-D centre."""
    cx, cy, cz = centre
    return (cx, cy + db, cz + da)

def _lift_velocity(va: float, vb: float) -> Tuple[float, float, float]:
    """Convert 2-D velocity (va=vz, vb=vy) to 3-D (0, vy, vz)."""
    return (0.0, vb, va)

# ──────────────────────────────────────────────────────────────────
# Point-pair generation  (axis-centric, YZ plane)
# ──────────────────────────────────────────────────────────────────

def axis_point_pairs(
    centre: Tuple[float, float, float],
    axis_angles_deg: List[float],
    radii: List[float],
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    For every (angle, radius) combination place two 3-D points symmetrically
    on opposite sides of *centre* along the 2-D axis defined by *angle*.

    The tearing plane is YZ (Y=horizontal, Z=vertical); X is fixed at centre[0].

    Parameters
    ----------
    centre          : (cx, cy, cz) — object centre in MPM 3-D space
    axis_angles_deg : angles (degrees from Z toward Y) defining tearing directions
    radii           : distances from centre to each grip point

    Returns
    -------
    List of (p1, p2) 3-D point pairs.  p1 is on the "negative-axis" side,
    p2 on the "positive-axis" side.
    """
    pairs = []
    for angle in axis_angles_deg:
        az, ay = _2d_axis(angle)
        for r in radii:
            p1 = _lift_point(centre, -r * az, -r * ay)   # negative side
            p2 = _lift_point(centre, +r * az, +r * ay)   # positive side
            pairs.append((p1, p2))
    return pairs

# ──────────────────────────────────────────────────────────────────
# Velocity computation  (guaranteed fracture)
# ──────────────────────────────────────────────────────────────────

# Minimum dot-product threshold to accept a combo as fracture-producing.
# pure_pull gives -1, shear gives 0, mixed gives ~-0.5.
# We only reject when there is *no* opposing component at all.
COS_THRESHOLD = -0.05   # must be strictly below 0 (i.e. opposing)

def compute_velocities(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    speed1: float,
    speed2: float,
    mode: str,
    perturb_deg: float = 0.0,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Return (v1_2d, v2_2d) for a fracture-producing boundary condition.

    Internal 2-D representation: (a, b) where a=Z (vertical), b=Y (horizontal).
    The tearing axis is derived from the p1→p2 offset in the YZ plane.
    Caller should lift to 3-D via _lift_velocity(va, vb) → (0, vy, vz).

    Velocities always satisfy dot(v1_unit, v2_unit) < 0.

    Perturbation sign convention:
      u1 rotated by -perturb_deg, u2 by +perturb_deg  (symmetric divergence)
    """
    # Extract 2-D offset in YZ plane: (Δz, Δy)
    dz = p2[2] - p1[2]   # vertical
    dy = p2[1] - p1[1]   # horizontal
    axis = _unit2((dz, dy))
    if axis is None:
        raise ValueError(f"Points too close to define an axis: {p1}, {p2}")
    az, ay = axis
    perp = (-ay, az)      # left-perpendicular in (z,y) space

    if mode == "pure_pull":
        u1 = (-az, -ay)
        u2 = ( az,  ay)

    elif mode == "shear":
        u1 = ( perp[0],  perp[1])
        u2 = (-perp[0], -perp[1])

    elif mode == "mixed_45":
        raw1 = (-az + perp[0], -ay + perp[1])
        raw2 = ( az - perp[0],  ay - perp[1])
        u1 = _unit2(raw1) or (-az, -ay)
        u2 = _unit2(raw2) or ( az,  ay)

    elif mode == "mixed_135":
        raw1 = (-az - perp[0], -ay - perp[1])
        raw2 = ( az + perp[0],  ay + perp[1])
        u1 = _unit2(raw1) or (-az, -ay)
        u2 = _unit2(raw2) or ( az,  ay)

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # Symmetric angular perturbation.
    # Convention: u1 rotated by -perturb_deg, u2 by +perturb_deg.
    # This keeps the midpoint direction fixed on the axis while spreading
    # the two grip directions symmetrically.
    if abs(perturb_deg) > 1e-9:
        u1 = _unit2(_rotate2(u1, -perturb_deg)) or u1
        u2 = _unit2(_rotate2(u2, +perturb_deg)) or u2

    v1 = (speed1 * u1[0], speed1 * u1[1])
    v2 = (speed2 * u2[0], speed2 * u2[1])
    return v1, v2


def _is_valid_fracture(u1, u2) -> bool:
    """Return True iff the two unit vectors have a sufficiently opposing component."""
    if u1 is None or u2 is None:
        return False
    return _dot2(u1, u2) < COS_THRESHOLD

# ──────────────────────────────────────────────────────────────────
# Grip-size computation
# ──────────────────────────────────────────────────────────────────

def compute_grip_size(
    axis_angle_deg: float,
    separation: float,
    slab_thickness: float = 0.05,
    wide_extent: float    = 1.0,
    gap_factor: float     = 0.4,
) -> Tuple[float, float, float]:
    """
    Return the axis-aligned 3-D size box (sx, sy, sz) for a grip point,
    guaranteed to avoid overlap with the opposing grip's slab.

    MPM convention:  X = depth (fixed),  Y = horizontal,  Z = vertical.
    Tearing plane is YZ.  axis_angle is measured from Z (vertical) toward Y.

    Physical model
    --------------
    Each grip should be:
      * thin along the tearing axis — stays on its own side of the fracture
        plane, not bleeding into the opposing grip's territory.
      * wide perpendicular to the axis — grabs the full cross-section.

    Because PhysGaussian uses an axis-aligned bounding box (AABB), the
    effective extent of a rotated (thin × wide) slab *along the axis* is:

        half_proj = thin/2 + (wide/2) * |sin(2θ)|

    At diagonal angles this is much larger than thin/2, causing the two
    slabs to overlap.  For example at θ=60°, wide=1.0, sep=0.2:
        half_proj ≈ 0.458  >>  sep/2 = 0.1   (4.6× overlap!)

    Fix: clamp wide_extent so that:
        half_proj ≤ (sep/2) × (1 − gap_factor)

    giving a guaranteed clear gap of gap_factor × sep between slabs.

        wide_eff = min(wide, (2 × max_half − thin) / |sin(2θ)|)

    For axis-aligned pulls (sin2θ ≈ 0) no clamping is needed.

    Axis-aligned sanity checks:
      θ =  0°  (vertical pull)   → sy = wide, sz = thin  ✓
      θ = 90°  (horizontal pull) → sy = thin, sz = wide  ✓
                                    matches reference: [1.0, 0.05, 1.0]

    Parameters
    ----------
    axis_angle_deg : angle of the tearing axis from Z (vertical), in degrees
    separation     : distance between the two grip points (MPM world units)
    slab_thickness : desired thin extent along the axis direction
    wide_extent    : desired wide extent perpendicular to the axis
    gap_factor     : fraction of separation kept as clear gap between slabs
                     (default 0.4 → 40% of separation stays particle-free)

    Returns
    -------
    (sx, sy, sz) — 3-D size to write into the config
    """
    rad  = math.radians(axis_angle_deg)
    c    = abs(math.cos(rad))          # Z projection of axis unit vector
    s    = abs(math.sin(rad))          # Y projection of axis unit vector
    sin2 = abs(math.sin(2.0 * rad))    # |sin(2θ)| — diagonal-leak factor

    # Maximum allowed AABB half-extent along the axis
    max_half = (separation / 2.0) * (1.0 - gap_factor)

    # Clamp wide so AABB projection stays within max_half
    # half_proj = thin/2 + (wide_eff/2)*sin2  ≤  max_half
    if sin2 > 1e-9:
        wide_eff = min(wide_extent,
                       (2.0 * max_half - slab_thickness) / sin2)
        wide_eff = max(wide_eff, slab_thickness)   # floor: never thinner than thin
    else:
        wide_eff = wide_extent    # axis-aligned: no diagonal leak, no clamping

    sx = wide_extent                               # X depth — always full wide
    sy = s * slab_thickness + c * wide_eff         # Y (horizontal) extent
    sz = c * slab_thickness + s * wide_eff         # Z (vertical) extent

    return (sx, sy, sz)

# ──────────────────────────────────────────────────────────────────
# Main sampler
# ──────────────────────────────────────────────────────────────────

def sample_fracture_combos(
    centre: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    # ── axis / geometry ──────────────────────────────────────────
    axis_angles_deg: List[float] = None,    # tearing-axis directions
    radii: List[float] = None,              # half-separation distances
    min_pair_dist: float = 0.1,             # minimum separation (2 * radius)
    # ── grip size ────────────────────────────────────────────────
    slab_thickness: float = 0.05,           # size along axis (thin dim)
    wide_extent: float    = 1.0,            # desired size perpendicular to axis
    gap_factor: float     = 0.4,            # min gap between slabs as fraction of separation
    # ── dynamics ─────────────────────────────────────────────────
    speed_min: float = 0.5,
    speed_max: float = 1.5,
    speed_step: float = 0.25,
    same_speed: bool = True,                # True → s1 == s2 (symmetric pull)
    modes: Tuple[str, ...] = ("pure_pull", "shear", "mixed_45", "mixed_135"),
    perturb_angles: Tuple[float, ...] = (-15.0, 0.0, 15.0),
    # ── misc ─────────────────────────────────────────────────────
    shuffle: bool = True,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Enumerate all valid fracture-producing (point_pair × mode × perturb × speed)
    combinations in the MPM YZ tearing plane (Y=horizontal, Z=vertical, X=depth).

    Every returned combo satisfies dot(v1_unit, v2_unit) < COS_THRESHOLD, and
    carries a pre-computed `grip_size` (sx, sy, sz) whose wide_extent is
    automatically clamped per-pair so that the two grip slabs never overlap,
    regardless of axis angle.

    Parameters
    ----------
    slab_thickness : thin extent of the slab along the tearing axis.
                     Default 0.05 matches the reference bread-tearing config.
    wide_extent    : desired perpendicular extent (will be clamped downward for
                     diagonal axes to prevent AABB overlap between the two grips).
    gap_factor     : fraction of the point separation kept as a clear gap between
                     the two grip slabs (default 0.4 = 40% of separation).
                     Increase for more separation, decrease to capture more particles.

    Returns
    -------
    List of combo dicts ready for ``generate_configs_from_combos``.
    Each dict contains: p1, p2, v1, v2, v1_unit, v2_unit, dot, dist,
    grip_size, mode, perturb_deg, speed1, speed2.
    """
    # Sensible defaults
    if axis_angles_deg is None:
        # 8 equally-spaced axis directions (0°–157.5°)
        axis_angles_deg = [i * 22.5 for i in range(8)]
    if radii is None:
        # Vary how far from centre each particle sits
        radii = [0.10, 0.15, 0.20, 0.25]

    rng = random.Random(seed)
    speeds = _frange(float(speed_min), float(speed_max), float(speed_step))

    pairs = axis_point_pairs(centre, axis_angles_deg, radii)

    # Filter pairs that are too close together — distance measured in YZ plane
    pairs = [
        (p1, p2) for p1, p2 in pairs
        if math.hypot(p2[2] - p1[2], p2[1] - p1[1]) >= float(min_pair_dist)
    ]

    if not pairs:
        raise RuntimeError(
            "No valid point pairs generated. "
            "Check radii vs min_pair_dist (need 2*radius >= min_pair_dist)."
        )

    # Warn if slab_thickness alone (ignoring wide_extent) already exceeds sep/2.
    # The wide_extent leakage is handled automatically by compute_grip_size clamping.
    min_sep = min(math.hypot(p2[2]-p1[2], p2[1]-p1[1]) for p1, p2 in pairs)
    if slab_thickness >= min_sep / 2.0:
        import warnings
        warnings.warn(
            f"slab_thickness={slab_thickness} >= half the minimum pair separation "
            f"({min_sep/2:.4f}). Even without wide-extent leakage the thin dimension "
            f"alone may overlap — consider reducing slab_thickness.",
            stacklevel=2,
        )

    # Build speed combinations
    if same_speed:
        speed_pairs = [(s, s) for s in speeds]
    else:
        speed_pairs = list(iproduct(speeds, speeds))

    combos = []
    skipped = 0

    for (p1, p2), mode, perturb, (s1, s2) in iproduct(
        pairs, modes, perturb_angles, speed_pairs
    ):
        # Recover axis angle from YZ offset: atan2(Δy, Δz)  (angle from Z toward Y)
        dz = p2[2] - p1[2]
        dy = p2[1] - p1[1]
        pair_angle_deg = math.degrees(math.atan2(dy, dz))

        try:
            v1_2d, v2_2d = compute_velocities(p1, p2, s1, s2, mode, perturb)
        except ValueError:
            skipped += 1
            continue

        u1 = _unit2(v1_2d)
        u2 = _unit2(v2_2d)

        if not _is_valid_fracture(u1, u2):
            skipped += 1
            continue

        dist = math.hypot(dz, dy)
        grip_size = compute_grip_size(pair_angle_deg, dist,
                                      slab_thickness, wide_extent, gap_factor)

        # Lift 2-D velocities to 3-D: (vz_component, vy_component) → (0, vy, vz)
        v1_3d = _lift_velocity(v1_2d[0], v1_2d[1])
        v2_3d = _lift_velocity(v2_2d[0], v2_2d[1])

        combos.append({
            "p1"          : p1,
            "p2"          : p2,
            "v1"          : v1_3d,
            "v2"          : v2_3d,
            "v1_unit"     : u1,
            "v2_unit"     : u2,
            "dot"         : _dot2(u1, u2),
            "dist"        : dist,            # separation in YZ plane
            "grip_size"   : grip_size,       # (sx, sy, sz) — overlap-free
            "mode"        : mode,
            "perturb_deg" : perturb,
            "speed1"      : s1,
            "speed2"      : s2,
        })

    if skipped:
        print(f"[sampler] Skipped {skipped} invalid combos (dot ≥ {COS_THRESHOLD}).")

    if shuffle:
        rng.shuffle(combos)

    return combos

# ──────────────────────────────────────────────────────────────────
# Config writer
# ──────────────────────────────────────────────────────────────────

def generate_configs_from_combos(
    template_path: str,
    out_dir: str,
    combos: List[Dict[str, Any]],
    filename_fmt: str = "config_{:04d}.json",
    frame_num: int = 100,
    max_configs: Optional[int] = None,
) -> None:
    """
    Write one PhysGaussian JSON config per combo.

    The template must contain exactly 2 entries of type
    ``"enforce_particle_translation"`` in ``boundary_conditions``.
    """
    with open(template_path, "r") as f:
        template_cfg = json.load(f)

    bc = template_cfg["boundary_conditions"]
    idx = [
        i for i, e in enumerate(bc)
        if e.get("type") == "enforce_particle_translation"
    ]
    if len(idx) != 2:
        raise ValueError(
            f"Expected exactly 2 'enforce_particle_translation' entries, "
            f"found {len(idx)}."
        )
    i0, i1 = idx

    os.makedirs(out_dir, exist_ok=True)

    write_combos = combos if max_configs is None else combos[:max_configs]

    for k, c in enumerate(write_combos):
        cfg = copy.deepcopy(template_cfg)
        cfg["frame_num"] = int(frame_num)

        cfg["boundary_conditions"][i0]["point"]    = list(map(float, c["p1"]))
        cfg["boundary_conditions"][i1]["point"]    = list(map(float, c["p2"]))
        cfg["boundary_conditions"][i0]["velocity"] = list(map(float, c["v1"]))
        cfg["boundary_conditions"][i1]["velocity"] = list(map(float, c["v2"]))
        # Axis-aware slab size: thin along the tearing axis, wide perpendicular.
        # Both grip points share the same axis so they get the same size box.
        cfg["boundary_conditions"][i0]["size"] = list(map(float, c["grip_size"]))
        cfg["boundary_conditions"][i1]["size"] = list(map(float, c["grip_size"]))

        fpath = os.path.join(out_dir, filename_fmt.format(k))
        with open(fpath, "w") as f:
            json.dump(cfg, f, indent=2)

    print(
        f"[writer] Wrote {len(write_combos)} config files → {out_dir}/"
        + (f"  (capped from {len(combos)})" if max_configs and max_configs < len(combos) else "")
    )

# ──────────────────────────────────────────────────────────────────
# Diagnostics / sanity check
# ──────────────────────────────────────────────────────────────────

def print_summary(combos: List[Dict[str, Any]]) -> None:
    if not combos:
        print("[summary] No combos generated.")
        return

    from collections import Counter
    mode_counts = Counter(c["mode"] for c in combos)
    dots        = [c["dot"] for c in combos if c["dot"] is not None]
    dists       = [c["dist"] for c in combos]

    print(f"\n{'─'*55}")
    print(f"  Total valid combos : {len(combos)}")
    print(f"  By mode            : {dict(mode_counts)}")
    print(f"  dot(v1,v2)  range  : [{min(dots):.3f}, {max(dots):.3f}]  "
          f"(all must be < {COS_THRESHOLD})")
    print(f"  pair dist   range  : [{min(dists):.3f}, {max(dists):.3f}]")
    print(f"  Example combo[0]   :")
    c = combos[0]
    print(f"    p1={c['p1']}, p2={c['p2']}")
    print(f"    v1={tuple(round(x,4) for x in c['v1'])}")
    print(f"    v2={tuple(round(x,4) for x in c['v2'])}")
    gs = c['grip_size']
    print(f"    grip_size=({gs[0]:.4f}, {gs[1]:.4f}, {gs[2]:.4f})  "
          f"[thin along axis, wide perpendicular]")
    print(f"    mode={c['mode']}, perturb={c['perturb_deg']}°, "
          f"speeds=({c['speed1']},{c['speed2']}), dot={c['dot']:.3f}")
    print(f"{'─'*55}\n")

    # Hard assertion: every combo must satisfy the fracture condition
    violations = [c for c in combos if c["dot"] is not None and c["dot"] >= COS_THRESHOLD]
    if violations:
        raise RuntimeError(
            f"BUG: {len(violations)} combos do NOT satisfy the fracture condition "
            f"dot < {COS_THRESHOLD}!"
        )
    print("  ✓  All combos satisfy the fracture guarantee.\n")

# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── USER CONFIG ──────────────────────────────────────────────
    TEMPLATE   = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src", "fracsim", "configs", "tear_bread_config.json")
    OUTDIR     = "./all_cfg_tear_bread"

    # MPM bounding-box centre (match your scene)
    MPM_CENTRE = (1.0, 1.0, 1.0)

    combos = sample_fracture_combos(
        centre          = MPM_CENTRE,

        # ── Axis sweep ──────────────────────────────────────────
        # 12 axes from 0° to 165° in 15° steps gives dense angular coverage.
        # Increase to 24 (7.5° steps) for even denser sampling.
        axis_angles_deg = [i * 15.0 for i in range(12)],

        # ── Radial offsets from centre ───────────────────────────
        # Vary how far the grip points sit from the object centre.
        # Values are in MPM simulation units; adjust to your object size.
        radii           = [0.10, 0.15, 0.20, 0.25],

        min_pair_dist   = 0.15,   # filter pairs closer than this

        # ── Grip size ────────────────────────────────────────────
        # slab_thickness: thin extent along the tearing axis.
        slab_thickness  = 0.05,
        # wide_extent: desired perpendicular extent (will be auto-clamped
        #   for diagonal axes to prevent the two slabs from overlapping).
        wide_extent     = 1.0,
        # gap_factor: fraction of the point separation kept as a clear
        #   particle-free gap between the two grip slabs (0.4 = 40%).
        gap_factor      = 0.4,

        # ── Speed sweep ──────────────────────────────────────────
        speed_min       = 0.5,
        speed_max       = 1.5,
        speed_step      = 0.25,
        same_speed      = True,   # symmetric: s1 == s2
                                  # set False to allow asymmetric pulling

        # ── Stress modes ─────────────────────────────────────────
        modes           = ("pure_pull", "shear", "mixed_45", "mixed_135"),

        # ── Angular perturbations ────────────────────────────────
        # Adds diversity without breaking the fracture condition.
        perturb_angles  = (-15.0, 0.0, 15.0),

        shuffle         = True,
        seed            = 42,
    )

    print_summary(combos)
    breakpoint()

    generate_configs_from_combos(
        template_path = TEMPLATE,
        out_dir       = OUTDIR,
        combos        = combos,
        frame_num     = 100,
        # max_configs = 500,   # uncomment to cap output
    )