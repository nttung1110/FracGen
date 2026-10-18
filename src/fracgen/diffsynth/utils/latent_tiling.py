"""
Canonical tile/untile helpers for combining RGB + physics-map VAE latents into one
spatially-taller latent (concatenated along the H axis), instead of the existing
channel-concat + auxiliary-head mechanism (with_phys_map, see diffsynth/diffusion/loss.py
and diffsynth/models/wan_video_dit.py's heads_aux).

All latents follow this codebase's (B, C, T, H, W) convention -- H is dim=3.
"""
from typing import Dict, List

import torch
from torch import Tensor

# Canonical physics-modality order -- single source of truth, imported by
# diffsynth/diffusion/loss.py instead of being redefined there. Training stacks tiles in
# this order; eval-side decode must chunk and reassign in this same order.
PHYS_KEYS = ["stress_video", "strain_video", "damage_video", "flow_video"]

# RGB first, then PHYS_KEYS.
TILE_ORDER = ["video"] + PHYS_KEYS


def tile_latents(latents_list: List[Tensor], dim: int = 3) -> Tensor:
    """Concatenate K latents of identical shape (except possibly dim) along `dim`.

    Asserts all tensors share B, C, T, W (every dim except `dim`), dtype, and device --
    no implicit casting, so shape/dtype bugs surface immediately rather than silently.
    """
    assert len(latents_list) > 0, "tile_latents requires at least one tensor"
    ref = latents_list[0]
    for i, t in enumerate(latents_list):
        assert t.dim() == ref.dim(), f"tensor {i} has {t.dim()} dims, expected {ref.dim()}"
        for d in range(t.dim()):
            if d == dim:
                continue
            assert t.shape[d] == ref.shape[d], (
                f"tensor {i} shape {tuple(t.shape)} mismatches tensor 0 shape "
                f"{tuple(ref.shape)} at dim {d} (only dim={dim} may differ)"
            )
        assert t.dtype == ref.dtype, f"tensor {i} dtype {t.dtype} != tensor 0 dtype {ref.dtype}"
        assert t.device == ref.device, f"tensor {i} device {t.device} != tensor 0 device {ref.device}"
    return torch.cat(latents_list, dim=dim)


def untile_latents(tiled: Tensor, num_tiles: int, dim: int = 3) -> List[Tensor]:
    """Exact inverse of tile_latents: split `tiled` back into `num_tiles` equal chunks
    along `dim`. Pure cat/chunk -- no precision loss, untile_latents(tile_latents(xs),
    len(xs)) must reproduce xs exactly.
    """
    assert tiled.shape[dim] % num_tiles == 0, (
        f"dim {dim} size {tiled.shape[dim]} not divisible by num_tiles={num_tiles}"
    )
    return list(torch.chunk(tiled, num_tiles, dim=dim))


def slice_tile(x: Tensor, key: str, tile_order: List[str], H_tile: int) -> Tensor:
    """Slice one modality tile out of a height-stacked tiled latent (B, C, T, K*H', W').

    `tile_order` is the ordered list of modality names used at tiling time (e.g. TILE_ORDER).
    `H_tile` is the single-tile spatial height (= H_total // len(tile_order)).
    """
    idx = tile_order.index(key)
    return x[:, :, :, idx * H_tile:(idx + 1) * H_tile, :]


def tile_latents_named(
    latents_dict: Dict[str, Tensor],
    order: List[str] = TILE_ORDER,
    dim: int = 3,
) -> Tensor:
    """Convenience wrapper bridging source_latents_dict/input_latents naming (as used in
    wan_video.py/loss.py) to the positional tile_latents above.
    """
    missing = [name for name in order if name not in latents_dict]
    assert not missing, f"tile_latents_named: missing keys in latents_dict: {missing}"
    return tile_latents([latents_dict[name] for name in order], dim=dim)
