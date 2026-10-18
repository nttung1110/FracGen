"""
# LPIPS on VAE-decoded RGB frames — backpropagates through the VAE decoder.
# Memory knobs (tune to your GPU budget):
#   tiled=True        → spatial tiling; cap per-tile VRAM at the cost of more decode calls
#   tile_stride=(h,w) → smaller stride → smoother seams, more tiles, more compute
#   use_checkpoint    → recompute VAE activations on backward (saves ~40 % activation mem)
#   lpips_chunk_size  → frames evaluated by LPIPS at once (smaller = less LPIPS mem)
#   with_grad=False   → fully detach (monitoring only, cheapest option)

Example:
pred_rgb = clean_latents[:, :16]
gt_rgb   = inputs["input_latents"][:, :16]
lpips_loss = compute_lpips_on_decoded_frames(
    pipe, pred_rgb, gt_rgb,
    with_grad=True,
    tiled=True,
    tile_size=(34, 34),
    tile_stride=(18, 16),
    use_checkpoint=False,
    lpips_chunk_size=4,
)
"""
from torch import Tensor
from typing import Tuple, List
import torch
import torch.nn.functional as F


_lpips_fn = None

def _get_lpips_fn(device):
    """Return a frozen LPIPS (AlexNet) model, lazily created on *device*."""
    global _lpips_fn
    if _lpips_fn is None:
        from tool_scripts.FVD.lpips import lpips
        _lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
        for p in _lpips_fn.parameters():
            p.requires_grad_(False)
    elif next(_lpips_fn.parameters()).device != device:
        _lpips_fn = _lpips_fn.to(device)
    return _lpips_fn


def _decode_for_grad(vae, latent_1b: Tensor, device, use_checkpoint: bool = False) -> Tensor:
    """Decode a single-item latent ``[1, C, T, H, W]`` while preserving gradients.

    The standard ``vae.decode()`` path breaks the autograd graph in two ways:
    (1) it moves latents to CPU before any computation, and (2) it uses the
    in-place ``clamp_``. This helper calls ``vae.model.decode`` directly,
    stays on *device*, and uses the out-of-place ``clamp``.

    Args:
        use_checkpoint: Gradient-checkpoint the VAE decoder to trade ~30 % extra
                        compute for a large reduction in activation memory.
    Returns:
        Decoded video ``[1, 3, T_out, H_out, W_out]`` in ``[-1, 1]``, with
        gradients flowing back to ``latent_1b``.
    """
    latent_1b = latent_1b.to(device)

    if use_checkpoint:
        from torch.utils.checkpoint import checkpoint as ck
        video = ck(lambda x: vae.model.decode(x, vae.scale), latent_1b, use_reentrant=False)
    else:
        video = vae.model.decode(latent_1b, vae.scale)

    return video.clamp(-1, 1)  # non-in-place: keeps autograd graph intact


def _decode_tiled_grad(
    vae, latent_1b: Tensor, device,
    tile_size: Tuple[int, int] = (34, 34),
    tile_stride: Tuple[int, int] = (18, 16),
    use_checkpoint: bool = False,
) -> Tensor:
    """Tiled spatial decode that **preserves gradients**.

    Splits the spatial latent into overlapping tiles, decodes each tile
    independently, then reassembles via **pad-and-add** (fully differentiable).
    This limits peak activation memory to one tile at a time while keeping
    the gradient path intact.

    The tradeoff vs. ``_decode_for_grad``:
    - *Less* peak VRAM during the forward decode.
    - Each padding step creates a full-resolution float tensor (``[1, 3, T, H, W]``);
      total memory of the reassembly step is O(n_tiles × full_frame_size).
    - Combine with ``use_checkpoint=True`` or AMP to further reduce memory.

    Args:
        tile_size:   ``(H_tile, W_tile)`` in *latent* coordinates.
        tile_stride: ``(H_stride, W_stride)`` in *latent* coordinates.
                     Smaller stride → more overlap → smoother seams.
        use_checkpoint: Gradient-checkpoint each tile's decode.
    """
    latent_1b = latent_1b.to(device)
    _, _, T, H, W = latent_1b.shape
    size_h, size_w = tile_size
    stride_h, stride_w = tile_stride
    up = vae.upsampling_factor
    out_T = T * 4 - 3  # Wan VAE temporal factor

    H_out, W_out = H * up, W * up

    if use_checkpoint:
        from torch.utils.checkpoint import checkpoint as ck

        def _ck_inner(z):
            return vae.model.decode(z, vae.scale)

        def run_decode(x):
            return ck(_ck_inner, x, use_reentrant=False)
    else:
        def run_decode(x):
            return vae.model.decode(x, vae.scale)

    tasks: List[Tuple[int, int, int, int]] = []
    for h in range(0, H, stride_h):
        if h - stride_h >= 0 and h - stride_h + size_h >= H:
            continue
        for w in range(0, W, stride_w):
            if w - stride_w >= 0 and w - stride_w + size_w >= W:
                continue
            tasks.append((h, min(h + size_h, H), w, min(w + size_w, W)))

    # Weight accumulator (no grad needed)
    weight = torch.zeros(1, 1, out_T, H_out, W_out, dtype=latent_1b.dtype, device=device)

    # Reassemble via out-of-place addition so the gradient graph is intact.
    # Each iteration creates a full-res padded tensor and adds it to `values`;
    # the previous `values` tensor is freed once the += completes.
    values = torch.zeros(1, 3, out_T, H_out, W_out, dtype=latent_1b.dtype, device=device)

    for h, h_, w, w_ in tasks:
        tile = latent_1b[:, :, :, h:h_, w:w_]
        decoded = run_decode(tile)                          # [1, 3, out_T, th, tw]
        th, tw = decoded.shape[3], decoded.shape[4]

        mask = vae.build_mask(
            decoded,
            is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
            border_width=((size_h - stride_h) * up, (size_w - stride_w) * up),
        ).to(dtype=latent_1b.dtype, device=device)         # [1, 1, out_T, th, tw]

        tgt_h, tgt_w = h * up, w * up

        # F.pad order for 5-D: (W_left, W_right, H_top, H_bottom)
        pad = (tgt_w, W_out - tgt_w - tw, tgt_h, H_out - tgt_h - th)
        values = values + F.pad(decoded * mask, pad)       # differentiable
        weight += F.pad(mask.detach(), pad)                # weight needs no grad

    return (values / weight.clamp(min=1e-6)).clamp(-1, 1)


def compute_lpips_on_decoded_frames(
    pipe, pred_latents_rgb: Tensor, gt_latents_rgb: Tensor,
    lpips_chunk_size: int = 4,
    with_grad: bool = True,
    tiled: bool = False,
    tile_size: Tuple[int, int] = (34, 34),
    tile_stride: Tuple[int, int] = (18, 16),
    use_checkpoint: bool = False,
) -> Tensor:
    """Decode RGB latents and return mean frame-level LPIPS.

    Memory-efficient design
    -----------------------
    * **Per-item decoding**: batch items are decoded one at a time (saves B×
      memory vs. decoding the full batch).
    * **GT decoded with no_grad**: ground-truth does not need a gradient, so
      its decode is wrapped in ``torch.no_grad()``.
    * **Temporal chunking**: LPIPS is evaluated on ``lpips_chunk_size`` frames
      at a time to limit the memory held by the perception network.
    * **Spatial tiling** (``tiled=True``): each tile's activation peak is
      bounded by the tile size; decoded tiles are assembled via pad-and-add.
    * **Gradient checkpointing** (``use_checkpoint=True``): recomputes VAE
      activations on the backward pass instead of storing them.
    * **no-grad mode** (``with_grad=False``): completely detaches the result;
      useful for monitoring/logging with minimal overhead.

    Args:
        pred_latents_rgb: ``[B, 16, T, H, W]`` predicted latents (in graph).
        gt_latents_rgb:   ``[B, 16, T, H, W]`` ground-truth latents.
        lpips_chunk_size: Frames evaluated by LPIPS in one forward pass.
        with_grad:        Keep the result in the computation graph so it can
                          be back-propagated. Set ``False`` for logging only.
        tiled:            Use spatial tiling to cap per-call VRAM usage.
        tile_size:        ``(H, W)`` of each tile in latent space.
        tile_stride:      ``(H, W)`` stride between tiles (smaller = smoother
                          seams, more tiles, more compute).
        use_checkpoint:   Gradient-checkpoint the VAE decoder forward pass.

    Returns:
        Scalar LPIPS loss. Detached when ``with_grad=False``.
    """
    device = pred_latents_rgb.device
    B = pred_latents_rgb.shape[0]

    pipe.vae.model.to(device)
    lpips_fn = _get_lpips_fn(device)

    total_lpips: Tensor = pred_latents_rgb.new_zeros(())
    total_frames: int = 0

    for b in range(B):
        pred_1b = pred_latents_rgb[b : b + 1]  # [1, 16, T, H, W]
        gt_1b   = gt_latents_rgb[b : b + 1]

        if with_grad:
            if tiled:
                decoded_pred = _decode_tiled_grad(pipe.vae, pred_1b, device, tile_size, tile_stride, use_checkpoint)
            else:
                decoded_pred = _decode_for_grad(pipe.vae, pred_1b, device, use_checkpoint)
            # GT never needs a gradient — decode cheaply
            with torch.no_grad():
                if tiled:
                    decoded_gt = _decode_tiled_grad(pipe.vae, gt_1b, device, tile_size, tile_stride, False)
                else:
                    decoded_gt = _decode_for_grad(pipe.vae, gt_1b, device, False)
        else:
            with torch.no_grad():
                if tiled:
                    decoded_pred = _decode_tiled_grad(pipe.vae, pred_1b, device, tile_size, tile_stride, False)
                    decoded_gt   = _decode_tiled_grad(pipe.vae, gt_1b,   device, tile_size, tile_stride, False)
                else:
                    decoded_pred = _decode_for_grad(pipe.vae, pred_1b, device, False)
                    decoded_gt   = _decode_for_grad(pipe.vae, gt_1b,   device, False)

        _, C, T_out, H_out, W_out = decoded_pred.shape

        # Temporal chunk loop keeps LPIPS network memory bounded
        for t in range(0, T_out, lpips_chunk_size):
            p = decoded_pred[0, :, t : t + lpips_chunk_size].permute(1, 0, 2, 3)  # [k, 3, H, W]
            g = decoded_gt[0,   :, t : t + lpips_chunk_size].permute(1, 0, 2, 3)
            k = p.shape[0]
            total_lpips = total_lpips + lpips_fn(p.float(), g.float()).sum()
            total_frames += k

    mean_lpips = total_lpips / max(total_frames, 1)
    return mean_lpips if with_grad else mean_lpips.detach()
