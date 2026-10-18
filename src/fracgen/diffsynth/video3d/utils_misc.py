import torch
from PIL import Image
from typing import List, Tuple
from jaxtyping import Float, Int32, UInt8, Bool
from torch import Tensor


def log_checkpoint_loading(ckpt_path, is_strict, state_dict, missing, unexpected, load_log_path):
    loaded_keys = list(state_dict)
    missing_keys = list(missing) if missing is not None else []
    unexpected_keys = list(unexpected) if unexpected is not None else []

    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print(f"[ckpt load] strict={is_strict} ## Missing_keys={len(missing_keys)} ## Unexpected_keys={len(unexpected_keys)}", flush=True)

    with open(load_log_path, "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Strict loading: {is_strict}\n\n")

        f.write(f"Loaded keys (from checkpoint state_dict): {len(loaded_keys)}\n")
        for k in sorted(loaded_keys):
            f.write(f"{k}\n")

        f.write("\n" + "=" * 80 + "\n\n")

        f.write(f"Missing keys (expected by model, absent in ckpt): {len(missing_keys)}\n")
        for k in sorted(missing_keys):
            f.write(f"{k}\n")

        f.write("\n" + "=" * 80 + "\n\n")

        f.write(f"Unexpected keys (present in ckpt, not used by model): {len(unexpected_keys)}\n")
        for k in sorted(unexpected_keys):
            f.write(f"{k}\n")
        
    print(f"[ckpt load] Loading report saved to: {load_log_path}", flush=True)
    


def stitch_frames_vertical(
    top_frames: List[Image.Image],
    bottom_frames: List[Image.Image],
    separator_px: int = 2,
    separator_color: Tuple[int, int, int] = (255, 255, 255),
) -> List[Image.Image]:
    if len(top_frames) == 0 or len(bottom_frames) == 0:
        return []
    n = min(len(top_frames), len(bottom_frames))
    stitched: List[Image.Image] = []
    for i in range(n):
        top = top_frames[i].convert("RGB")
        bottom = bottom_frames[i].convert("RGB")

        target_w = max(top.width, bottom.width)
        if top.width != target_w:
            top_h = max(1, int(round(top.height * (target_w / top.width))))
            top = top.resize((target_w, top_h), resample=Image.BILINEAR)
        if bottom.width != target_w:
            bottom_h = max(1, int(round(bottom.height * (target_w / bottom.width))))
            bottom = bottom.resize((target_w, bottom_h), resample=Image.BILINEAR)

        sep = max(0, int(separator_px))
        out_h = top.height + sep + bottom.height
        out = Image.new("RGB", (target_w, out_h), color=separator_color)
        out.paste(top, (0, 0))
        out.paste(bottom, (0, top.height + sep))
        stitched.append(out)
    return stitched


def vis_from_bf3hw(x_bf3hw: torch.Tensor, scale_b1111 = None, mask_bfhw = None) -> List:
    """
    Visualize a (B,F,3,H,W) tensor as RGB frames.
    - If `scale_b1111` is provided (e.g. pts3d_norm_scale / disp_norm_scale), values are assumed in [-1, 1]
        and mapped to [0,1] via (x/(2*scale)+0.5).
    - Otherwise, uses per-video max-abs for a similar symmetric mapping.
    """
    x = x_bf3hw
    if x.dim() != 5 or x.shape[2] != 3:
        raise ValueError(f"Expected (B,F,3,H,W), got {tuple(x.shape)}")

    # Apply validity mask (invalid pixels -> 0) before visualization.
    if mask_bfhw is not None:
        m: Bool[Tensor, "B F H W"] = mask_bfhw
        assert m.dim() == 4
        # (B,F,H,W) -> (B,F,1,H,W)
        m = m.unsqueeze(2)
        if m.shape[0] != x.shape[0] or m.shape[1] != x.shape[1] or m.shape[3] != x.shape[3] or m.shape[4] != x.shape[4]:
            raise ValueError(f"Mask shape {tuple(m.shape)} incompatible with x {tuple(x.shape)}")
        m = m.to(device=x.device, dtype=x.dtype)
    else:
        m = torch.ones((x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4]), device=x.device, dtype=x.dtype)

    if scale_b1111 is None:
        scale_b1111 = x.abs().amax(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
    else:
        scale_b1111 = scale_b1111.to(device=x.device, dtype=x.dtype).clamp_min(1e-6)
    x01 = (x / (2.0 * scale_b1111) + 0.5).clamp(0.0, 1.0) * m
    # convert_tensor_to_images expects (F,C,H,W) or similar; take batch 0
    return convert_tensor_to_images(x01[0])

def convert_tensor_to_images(imgs):
    # Supports (F, C, H, W), (F, H, W, C), or (F, H, W).
    if isinstance(imgs, torch.Tensor):
        frames: List[Image.Image] = []
        for frame in imgs:
            frame = frame.detach().cpu()

            # If float, assume [0,1] (common) and scale to [0,255]; otherwise clamp.
            if torch.is_floating_point(frame):
                if frame.max() <= 1.0:
                    frame = frame * 255.0
                frame = frame.clamp(0, 255).to(torch.uint8)
            else:
                frame = frame.to(torch.uint8)

            if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
                # CHW -> HWC
                frame = frame.permute(1, 2, 0).contiguous()
            elif frame.ndim == 2:
                # HW -> HWC (grayscale)
                frame = frame.unsqueeze(-1)

            arr = frame.numpy()
            img = Image.fromarray(arr)
            frames.append(img.convert("RGB"))

        return frames