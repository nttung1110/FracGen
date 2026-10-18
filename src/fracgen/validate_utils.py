from __future__ import annotations
import argparse
from typing import Optional, Tuple, List
from PIL import Image
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
import torch
import numpy as np

MODALITY_KEYS = ["rgb", "flow", "damage", "strain", "stress"]


def split_latents_by_stream(latents_bcfhw: torch.Tensor, z_dim: int):
    """
    Split packed latents into streams based on channel count.
    Supported layouts:
    - (B, z,   F, H, W): rgb-only (standard WanModel)
    - (B, 3*z, F, H, W): rgb + pts3d + displacement (JointGen)
    """
    assert latents_bcfhw.dim() == 5, f"Expected latents with shape (B,C,F,H,W), got {tuple(latents_bcfhw.shape)}"
    c = int(latents_bcfhw.shape[1])
    if c == z_dim:
        return {"rgb": latents_bcfhw, "pts": None, "disp": None}
    elif c == 3 * z_dim:
        return {
            "rgb":    latents_bcfhw[:, 0 : z_dim],
            "stress": None,
            "flow":   latents_bcfhw[:, z_dim : 2 * z_dim],
            "damage": latents_bcfhw[:, 2 * z_dim : 3 * z_dim],
        }
    elif c == len(MODALITY_KEYS) * z_dim:
        return {
            key: latents_bcfhw[:, i * z_dim:(i + 1) * z_dim]
            for i, key in enumerate(MODALITY_KEYS)
        }
    elif c == 2 * z_dim:
        return {
            "rgb":    latents_bcfhw[:, 0:z_dim],
            "stress": None,
            "flow":   latents_bcfhw[:, z_dim:2 * z_dim],
            "damage": None,
        }
    else:
        raise ValueError(f"Unsupported latent channel count: C={c}. Expected C==z_dim ({z_dim}) or C==3*z_dim ({3*z_dim}).")


def parse_hw_csv(s: str) -> Tuple[int, int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected H,W (e.g. 480,480), got: {s!r}")
    return int(parts[0]), int(parts[1])


def infer_ckpt_type(ckpt_path: Optional[str], ckpt_type: str = "auto") -> str:
    ckpt_type = (ckpt_type or "auto").lower()
    if ckpt_type != "auto":
        return ckpt_type
    # Decide from the tensor names inside the file, not from the path.
    if ckpt_path and ckpt_path.endswith(".safetensors"):
        try:
            from safetensors import safe_open
            with safe_open(ckpt_path, framework="pt") as f:
                if any("lora_A" in k or "lora_B" in k for k in f.keys()):
                    return "lora"
            return "full"
        except Exception:
            pass
    p = (ckpt_path or "").lower()
    if "lora" in p or "train_lora" in p:
        assert "sft" not in p, "SFT ckpt should not be a LoRA ckpt"
        return "lora"
    return "full"


def load_diff_pipe(
    model_id: str,
    device: Optional[object] = "auto",
    torch_dtype: Optional[object] = "bfloat16",
    redirect_common_files: bool = False,
    ckpt_path: Optional[str] = None,
    strict: bool = False,
    scheduler_cls=None,
    ckpt_type: str = "auto",
    lora_alpha: float = 1.0,
):
    """
    Shared pipeline + checkpoint loader used by validation scripts.

    Returns:
        pipe, (missing_keys, unexpected_keys, loaded_state_dict_keys)
    """
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id=model_id, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
        redirect_common_files=redirect_common_files,
        scheduler_cls=scheduler_cls,
    )

    resolved_type = infer_ckpt_type(ckpt_path, ckpt_type=ckpt_type)
    if ckpt_path:
        if resolved_type == "lora":
            lora_true_keys, extra_sd = pipe.load_lora(pipe.dit, ckpt_path, alpha=float(lora_alpha))
            loaded_state_dict_keys = list(lora_true_keys)
            missing, unexpected = [], list(extra_sd.keys())
            if len(extra_sd) > 0:
                # Try to load any remaining (non-LoRA) tensors from the same checkpoint.
                _, unexpected_extra = pipe.dit.load_state_dict(extra_sd, strict=False)
                loaded_extra_keys = [k for k in extra_sd.keys() if k not in set(unexpected_extra)]
                if loaded_extra_keys:
                    loaded_state_dict_keys = list(dict.fromkeys(list(loaded_state_dict_keys) + list(loaded_extra_keys)))
                unexpected = list(unexpected_extra)
        elif resolved_type == "full":
            state_dict = load_state_dict(ckpt_path)
            loaded_state_dict_keys = list(state_dict.keys())
            missing, unexpected = pipe.dit.load_state_dict(state_dict, strict=bool(strict))
        else:
            raise ValueError(f"Unknown ckpt_type: {ckpt_type!r}. Use auto|lora|full.")
    else:
        loaded_state_dict_keys = []
        missing, unexpected = [], []

    return pipe, (missing, unexpected, loaded_state_dict_keys)


def stitch_videos_vertical(
    videos,
    separator_px: int = 2,
    separator_color: Tuple[int, int, int] = (255, 255, 255),
) -> List[Image.Image]:
    """
    Stack videos into a single video.

    Supported input formats:
    - 1 column (backward compatible): `videos = [videoA, videoB, ...]`
      where each `videoX` is a `List[PIL.Image]`. Output is a vertical stack.
    - Multi-column grid (e.g. 2 columns): `videos = [[col0_video0, col0_video1, ...], [col1_video0, ...], ...]`
      Output is columns stitched horizontally; each column is stitched vertically.
    """
    def _is_frame(x) -> bool:
        return isinstance(x, Image.Image)

    def _is_video(x) -> bool:
        if not isinstance(x, list) or len(x) == 0:
            return False
        first = next((y for y in x if y is not None), None)
        return _is_frame(first)

    def _stitch_vertical_one_col(videos_1col: List[List[Image.Image]]) -> List[Image.Image]:
        vids_in = videos_1col or []
        vids = [v for v in vids_in if _is_video(v)]
        if not vids:
            return []
        n = min(len(v) for v in vids)
        if n <= 0:
            return []

        sep = max(0, int(separator_px))
        out: List[Image.Image] = []
        for fi in range(n):
            frames_maybe: List[Optional[Image.Image]] = []
            for v in vids_in:
                if v is None:
                    frames_maybe.append(None)
                elif _is_video(v):
                    frames_maybe.append(v[fi].convert("RGB"))

            ref = next((im for im in frames_maybe if im is not None), None)
            if ref is None:
                continue
            placeholder = Image.new("RGB", (ref.width, ref.height), color=(255, 0, 255))
            frames = [(im if im is not None else placeholder) for im in frames_maybe]
            target_w = max(im.width for im in frames)

            resized: List[Image.Image] = []
            heights: List[int] = []
            for im in frames:
                if im.width != target_w:
                    new_h = max(1, int(round(im.height * (target_w / im.width))))
                    im = im.resize((target_w, new_h), resample=Image.BILINEAR)
                resized.append(im)
                heights.append(im.height)

            out_h = sum(heights) + sep * (len(resized) - 1)
            canvas = Image.new("RGB", (target_w, out_h), color=separator_color)

            y = 0
            for j, im in enumerate(resized):
                canvas.paste(im, (0, y))
                y += im.height
                if j != len(resized) - 1:
                    y += sep

            out.append(canvas)
        return out

    def _stitch_horizontal(videos_row: List[List[Image.Image]]) -> List[Image.Image]:
        vids_in = videos_row or []
        vids = [v for v in vids_in if _is_video(v)]
        if not vids:
            return []
        n = min(len(v) for v in vids)
        if n <= 0:
            return []

        sep = max(0, int(separator_px))
        out: List[Image.Image] = []
        for fi in range(n):
            frames_maybe: List[Optional[Image.Image]] = []
            for v in vids_in:
                if v is None:
                    frames_maybe.append(None)
                elif _is_video(v):
                    frames_maybe.append(v[fi].convert("RGB"))

            ref = next((im for im in frames_maybe if im is not None), None)
            if ref is None:
                continue
            placeholder = Image.new("RGB", (ref.width, ref.height), color=(255, 0, 255))
            frames = [(im if im is not None else placeholder) for im in frames_maybe]
            target_h = max(im.height for im in frames)

            resized: List[Image.Image] = []
            widths: List[int] = []
            for im in frames:
                if im.height != target_h:
                    new_w = max(1, int(round(im.width * (target_h / im.height))))
                    im = im.resize((new_w, target_h), resample=Image.BILINEAR)
                resized.append(im)
                widths.append(im.width)

            out_w = sum(widths) + sep * (len(resized) - 1)
            canvas = Image.new("RGB", (out_w, target_h), color=separator_color)

            x = 0
            for j, im in enumerate(resized):
                canvas.paste(im, (x, 0))
                x += im.width
                if j != len(resized) - 1:
                    x += sep

            out.append(canvas)
        return out

    vids_in = videos or []
    first = next((v for v in vids_in if v is not None), None)
    if first is None:
        return []

    # 1-col mode: [video0, video1, ...]
    if _is_video(first):
        return _stitch_vertical_one_col(vids_in)

    # multi-col mode: [[col0_vid0, col0_vid1, ...], [col1_vid0, ...], ...]
    if isinstance(first, list):
        cols: List[List[Image.Image]] = []
        for col in vids_in:
            if not isinstance(col, list) or not col:
                continue
            col_video = _stitch_vertical_one_col(col)
            if col_video:
                cols.append(col_video)
        return _stitch_horizontal(cols)

    return []


def pil_video_to_tchw01(video_frames: List[Image.Image]) -> torch.Tensor:
    # video_frames: List[PIL.Image.Image] (assumed RGB-compatible)
    arr = np.stack([np.asarray(im.convert("RGB")) for im in video_frames], axis=0)  # [T,H,W,3] uint8
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float().div(255.0)  # [T,3,H,W] in [0,1]

def evaluate_video_metrics(
    ground_truth_frames,
    generated_frames,
    only_final: bool = False,
    fvd_method: str = "styleganv",
    resize_hw=None,
):
    """
    Wrapper around `tool_scripts/FVD/cal_4metrics.py` to compute SSIM/PSNR/LPIPS.

    Notes:
    - The underlying implementation expects inputs in [0,1] and operates on CPU
      (SSIM/PSNR convert tensors to numpy).
    - `fvd_method` is accepted for API compatibility with existing callers.
    """
    # Local import to keep validate utils lightweight unless metrics are requested.
    from tool_scripts.FVD.cal_4metrics import evaluate_video_metrics as _eval_impl

    return _eval_impl(
        ground_truth_frames=ground_truth_frames,
        generated_frames=generated_frames,
        device="cpu",
        only_final=only_final,
        fvd_method=fvd_method,
        resize_hw=resize_hw,
    )
