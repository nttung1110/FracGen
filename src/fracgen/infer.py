import argparse
import os
import sys
import statistics
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Env vars must be set before importing diffsynth.
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ckpts", "wan"))
os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_REPO_ROOT = Path(__file__).resolve().parent  # src/fracgen
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from diffsynth.core import UnifiedDataset, load_state_dict  # noqa: E402
from diffsynth.diffusion.flow_match import BridgeMatchScheduler, FlowMatchScheduler  # noqa: E402
from diffsynth.diffusion.training_module import DiffusionTrainingModule  # noqa: E402
from diffsynth.utils.data import save_video  # noqa: E402
from diffsynth.utils.latent_tiling import PHYS_KEYS, TILE_ORDER, tile_latents, untile_latents  # noqa: E402
from train import WanTrainingModule  # noqa: E402
from diffsynth.video3d.utils_misc import log_checkpoint_loading  # noqa: E402
from validate_utils import (  # noqa: E402
    load_diff_pipe,
    parse_hw_csv,
    stitch_videos_vertical,
    evaluate_video_metrics,
    pil_video_to_tchw01,
)


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def load_phys_head_checkpoint(module: "WanInferenceModule", ckpt_path: str):
    state_dict = load_state_dict(ckpt_path, torch_dtype=module.pipe.torch_dtype, device="cpu")
    mapped_state_dict = {}
    for key, value in state_dict.items():
        key = key.removeprefix("pipe.dit.").removeprefix("dit.")
        mapped_state_dict[key] = value
    missing, unexpected = module.pipe.dit.load_state_dict(mapped_state_dict, strict=False)
    loaded_keys = list(mapped_state_dict.keys())
    return loaded_keys, list(missing), list(unexpected)


class WanInferenceModule(WanTrainingModule):
    """
    Inference-only module that reuses input formatting from `WanTrainingModule`,
    but performs force-conditioned denoising and video decoding.
    """

    def __init__(
        self,
        model_id: str,
        ckpt: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        ckpt_type: str = "auto",
        lora_alpha: float = 1.0,
        combined_latent_tiling: bool = False,
        phys_keys=None,
        strict: bool = False,
        redirect_common_files: bool = False,
        extra_inputs: Optional[str] = "input_image",
        max_timestep_boundary: float = 1.0,
        min_timestep_boundary: float = 0.0,
    ):
        # Reuse `WanTrainingModule` helpers without invoking its training initialization.
        DiffusionTrainingModule.__init__(self)

        self.combined_latent_tiling = combined_latent_tiling
        self.phys_keys = phys_keys if phys_keys is not None else PHYS_KEYS
        self.tile_order = ["video"] + self.phys_keys
        self.pipe, state_keys = load_diff_pipe(
            model_id=model_id,
            ckpt_path=ckpt,
            device=device,
            torch_dtype=torch_dtype,
            redirect_common_files=redirect_common_files,
            strict=strict,
            scheduler_cls=FlowMatchScheduler,
            ckpt_type=ckpt_type,
            lora_alpha=lora_alpha,
        )
        self.missing_keys, self.unexpected_keys, self.loaded_state_dict_keys = state_keys

        self.task = "inference"
        self.use_gradient_checkpointing = False
        self.use_gradient_checkpointing_offload = False
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = float(max_timestep_boundary)
        self.min_timestep_boundary = float(min_timestep_boundary)
        self.constitutive_loss = None  # no physics loss at inference
        self.effective_lambda_const = 0.0
        self.gamma_const = 2.0
        self.lambda_const = 0.0
        self.const_warmup_steps = 0

        self.pipe.freeze_except([])
        self.pipe.dit.eval()


    def build_inputs(self, sample: dict, **kwargs):
        prompt = kwargs.get("prompt") or sample.get("prompt", "")
        data = {**sample, "prompt": prompt}  # `get_pipeline_inputs()` expects `prompt`, `video`, `force_video`.
        inputs_shared, inputs_posi, _ = self.get_pipeline_inputs(data)

        inputs_posi["prompt"] = prompt
        inputs_nega = {"negative_prompt": kwargs.get("negative_prompt", "")}

        # Preserve original validate script behavior: no `input_video`
        inputs_shared["input_video"] = None
        inputs_shared["control_video"] = sample["force_video"]
        inputs_shared["reference_image"] = sample["video"][0]
        inputs_shared["combined_latent_tiling"] = self.combined_latent_tiling

        for k in (
            "height", "width", "num_frames",
            "cfg_scale",
            "denoising_strength",
            "sigma_shift",
            "rand_device", "seed",
            "tiled", "tile_size", "tile_stride",
        ):
            v = kwargs.get(k)
            if v is not None:
                inputs_shared[k] = v

        return inputs_shared, inputs_posi, inputs_nega

    @torch.no_grad()
    def infer(
        self,
        sample: dict,
        prompt: str,
        negative_prompt: str,
        seed: int,
        rand_device: str,
        height: int,
        width: int,
        num_frames: int,
        cfg_scale: float,
        denoising_strength: float,
        num_inference_steps: int,
        sigma_shift: float,
        tiled: bool,
        tile_size: Tuple[int, int],
        tile_stride: Tuple[int, int],
    ):
        pipe = self.pipe
        pipe.scheduler.set_timesteps(
            int(num_inference_steps),
            denoising_strength=float(denoising_strength),
            shift=float(sigma_shift),
        )

        inputs = self.build_inputs(
            sample,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            rand_device=rand_device,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_scale=cfg_scale,
            denoising_strength=denoising_strength,
            sigma_shift=sigma_shift,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        inputs_shared, inputs_posi, inputs_nega = self.run_pipeline_units(inputs)

        z_dim = int(self.pipe.vae.model.z_dim)

        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

        if self.combined_latent_tiling:
            # Tile RGB + physics noise along the spatial-H axis -- matches
            # training's FlowMatchSFTLoss, which draws one noise tensor over the whole
            # combined (RGB+physics) clean latent.
            rgb_lat = inputs_shared["latents"]
            phys_noise = [torch.randn_like(rgb_lat) for _ in self.phys_keys]
            inputs_shared["latents"] = tile_latents([rgb_lat] + phys_noise, dim=3)

        for progress_id, timestep in enumerate(tqdm(pipe.scheduler.timesteps, desc="denoise")):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred_posi = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_nega = pipe.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # scheduler.step is channel-independent — apply to full joint tensor at once
            inputs_shared["latents"] = pipe.scheduler.step(
                noise_pred, pipe.scheduler.timesteps[progress_id], inputs_shared["latents"]
            )

            if "first_frame_latents" in inputs_shared:
                if self.combined_latent_tiling:
                    # RGB tile = top H' rows (H' = first_frame_latents' own height) --
                    # every tile shares the same z_dim channel count under H-tiling, so
                    # the RGB tile is identified by row position, not channel slice.
                    H_prime = inputs_shared["first_frame_latents"].shape[3]
                    inputs_shared["latents"][:, :, 0:1, :H_prime, :] = inputs_shared["first_frame_latents"]
                else:
                    # Only fix RGB first frame; physics has no reference first frame
                    inputs_shared["latents"][:, :z_dim, 0:1] = inputs_shared["first_frame_latents"]

        pipe.load_models_to_device(["vae"])
        gen_videos = {}

        def _decode(latents):
            return pipe.vae_output_to_video(pipe.vae.decode(
                latents, device=pipe.device,
                tiled=bool(tiled), tile_size=tile_size, tile_stride=tile_stride,
            ))

        if self.combined_latent_tiling:
            # Row-slice (not channel-slice): every tile shares the same z_dim channels,
            # only the H axis distinguishes them. Display name derived from TILE_ORDER
            # itself (stripping "_video", "video" -> "rgb") rather than a hand-maintained
            # dict, so adding a modality to PHYS_KEYS doesn't require updating this too.
            def _display_name(key: str) -> str:
                return "rgb" if key == "video" else key[: -len("_video")]

            tile_tensors = untile_latents(inputs_shared["latents"], len(self.tile_order), dim=3)
            for key, tensor in zip(self.tile_order, tile_tensors):
                gen_videos[_display_name(key)] = _decode(tensor)
        else:
            gen_videos["rgb"] = _decode(inputs_shared["latents"][:, :z_dim])

        pipe.load_models_to_device([])
        return gen_videos
    
    def create_visual_outputs(self, save_dir, save_prefix, gt_videos: dict, gen_videos: dict, fps=5, quality=5):
        """
        Save per-modality predicted videos in subfolders, plus a combined stitched video.

        Output structure:
          save_dir/
            fracture/{save_prefix}.mp4          # predicted RGB
            stress/{save_prefix}.mp4
            strain/{save_prefix}.mp4
            damage/{save_prefix}.mp4
            flow/{save_prefix}.mp4
            joint_inspect/{save_prefix}.mp4     # horizontal: strain | stress | damage | fracture
            combined/{save_prefix}_stitched.mp4 # GT + gen all modalities side-by-side
        """
        # "rgb" maps to the "fracture" subfolder; all other keys use their own name.
        _SUBFOLDER = {"rgb": "fracture"}

        for key, frames in gen_videos.items():
            if frames is None:
                continue
            mod_dir = os.path.join(save_dir, _SUBFOLDER.get(key, key))
            os.makedirs(mod_dir, exist_ok=True)
            save_video(frames, os.path.join(mod_dir, f"{save_prefix}.mp4"), fps=int(fps), quality=int(quality))

        # Horizontal inspection strip: strain | stress | damage | fracture (predicted only).
        joint_clips = [
            gen_videos.get("strain"),
            gen_videos.get("stress"),
            gen_videos.get("damage"),
            gen_videos.get("rgb"),
        ]
        if any(c is not None for c in joint_clips):
            joint_strip = stitch_videos_vertical([[c] for c in joint_clips])
            joint_dir = os.path.join(save_dir, "joint_inspect")
            os.makedirs(joint_dir, exist_ok=True)
            save_video(joint_strip, os.path.join(joint_dir, f"{save_prefix}.mp4"), fps=int(fps), quality=int(quality))

        # Combined stitched view: GT column | gen column, one row per modality.
        if self.combined_latent_tiling:
            keys = ["force"] + [k[: -len("_video")] for k in self.phys_keys] + ["rgb"]
        else:
            keys = ["force", "rgb"]

        gt_col = [gt_videos.get(k, None) for k in keys]
        gen_col = [gen_videos.get(k, None) for k in keys]
        stitched = stitch_videos_vertical([gt_col, gen_col])
        combined_dir = os.path.join(save_dir, "combined")
        os.makedirs(combined_dir, exist_ok=True)
        save_video(
            stitched,
            os.path.join(combined_dir, f"{save_prefix}_stitched.mp4"),
            fps=int(fps),
            quality=int(quality),
        )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="PAI/Wan2.1-Fun-V1.1-14B-InP")
    parser.add_argument("--ckpt", type=str, required=True, help="LoRA checkpoint or full weights (.safetensors)")
    parser.add_argument("--ckpt_type", type=str, default="auto", choices=["auto", "lora", "full"])
    parser.add_argument(
        "--phys_head_ckpt",
        type=str,
        default=None,
        help="Optional aux physics-head checkpoint to overlay after loading the base/full/LoRA checkpoint.",
    )
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--strict", action="store_true", help="Strict state_dict loading (full ckpt only).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--extra_inputs", type=str, default="input_image", help="Additional model inputs, comma-separated.")

    # Dataset (UnifiedDataset) config: must include columns `video` and `force_video`.
    parser.add_argument("--dataset_base_path", type=str, default="data/example_force_dataset")
    parser.add_argument("--dataset_metadata_path", type=str, default="data/example_force_dataset/metadata.csv")
    parser.add_argument("--data_file_keys", type=str, default="video,force_video")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--max_pixels", type=int, default=1920 * 1080)

    # Inference config
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_save", type=int, default=20)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="", help="Override prompt. If empty, uses sample['prompt'] when available.")
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--denoising_strength", type=float, default=1.0)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=parse_hw_csv, default=parse_hw_csv("30,52"))
    parser.add_argument("--tile_stride", type=parse_hw_csv, default=parse_hw_csv("15,26"))
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--rand_device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--combined_latent_tiling",
        default=False,
        action="store_true",
        help="Generate RGB + physics maps jointly via combined_latent_tiling -- "
             "must match how the checkpoint was trained. "
    )
    parser.add_argument(
        "--phys_keys",
        type=str,
        default=None,
        help="Comma-separated physics modality keys (e.g. 'flow_video' or "
             "'stress_video,strain_video,damage_video,flow_video'). Must match "
             "--phys_keys used during training. Defaults to all 4 maps.",
    )
    args = parser.parse_args()

    # Disambiguate output_dir by the eval dataset used, e.g.
    # ".../infer_epoch-5" + dataset_base_path=".../benchmarks/output_test_set"
    # -> ".../infer_epoch-5_output_test_set" -- otherwise re-running the same
    # checkpoint against a different --dataset_base_path silently overwrites
    # the previous eval's results in the same output_dir.
    dataset_suffix = os.path.basename(os.path.normpath(args.dataset_base_path))
    if dataset_suffix:
        args.output_dir = f"{args.output_dir.rstrip('/')}_{dataset_suffix}"
    print(f"Output dir (suffixed with dataset '{dataset_suffix}'): {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.torch_dtype]
    module = WanInferenceModule(
        model_id=args.model_id,
        ckpt=args.ckpt,
        device=str(args.device),
        torch_dtype=torch_dtype,
        strict=args.strict,
        ckpt_type=args.ckpt_type,
        lora_alpha=args.lora_alpha,
        combined_latent_tiling=args.combined_latent_tiling,
        phys_keys=args.phys_keys.split(",") if args.phys_keys else None,
        extra_inputs=args.extra_inputs,
    )
    load_log_path = os.path.join(args.output_dir, "ckpt_load_report.txt")
    log_checkpoint_loading(
        args.ckpt,
        args.strict,
        module.loaded_state_dict_keys,
        module.missing_keys,
        module.unexpected_keys,
        load_log_path,
    )
    if args.phys_head_ckpt:
        loaded, missing, unexpected = load_phys_head_checkpoint(module, args.phys_head_ckpt)
        phys_head_log_path = os.path.join(args.output_dir, "phys_head_load_report.txt")
        with open(phys_head_log_path, "w") as f:
            f.write(f"phys_head_ckpt: {args.phys_head_ckpt}\n")
            f.write(f"loaded_keys ({len(loaded)}):\n")
            for key in loaded:
                f.write(f"  {key}\n")
            f.write(f"missing_keys ({len(missing)}):\n")
            for key in missing:
                f.write(f"  {key}\n")
            f.write(f"unexpected_keys ({len(unexpected)}):\n")
            for key in unexpected:
                f.write(f"  {key}\n")
        print(
            f"Loaded phys head checkpoint: {args.phys_head_ckpt} "
            f"(loaded={len(loaded)}, missing={len(missing)}, unexpected={len(unexpected)})"
        )

    data_file_keys = tuple(k.strip() for k in args.data_file_keys.split(",") if k.strip())
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=1,
        data_file_keys=data_file_keys,
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "young_modulus": lambda x: x,      # passthrough scalars
            "poisson_ratio": lambda x: x,
            "lambda_onset": lambda x: x,
            "lambda_crit": lambda x: x,
            "category": lambda x: x,
            "young_modulus_map": lambda x: torch.tensor(
                np.load(os.path.join(args.dataset_base_path, x)), dtype=torch.float32
            ) if x else None,
        }
    )

    end_i = min(args.start_idx + args.num_save, len(dataset))
    metrics_accum = {}  # metric_name -> List[float]
    for i in tqdm(range(args.start_idx, end_i), desc="infer"):
        sample = dataset[i]

        prompt = args.prompt if args.prompt.strip() else str(sample.get("prompt", ""))
        seed_i = int(args.seed) + int(i)
        gen = module.infer(
            sample,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            seed=seed_i,
            rand_device=args.rand_device,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            cfg_scale=args.cfg_scale,
            denoising_strength=args.denoising_strength,
            num_inference_steps=args.num_inference_steps,
            sigma_shift=args.sigma_shift,
            tiled=bool(args.tiled),
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
        )
        raw_video_path = dataset.data[i % len(dataset.data)].get("video", "")
        prefix = Path(raw_video_path).stem if raw_video_path else f"{i:04d}"
        gt_videos = {
            "rgb":    sample["video"],
            "force":  sample["force_video"],
            "stress": sample.get("stress_video"),
            "strain": sample.get("strain_video"),
            "damage": sample.get("damage_video"),
            "flow":   sample.get("flow_video"),
        }
        module.create_visual_outputs(
            save_dir=args.output_dir,
            save_prefix=prefix,
            gt_videos=gt_videos,
            gen_videos=gen,
            fps=args.fps,
            quality=args.quality,
        )
        # frames = {
        #     "gt_frames": [pil_video_to_tchw01(sample["video"])],
        #     "gen_frames": [pil_video_to_tchw01(gen["rgb"])],
        # }

        # 4. Evaluation metrics
        # metrics_dict = evaluate_video_metrics(
        #     ground_truth_frames=frames['gt_frames'],
        #     generated_frames=frames['gen_frames'],
        #     only_final=True,         # keep same behavior
        # )
        # for key, value in metrics_dict.items():
        #     metrics_accum.setdefault(key, []).append(float(value["value"][0]))

        # print(f"Saved sample {i:04d} into {args.output_dir}\n")

    # if metrics_accum:
    #     print("\n=== Metrics summary (mean ± std over samples) ===")
    #     for k in sorted(metrics_accum.keys()):
    #         vals = metrics_accum.get(k, [])
    #         if not vals:
    #             continue
    #         mean = statistics.mean(vals)
    #         std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    #         print(f"{k}: {mean:.6f} ± {std:.6f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
