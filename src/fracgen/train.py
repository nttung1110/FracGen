import os
# Env vars must be set before importing diffsynth.
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ckpts", "wan"))
os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent  # src/fracgen, so `import diffsynth` works without installing
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import numpy as np
import torch, argparse, accelerate, warnings
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.models.wan_video_jointGen import JointWanTransformer3DModel, WanModel


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        train_mode="lora",
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        accelerator=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        combined_latent_tiling=False,
        phys_keys=None,
        lambda_const=0.0,
        gamma_const=2.0,
        const_warmup_steps=500,
    ):
        super().__init__()
        self.combined_latent_tiling = combined_latent_tiling
        from diffsynth.utils.latent_tiling import PHYS_KEYS as _DEFAULT_PHYS_KEYS
        self.phys_keys = phys_keys if phys_keys is not None else _DEFAULT_PHYS_KEYS

        # Constitutive consistency loss (learnable A_vol / A_dev operators).
        self.lambda_const = lambda_const
        self.gamma_const = gamma_const
        self.const_warmup_steps = const_warmup_steps
        self.effective_lambda_const = 0.0 if const_warmup_steps > 0 else lambda_const
        required_for_const = {"stress_video", "strain_video", "damage_video"}
        if combined_latent_tiling and lambda_const > 0 and required_for_const.issubset(set(self.phys_keys)):
            from diffsynth.diffusion.constitutive_loss import ConstitutiveLoss
            self.constitutive_loss = ConstitutiveLoss(C_latent=16)
        else:
            self.constitutive_loss = None
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        # Full fine-tuning: train the base model directly instead of injecting a LoRA
        # adapter into it. Reuse `lora_base_model` to name which model to unfreeze
        # (e.g. "dit"), then clear it so `switch_pipe_to_training_mode` skips LoRA setup.
        if train_mode == "full":
            if lora_base_model is not None:
                trainable_models = lora_base_model if trainable_models is None else trainable_models
            lora_base_model = None
        self.train_mode = train_mode

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(audio_processor_path)
        # if accelerator.num_processes > 1:
        #     # On multi-GPU: rank 0 loads first, then others load one at a time
        #     # This prevents all ranks hammering WEKA simultaneously
        #     if not accelerator.is_main_process:
        #         accelerator.wait_for_everyone()

        #     if accelerator.is_main_process:
        #         accelerator.wait_for_everyone()
        if accelerator.num_processes > 1:
            for rank_i in range(accelerator.num_processes):
                if accelerator.process_index == rank_i:
                    print(f"[rank{rank_i}] loading model...", flush=True)
                    self.pipe = WanVideoPipeline.from_pretrained(
                        torch_dtype=torch.bfloat16,
                        device=device,
                        model_configs=model_configs,
                        tokenizer_config=tokenizer_config,
                        audio_processor_config=audio_processor_config,
                        redirect_common_files=False,
                        scheduler_cls=FlowMatchScheduler,
                    )
                accelerator.wait_for_everyone()  # each rank waits for the previous
        else:
            # Single GPU: load normally
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=device,
                model_configs=model_configs,
                tokenizer_config=tokenizer_config,
                audio_processor_config=audio_processor_config,
                redirect_common_files=False,
                scheduler_cls=FlowMatchScheduler,
            )

        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft":       lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                # inputs_shared[extra_input] = data[extra_input][0]
                inputs_shared[extra_input] = data["video"][0] # modify: treat reference_image as the first frame of the video
            elif extra_input == "control_video":       # for control_video
                inputs_shared[extra_input] = data["force_video"] # modify: treat force_video as control_video
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],       # is List[PIL.Image.Image]
            "height": data["video"][0].size[1],
            "width":  data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared["combined_latent_tiling"] = self.combined_latent_tiling
        inputs_shared["phys_keys"] = self.phys_keys
        inputs_shared["constitutive_loss_fn"]   = self.constitutive_loss
        inputs_shared["effective_lambda_const"] = self.effective_lambda_const
        inputs_shared["gamma_const"]            = self.gamma_const
        inputs_shared["video_dict"] = {
            "stress_video": data["stress_video"],
            "strain_video": data["strain_video"],
            "damage_video": data["damage_video"],
            "flow_video":   data["flow_video"],
        }
        if "young_modulus" in data:
            inputs_shared.update({
                "young_modulus": data["young_modulus"],
            })
        young_modulus_map = data.get("young_modulus_map", None)
        inputs_shared["young_modulus_map"] = young_modulus_map
        inputs_shared["is_homo_young"] = young_modulus_map is None
        if "poisson_ratio" in data:
            inputs_shared.update({
                "poisson_ratio": data["poisson_ratio"],
            })
        if "lambda_onset" in data:
            inputs_shared.update({
                "lambda_onset": data["lambda_onset"],
            })
        if "lambda_crit" in data:
            inputs_shared.update({
                "lambda_crit": data["lambda_crit"],
            })
        if "category" in data:
            inputs_shared.update({
                "category": data["category"],
            })
        
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)

        return inputs_shared, inputs_posi, inputs_nega

    def modify_inputs(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        """Modify inputs_shared according to the model type"""

        # if type(pipe.dit) is WanModel:
        #     if ["reference_image"] == self.extra_inputs: # enable fun-control model:
        #         assert "reference_latents" in inputs_shared
        #         assert False, "Not implemented yet. Should only use fun-Inp model for force conditioning."

        #     elif ["input_image"] == self.extra_inputs:   # enable fun-Inp model:
        #         assert "input_latents" in inputs_shared
        #     else:
        #         raise NotImplementedError(f"Unsupported extra_inputs for WanModel: {self.extra_inputs}")

        # elif type(pipe.dit) is JointWanTransformer3DModel:
        #     pass
        # else:
        #     raise NotImplementedError(f"Unsupported model type: {type(pipe.dit)}")

        # if type(pipe.scheduler) is BridgeMatchScheduler:
        #     assert "source_latents_dict" in inputs_shared
        #     assert type(pipe.dit) is WanModel

        inputs_shared["combined_latent_tiling"] = self.combined_latent_tiling
        inputs_shared["phys_keys"] = self.phys_keys
        # Always inject constitutive-loss context so it works whether training
        # from cache (get_pipeline_inputs is skipped) or directly from raw data.
        inputs_shared["constitutive_loss_fn"]   = self.constitutive_loss
        inputs_shared["effective_lambda_const"] = self.effective_lambda_const
        inputs_shared["gamma_const"]            = self.gamma_const

        inputs = (inputs_shared, inputs_posi, inputs_nega)
        return inputs


    def run_pipeline_units(self, inputs):
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)  # (shared, posi, nega) -> (shared, posi, nega)
        return inputs

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
            inputs = self.run_pipeline_units(inputs)
        # When inputs is pre-loaded from cache (load_from_cache=True), skip
        # run_pipeline_units: the cache already contains the full 32-ch y conditioning
        # (force video 16ch + material props 16ch).  Re-running the pipeline would let
        # WanVideoUnitMaterialPropEmbedder overwrite y with a fresh 16-ch material-only
        # tensor (control_video is not saved in cache, so WanVideoUnit_FunControl is
        # skipped), causing a channel mismatch in patch_embedding (expects 48ch, gets 32ch).
        inputs = self.modify_inputs(self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument(
        "--audio_processor_path",
        type=str,
        default=None,
        help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.",
    )
    parser.add_argument(
        "--max_timestep_boundary",
        type=float,
        default=1.0,
        help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).",
    )
    parser.add_argument(
        "--min_timestep_boundary",
        type=float,
        default=0.0,
        help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).",
    )
    parser.add_argument(
        "--initialize_model_on_cpu",
        default=False,
        action="store_true",
        help="Whether to initialize models on CPU.",
    )
    parser.add_argument(
        "--combined_latent_tiling",
        default=False,
        action="store_true",
        help="Predict RGB + physics maps (stress/strain/damage/flow, see PHYS_KEYS in "
             "diffsynth/utils/latent_tiling.py) jointly by concatenating "
             "their VAE latents along the spatial-H axis into one taller latent. "
             "Compatible with force/material conditioning (model_fn_wan_video broadcasts "
             "the conditioning to every tile-row).",
    )
    return parser


if __name__ == "__main__":
    from diffsynth.video3d.logger import setup_logger
    parser = wan_parser()
    args = parser.parse_args()

    setup_logger(output=os.path.join(args.output_path, f"{args.task}.log"))
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
        log_with=("wandb" if getattr(args, "wandb", False) else None),
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
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
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        train_mode=args.train_mode,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        accelerator=accelerator,
        combined_latent_tiling=args.combined_latent_tiling,
        phys_keys=args.phys_keys.split(",") if args.phys_keys else None,
        lambda_const=args.lambda_const,
        gamma_const=args.gamma_const,
        const_warmup_steps=args.const_warmup_steps,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
    print(f"Training for task {args.task} completed!")
