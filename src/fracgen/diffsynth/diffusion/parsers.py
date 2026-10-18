import argparse


def add_dataset_base_config(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    return parser

def add_image_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution.")
    return parser

def add_video_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    return parser

def add_model_config(parser: argparse.ArgumentParser):
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--fp8_models", default=None, help="Models with FP8 precision, comma-separated.")
    parser.add_argument("--offload_models", default=None, help="Models with offload, comma-separated. Only used in splited training.")
    return parser

def add_training_config(parser: argparse.ArgumentParser):
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--phys_head_lr", type=float, default=None, help="Learning rate for physics aux heads (head.heads_aux). If omitted, uses --learning_rate.")
    parser.add_argument("--phys_keys", type=str, default=None, help="Comma-separated physics modality keys to use for combined_latent_tiling (e.g. 'flow_video' or 'stress_video,strain_video,damage_video,flow_video'). Defaults to all 4 maps. Only used when --combined_latent_tiling is set.")
    parser.add_argument("--lambda_const", type=float, default=0.0, help="Weight for the constitutive consistency loss (L_const). 0 disables it. Only active when --combined_latent_tiling is set and stress/strain/damage tiles are all present.")
    parser.add_argument("--gamma_const", type=float, default=2.0, help="Exponent for timestep weighting of L_const: w(t) = (1 - t/1000)^gamma. Higher gamma focuses the loss more on low-noise timesteps.")
    parser.add_argument("--const_warmup_steps", type=int, default=500, help="Steps over which lambda_const ramps from 0 to its target value (cosine schedule). 0 applies lambda_const immediately from step 1.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", '
            '"constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=150,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    parser.add_argument("--exp_id", type=str, default="DEBUG_2026", required=True, help="Experiment ID.")
    return parser

def add_output_config(parser: argparse.ArgumentParser):
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="auto",
        help=(
            "Resume training from a checkpoint. "
            "Supported values: 'auto' (resume latest if exists under --output_path), "
            "'latest' (alias of auto), 'none' (disable), or a path/filename to a checkpoint file."
        ),
    )
    parser.add_argument(
        "--no_resume_strict",
        default=False,
        action="store_true",
        help="Use strict=False when loading checkpoint weights (will not error on missing/unexpected keys).",
    )
    return parser

def add_lora_config(parser: argparse.ArgumentParser):
    parser.add_argument("--train_mode", type=str, default="lora", choices=["lora", "full"], help="Whether to train a LoRA adapter ('lora') or fully fine-tune the base model ('full'). In 'full' mode, --lora_base_model names the model to unfreeze and train directly, and all other --lora_* flags are ignored.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--preset_lora_path", type=str, default=None, help="Path to the preset LoRA checkpoint. If provided, this LoRA will be fused to the base model.")
    parser.add_argument("--preset_lora_model", type=str, default=None, help="Which model the preset LoRA is fused to.")
    return parser

def add_gradient_config(parser: argparse.ArgumentParser):
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping (<=0 disables).")
    return parser

def add_wandb_config(parser: argparse.ArgumentParser):
    parser.add_argument("--wandb", default=False, action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="DiffSynth_video3d", help="WandB project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity/team.")
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"], help="WandB mode.")
    parser.add_argument(
        "--wandb_proxy",
        type=str,
        default=None,
        help="Proxy URL for W&B (e.g. http://host:port or socks5h://host:port).",
    )
    parser.add_argument("--wandb_log_interval", type=int, default=1, help="Log every N optimizer steps.")
    return parser

def add_general_config(parser: argparse.ArgumentParser):
    parser = add_dataset_base_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_wandb_config(parser)
    return parser
