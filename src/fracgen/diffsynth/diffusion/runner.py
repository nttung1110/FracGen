import os
import math
import torch
from tqdm.auto import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from ..utils.latent_tiling import PHYS_KEYS as _DEFAULT_PHYS_KEYS
from diffsynth.video3d.utils_train import (
    configure_optimizer_scheduler,
    get_train_params,
)


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    max_grad_norm: float = 5.0,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
    dataloader_kwargs={},
):
    if args is not None:
        max_grad_norm = args.max_grad_norm
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    # Optional Weights & Biases logging
    if args.wandb: init_wandb(accelerator, args)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        shuffle=True, 
        collate_fn=lambda x: x[0], 
        num_workers=num_workers,
        **dataloader_kwargs,
    )

    # cond_list = [
    #     "('after_proj_main_block' in name)",
    #     "('geo_' in name)",

    #     # "('text_embedding' in name)",
    #     # "('time_embedding' in name)",
    #     # "('time_projection' in name)",
    #     # "('img_emb' in name)",
    # ]
    # cond_text = " or ".join(cond_list)
    # cond_text = "True"      

    ### Original behavior (LoRA): PEFT already set requires_grad=True on adapter
    ### params; just also pick up the (zero-init) physics aux-head params.
    ### Full fine-tuning: `freeze_except` already unfroze exactly the model(s)
    ### named by --trainable_models, so train everything left unfrozen.
    if getattr(args, "train_mode", "lora") == "full":
        cond_text = "param.requires_grad"
        added_cond_text = "False"
    else:
        cond_text = None
        added_cond_text = "'_aux' in name"

    ### For diagnose
    # cond_text = (
    #     "name in {"
    #     "'pipe.dit.head.heads_aux.0.weight', "
    #     "'pipe.dit.head.heads_aux.0.bias', "
    #     "'pipe.dit.head.heads_aux.1.weight', "
    #     "'pipe.dit.head.heads_aux.1.bias', "
    #     "'pipe.dit.head.heads_aux.2.weight', "
    #     "'pipe.dit.head.heads_aux.2.bias', "
    #     "'pipe.dit.head.heads_aux.3.weight', "
    #     "'pipe.dit.head.heads_aux.3.bias', "
    #     "'pipe.dit.head.modulations_aux.0', "
    #     "'pipe.dit.head.modulations_aux.1', "
    #     "'pipe.dit.head.modulations_aux.2', "
    #     "'pipe.dit.head.modulations_aux.3'"
    #     "}"
    # )
    # added_cond_text = "False"

    optimizer_cls = torch.optim.AdamW
    parameters_list = get_train_params(
        model,
        trainable_params=cond_text,
        added_params=added_cond_text,
        output_dir=args.output_path,
    )

    # Optionally use a higher LR for physics aux heads than for LoRA weights.
    phys_head_lr = getattr(args, "phys_head_lr", None)
    if phys_head_lr is not None and phys_head_lr != args.learning_rate:
        phys_ids = {id(p) for name, p in model.named_parameters() if "_aux" in name}
        lora_params = [p for p in parameters_list if id(p) not in phys_ids]
        phys_params = [p for p in parameters_list if id(p) in phys_ids]
        parameters_list = [
            {"params": lora_params, "lr": args.learning_rate},
            {"params": phys_params, "lr": phys_head_lr},
        ]
        print(f"Using separate LRs — LoRA: {args.learning_rate}, head.heads_aux: {phys_head_lr}")

    # Scheduler (configurable).
    per_process_batches_per_epoch = math.ceil(len(dataloader) / accelerator.num_processes)
    num_update_steps_per_epoch = math.ceil(per_process_batches_per_epoch / args.gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    args.max_train_steps = max_train_steps

    optimizer, scheduler = configure_optimizer_scheduler(
        args, accelerator.num_processes,
        optimizer_cls, parameters_list,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    # 1. Resume training
    start_epoch, global_step = model_logger.maybe_resume(
        accelerator,
        model,
        args,
        optimizer=optimizer,
        scheduler=scheduler,
        num_update_steps_per_epoch=num_update_steps_per_epoch,
    )

    # 2. Training loop
    for epoch_id in range(start_epoch, num_epochs):
        progress_bar = tqdm(
            dataloader,
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
            desc=f"Epoch {epoch_id + 1}/{num_epochs}",
        )
        for step_id, data in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # Cosine ramp-up for constitutive loss weight.
                unwrapped = accelerator.unwrap_model(model)
                if (hasattr(unwrapped, "const_warmup_steps")
                        and unwrapped.const_warmup_steps > 0
                        and unwrapped.constitutive_loss is not None):
                    progress = min(global_step / unwrapped.const_warmup_steps, 1.0)
                    cosine_weight = (1.0 - math.cos(math.pi * progress)) / 2.0
                    unwrapped.effective_lambda_const = unwrapped.lambda_const * cosine_weight

                if dataset.load_from_cache:
                    result = model({}, inputs=data)
                else:
                    result = model(data)
                if isinstance(result, tuple):
                    loss = result[0]
                    raw_phys = result[1] if len(result) > 1 else None
                    const_loss_item = result[2] if len(result) > 2 else None
                else:
                    loss, raw_phys, const_loss_item = result, None, None
                loss_for_log = accelerator.reduce(loss.detach(), reduction="mean").float()
                raw_phys_for_log = (
                    accelerator.reduce(raw_phys.detach(), reduction="mean").float()
                    if raw_phys is not None else None
                )
                const_loss_for_log = (
                    accelerator.reduce(const_loss_item.detach(), reduction="mean").float()
                    if const_loss_item is not None else None
                )

                # assert loss.dim() == 1 and loss.shape[0] >= 3, f"Expected loss shape [>=3], got {loss.shape}"
                accelerator.backward(loss.mean())
                # if accelerator.sync_gradients:
                #     grad_norm = accelerator.clip_grad_norm_(parameters_list, max_grad_norm)
                #     if not torch.isfinite(grad_norm):
                #         if accelerator.is_local_main_process:
                #             accelerator.print(f"[WARRNING] Non-finite grad norm at step={global_step} (epoch_step={step_id}): {grad_norm}")
                #         optimizer.zero_grad()
                #         continue

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(
                    accelerator, model, save_steps,
                    optimizer=optimizer, scheduler=scheduler,
                    epoch_id=epoch_id, global_step=global_step,
                )
                if accelerator.sync_gradients:
                    global_step += 1

            # log and update progress bar:
            if accelerator.is_local_main_process:
                lr = optimizer.param_groups[0].get("lr", None)
                postfix = {
                    "step": global_step,
                    "loss": f"{loss_for_log.mean().item():.4f}",
                    "lr": f"{lr:.2e}" if lr is not None else "n/a",
                }
                progress_bar.set_postfix(postfix)
                if const_loss_for_log is not None and global_step <= 2:
                    # Printed regardless of wandb so a run shows the constitutive loss is active.
                    accelerator.print(
                        f"[constitutive_loss] step={global_step} "
                        f"loss_const_raw={const_loss_for_log.item():.6f}  "
                        f"effective_lambda_const={accelerator.unwrap_model(model).effective_lambda_const:.6f}"
                    )
                if args.wandb and (global_step % args.wandb_log_interval == 0):
                    _phys_keys_arg = getattr(args, "phys_keys", None)
                    _phys_keys = _phys_keys_arg.split(",") if _phys_keys_arg else _DEFAULT_PHYS_KEYS
                    PHYS_NAMES = [k[: -len("_video")] if k != "video" else k for k in ["video"] + _phys_keys]
                    metrics = {
                        # Under combined_latent_tiling this is the one joint loss over the
                        # whole combined tensor, not RGB-only -- see loss_video_raw below
                        # (from raw_phys_for_log) for the RGB-tile-only diagnostic in that mode.
                        "loss_rgb": loss_for_log[0].item(),
                    }
                    # Scaled physics losses (what the optimizer sees)
                    for i in range(len(loss_for_log) - 1):
                        name = PHYS_NAMES[i] if i < len(PHYS_NAMES) else str(i)
                        metrics[f"loss_{name}_scaled"] = loss_for_log[i + 1].item()
                    # Raw unscaled per-tile losses (diagnostic only, no gradient effect)
                    if raw_phys_for_log is not None:
                        for i in range(len(raw_phys_for_log)):
                            name = PHYS_NAMES[i] if i < len(PHYS_NAMES) else str(i)
                            metrics[f"loss_{name}_raw"] = raw_phys_for_log[i].item()
                    if const_loss_for_log is not None:
                        _clv = const_loss_for_log.item()
                        _eff_lam = accelerator.unwrap_model(model).effective_lambda_const
                        metrics["loss_const_raw"] = _clv
                        metrics["effective_lambda_const"] = _eff_lam
                    metrics.update({
                        "loss_all": loss_for_log.mean().item(),
                        "lr": lr,
                    })
                    accelerator.log(metrics, step=global_step)
        if save_steps is None:
            model_logger.save_epoch_checkpoint(
                accelerator,
                model,
                epoch_id,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
            )
    model_logger.on_training_end(accelerator, model, save_steps)

def init_wandb(accelerator, args):
    wandb_project = args.wandb_project
    wandb_proxy = getattr(args, "wandb_proxy", None)
    if wandb_proxy:
        # Let users route W&B through a proxy (e.g. ssh SOCKS tunnel to another node).
        os.environ.setdefault("HTTP_PROXY", wandb_proxy)
        os.environ.setdefault("HTTPS_PROXY", wandb_proxy)
        os.environ.setdefault("ALL_PROXY", wandb_proxy)

    wandb_init = {
        "name": args.exp_id,
        "dir": args.output_path,
    }
    if args.wandb_entity:
        wandb_init["entity"] = args.wandb_entity
    tags = [t for t in str(args.exp_id).split("_") if t]
    if tags:
        # De-duplicate while preserving order.
        seen = set()
        deduped = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        wandb_init["tags"] = deduped
    if args.wandb_mode:
        wandb_init["mode"] = args.wandb_mode
    init_kwargs = {"wandb": wandb_init} if wandb_init else None
    accelerator.init_trackers(wandb_project, config=vars(args), init_kwargs=init_kwargs)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
    dataloader_kwargs = {},
    saved_keys = [
        "context", "prompt", 
        "height", "width", "num_frames",
        "young_modulus", "poisson_ratio",
        "lambda_onset", "lambda_crit", "category",
        # 1. default params for WanVideoPipeline:
        "cfg_scale",
        "tiled",
        "rand_device",
        "use_gradient_checkpointing",
        "use_gradient_checkpointing_offload",
        "cfg_merge",
        "vace_scale",
        "max_timestep_boundary",
        "min_timestep_boundary",
        # 2. required params for WanVideoPipeline dit in training:
        # "noise", ~ "latents"
        # "latents",
        "input_latents", # video rgb latents
        "source_latents_dict",      # video physics latents dict: {"stress", "strain", "damage", "flow"}
        "reference_latents",
        "y", # condition (control video)
        "clip_feature",
        # 3. for JointWanTransformer3DModel:
        # "y_geo",
        "pts3d_norm_scale",
        "disp_norm_scale",
        "pts_valid_mask",
        "disp_valid_mask",
    ],
):
    if args is not None:
        num_workers = args.dataset_num_workers

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x[0],
        **dataloader_kwargs,
    )
    model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                if os.path.exists(save_path):
                    continue
                data = model(data)
                # Keep only selected keys to reduce cache size.
                if isinstance(data, dict):
                    data = {k: v for k, v in data.items() if k in saved_keys}
                else:
                    filtered = []
                    for elem in data:
                        assert isinstance(elem, dict)
                        filtered.append({k: v for k, v in elem.items() if k in saved_keys})
                    data = tuple(filtered) if isinstance(data, tuple) else filtered
                torch.save(data, save_path)
    
    accelerator.print(f"Data processing completed! Processed data are saved to folder: {model_logger.output_path}")
