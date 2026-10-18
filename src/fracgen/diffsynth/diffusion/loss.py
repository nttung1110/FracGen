from .base_pipeline import BasePipeline
import torch
from jaxtyping import Float, Int32, UInt8
from numpy.typing import NDArray
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple, Union
from .flow_match import BridgeMatchScheduler
from ..models.wan_video_jointGen import JointWanTransformer3DModel
from .pixel_loss import compute_lpips_on_decoded_frames
from ..utils.latent_tiling import PHYS_KEYS, TILE_ORDER, tile_latents_named, untile_latents
import os
import warnings
import torch.nn.functional as F


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    """
    Plain single-stream flow matching by default. When `combined_latent_tiling` is set,
    RGB + physics-map clean latents (already VAE-encoded in source_latents_dict by
    WanVideoUnit_PhysVideoEmbedder, regardless of this flag) are concatenated along the
    latent H axis into one taller tensor before noising -- the model then denoises one
    ordinary-shaped tensor, no aux heads/channel-concat involved.

    Supersedes the old with_phys_map channel-concat + auxiliary-head mechanism (removed).
    """
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    rgb_latents = inputs["input_latents"]   # (B, C, T, H, W) clean RGB
    dim_latent = rgb_latents.shape[1]

    combined_latent_tiling = inputs.get("combined_latent_tiling", False)

    phys_keys = inputs.get("phys_keys", PHYS_KEYS)
    tile_order = ["video"] + phys_keys

    if combined_latent_tiling:
        source_latents_dict = inputs["source_latents_dict"]
        clean_combined = tile_latents_named(
            {"video": rgb_latents, **{k: source_latents_dict[k] for k in phys_keys}},
            order=tile_order,
        )
    else:
        clean_combined = rgb_latents

    noise = torch.randn_like(clean_combined)
    noisy = pipe.scheduler.add_noise(clean_combined, noise, timestep)
    target = pipe.scheduler.training_target(clean_combined, noise, timestep)

    inputs["latents"] = noisy
    if "first_frame_latents" in inputs:
        if combined_latent_tiling:
            # RGB tile = top H' rows of frame 0 (H' = rgb_latents' own height) -- under
            # H-tiling every tile shares the same dim_latent channel count, so the RGB
            # tile is identified by row position, not by channel slice.
            H_prime = rgb_latents.shape[3]
            inputs["latents"][:, :, 0:1, :H_prime, :] = inputs["first_frame_latents"]
        else:
            inputs["latents"][:, :dim_latent, 0:1] = inputs["first_frame_latents"]

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    first_frame = "first_frame_latents" in inputs
    if first_frame:
        noise_pred = noise_pred[:, :, 1:]
        target = target[:, :, 1:]

    weight = pipe.scheduler.training_weight(timestep)
    loss = F.mse_loss(noise_pred.float(), target.float()) * weight

    # Constitutive consistency loss -- latent-space Lamé relation between predicted
    # stress, strain, and damage tiles. Only active when tiling is on, the three
    # required tiles are present, and lambda_const > 0.
    const_loss_fn      = inputs.get("constitutive_loss_fn", None)
    effective_lambda_const = float(inputs.get("effective_lambda_const", 0.0))
    L_const_raw = None
    _has_ym = "young_modulus" in inputs
    _has_pr = "poisson_ratio" in inputs
    if (const_loss_fn is not None
            and combined_latent_tiling
            and _has_ym
            and _has_pr):
        # Recover predicted clean latent: x̂₀ = z_t − (t/1000) · v_θ
        # Works for the flow-matching convention z_t = (1−σ)·x0 + σ·noise, v = noise − x0.
        sigma = (timestep.float() / 1000.0).view(-1, 1, 1, 1, 1)
        noisy_frames = noisy[:, :, 1:] if first_frame else noisy
        x0_pred = noisy_frames - sigma * noise_pred
        L_const = const_loss_fn(
            x0_pred,
            tile_order,
            inputs["young_modulus"],
            inputs["poisson_ratio"],
            timestep,
            float(inputs.get("gamma_const", 2.0)),
        )
        L_const_raw = L_const.detach()
        loss = loss + effective_lambda_const * L_const
    elif const_loss_fn is not None and combined_latent_tiling:
        # Warn once so training logs reveal why the metric is absent.
        warnings.warn(
            f"[FlowMatchSFTLoss] constitutive_loss_fn present but skipped: "
            f"young_modulus in inputs={_has_ym}, poisson_ratio in inputs={_has_pr}",
            RuntimeWarning, stacklevel=2,
        )

    if not combined_latent_tiling:
        return torch.stack([loss]), None, None

    # Diagnostic only, no gradient effect: per-tile MSE for trend-watching in W&B (read as
    # smoothed curves over training, not single-batch values -- flow-matching loss is
    # timestep-dependent and inherently noisy per batch).
    with torch.no_grad():
        pred_tiles = untile_latents(noise_pred.detach(), len(tile_order), dim=3)
        target_tiles = untile_latents(target.detach(), len(tile_order), dim=3)
        per_tile_mse = torch.stack([
            F.mse_loss(p.float(), t.float()) * weight for p, t in zip(pred_tiles, target_tiles)
        ])

    return torch.stack([loss]), per_tile_mse, L_const_raw

def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss

class PhysicsMapMatchingLoss(torch.nn.Module):
    def __init__(self, w_stress=1, w_strain=1, w_damage=1, w_vel=1):
        '''
            w: scalar weights balancing physical loss
        '''
        super().__init__()
        self.map2w = {"stress_map": w_stress, "strain_map": w_strain, "damage_map": w_damage, "vel_map": w_vel}

    def forward(self, pipe: BasePipeline, inputs: dict):
        outputs_map = pipe.inference_phys_map

        loss = torch.tensor(0.0, device=next(pipe.parameters()).device)

        for phys_map_key in self.list_phys_map:
            gt_latents = inputs[phys_map_key]          # [bs, C, H, W]
            pred_latents = outputs_map[phys_map_key]   # [bs, C, H, W]

            assert pred_latents.shape == gt_latents.shape, f"{phys_map_key} shape mismatch: {pred_latents.shape} vs {gt_latents.shape}"

            weight = self.wap2w[phys_map_key]
            loss = loss + weight * F.mse_loss(pred_latents, gt_latents)

        return loss

