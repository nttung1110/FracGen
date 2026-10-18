import torch, math
import numpy as np
from typing_extensions import Literal
from functools import partial



class FlowMatchScheduler():

    def __init__(self, template: Literal["FLUX.1", "Wan", "Qwen-Image", "FLUX.2", "Z-Image"] = "FLUX.1"):
        self.set_timesteps_fn = {
            "FLUX.1": FlowMatchScheduler.set_timesteps_flux,
            "Wan": FlowMatchScheduler.set_timesteps_wan,
            "Qwen-Image": FlowMatchScheduler.set_timesteps_qwen_image,
            "FLUX.2": FlowMatchScheduler.set_timesteps_flux2,
            "Z-Image": FlowMatchScheduler.set_timesteps_z_image,
        }.get(template, FlowMatchScheduler.set_timesteps_flux)
        self.num_train_timesteps = 1000

    @staticmethod
    def set_timesteps_flux(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.003/1.002
        sigma_max = 1.0
        shift = 3 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def set_timesteps_wan(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 5 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def _calculate_shift_qwen_image(image_seq_len, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
    
    @staticmethod
    def set_timesteps_qwen_image(num_inference_steps=100, denoising_strength=1.0, exponential_shift_mu=None, dynamic_shift_len=None):
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        shift_terminal = 0.02
        # Sigmas
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        # Mu
        if exponential_shift_mu is not None:
            mu = exponential_shift_mu
        elif dynamic_shift_len is not None:
            mu = FlowMatchScheduler._calculate_shift_qwen_image(dynamic_shift_len)
        else:
            mu = 0.8
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        # Shift terminal
        one_minus_z = 1 - sigmas
        scale_factor = one_minus_z[-1] / (1 - shift_terminal)
        sigmas = 1 - (one_minus_z / scale_factor)
        # Timesteps
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def compute_empirical_mu(image_seq_len, num_steps):
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            mu = a2 * image_seq_len + b2
            return float(mu)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1

        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * num_steps + b

        return float(mu)
    
    @staticmethod
    def set_timesteps_flux2(num_inference_steps=100, denoising_strength=1.0, dynamic_shift_len=1024//16*1024//16):
        sigma_min = 1 / num_inference_steps
        sigma_max = 1.0
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
        mu = FlowMatchScheduler.compute_empirical_mu(dynamic_shift_len, num_inference_steps)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_z_image(num_inference_steps=100, denoising_strength=1.0, shift=None, target_timesteps=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 3 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        if target_timesteps is not None:
            target_timesteps = target_timesteps.to(dtype=timesteps.dtype, device=timesteps.device)
            for timestep in target_timesteps:
                timestep_id = torch.argmin((timesteps - timestep).abs())
                timesteps[timestep_id] = timestep
        return sigmas, timesteps
    
    def set_training_weight(self):
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            # This is an empirical formula.
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        self.linear_timesteps_weights = bsmntw_weighing
        
    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, **kwargs):
        self.sigmas, self.timesteps = self.set_timesteps_fn(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            **kwargs,
        )
        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    
    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample
    
    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target
    
    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights


class BridgeMatchScheduler(FlowMatchScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.lbm_select_num = 50
        # eg. [250, 500, 750, 1000]
        self.bridge_noise_sigma = 0.005


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, **kwargs):
        super().set_timesteps(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            training=training,
            **kwargs,
        )
        # Equally sample lbm_select_num timesteps from self.timesteps
        if not self.training:
            assert self.lbm_select_num == num_inference_steps or num_inference_steps == 1
        else:
            _, self.selected_timesteps = self.set_timesteps_fn(
                num_inference_steps=self.lbm_select_num,
                denoising_strength=denoising_strength,
                **kwargs,
            )

    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise

        # add bridge noise:
        sample = (
            sample + self.bridge_noise_sigma 
                * (sigma * (1.0 - sigma)) ** 0.5
                * torch.randn_like(sample)
        )
        return sample
    
    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)

        # if timestep is not the last step, add bridge noise
        prev_sample = (
            prev_sample + self.bridge_noise_sigma 
                * (sigma_ * (1.0 - sigma_)) ** 0.5
                * torch.randn_like(sample)
        )
        return prev_sample

class BridgeMatchSchedulerReparam(FlowMatchScheduler):
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
 
        self.lbm_select_num = 50
        self.bridge_noise_sigma = 0.005
 
    def _bridge_coeffs(self, sigma: float):
        """
        Compute bridge process coefficients at noise level sigma.
 
        Bridge interpolant:
            z_t = a_t * z0 + b_t * z1 + c_t * eps
 
        where (following LBM eq. 5 with t -> sigma in FM convention):
            a_t = 1 - sigma          (weight on clean data z0)
            b_t = sigma              (weight on source/target z1)
            c_t = bridge_sigma * sqrt(sigma * (1 - sigma))   (stochastic term)
        """
        a_t = 1.0 - sigma
        b_t = sigma
        c_t = self.bridge_noise_sigma * (sigma * (1.0 - sigma)) ** 0.5
        return a_t, b_t, c_t
 
    def _aligned_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Find the aligned FM timestep sigma_tilde such that:
            SNR_FM(sigma_tilde) == SNR_bridge_eff(sigma)
 
        Accepts and returns a 1D tensor so timestep_aligned always has
        shape (B,) matching the input timestep batch dimension.
 
        SNR_bridge_eff = (1 - sigma) / (bridge_sigma^2 * sigma)
        SNR_FM         = (1 - sigma_tilde)^2 / sigma_tilde^2
 
        Solving: sigma_tilde = 1 / (1 + sqrt(SNR_bridge_eff))
        """
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, dtype=torch.float32)
        
        # Ensure at least 1D so .expand() works downstream
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
 
        sigma = sigma.float().clamp(1e-6, 1.0 - 1e-6)
        snr_eff = (1.0 - sigma) / (self.bridge_noise_sigma ** 2 * sigma)
        sigma_tilde = 1.0 / (1.0 + snr_eff ** 0.5)
        return sigma_tilde  # shape: (B,) or (1,)
 
    def _sigma_to_aligned_sigma_tensor(self, sigma_tensor: torch.Tensor) -> torch.Tensor:
        """Vectorized version of _aligned_sigma for batched timesteps."""
        sigma = sigma_tensor.float()
        # clamp to avoid division by zero at boundaries
        sigma = sigma.clamp(1e-6, 1.0 - 1e-6)
        snr_eff = (1.0 - sigma) / (self.bridge_noise_sigma ** 2 * sigma)
        sigma_tilde = 1.0 / (1.0 + snr_eff ** 0.5)
        return sigma_tilde
 
    def set_timesteps(
        self,
        num_inference_steps=100,
        denoising_strength=1.0,
        training=False,
        **kwargs,
    ):
        super().set_timesteps(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            training=training,
            **kwargs,
        )
 
        if not self.training:
            assert (
                self.lbm_select_num == num_inference_steps or num_inference_steps == 1
            ), (
                f"At inference, num_inference_steps must equal lbm_select_num "
                f"({self.lbm_select_num}) or be 1, got {num_inference_steps}."
            )
        else:
            _, self.selected_timesteps = self.set_timesteps_fn(
                num_inference_steps=self.lbm_select_num,
                denoising_strength=denoising_strength,
                **kwargs,
            )
 
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,         # shape: (B,) or scalar tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z_tilde        : torch.Tensor  shape (B, C, F, H, W)
        timestep_aligned : torch.Tensor  shape (B,)  — 1D, same batch size as input
        """
        timestep_cpu = timestep.cpu()
 
        timestep_id = torch.argmin((self.timesteps - timestep_cpu).abs())
        sigma = self.sigmas[timestep_id]           # scalar tensor
 
        a_t = 1.0 - sigma
        c_t = self.bridge_noise_sigma * (sigma * (1.0 - sigma)) ** 0.5
        denom = (a_t ** 2 + c_t ** 2) ** 0.5
 
        eps = torch.randn_like(original_samples)
        z_tilde = (a_t / denom) * original_samples + (c_t / denom) * eps
 
        # _aligned_sigma now works on a 1D tensor.
        # Ensure timestep_cpu is at least 1D before passing.
        sigma_1d = sigma.unsqueeze(0) if sigma.dim() == 0 else sigma  # shape (1,) or (B,)
        sigma_tilde_1d = self._aligned_sigma(sigma_1d)                # shape (1,) or (B,)
 
        # Broadcast to match the batch size of original_samples.
        batch_size = original_samples.shape[0]
        timestep_aligned = sigma_tilde_1d.expand(batch_size).to(dtype=timestep.dtype)
 
        return z_tilde, timestep_aligned
 
    def step(
        self,
        model_output: torch.Tensor,     # velocity prediction v = z1 - z0  (FM convention)
        timestep: torch.Tensor,
        sample: torch.Tensor,           # z_tilde: reparameterized noisy latent
        source: torch.Tensor,           # z1: source latent (bridge prior)
        to_final: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Bridge backward step using the model's z0 prediction.
 
        The pre-trained flow matching model predicts velocity:
            v = model_output ≈ z1 - z0
 
        From this we recover z0_hat, then construct the next bridge latent
        at sigma_next directly from the bridge interpolant:
 
            z_{t'} = a_{t'} * z0_hat + b_{t'} * z1
 
        plus optional bridge noise c_{t'} * eps.
 
        This is the DDIM-style "jump to interpolant at next timestep" update,
        which is the standard approach for flow/bridge models and is consistent
        with how LBM trains (the model learns the full z0, not an incremental step).
 
        Using the bridge drift v=(z1-z_t)/(1-sigma) instead (as in the previous
        version) ignores z0_hat entirely and just slides toward z1, producing
        black/collapsed outputs because the model prediction is never used.
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
 
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = float(self.sigmas[timestep_id])
 
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_next = 0.0
        else:
            sigma_next = float(self.sigmas[timestep_id + 1])
 
        # ------------------------------------------------------------------ #
        # Step 1: recover z0_hat from the velocity prediction.               #
        #                                                                      #
        # FM velocity convention: v = z1 - z0                                 #
        # From z_tilde at sigma_tilde:                                         #
        #   z_tilde = alpha_tilde * z0 + sigma_tilde * eps                    #
        #   v_pred  = z1 - z0_hat                                             #
        #                                                                      #
        # Recover z0_hat via the standard FM x0-prediction from v:            #
        #   z0_hat = z_tilde / alpha_tilde - sigma_tilde / alpha_tilde * eps  #
        #                                                                      #
        # But since we don't have eps separately, use the velocity directly:  #
        #   z0_hat = z1 - v_pred    (when model predicts v = z1 - z0)         #
        #                                                                      #
        # If your model predicts eps instead, swap to:                        #
        #   z0_hat = (z_tilde - sigma_tilde * model_output) / alpha_tilde     #
        # ------------------------------------------------------------------ #
        sigma_tilde = float(self._aligned_sigma(torch.tensor(sigma)).squeeze())
        alpha_tilde = 1.0 - sigma_tilde
 
        # Velocity prediction: v ≈ z1 - z0  =>  z0_hat = z1 - v
        # This is the standard WanVideo / flow matching convention.
        z0_hat = source - model_output
 
        # ------------------------------------------------------------------ #
        # Step 2: construct next bridge latent from interpolant at sigma_next.#
        #                                                                      #
        # Bridge interpolant (LBM eq. 5, sigma=0 is clean, sigma=1 is source):#
        #   z_t = a_t * z0 + b_t * z1    (ignoring noise term for mean)       #
        # where a_t = 1 - sigma, b_t = sigma.                                  #
        #                                                                      #
        # At sigma_next:                                                       #
        #   z_{t'} = (1 - sigma_next) * z0_hat + sigma_next * z1              #
        #                                                                      #
        # At sigma_next=0 (final step): z_{t'} = z0_hat  (clean prediction). #
        # ------------------------------------------------------------------ #
        a_next = 1.0 - sigma_next
        b_next = sigma_next
 
        z_next = a_next * z0_hat + b_next * source
 
        # ------------------------------------------------------------------ #
        # Step 3: optionally add bridge stochastic noise at sigma_next.       #
        #                                                                      #
        # Only inject noise for intermediate steps. At the final step         #
        # (sigma_next=0) we return the clean z0_hat directly.                 #
        # ------------------------------------------------------------------ #
        if sigma_next > 0.0 and not to_final:
            _, _, c_next = self._bridge_coeffs(sigma_next)
            z_next = z_next + c_next * torch.randn_like(z_next)
 
        return z_next

    def reparam_for_inference(
        self,
        z_t_bridge: torch.Tensor,
        source: torch.Tensor,
        timestep: torch.Tensor,         # shape: (B,) or (1,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Deterministic reparameterization for inference.
 
        Returns
        -------
        z_tilde          : torch.Tensor  shape (B, C, F, H, W)
        timestep_aligned : torch.Tensor  shape (B,)  — 1D
        """
        timestep_cpu = timestep.cpu()
 
        timestep_id = torch.argmin((self.timesteps - timestep_cpu).abs())
        sigma = self.sigmas[timestep_id]           # scalar tensor
 
        a_t = 1.0 - sigma
        b_t = sigma
        c_t = self.bridge_noise_sigma * (sigma * (1.0 - sigma)) ** 0.5
        denom = (a_t ** 2 + c_t ** 2) ** 0.5
 
        z_tilde = (z_t_bridge - b_t * source) / denom
 
        sigma_1d = sigma.unsqueeze(0) if sigma.dim() == 0 else sigma  # (1,) or (B,)
        sigma_tilde_1d = self._aligned_sigma(sigma_1d)                # (1,) or (B,)
 
        batch_size = z_t_bridge.shape[0]
        timestep_aligned = sigma_tilde_1d.expand(batch_size).to(dtype=timestep.dtype)
 
        return z_tilde, timestep_aligned