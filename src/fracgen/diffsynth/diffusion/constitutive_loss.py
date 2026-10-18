"""
Constitutive consistency loss for physics-informed video generation.

Ties predicted stress, strain, and damage tiles together via a damage-gated linear
elastic (Lamé) relation, enforced entirely in latent space -- no decoding step.

L_const = ‖ x̂₀^σ − (1 − sigmoid(x̂₀^d)) · (λ·A_vol(x̂₀^ε) + 2μ·A_dev(x̂₀^ε)) ‖²

Runs only when combined_latent_tiling is active and stress/strain/damage are all in
the active phys_keys. Ground-truth supervision stays in FlowMatchSFTLoss (L_fm);
L_const only couples the predicted modalities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List


# Reference scale for Young's modulus to keep Lamé params O(1).
# Typical E range in this dataset: 1e3 – 1e9 Pa; 1e6 puts mid-range at ~1.
_E_SCALE = 1e6


class ConstitutiveLoss(nn.Module):
    """
    Learnable latent-space constitutive consistency loss.

    New parameters:
      A_vol  (Conv3d C→C, 1×1×1): maps strain latent to volumetric stress component
      A_dev  (Conv3d C→C, 1×1×1): maps strain latent to deviatoric stress component

    Both initialized to zero so training starts without constitutive pressure;
    the model learns the appropriate latent-space operator from data.
    """

    def __init__(self, C_latent: int = 16):
        super().__init__()
        self.A_vol = nn.Conv3d(C_latent, C_latent, kernel_size=1, bias=False)
        self.A_dev = nn.Conv3d(C_latent, C_latent, kernel_size=1, bias=False)
        nn.init.zeros_(self.A_vol.weight)
        nn.init.zeros_(self.A_dev.weight)

    def forward(
        self,
        x0_pred: Tensor,
        tile_order: List[str],
        young_modulus: Tensor,
        poisson_ratio: Tensor,
        timestep: Tensor,
        gamma: float = 2.0,
    ) -> Tensor:
        """
        Args:
            x0_pred:       (B, C, T, K*H', W') predicted clean tiled latent
            tile_order:    modality names in H-stacking order (e.g. ["video", "stress_video", ...])
            young_modulus: (B,) Young's modulus per sample (raw Pa)
            poisson_ratio: (B,) Poisson's ratio per sample (dimensionless)
            timestep:      scalar or (B,) in [0, 1000]; t=0 clean, t=1000 noisy
            gamma:         exponent for timestep weighting w(t) = (1-t/1000)^gamma

        Returns:
            Scalar constitutive loss (timestep-weighted, unscaled by lambda_const).
        """
        required = {"stress_video", "strain_video", "damage_video"}
        if not required.issubset(set(tile_order)):
            return x0_pred.new_zeros(())

        from diffsynth.utils.latent_tiling import slice_tile
        num_tiles = len(tile_order)
        H_tile = x0_pred.shape[3] // num_tiles

        x_stress = slice_tile(x0_pred, "stress_video", tile_order, H_tile).float()
        x_strain = slice_tile(x0_pred, "strain_video", tile_order, H_tile).float()
        x_damage = slice_tile(x0_pred, "damage_video", tile_order, H_tile).float()

        # Lamé parameters from per-sample material properties.
        # Divide E by _E_SCALE so the scalars are O(1) and the identity initialization
        # of A_vol/A_dev gives a physically reasonable starting point.
        # Coerce scalar inputs (Python float, numpy float64, 0-dim tensor) to (B, 1,1,1,1).
        dev = x0_pred.device
        if not isinstance(young_modulus, torch.Tensor):
            young_modulus = torch.tensor([young_modulus], dtype=torch.float32, device=dev)
        if not isinstance(poisson_ratio, torch.Tensor):
            poisson_ratio = torch.tensor([poisson_ratio], dtype=torch.float32, device=dev)
        E  = young_modulus.float().to(dev).view(-1, 1, 1, 1, 1) / _E_SCALE
        nu = poisson_ratio.float().to(dev).clamp(max=0.49).view(-1, 1, 1, 1, 1)
        lame_lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
        lame_mu     = E / (2 * (1 + nu))

        # Elastic prediction in latent space.
        elastic = lame_lambda * self.A_vol(x_strain) + 2 * lame_mu * self.A_dev(x_strain)

        # Damage gate: sigmoid applied directly to the C-channel damage latent.
        # Keeps C channels (same dim as stress/strain) -- no dimensionality reduction
        # needed because the physical "scalar damage" interpretation doesn't survive the
        # VAE encoder. Per-channel gating is more expressive and parameter-free.
        d_gate = torch.sigmoid(x_damage)
        elastic_gated = (1 - d_gate) * elastic

        L_const = F.mse_loss(x_stress, elastic_gated)

        # Timestep weighting: physics loss is off at high noise, on near clean.
        t_norm = (timestep.float() / 1000.0).clamp(0, 1).mean()
        w_t = (1 - t_norm) ** gamma

        return w_t * L_const
