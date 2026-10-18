import torch
from .wan_video_dit import DiTBlock, WanModel, Head, sinusoidal_embedding_1d
import torch.nn as nn
from typing import Tuple, Optional
from einops import rearrange


# ================================= New added =================================
class DiTBlockJointGen_Sub(DiTBlock):
    def __init__(
        self,
        has_image_input,
        dim,
        num_heads,
        ffn_dim,
        eps=1e-6,
        block_id=None,
        enable_after_proj=False
    ):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        self.enable_after_proj = enable_after_proj
        if enable_after_proj:
            self.after_proj_vace_block = torch.nn.Linear(self.dim, self.dim)
            torch.nn.init.zeros_(self.after_proj_vace_block.weight)
            torch.nn.init.zeros_(self.after_proj_vace_block.bias)

    def forward(self, c, main_hint, scale=1.0, **kwargs):
        if main_hint is not None:
            c = c + main_hint * scale

        c = super().forward(c, **kwargs)

        if self.enable_after_proj:
            vace_hint = self.after_proj_vace_block(c)
        else:
            vace_hint = None

        return c, vace_hint
    

class DiTBlockJointGen(DiTBlock):
    def __init__(
        self,
        has_image_input,
        dim,
        num_heads,
        ffn_dim,
        eps=1e-6,
        block_id=None,
        enable_after_proj=False
    ):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        self.enable_after_proj = enable_after_proj
        if enable_after_proj:
            # Produce "main -> geo" hints from main features.
            self.after_proj_main_block = torch.nn.Linear(self.dim, self.dim)
            torch.nn.init.zeros_(self.after_proj_main_block.weight)
            torch.nn.init.zeros_(self.after_proj_main_block.bias)

    def forward(self, x, side_hint, scale=1.0, **kwargs):
        if side_hint is not None:
            x = x + side_hint * scale

        x = super().forward(x, **kwargs)

        if self.enable_after_proj:
            main_hint = self.after_proj_main_block(x)
        else:
            main_hint = None

        return x, main_hint
    

class JointWanTransformer3DModel(WanModel):
    def __init__(
        self,
        vace_interval: int,
        vace_in_dim: int,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            in_dim=in_dim,
            ffn_dim=ffn_dim,
            out_dim=out_dim,
            text_dim=text_dim,
            freq_dim=freq_dim,
            eps=eps,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
            has_image_input=has_image_input,
            has_image_pos_emb=has_image_pos_emb,
            has_ref_conv=has_ref_conv,
            add_control_adapter=add_control_adapter,
            in_dim_control_adapter=in_dim_control_adapter,
            seperated_timestep=seperated_timestep,
            require_vae_embedding=require_vae_embedding,
            require_clip_embedding=require_clip_embedding,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        self.vace_interval = vace_interval
        self.vace_layers = [i for i in range(0, num_layers, vace_interval)]
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim

        assert 0 in self.vace_layers
        # {0, 2, 4, ..} -> {block_id: vace_block_list_index}
        # (geo blocks are stored compactly in `self.geo_blocks`)
        self.vace_layers_mapping = {i: i for i in self.vace_layers}
        self.residual_layers_in_main = [i-1 for i in self.vace_layers if i - 1 >= 0]

        # blocks
        self.blocks = nn.ModuleList([
            DiTBlockJointGen(has_image_input, dim, num_heads, ffn_dim, eps,
                            block_id=i,
                            enable_after_proj=True if i in self.residual_layers_in_main else False)
            for i in range(num_layers)
        ])
        # geo blocks
        self.geo_blocks = nn.ModuleList([
            DiTBlockJointGen_Sub(has_image_input, dim, num_heads, ffn_dim, eps,
                                block_id=i,
                                enable_after_proj=True if idx+1 != len(self.vace_layers) else False)
            for idx, i in enumerate(self.vace_layers)
        ])
        # self.motion_blocks = nn.ModuleList([
        #     DiTBlockJointGen_Sub(has_image_input, dim, num_heads, ffn_dim, eps,
        #                         block_id=i,
        #                         enable_after_proj=True if idx+1 != len(self.vace_layers) else False)
        #     for idx, i in enumerate(self.vace_layers)
        # ])

        # geo input patch embeddings
        self.geo_patch_embedding = nn.Conv3d(
            vace_in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        # geo output head
        self.geo_head = Head(dim, out_dim*2, patch_size, eps)


    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def patchify_geo(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.geo_patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )
# ================================= End added =================================
    

def model_fn_wan_jointGen(
    dit: JointWanTransformer3DModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    x_geo: Optional[torch.Tensor] = None,
    y_geo: Optional[torch.Tensor] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    use_unified_sequence_parallel: bool = False,
    **kwargs,
):
    """
    JointGen model_fn in the same *format* as `model_fn_wan_video` (see `diffsynth/pipelines/wan_video.py`),
    but producing a dual-stream output: `[rgb_pred, geo_pred]`.
    """
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep (kept consistent with `model_fn_wan_video`)
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))


    # # Motion Controller
    # if motion_bucket_id is not None and motion_controller is not None:
    #     t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)
    x = latents
    # Merged cfg (kept consistent with `model_fn_wan_video`)
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding (kept consistent with `model_fn_wan_video`)
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    # Camera control
    x = dit.patchify(x, control_camera_latents_input)

    # # Animate
    # if pose_latents is not None and face_pixel_values is not None:
    #     x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)
    
    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    # # Reference image
    # if reference_latents is not None:
    #     if len(reference_latents.shape) == 5:
    #         reference_latents = reference_latents[:, :, 0]
    #     reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
    #     x = torch.concat([reference_latents, x], dim=1)
    #     f += 1

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # Geo branch (optional): mirror the i2v-style conditioning (concat on channel dim)
    geo_state = None
    if x_geo is not None and y_geo is not None:
        geo_state = torch.cat([x_geo, y_geo], dim=1)
        geo_state = dit.patchify_geo(geo_state, control_camera_latents_input)
        gf, gh, gw = geo_state.shape[2:]
        if (gf, gh, gw) != (f, h, w):
            raise ValueError(
                f"Geo grid {(gf, gh, gw)} must match main grid {(f, h, w)} "
                f"(patch_size={dit.patch_size})."
            )
        geo_state = rearrange(geo_state, "b c f h w -> b (f h w) c").contiguous()

    # Unified Sequence Parallel (match `model_fn_wan_video` behavior)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [
                torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1] - chunk.shape[1]), value=0)
                for chunk in chunks
            ]
            x = chunks[get_sequence_parallel_rank()]
            if geo_state is not None:
                geo_chunks = torch.chunk(geo_state, get_sequence_parallel_world_size(), dim=1)
                geo_chunks = [
                    torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1] - chunk.shape[1]), value=0)
                    for chunk in geo_chunks
                ]
                geo_state = geo_chunks[get_sequence_parallel_rank()]

    #  Geo branch blocks (main <-> geo exchange)
    def create_custom_forward(module, **static_kwargs):
        def custom_forward(*inputs):
            return module(*inputs, **static_kwargs)
        return custom_forward

    static_kwargs = dict(context=context, t_mod=t_mod, freqs=freqs)
    staged_main_hints = [False] * len(dit.blocks)
    staged_geo_hints = [False] * len(dit.blocks)

    for block_id, block in enumerate(dit.blocks):
        # Geo -> Main hint arrives delayed by `vace_interval`
        geo_hint_idx = block_id - dit.vace_interval
        if geo_hint_idx >= 0:
            curr_geo_hint = staged_geo_hints[geo_hint_idx]
            assert curr_geo_hint is not False
        else:
            curr_geo_hint = None

        # Main block: (x, main_hint)
        if use_gradient_checkpointing:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x, main_hint = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block, **static_kwargs),
                        x,
                        curr_geo_hint,
                        use_reentrant=False,
                    )
            else:
                x, main_hint = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, **static_kwargs),
                    x,
                    curr_geo_hint,
                    use_reentrant=False,
                )
        else:
            x, main_hint = block(x, curr_geo_hint, **static_kwargs)
        staged_main_hints[block_id] = main_hint

        # Paired geo block
        if geo_state is not None and block_id in dit.vace_layers:
            geo_block_idx = dit.vace_layers_mapping[block_id]

            main_hint_idx = block_id - 1
            if main_hint_idx >= 0:
                curr_main_hint = staged_main_hints[main_hint_idx]
                assert curr_main_hint is not False and curr_main_hint is not None
            else:
                curr_main_hint = None

            geo_block = dit.geo_blocks[geo_block_idx]
            if use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        geo_state, geo_hint = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(geo_block, **static_kwargs),
                            geo_state,
                            curr_main_hint,
                            use_reentrant=False,
                        )
                else:
                    geo_state, geo_hint = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(geo_block, **static_kwargs),
                        geo_state,
                        curr_main_hint,
                        use_reentrant=False,
                    )
            else:
                geo_state, geo_hint = geo_block(geo_state, curr_main_hint, **static_kwargs)
            staged_geo_hints[block_id] = geo_hint
        else:
            staged_geo_hints[block_id] = None

    # heads -> unpatchify
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    x = dit.unpatchify(x, (f, h, w))

    if geo_state is not None:
        geo_state = dit.geo_head(geo_state, t)
        if use_unified_sequence_parallel:
            if dist.is_initialized() and dist.get_world_size() > 1:
                geo_state = get_sp_group().all_gather(geo_state, dim=1)
                geo_state = geo_state[:, :-pad_shape] if pad_shape > 0 else geo_state
        geo_state = dit.unpatchify(geo_state, (f, h, w))

    return [x, geo_state]

