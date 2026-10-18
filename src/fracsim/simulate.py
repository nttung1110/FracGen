import sys
import os
import argparse
import math
import cv2
import torch
import numpy as np
import subprocess
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "gaussian-splatting"))

# Gaussian splatting dependencies
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
import warp as wp
import taichi as ti

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
from my_utils import get_gaussian_blob, ffmpeg_bin

def resize_map_if_needed(np_map, target_h, target_w):
    """Resize a 2D or 3D numpy map to (target_h, target_w) if dimensions differ."""
    if target_h is None or target_w is None:
        return np_map
    if np_map.shape[0] == target_h and np_map.shape[1] == target_w:
        return np_map
    if np_map.ndim == 2:
        return cv2.resize(np_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    else:  # e.g. [H, W, 2] flow map
        return cv2.resize(np_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

def project_world_to_pixel_and_ndc(world_xyz, cam):
    pts = torch.cat([world_xyz, torch.ones_like(world_xyz[:, :1])], dim=-1)  # [N,4]
    clip = (cam.full_proj_transform.T.unsqueeze(0) @ pts.unsqueeze(-1)).squeeze(-1)  # [N,4]
    view = (cam.world_view_transform.T.unsqueeze(0) @ pts.unsqueeze(-1)).squeeze(-1)  # [N,4]

    w = clip[:, 3:4]
    w_safe = torch.where(w.abs() < 1e-8, torch.full_like(w, 1e-8), w)  # keep sign
    ndc = clip[:, :3] / w_safe

    W = float(cam.image_width)
    H = float(cam.image_height)
    u = ((ndc[:, 0] + 1.0) * 0.5) * (W - 1.0)
    # Match rasterizer ndc2Pix() convention used by diff-gaussian-rasterization.
    v = ((ndc[:, 1] + 1.0) * 0.5) * (H - 1.0)

    # Match rasterizer in_frustum() front test (view-space z > 0.2).
    in_front = view[:, 2] > 0.2
    in_frame = in_front & (u >= 0) & (u <= W - 1.0) & (v >= 0) & (v <= H - 1.0)
    return u, v, ndc[:, 2], in_frame

def mpm_pos_to_world(pos_mpm):
    # same transform used for rendered particle positions
    return apply_inverse_rotations(
        undotransform2origin(undoshift2center111(pos_mpm), scale_origin, original_mean_pos),
        rotation_matrices,
    )

def mpm_vel_to_world_vel(v_mpm):
    # inverse of world_vel_to_mpm_vel: v_mpm = scale_origin * Rot(v_world)
    return apply_inverse_rotations(v_mpm / scale_origin, rotation_matrices)

class PipelineParamsNoparse:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(checkpt_dir, f"iteration_{iteration}", "point_cloud.ply")

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians

def validate_time_params(frame_dt, substep_dt):
    if frame_dt <= 0 or substep_dt <= 0:
        raise ValueError(
            f"Invalid time parameters: frame_dt={frame_dt}, substep_dt={substep_dt}. Both must be > 0."
        )
    steps_exact = frame_dt / substep_dt
    step_per_frame = int(round(steps_exact))
    if step_per_frame < 1:
        raise ValueError(
            f"Invalid integration setup: frame_dt / substep_dt = {steps_exact}, which yields < 1 step per frame."
        )
    if not math.isclose(steps_exact, step_per_frame, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"frame_dt / substep_dt must be an integer to avoid time drift. "
            f"Got frame_dt={frame_dt}, substep_dt={substep_dt}, ratio={steps_exact}."
        )
    return step_per_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./output_simulation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--render_img", action="store_true")
    parser.add_argument("--compile_video", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--physics_chunk_size",
        type=int,
        default=32768,
        help="Particles per chunk for physics-map computation. Lower values reduce peak CUDA memory.",
    )
    parser.add_argument(
        "--render_width",
        type=int,
        default=None,
        help="Output render width in pixels. If only one dimension is set, aspect ratio is preserved.",
    )
    parser.add_argument(
        "--render_height",
        type=int,
        default=None,
        help="Output render height in pixels. If only one dimension is set, aspect ratio is preserved.",
    )

    args = parser.parse_args()

    args.render_phys_map = True  # FracSim always renders the physics maps alongside RGB

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 3D Gaussian Splatting and the MPM simulation pipeline.")
    # device = "cuda:0"
    device = f"cuda:{torch.cuda.current_device()}"

    # Initialize GPU backends only after CUDA availability is verified.
    wp.init()
    wp.config.verify_cuda = True
    ti.init(arch=ti.cuda, device_memory_GB=8.0)

    if not os.path.exists(args.model_path):
        raise AssertionError("Model path does not exist!")
    if not os.path.exists(args.config):
        raise AssertionError("Scene config does not exist!")

    os.makedirs(args.output_path, exist_ok=True)

    path_img_frames_folder = os.path.join(args.output_path, "img_frames")
    path_force_map_folder = os.path.join(args.output_path, "2d_force_map")
    path_velocity_map_folder = os.path.join(args.output_path, "2d_vel_map")
    path_stress_map_folder = os.path.join(args.output_path, "2d_stress_map")
    path_strain_map_folder = os.path.join(args.output_path, "2d_strain_map")
    path_damage_map_folder = os.path.join(args.output_path, "2d_damage_map")

    os.makedirs(path_img_frames_folder, exist_ok=True)
    os.makedirs(path_force_map_folder, exist_ok=True)
    os.makedirs(path_velocity_map_folder, exist_ok=True)
    os.makedirs(path_stress_map_folder, exist_ok=True)
    os.makedirs(path_strain_map_folder, exist_ok=True)
    os.makedirs(path_damage_map_folder, exist_ok=True)

    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.config)

    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    )

    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]

    if args.debug:
        os.makedirs("./log", exist_ok=True)
        particle_position_tensor_to_ply(init_pos, "./log/init_particles.ply")

    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    if args.debug:
        particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")

    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (None, None, None, None)
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool, device=rotated_pos.device)
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(
        rotated_pos, preprocessing_params["scale"]
    )

    if args.debug:
        particle_position_tensor_to_ply(transformed_pos, "./log/transformed_particles_before_shift.ply")

    transformed_pos = shift2center111(transformed_pos)

    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(transformed_pos, "./log/transformed_particles.ply")

    gs_num = transformed_pos.shape[0]
    filling_params = preprocessing_params["particle_filling"]

    if filling_params is not None:
        print("Filling internal particles...")
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)

        filled_particle_count = mpm_init_pos.shape[0]

        if args.debug:
            particle_position_tensor_to_ply(mpm_init_pos, "./log/filled_particles.ply")
    else:
        mpm_init_pos = transformed_pos.to(device=device)
        filled_particle_count = gs_num

    print("Initializing MPM solver and setting up boundary conditions...")
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)

    if filling_params is not None and filling_params["visualize"]:
        shs, opacity, filled_cov = init_filled_particles(
            mpm_init_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            mpm_init_pos[gs_num:filled_particle_count],
            fill_cov_scale=filling_params["fill_cov_scale"],
        )
        shs = shs.to(device=device)
        opacity = opacity.to(device=device)
        filled_cov = filled_cov.to(device=device)

        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device, dtype=filled_cov.dtype)
        if filled_cov.shape[0] == filled_particle_count:
            mpm_init_cov[:filled_particle_count] = filled_cov
        elif filled_cov.shape[0] == mpm_init_pos.shape[0]:
            mpm_init_cov = filled_cov
        else:
            raise ValueError(
                f"init_filled_particles returned unexpected cov size {filled_cov.shape[0]}, "
                f"expected {filled_particle_count} or {mpm_init_pos.shape[0]}."
            )

        if shs.shape[0] < filled_particle_count or opacity.shape[0] < filled_particle_count:
            raise ValueError("init_filled_particles returned fewer SH/opacity entries than filled particles.")

        shs = shs[:filled_particle_count]
        opacity = opacity[:filled_particle_count]
        gs_num = filled_particle_count
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity

    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    mpm_solver.set_parameters_dict(material_params)

    mpm_space_viewpoint_center = torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).to(device)
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"]).reshape((1, 3)).to(device)
    )
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )

    set_boundary_conditions(mpm_solver, bc_params, time_params)
    mpm_solver.finalize_mu_lam()

    if args.output_ply or args.output_h5:
        directory_to_save = os.path.join(args.output_path, "simulation_ply")
        os.makedirs(directory_to_save, exist_ok=True)
        save_data_at_frame(
            mpm_solver,
            directory_to_save,
            0,
            save_to_ply=args.output_ply,
            save_to_h5=args.output_h5,
        )

    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = validate_time_params(frame_dt, substep_dt)

    opacity_render = opacity
    shs_render = shs
    height = None
    width = None

    for frame in tqdm(range(frame_num)):
        current_camera = get_camera_view(
            model_path,
            default_camera_index=camera_params["default_camera_index"],
            center_view_world_space=viewpoint_center_worldspace,
            observant_coordinates=observant_coordinates,
            show_hint=camera_params["show_hint"],
            init_azimuthm=camera_params["init_azimuthm"],
            init_elevation=camera_params["init_elevation"],
            init_radius=camera_params["init_radius"],
            move_camera=camera_params["move_camera"],
            current_frame=frame,
            delta_a=camera_params["delta_a"],
            delta_e=camera_params["delta_e"],
            delta_r=camera_params["delta_r"],
        )

        rasterize = initialize_resterize(current_camera, gaussians, pipeline, background)

        # define once outside the frame loop
        background_phys = torch.zeros(3, dtype=torch.float32, device=device)  # [0,0,0] -> this is important as we want rasterizer to fill 0 zero for background region
        rasterize_phys = initialize_resterize(current_camera, gaussians, pipeline, background_phys)

        for _ in range(step_per_frame):
            mpm_solver.p2g2p(frame, substep_dt, device=device)
        
        if args.output_ply or args.output_h5:
            save_data_at_frame(
                mpm_solver,
                directory_to_save,
                frame + 1,
                save_to_ply=args.output_ply,
                save_to_h5=args.output_h5,
            )
        
        particle_x = mpm_solver.export_particle_x_to_torch().to(device)
        pos = particle_x[:gs_num]

        cov3D = mpm_solver.export_particle_cov_to_torch().view(-1, 6)[:gs_num].to(device)
        rot = mpm_solver.export_particle_R_to_torch().view(-1, 3, 3)[:gs_num].to(device)

        #### Rendering Image
        pos = apply_inverse_rotations(
            undotransform2origin(
                undoshift2center111(pos), scale_origin, original_mean_pos
            ),
            rotation_matrices,
        )
        cov3D = cov3D / (scale_origin * scale_origin)
        cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
        opacity = opacity_render
        shs = shs_render
        if preprocessing_params["sim_area"] is not None:
            pos = torch.cat([pos, unselected_pos], dim=0)
            cov3D = torch.cat([cov3D, unselected_cov], dim=0)
            opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
            shs = torch.cat([shs_render, unselected_shs], dim=0)

        colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
        rendering, raddi = rasterize(
            means3D=pos,
            means2D=init_screen_points,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=None,
            rotations=None,
            cov3D_precomp=cov3D,
        )
        cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        # --- resize if requested ---
        if args.render_width is not None or args.render_height is not None:
            orig_h, orig_w = cv2_img.shape[:2]
            if args.render_width is not None and args.render_height is not None:
                target_w, target_h = args.render_width, args.render_height
            elif args.render_width is not None:
                target_w = args.render_width
                target_h = int(round(orig_h * target_w / orig_w))
            else:
                target_h = args.render_height
                target_w = int(round(orig_w * target_h / orig_h))
            cv2_img = cv2.resize(cv2_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if height is None or width is None:
            height = cv2_img.shape[0] // 2 * 2
            width  = cv2_img.shape[1] // 2 * 2
        assert args.output_path is not None
        cv2.imwrite(
            os.path.join(path_img_frames_folder, f"{frame}.png".rjust(8, "0")),
            255 * cv2_img,
        )

        if args.render_phys_map:
            ## Compute phys map
            vel = mpm_solver.export_particle_v_to_torch()[:gs_num].to(device)          # [N, 3]

            vm_stress = (
                mpm_solver.export_vm_scale_particle_stress_to_torch()[:gs_num]                 # [N]
                .to(device).unsqueeze(1).repeat(1, 3)                                  # [N, 3]
            )

            strain = (
                mpm_solver.export_particle_strain_to_torch()[:gs_num]                  # [N]
                .to(device).unsqueeze(1).repeat(1, 3)                                  # [N, 3]
            )
            damage = (
                mpm_solver.export_particle_damage_to_torch()[:gs_num]                  # [N]
                .to(device).unsqueeze(1).repeat(1, 3)                                  # [N, 3]
            )

            pos_gs  = pos[:gs_num]    if preprocessing_params["sim_area"] is not None else pos
            cov3D_gs = cov3D[:gs_num] if preprocessing_params["sim_area"] is not None else cov3D

            vel = apply_inverse_rotations(vel / scale_origin, rotation_matrices)

            vm_stress_rendering, _ = rasterize_phys(
                means3D=pos_gs,
                means2D=init_screen_points,
                shs=None,
                colors_precomp=vm_stress,
                opacities=opacity_render,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D_gs,
            )

            strain_rendering, _ = rasterize_phys(
                means3D=pos_gs,
                means2D=init_screen_points,
                shs=None,
                colors_precomp=strain,
                opacities=opacity_render,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D_gs,
            )

            damage_rendering, _ = rasterize_phys(
                means3D=pos_gs,
                means2D=init_screen_points,
                shs=None,
                colors_precomp=damage,
                opacities=opacity_render,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D_gs,
            )

            # Process enforce_particle_translation (forces) point/velocity in MPM coordinates
            # for later force map saving
            enforce_bcs = [bc for bc in bc_params if bc.get("type") == "enforce_particle_translation"]
            dt = float(time_params["frame_dt"])
            
            # ── Force map ──────────────────────────────────────────────
            np_multi_force_point_map = None
            for bc in enforce_bcs:
                p_mpm = torch.tensor(bc["point"], dtype=torch.float32, device=device).view(1, 3)
                v_mpm = torch.tensor(bc["velocity"], dtype=torch.float32, device=device).view(1, 3)
                p_world = mpm_pos_to_world(p_mpm)
                v_world = mpm_vel_to_world_vel(v_mpm)

                u0, v0, z0, in0 = project_world_to_pixel_and_ndc(p_world, current_camera)

                t = frame * dt
                start_t = float(bc.get("start_time", -1e30))
                end_t   = float(bc.get("end_time",   1e30))
                if t < start_t or t > end_t or (not bool(in0.item())):
                    continue

                p_world_next = p_world + v_world * t
                u1, v1, z1, in1 = project_world_to_pixel_and_ndc(p_world_next, current_camera)
                u1, v1 = int(u1.detach()), int(v1.detach())

                # blob built at NATIVE rasterizer resolution, resize after accumulation
                blob = get_gaussian_blob(
                    [u1, v1],
                    shape=(1, rendering.shape[1], rendering.shape[2])  # native res
                ).numpy()

                np_multi_force_point_map = blob if np_multi_force_point_map is None \
                                           else np_multi_force_point_map + blob

            if np_multi_force_point_map is None:
                # No BC active this frame (e.g. force stopped mid-video)
                np_multi_force_point_map = np.zeros(
                    (1, rendering.shape[1], rendering.shape[2]), dtype=np.float32
                )
            else:
                np_multi_force_point_map /= len(enforce_bcs)
            # [1,H,W] -> [H,W] -> resize -> [1,H',W']
            force_map_2d = np_multi_force_point_map[0]
            force_map_2d = resize_map_if_needed(force_map_2d, height, width)

            np.savez_compressed(os.path.join(path_force_map_folder, f"{frame:04d}.npz"),
                    data=force_map_2d[np.newaxis])

            # ── Velocity / flow map (rasterizer-based for smooth Gaussian footprint) ──
            # Each particle's 2D pixel displacement is rendered through rasterize_phys
            # so it gets the same Gaussian footprint as stress/strain/damage.
            # The old scatter approach (index_add_ to a single integer pixel per particle)
            # produced isolated dots; this gives area coverage matching the visual render.
            ix1_n, iy1_n, _, in1 = project_world_to_pixel_and_ndc(pos_gs, current_camera)
            pos_next_gs = pos_gs + vel * frame_dt
            ix2_n, iy2_n, _, in2 = project_world_to_pixel_and_ndc(pos_next_gs, current_camera)

            native_w = float(rendering.shape[2])
            native_h = float(rendering.shape[1])
            scale_x = width / native_w
            scale_y = height / native_h

            # Particles not visible in both frames are excluded from flow rendering.
            # We zero their opacity rather than zeroing their (du,dv): if we kept
            # full opacity but set color=(0,0), the rasterizer would alpha-composite
            # that zero into neighboring pixels and pull their velocity estimate
            # toward zero incorrectly.
            valid_mask = (in1 & in2).float()
            du_pp = (ix2_n - ix1_n) * scale_x
            dv_pp = (iy2_n - iy1_n) * scale_y
            flow_precomp = torch.stack([du_pp, dv_pp, torch.zeros_like(du_pp)], dim=1)
            flow_opacity = opacity_render * valid_mask.unsqueeze(1)

            # Render (du, dv, 0) as per-particle colors; rasterizer alpha-composites
            # over each particle's Gaussian footprint → smooth 2D flow field
            flow_rendering, _ = rasterize_phys(
                means3D=pos_gs,
                means2D=init_screen_points,
                shs=None,
                colors_precomp=flow_precomp,
                opacities=flow_opacity,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D_gs,
            )
            # [3, H_native, W_native] → channels 0,1 = (du, dv) → resize to target
            flow_uv = flow_rendering[:2].permute(1, 2, 0).detach().cpu().numpy()
            flow_uv = resize_map_if_needed(flow_uv, height, width)
            np.savez_compressed(os.path.join(path_velocity_map_folder, f"{frame:04d}.npz"),
                    data=flow_uv)

            # ── Scalar physics maps (stress / strain / damage) ─
            # These come from the rasterizer at NATIVE resolution → resize to target
            np_stress_map = vm_stress_rendering.permute(1,2,0).detach().cpu().numpy().mean(axis=-1)
            np_stress_map = resize_map_if_needed(np_stress_map, height, width)
            np.savez_compressed(os.path.join(path_stress_map_folder, f"{frame:04d}.npz"),
                    data=np_stress_map)

            np_strain_map = strain_rendering.permute(1,2,0).detach().cpu().numpy().mean(axis=-1)
            np_strain_map = resize_map_if_needed(np_strain_map, height, width)
            np.savez_compressed(os.path.join(path_strain_map_folder, f"{frame:04d}.npz"),
                    data=np_strain_map)

            np_damage_map = damage_rendering.permute(1,2,0).detach().cpu().numpy().mean(axis=-1)
            np_damage_map = resize_map_if_needed(np_damage_map, height, width)
            np.savez_compressed(os.path.join(path_damage_map_folder, f"{frame:04d}.npz"),
                    data=np_damage_map)

    if args.render_img and args.compile_video:
        fps = min(60, max(1, int(round(1.0 / frame_dt))))
        input_pattern = os.path.join(path_img_frames_folder, "%04d.png")
        output_file = os.path.join(args.output_path, "output.mp4")

        if width is None or height is None:
            print("No rendered frames were produced; skipping video compilation.")
            sys.exit(0)

        print(f"Compiling video at {fps} FPS...")
        ffmpeg_cmd = [
            ffmpeg_bin(),
            "-y",
            "-framerate",
            str(fps),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-s",
            f"{width}x{height}",
            output_file,
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Video successfully saved to {output_file}")
        except FileNotFoundError:
            print("FFmpeg not found! Please install ffmpeg and ensure it is in your system PATH.")
        except subprocess.CalledProcessError as e:
            print("FFmpeg compilation failed.")
            print(e.stderr.decode("utf-8"))
