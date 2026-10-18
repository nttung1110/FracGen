import warp as wp
from warp_utils import *
import numpy as np
import math

@wp.func
def clamp01(x: float):
    return wp.max(0.0, wp.min(1.0, x))


@wp.func
def tension_energy_from_F(F: wp.mat33, mu: float, lam: float):
    """
    Simple tension-only energy proxy using principal log strains:
      eps_i = log(sig_i)
      eps_i+ = max(eps_i, 0)
      psi+ = mu * sum(eps_i+^2) + 0.5 * lam * (sum eps_i+)^2
    """
    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F, U, sig, V)

    # prevent log(0)
    s0 = wp.max(sig[0], 1e-6)
    s1 = wp.max(sig[1], 1e-6)
    s2 = wp.max(sig[2], 1e-6)

    e0 = wp.log(s0)
    e1 = wp.log(s1)
    e2 = wp.log(s2)

    ep0 = wp.max(e0, 0.0)
    ep1 = wp.max(e1, 0.0)
    ep2 = wp.max(e2, 0.0)

    trp = ep0 + ep1 + ep2
    psi = mu * (ep0 * ep0 + ep1 * ep1 + ep2 * ep2) + 0.5 * lam * (trp * trp)
    return psi

# compute stress from F
@wp.func
def kirchoff_stress_FCR(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    R = U * wp.transpose(V)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return 2.0 * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - 1.0)


@wp.func
def kirchoff_stress_neoHookean(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, sig: wp.vec3, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    b = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    b_hat = b - wp.vec3(
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
    )
    tau = mu * J ** (-2.0 / 3.0) * b_hat + lam / 2.0 * (J * J - 1.0) * wp.vec3(
        1.0, 1.0, 1.0
    )
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_StVK(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    sig = wp.vec3(
        wp.max(sig[0], 0.01), wp.max(sig[1], 0.01), wp.max(sig[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    ONE = wp.vec3(1.0, 1.0, 1.0)
    tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )


@wp.func
def kirchoff_stress_drucker_prager(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    center00 = 2.0 * mu * wp.log(sig[0]) * (1.0 / sig[0]) + lam * log_sig_sum * (
        1.0 / sig[0]
    )
    center11 = 2.0 * mu * wp.log(sig[1]) * (1.0 / sig[1]) + lam * log_sig_sum * (
        1.0 / sig[1]
    )
    center22 = 2.0 * mu * wp.log(sig[2]) * (1.0 / sig[2]) + lam * log_sig_sum * (
        1.0 / sig[2]
    )
    center = wp.mat33(center00, 0.0, 0.0, 0.0, center11, 0.0, 0.0, 0.0, center22)
    return U * center * wp.transpose(V) * wp.transpose(F)


@wp.func
def von_mises_return_mapping(F_trial: wp.mat33, model: MPMModelStruct, p: int):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


@wp.func
def von_mises_return_mapping_with_damage(
    F_trial: wp.mat33, model: MPMModelStruct, p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    # # Testing damage control
    # lambda_max = wp.max(sig_old[0], wp.max(sig_old[1], sig_old[2]))

    # if lambda_max < model.lambda_onset:
    #     return F_trial  # pure elastic, untouched

    # if lambda_max >= model.lambda_crit:
    #     model.mu[p] = 0.0
    #     model.lam[p] = 0.0
    #     return F_trial  # fully dead

    # Testing damage control

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[p]:
        if model.yield_stress[p] <= 0:
            return F_trial
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu[p])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        model.yield_stress[p] = model.yield_stress[p] - model.softening * wp.length(
            (delta_gamma / epsilon_hat_norm) * epsilon_hat
        )
        if model.yield_stress[p] <= 0:
            model.mu[p] = 0.0
            model.lam[p] = 0.0
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[p] = (
                model.yield_stress[p] + 2.0 * model.mu[p] * model.xi * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


# for toothpaste
@wp.func
def viscoplasticity_return_mapping_with_StVK(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, dt: float
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    b_trial = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    epsilon_hat = epsilon - wp.vec3(
        trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
    )
    s_trial = 2.0 * model.mu[p] * epsilon_hat
    s_trial_norm = wp.length(s_trial)
    y = s_trial_norm - wp.sqrt(2.0 / 3.0) * model.yield_stress[p]
    if y > 0:
        mu_hat = model.mu[p] * (b_trial[0] + b_trial[1] + b_trial[2]) / 3.0
        s_new_norm = s_trial_norm - y / (
            1.0 + model.plastic_viscosity / (2.0 * mu_hat * dt)
        )
        s_new = (s_new_norm / s_trial_norm) * s_trial
        epsilon_new = 1.0 / (2.0 * model.mu[p]) * s_new + wp.vec3(
            trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
        )
        sig_elastic = wp.mat33(
            wp.exp(epsilon_new[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        return F_elastic
    else:
        return F_trial


@wp.func
def sand_return_mapping(
    F_trial: wp.mat33, state: MPMStateStruct, model: MPMModelStruct, p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    epsilon = wp.vec3(
        wp.log(wp.max(wp.abs(sig[0]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[1]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[2]), 1e-14)),
    )
    sigma_out = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    tr = epsilon[0] + epsilon[1] + epsilon[2]  # + state.particle_Jp[p]
    epsilon_hat = epsilon - wp.vec3(tr / 3.0, tr / 3.0, tr / 3.0)
    epsilon_hat_norm = wp.length(epsilon_hat)
    delta_gamma = (
        epsilon_hat_norm
        + (3.0 * model.lam[p] + 2.0 * model.mu[p])
        / (2.0 * model.mu[p])
        * tr
        * model.alpha
    )

    if delta_gamma <= 0:
        F_elastic = F_trial

    if delta_gamma > 0 and tr > 0:
        F_elastic = U * wp.transpose(V)

    if delta_gamma > 0 and tr <= 0:
        H = epsilon - epsilon_hat * (delta_gamma / epsilon_hat_norm)
        s_new = wp.vec3(wp.exp(H[0]), wp.exp(H[1]), wp.exp(H[2]))

        F_elastic = U * wp.diag(s_new) * wp.transpose(V)
    return F_elastic

@wp.kernel
def compute_mu_lam_from_E_nu(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()
    model.mu[p] = model.E[p] / (2.0 * (1.0 + model.nu[p]))
    model.lam[p] = (
        model.E[p] * model.nu[p] / ((1.0 + model.nu[p]) * (1.0 - 2.0 * model.nu[p]))
    )


@wp.kernel
def zero_grid(state: MPMStateStruct, model: MPMModelStruct):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_m[grid_x, grid_y, grid_z] = 0.0
    state.grid_v_in[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def zero_grid_dual(state: MPMStateStruct, model: MPMModelStruct):
    gx, gy, gz = wp.tid()

    # field 0
    state.grid_m[gx, gy, gz] = 0.0
    state.grid_v_in[gx, gy, gz] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_v_out[gx, gy, gz] = wp.vec3(0.0, 0.0, 0.0)

    # field 1
    state.grid_m1[gx, gy, gz] = 0.0
    state.grid_v_in1[gx, gy, gz] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_v_out1[gx, gy, gz] = wp.vec3(0.0, 0.0, 0.0)

# for fracture
@wp.kernel
def zero_damage_grids(grid_d_sum: wp.array(dtype=float, ndim=3),
                      grid_d_w: wp.array(dtype=float, ndim=3)):
    i, j, k = wp.tid()
    grid_d_sum[i, j, k] = 0.0
    grid_d_w[i, j, k] = 0.0

# for fracture
@wp.kernel
def p2g_damage(
    state: MPMStateStruct,
    model: MPMModelStruct,
    grid_d_sum: wp.array(dtype=float, ndim=3),
    grid_d_w: wp.array(dtype=float, ndim=3),
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        d = state.particle_damage[p]

        grid_pos = state.particle_x[p] * model.inv_dx
        base_x = wp.int(grid_pos[0] - 0.5)
        base_y = wp.int(grid_pos[1] - 0.5)
        base_z = wp.int(grid_pos[2] - 0.5)

        fx = grid_pos - wp.vec3(wp.float(base_x), wp.float(base_y), wp.float(base_z))
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)

        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    ix = base_x + i
                    iy = base_y + j
                    iz = base_z + k
                    weight = w[0, i] * w[1, j] * w[2, k]

                    wp.atomic_add(grid_d_sum, ix, iy, iz, weight * d)
                    wp.atomic_add(grid_d_w,   ix, iy, iz, weight)

# for fracture
@wp.kernel
def normalize_damage_grid(
    grid_d_sum: wp.array(dtype=float, ndim=3),
    grid_d_w: wp.array(dtype=float, ndim=3),
):
    i, j, k = wp.tid()
    w = grid_d_w[i, j, k]
    if w > 1e-12:
        grid_d_sum[i, j, k] = grid_d_sum[i, j, k] / w
    else:
        grid_d_sum[i, j, k] = 0.0

# for fracture
@wp.kernel
def g2p_damage_smooth(
    state: MPMStateStruct,
    model: MPMModelStruct,
    grid_d: wp.array(dtype=float, ndim=3),
    alpha: float,
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_x = wp.int(grid_pos[0] - 0.5)
        base_y = wp.int(grid_pos[1] - 0.5)
        base_z = wp.int(grid_pos[2] - 0.5)

        fx = grid_pos - wp.vec3(wp.float(base_x), wp.float(base_y), wp.float(base_z))
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)

        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )

        d_smooth = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    ix = base_x + i
                    iy = base_y + j
                    iz = base_z + k
                    weight = w[0, i] * w[1, j] * w[2, k]
                    d_smooth += weight * grid_d[ix, iy, iz]

        d_old = state.particle_damage[p]
        d_new = (1.0 - alpha) * d_old + alpha * d_smooth
        state.particle_damage[p] = clamp01(d_new)

# for fracture
@wp.kernel
def assign_field_on_break(
    state: MPMStateStruct,
    model: MPMModelStruct,
    break_d: float,
):
    p = wp.tid()
    if state.particle_selection[p] != 0:
        return

    # only assign when "broken enough"
    if state.particle_damage[p] < break_d:
        return

    x = state.particle_x[p][0]
    mid = 0.5 * model.grid_lim
    if x >= mid:
        state.particle_field[p] = 1
    else:
        state.particle_field[p] = 0

# for fracture
@wp.kernel
def apply_full_break(
    state: MPMStateStruct,
    model: MPMModelStruct,
    d_break: float,   # e.g., 0.99
):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        d = state.particle_damage[p]
        if d >= d_break:
            # Remove elastic response permanently.
            model.mu[p] = 0.0
            model.lam[p] = 0.0

            # Ensure no elastic force is generated this step.
            state.particle_stress[p] = wp.mat33(0.0)

            # F and F_trial are intentionally NOT reset to identity here.
            # Keeping the large deformation gradient preserves the stretched
            # Gaussian covariance (cov = F * init_cov * F^T), which is needed
            # for dense damage map rendering around the crack region.
            # F_trial growth after break is bounded by clamp_F_trial_singular_values
            # in the p2g2p loop.

@wp.func
def compute_dweight(
    model: MPMModelStruct, w: wp.mat33, dw: wp.mat33, i: int, j: int, k: int
):
    dweight = wp.vec3(
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k],
    )
    return dweight * model.inv_dx


@wp.func
def update_cov(state: MPMStateStruct, p: int, grad_v: wp.mat33, dt: float):
    cov_n = wp.mat33(0.0)
    cov_n[0, 0] = state.particle_cov[p * 6]
    cov_n[0, 1] = state.particle_cov[p * 6 + 1]
    cov_n[0, 2] = state.particle_cov[p * 6 + 2]
    cov_n[1, 0] = state.particle_cov[p * 6 + 1]
    cov_n[1, 1] = state.particle_cov[p * 6 + 3]
    cov_n[1, 2] = state.particle_cov[p * 6 + 4]
    cov_n[2, 0] = state.particle_cov[p * 6 + 2]
    cov_n[2, 1] = state.particle_cov[p * 6 + 4]
    cov_n[2, 2] = state.particle_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    state.particle_cov[p * 6] = cov_np1[0, 0]
    state.particle_cov[p * 6 + 1] = cov_np1[0, 1]
    state.particle_cov[p * 6 + 2] = cov_np1[0, 2]
    state.particle_cov[p * 6 + 3] = cov_np1[1, 1]
    state.particle_cov[p * 6 + 4] = cov_np1[1, 2]
    state.particle_cov[p * 6 + 5] = cov_np1[2, 2]


@wp.kernel
def p2g_apic_with_stress(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    # input given to p2g:   particle_stress
    #                       particle_x
    #                       particle_v
    #                       particle_C
    p = wp.tid()
    if state.particle_selection[p] == 0:
        stress = state.particle_stress[p]

        # for fracture: degrading stress
        if model.damage_enabled != 0:
            d = state.particle_damage[p]
            wd = 1.0 - d
            stress = stress * wd

        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))

        # for fracture
        # id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        # sigma_stress = stress
        # stress = -dt * 4.0 * state.particle_vol[p] * sigma_stress / wp.pow(model.dx, 2.0)
        # affine = stress * id + state.particle_mass[p] * state.particle_C[p]

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    dpos = (
                        wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    ) * model.dx
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    C = state.particle_C[p]
                    # if model.rpic = 0, standard apic
                    C = (1.0 - model.rpic_damping) * C + model.rpic_damping / 2.0 * (
                        C - wp.transpose(C)
                    )
                    if model.rpic_damping < -0.001:
                        # standard pic
                        C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                    elastic_force = -state.particle_vol[p] * stress * dweight
                    
                    v_in_add = (
                        weight
                        * state.particle_mass[p]
                        * (state.particle_v[p] + C * dpos)
                        + dt * elastic_force
                    )

                    # for fracture
                    # v_in_add = weight * (state.particle_mass[p] * state.particle_v[p] + affine @ dpos)

                    wp.atomic_add(state.grid_v_in, ix, iy, iz, v_in_add)
                    wp.atomic_add(
                        state.grid_m, ix, iy, iz, weight * state.particle_mass[p]
                    )


# add gravity
@wp.kernel
def grid_normalization_and_gravity(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_out = state.grid_v_in[grid_x, grid_y, grid_z] * (
            1.0 / state.grid_m[grid_x, grid_y, grid_z]
        )
        # add gravity
        v_out = v_out + dt * model.gravitational_accelaration
        state.grid_v_out[grid_x, grid_y, grid_z] = v_out

# for fracture
@wp.kernel
def grid_normalization_and_gravity_dual(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    gx, gy, gz = wp.tid()

    # field 0
    if state.grid_m[gx, gy, gz] > 1e-15:
        v0 = state.grid_v_in[gx, gy, gz] * (1.0 / state.grid_m[gx, gy, gz])
        v0 = v0 + dt * model.gravitational_accelaration
        state.grid_v_out[gx, gy, gz] = v0

    # field 1
    if state.grid_m1[gx, gy, gz] > 1e-15:
        v1 = state.grid_v_in1[gx, gy, gz] * (1.0 / state.grid_m1[gx, gy, gz])
        v1 = v1 + dt * model.gravitational_accelaration
        state.grid_v_out1[gx, gy, gz] = v1

@wp.kernel
def g2p(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        new_v = wp.vec3(0.0, 0.0, 0.0)
        new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = state.grid_v_out[ix, iy, iz]
                    new_v = new_v + grid_v * weight
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )
                    dweight = compute_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        state.particle_v[p] = new_v
        state.particle_x[p] = state.particle_x[p] + dt * new_v
        state.particle_C[p] = new_C
        I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        F_tmp = (I33 + new_F * dt) * state.particle_F[p]
        state.particle_F_trial[p] = F_tmp

        if model.update_cov_with_F:
            update_cov(state, p, new_F, dt)

@wp.func
def deviatoric(A: wp.mat33) -> wp.mat33:
    tr = A[0, 0] + A[1, 1] + A[2, 2]
    m = tr / 3.0
    return A - wp.mat33(m, 0.0, 0.0,
                        0.0, m, 0.0,
                        0.0, 0.0, m)

@wp.func
def von_mises_from_cauchy(sigma: wp.mat33) -> float:
    # sigma_vm = sqrt(3/2 * s:s), where s is deviatoric part
    s = deviatoric(sigma)
    ss = 0.0
    # Frobenius inner product s:s
    ss += s[0,0]*s[0,0] + s[0,1]*s[0,1] + s[0,2]*s[0,2]
    ss += s[1,0]*s[1,0] + s[1,1]*s[1,1] + s[1,2]*s[1,2]
    ss += s[2,0]*s[2,0] + s[2,1]*s[2,1] + s[2,2]*s[2,2]
    return wp.sqrt(wp.max(0.0, 1.5 * ss))

@wp.func
def frob_norm_sq(A: wp.mat33) -> float:
    s = 0.0
    s += A[0,0]*A[0,0] + A[0,1]*A[0,1] + A[0,2]*A[0,2]
    s += A[1,0]*A[1,0] + A[1,1]*A[1,1] + A[1,2]*A[1,2]
    s += A[2,0]*A[2,0] + A[2,1]*A[2,1] + A[2,2]*A[2,2]
    return s

@wp.func
def max3(a: float, b: float, c: float) -> float:
    return wp.max(a, wp.max(b, c))

@wp.func
def update_damage_from_stretch(
    sig: wp.vec3,
    d_prev: float,
    lambda_onset: float,
    lambda_crit: float,
) -> float:
    # wp.svd3 does NOT guarantee descending order, so take explicit max
    lam_max = wp.max(sig[0], wp.max(sig[1], sig[2]))

    denom = wp.max(lambda_crit - lambda_onset, 1e-8)
    d_raw = wp.clamp((lam_max - lambda_onset) / denom, 0.0, 1.0)

    # irreversibility: damage can only grow
    return wp.max(d_prev, d_raw)

@wp.kernel
def clamp_F_trial_singular_values(
    state: MPMStateStruct,
    s_min: float,
    s_max: float,
):
    p = wp.tid()
    if state.particle_selection[p] != 0:
        return

    F = state.particle_F_trial[p]
    U   = wp.mat33(0.0)
    V   = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F, U, sig, V)

    s0 = wp.clamp(sig[0], s_min, s_max)
    s1 = wp.clamp(sig[1], s_min, s_max)
    s2 = wp.clamp(sig[2], s_min, s_max)

    state.particle_F_trial[p] = U * wp.mat33(
        s0,  0.0, 0.0,
        0.0, s1,  0.0,
        0.0, 0.0, s2,
    ) * wp.transpose(V)

@wp.func
def green_lagrange_strain_norm(F: wp.mat33) -> float:
    I = wp.mat33(1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0)
    FtF = wp.transpose(F) * F
    # E = 0.5*(FtF - I)
    E = (FtF - I) * 0.5
    return wp.sqrt(wp.max(frob_norm_sq(E), 0.0))

# compute (Kirchhoff) stress = stress(returnMap(F_trial))
@wp.kernel
def compute_stress_from_F_trial(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):  
    p = wp.tid()
    if state.particle_selection[p] == 0:
        # apply return mapping
        if model.material == 1:  # metal
            state.particle_F[p] = von_mises_return_mapping(
                state.particle_F_trial[p], model, p
            )
        elif model.material == 2:  # sand
            state.particle_F[p] = sand_return_mapping(
                state.particle_F_trial[p], state, model, p
            )
        elif model.material == 3:  # visplas, with StVk+VM, no thickening
            state.particle_F[p] = viscoplasticity_return_mapping_with_StVK(
                state.particle_F_trial[p], model, p, dt
            )
        elif model.material == 5:
            state.particle_F[p] = von_mises_return_mapping_with_damage(
                state.particle_F_trial[p], model, p
            )
        else:  # elastic
            state.particle_F[p] = state.particle_F_trial[p]

        # also compute stress here
        U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        sig = wp.vec3(0.0)
        wp.svd3(state.particle_F[p], U, sig, V)

        ##### add this clamping to avoid numerical instability
        # Only clamp heavily damaged particles (physics meaningless, avoid Gaussian blowup).
        # Upper bound is lambda_crit * 1.5 so rubber (high lambda_crit) can reach its break
        # threshold before being clamped — a hardcoded 3.0 would freeze damage at ~0.9 for
        # any material whose lambda_crit exceeds 2.
        # if state.particle_damage[p] > 0.9:
        #     clamp_max = model.lambda_crit * 1.5
        #     sig_clamped = wp.vec3(
        #         wp.clamp(sig[0], 0.2, clamp_max),
        #         wp.clamp(sig[1], 0.2, clamp_max),
        #         wp.clamp(sig[2], 0.2, clamp_max),
        #     )
        #     state.particle_F[p] = U * wp.mat33(
        #         sig_clamped[0], 0.0, 0.0,
        #         0.0, sig_clamped[1], 0.0,
        #         0.0, 0.0, sig_clamped[2]
        #     ) * wp.transpose(V)
        ##### add this clamping to avoid numerical instability
                
        J = wp.determinant(state.particle_F[p])
        
        stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if model.material == 0 or model.material == 5:
            stress = kirchoff_stress_FCR(
                state.particle_F[p], U, V, J, model.mu[p], model.lam[p]
            )
        if model.material == 1:
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if model.material == 2:
            stress = kirchoff_stress_drucker_prager(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if model.material == 3:
            # temporarily use stvk, subject to change
            stress = kirchoff_stress_StVK(
                state.particle_F[p], U, V, sig, model.mu[p], model.lam[p]
            )
        if model.material == 6:
            stress = kirchoff_stress_neoHookean(
                state.particle_F[p], U, V, J, sig, model.mu[p], model.lam[p]
            )
        F = state.particle_F[p]
        J = wp.determinant(F)
        eps = 1e-12
        invJ = 1.0 / wp.max(J, eps)

        U = wp.mat33(0.0)
        V = wp.mat33(0.0)
        sig = wp.vec3(0.0)
        wp.svd3(F, U, sig, V)

        # # Undegraded stress: Convert to cauchy stress
        # stress = (stress + wp.transpose(stress)) / 2.0   # symmetrize first
        # cauchy_stress = stress * invJ
        # vm_scalar_stress = von_mises_from_cauchy(cauchy_stress)

        # Compute strain
        strain = green_lagrange_strain_norm(F)

        # Compute energy
        R = U * wp.transpose(V)
        A = F - R
        term_dev  = model.mu[p] * frob_norm_sq(A)
        J_clamped = wp.clamp(J, 1e-6, 1e6)
        term_vol  = 0.5 * model.lam[p] * (J_clamped - 1.0) * (J_clamped - 1.0)
        psi_energy = wp.clamp(term_dev + term_vol, 0.0, 1e12)

        # Compute damage from maximum principal stretch when fracture is enabled.
        d_new = state.particle_damage[p]
        if model.damage_enabled != 0:
            d_new = update_damage_from_stretch(
                sig,
                state.particle_damage[p],
                model.lambda_onset,
                model.lambda_crit,
            )

        # Degraded stress -> cauchy stress -> von mises stress
        stress = stress * (1.0 - d_new)
        stress = (stress + wp.transpose(stress)) / 2.0   # symmetrize first
        cauchy_stress = stress * invJ
        vm_scalar_stress = von_mises_from_cauchy(cauchy_stress)

        # store as (N,) scalar
        state.particle_strain[p] = strain
        state.particle_damage[p] = d_new
        state.particle_stress[p] = stress
        state.particle_vm_scalar_stress[p] = vm_scalar_stress
        state.particle_scalar_energy[p] = psi_energy

@wp.kernel
def compute_cov_from_F(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    F = state.particle_F_trial[p]

    init_cov = wp.mat33(0.0)
    init_cov[0, 0] = state.particle_init_cov[p * 6]
    init_cov[0, 1] = state.particle_init_cov[p * 6 + 1]
    init_cov[0, 2] = state.particle_init_cov[p * 6 + 2]
    init_cov[1, 0] = state.particle_init_cov[p * 6 + 1]
    init_cov[1, 1] = state.particle_init_cov[p * 6 + 3]
    init_cov[1, 2] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 0] = state.particle_init_cov[p * 6 + 2]
    init_cov[2, 1] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 2] = state.particle_init_cov[p * 6 + 5]

    cov = F * init_cov * wp.transpose(F)

    state.particle_cov[p * 6] = cov[0, 0]
    state.particle_cov[p * 6 + 1] = cov[0, 1]
    state.particle_cov[p * 6 + 2] = cov[0, 2]
    state.particle_cov[p * 6 + 3] = cov[1, 1]
    state.particle_cov[p * 6 + 4] = cov[1, 2]
    state.particle_cov[p * 6 + 5] = cov[2, 2]


@wp.kernel
def compute_R_from_F(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()

    F = state.particle_F_trial[p]

    # polar svd decomposition
    U = wp.mat33(0.0)
    V = wp.mat33(0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F, U, sig, V)

    if wp.determinant(U) < 0.0:
        U[0, 2] = -U[0, 2]
        U[1, 2] = -U[1, 2]
        U[2, 2] = -U[2, 2]

    if wp.determinant(V) < 0.0:
        V[0, 2] = -V[0, 2]
        V[1, 2] = -V[1, 2]
        V[2, 2] = -V[2, 2]

    # compute rotation matrix
    R = U * wp.transpose(V)
    state.particle_R[p] = wp.transpose(R)


@wp.kernel
def add_damping_via_grid(state: MPMStateStruct, scale: float):
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_v_out[grid_x, grid_y, grid_z] = (
        state.grid_v_out[grid_x, grid_y, grid_z] * scale
    )

# for fracture
@wp.kernel
def add_damping_via_grid_dual(state: MPMStateStruct, scale: float):
    gx, gy, gz = wp.tid()
    state.grid_v_out[gx, gy, gz] = state.grid_v_out[gx, gy, gz] * scale
    state.grid_v_out1[gx, gy, gz] = state.grid_v_out1[gx, gy, gz] * scale

@wp.kernel
def apply_additional_params(
    state: MPMStateStruct,
    model: MPMModelStruct,
    params_modifier: MaterialParamsModifier,
):
    p = wp.tid()
    pos = state.particle_x[p]
    if (
        pos[0] > params_modifier.point[0] - params_modifier.size[0]
        and pos[0] < params_modifier.point[0] + params_modifier.size[0]
        and pos[1] > params_modifier.point[1] - params_modifier.size[1]
        and pos[1] < params_modifier.point[1] + params_modifier.size[1]
        and pos[2] > params_modifier.point[2] - params_modifier.size[2]
        and pos[2] < params_modifier.point[2] + params_modifier.size[2]
    ):
        model.E[p] = params_modifier.E
        model.nu[p] = params_modifier.nu
        state.particle_density[p] = params_modifier.density


@wp.kernel
def selection_add_impulse_on_particles(
    state: MPMStateStruct, impulse_modifier: Impulse_modifier
):
    p = wp.tid()
    offset = state.particle_x[p] - impulse_modifier.point
    if (
        wp.abs(offset[0]) < impulse_modifier.size[0]
        and wp.abs(offset[1]) < impulse_modifier.size[1]
        and wp.abs(offset[2]) < impulse_modifier.size[2]
    ):
        impulse_modifier.mask[p] = 1
    else:
        impulse_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_translation(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point
    if (
        wp.abs(offset[0]) < velocity_modifier.size[0]
        and wp.abs(offset[1]) < velocity_modifier.size[1]
        and wp.abs(offset[2]) < velocity_modifier.size[2]
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_cylinder(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point

    vertical_distance = wp.abs(wp.dot(offset, velocity_modifier.normal))

    horizontal_distance = wp.length(
        offset - wp.dot(offset, velocity_modifier.normal) * velocity_modifier.normal
    )
    if (
        vertical_distance < velocity_modifier.half_height_and_radius[0]
        and horizontal_distance < velocity_modifier.half_height_and_radius[1]
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0
