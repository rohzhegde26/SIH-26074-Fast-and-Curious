"""
src/models/samplers.py

Sprint 7 Diffusion-Step & Sampler Frontier:
Higher-Order ODE Solvers and Canonical Timestep Schedulers for Spatiotemporal Weather Downscaling.

Mathematical Specifications:
  1. Canonical Standard Timestep Discretization (Song et al. 2021):
     tau_k = round(k * (T - 1) / (S - 1)), for k in {0, 1, ..., S-1}.
     Descending reverse trajectory: tau_desc = [99, ..., 0] spanning the entire cumulative variance schedule.
  2. Exact Analytical v-Prediction Identities (Salimans & Ho 2022):
     r_0 = sqrt(alpha_bar_t) * r_t - sqrt(1 - alpha_bar_t) * v_t
     eps = sqrt(1 - alpha_bar_t) * r_t + sqrt(alpha_bar_t) * v_t
  3. Solvers Supported:
     - DDIM (1st-Order Pseudo-Euler ODE)
     - DPM-Solver++ (2M) (2nd-Order Multi-Step ODE in data/residual space)
     - PNDM (4th-Order Linear Multi-Step Adams-Bashforth in noise space)
"""

import math
from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F


def get_sampling_timesteps(
    total_timesteps: int = 100,
    num_steps: int = 32,
    schedule_type: str = "standard",
) -> List[int]:
    """
    Computes reverse diffusion timesteps (strictly descending from max to min).

    Args:
        total_timesteps: Training timesteps T (default 100, indices 0..99).
        num_steps: Requested inference steps S (e.g. 4, 8, 16, 32, 64).
        schedule_type: "standard" (canonical Song et al.) or "legacy" (Sprint 5/6 stride slicing).

    Returns:
        List of integers in strictly descending order (e.g., [99, ..., 0]).
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")

    if schedule_type == "legacy":
        step_stride = total_timesteps // num_steps
        time_seq = list(range(0, total_timesteps, step_stride))
        time_seq = time_seq[:num_steps]
        return list(reversed(time_seq))

    # Standard canonical discretization
    if num_steps == 1:
        return [0]

    # tau_k = round(k * (T - 1) / (S - 1)) for k in 0..S-1
    t_max = total_timesteps - 1
    tau = [round(k * t_max / (num_steps - 1)) for k in range(num_steps)]
    # Ensure monotonicity and exact endpoints
    tau[0] = 0
    tau[-1] = t_max
    # Make strictly ascending if round caused any duplicate
    for i in range(1, len(tau)):
        if tau[i] <= tau[i - 1]:
            tau[i] = tau[i - 1] + 1
    # Reverse to descending trajectory
    return list(reversed(tau))


def v_to_r0_and_eps(
    r_t: torch.Tensor,
    v_t: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact analytical conversion from velocity prediction v_t to (r_0, eps).

    r_0 = sqrt(alpha_bar) * r_t - sqrt(1 - alpha_bar) * v_t
    eps = sqrt(1 - alpha_bar) * r_t + sqrt(alpha_bar) * v_t
    """
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar)
    r_0 = sqrt_alpha_bar * r_t - sqrt_one_minus * v_t
    eps = sqrt_one_minus * r_t + sqrt_alpha_bar * v_t
    return r_0, eps


def eps_to_r0_and_v(
    r_t: torch.Tensor,
    eps: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact conversion from noise prediction eps to (r_0, v_t).

    r_0 = (r_t - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)
    v_t = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * r_0
    """
    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar)
    r_0 = (r_t - sqrt_one_minus * eps) / (sqrt_alpha_bar + 1e-8)
    v_t = sqrt_alpha_bar * eps - sqrt_one_minus * r_0
    return r_0, v_t


def sample_ddim_trajectory(
    denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    r_init: torch.Tensor,
    timesteps_desc: List[int],
    alphas_cumprod: torch.Tensor,
    prediction_type: str = "v_prediction",
    eta: float = 0.0,
) -> torch.Tensor:
    """
    Executes first-order DDIM reverse trajectory across specified descending timesteps.
    """
    b = r_init.shape[0]
    device = r_init.device
    r_cur = r_init.clone()

    for i, t_curr in enumerate(timesteps_desc):
        t_tensor = torch.full((b,), t_curr, device=device, dtype=torch.long)
        model_out = denoise_fn(r_cur, t_tensor)

        alpha_bar_curr = alphas_cumprod[t_curr]
        alpha_bar_curr = alpha_bar_curr.view(1, 1, 1, 1, 1)

        if prediction_type == "v_prediction":
            r_0_pred, eps_pred = v_to_r0_and_eps(r_cur, model_out, alpha_bar_curr)
        else:
            eps_pred = model_out
            r_0_pred, _ = eps_to_r0_and_v(r_cur, eps_pred, alpha_bar_curr)

        if i < len(timesteps_desc) - 1:
            t_prev = timesteps_desc[i + 1]
            alpha_bar_prev = alphas_cumprod[t_prev].view(1, 1, 1, 1, 1)
            if eta == 0.0:
                r_cur = torch.sqrt(alpha_bar_prev) * r_0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps_pred
            else:
                sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_curr) * (1 - alpha_bar_curr / alpha_bar_prev))
                noise = torch.randn_like(r_cur)
                r_cur = torch.sqrt(alpha_bar_prev) * r_0_pred + torch.sqrt(1.0 - alpha_bar_prev - sigma**2) * eps_pred + sigma * noise
        else:
            r_cur = r_0_pred

    return r_cur


def sample_dpm_solver_2m_trajectory(
    denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    r_init: torch.Tensor,
    timesteps_desc: List[int],
    alphas_cumprod: torch.Tensor,
    prediction_type: str = "v_prediction",
) -> torch.Tensor:
    """
    Executes second-order DPM-Solver++(2M) multi-step reverse trajectory in data space.
    Reference: Lu et al. (NeurIPS 2022) Fast ODE Sampling of Diffusion Models.
    """
    b = r_init.shape[0]
    device = r_init.device
    r_cur = r_init.clone()

    # Precalculate log-SNR lambda_t = 0.5 * log(alpha_bar / (1 - alpha_bar))
    lambda_vals = {}
    for t in timesteps_desc:
        ab = alphas_cumprod[t].item()
        ab = max(min(ab, 1.0 - 1e-6), 1e-6)
        lambda_vals[t] = 0.5 * math.log(ab / (1.0 - ab))

    r0_history = []
    h_history = []

    for i, t_curr in enumerate(timesteps_desc):
        t_tensor = torch.full((b,), t_curr, device=device, dtype=torch.long)
        model_out = denoise_fn(r_cur, t_tensor)

        alpha_bar_curr = alphas_cumprod[t_curr].view(1, 1, 1, 1, 1)
        if prediction_type == "v_prediction":
            r_0_pred, _ = v_to_r0_and_eps(r_cur, model_out, alpha_bar_curr)
        else:
            r_0_pred, _ = eps_to_r0_and_v(r_cur, model_out, alpha_bar_curr)

        if i == len(timesteps_desc) - 1:
            # Final terminal step
            r_cur = r_0_pred
            break

        t_next = timesteps_desc[i + 1]
        lambda_curr = lambda_vals[t_curr]
        lambda_next = lambda_vals[t_next]
        # In reverse diffusion t_curr > t_next, so alpha_bar increases and lambda increases (lambda_next > lambda_curr)
        # Defining h = lambda_curr - lambda_next < 0
        h = lambda_curr - lambda_next

        alpha_bar_next = alphas_cumprod[t_next].view(1, 1, 1, 1, 1)
        sqrt_1_minus_curr = torch.sqrt(1.0 - alpha_bar_curr)
        sqrt_1_minus_next = torch.sqrt(1.0 - alpha_bar_next)
        sqrt_ab_next = torch.sqrt(alpha_bar_next)

        exp_h_minus_1 = math.expm1(h)  # e^h - 1 < 0

        if i == 0 or len(h_history) == 0:
            # 1st-order Euler warmup step
            d_i = r_0_pred
        else:
            # 2nd-order multi-step extrapolation
            h_prev = h_history[-1]
            r_i = h / (h_prev + 1e-8)
            d_i = (1.0 + 0.5 / r_i) * r_0_pred - (0.5 / r_i) * r0_history[-1]

        r_cur = (sqrt_1_minus_next / sqrt_1_minus_curr) * r_cur - sqrt_ab_next * exp_h_minus_1 * d_i

        r0_history.append(r_0_pred)
        h_history.append(h)

    return r_cur


def sample_pndm_trajectory(
    denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    r_init: torch.Tensor,
    timesteps_desc: List[int],
    alphas_cumprod: torch.Tensor,
    prediction_type: str = "v_prediction",
) -> torch.Tensor:
    """
    Executes Pseudo Numerical Methods for Diffusion Models (PNDM) using Adams-Bashforth multi-step in eps space.
    Reference: Liu et al. (ICLR 2022).
    """
    b = r_init.shape[0]
    device = r_init.device
    r_cur = r_init.clone()
    eps_history: List[torch.Tensor] = []

    for i, t_curr in enumerate(timesteps_desc):
        t_tensor = torch.full((b,), t_curr, device=device, dtype=torch.long)
        model_out = denoise_fn(r_cur, t_tensor)

        alpha_bar_curr = alphas_cumprod[t_curr].view(1, 1, 1, 1, 1)

        if prediction_type == "v_prediction":
            r_0_curr, eps_curr = v_to_r0_and_eps(r_cur, model_out, alpha_bar_curr)
        else:
            eps_curr = model_out
            r_0_curr, _ = eps_to_r0_and_v(r_cur, eps_curr, alpha_bar_curr)

        eps_history.append(eps_curr)

        if i == len(timesteps_desc) - 1:
            # Final step returns pure data prediction
            r_cur = r_0_curr
            break

        # Linear multi-step Adams-Bashforth gradient combination
        if len(eps_history) == 1:
            g_i = eps_history[-1]
        elif len(eps_history) == 2:
            g_i = 1.5 * eps_history[-1] - 0.5 * eps_history[-2]
        elif len(eps_history) == 3:
            g_i = (23.0 * eps_history[-1] - 16.0 * eps_history[-2] + 5.0 * eps_history[-3]) / 12.0
        else:
            g_i = (55.0 * eps_history[-1] - 59.0 * eps_history[-2] + 37.0 * eps_history[-3] - 9.0 * eps_history[-4]) / 24.0

        t_next = timesteps_desc[i + 1]
        alpha_bar_next = alphas_cumprod[t_next].view(1, 1, 1, 1, 1)

        # Estimate r_0 with combined gradient g_i
        r_0_transfer = (r_cur - torch.sqrt(1.0 - alpha_bar_curr) * g_i) / torch.sqrt(alpha_bar_curr)
        r_cur = torch.sqrt(alpha_bar_next) * r_0_transfer + torch.sqrt(1.0 - alpha_bar_next) * g_i

    return r_cur
