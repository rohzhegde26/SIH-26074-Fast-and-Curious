"""
src/models/ensemble.py

Sprint 8 Ensemble and Test-Time Scaling Module.
Provides:
  1. Nested Seed Manifest Generator (Seeds(K1) subset Seeds(K2)).
  2. Member-Wise Physical Clipping & Repair Diagnostics (P >= 0, 0 <= RH <= 100, Tmin <= Tmax).
  3. Proper Probabilistic Metrics:
     - Fair-CRPS (unbiased estimator with 2K(K-1) denominator for K >= 2)
     - Deterministic CRPS fallback (= MAE for K = 1, explicitly flagged)
     - Brier Score, Climatological BSS (training & sample), and Murphy Decomposition
     - Pairwise Diversity Diagnostics (pairwise RMSE, spatial correlation, ensemble variance, EDR)
     - Spread-Skill Ratio (SSR)
     - Prediction Interval Coverage and Sharpness
     - Multivariate Energy Score
  4. Memory-Safe Chunked Ensemble Generator (supporting chunk_size in {1, 2, 4}).
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

# Empirical climatology base rates computed across 854 training samples (2015-2021)
TRAINING_CLIMATOLOGY_RATES = {
    "p2_5": 0.273324,
    "p15": 0.110322,
    "p30": 0.058157,
    "p50": 0.032533,
}

CHANNEL_NAMES_6CH = ["precipitation", "tmax", "tmin", "rh", "wind_u", "wind_v"]
CHANNEL_UNITS_6CH = {
    "precipitation": "mm/day",
    "tmax": "degC",
    "tmin": "degC",
    "rh": "%",
    "wind_u": "m/s",
    "wind_v": "m/s",
}


def generate_nested_seeds(
    num_members: int,
    base_seed: int = 20260927,
    batch_idx: int = 0,
) -> List[int]:
    """
    Generates deterministic, strictly nested seeds for ensemble members.
    For any K1 < K2: generate_nested_seeds(K1) == generate_nested_seeds(K2)[:K1].
    """
    if num_members <= 0:
        raise ValueError(f"num_members must be positive, got {num_members}")
    return [base_seed + 1000 * (k + 1) + batch_idx for k in range(num_members)]


def apply_member_wise_physical_bounds(
    members_phys: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Enforces non-linear physical constraints member-by-member before ensemble reduction.
    Channel indexing:
      0: Precipitation (P, mm/day, >= 0)
      1: Maximum Temperature (Tmax, deg C, >= Tmin)
      2: Minimum Temperature (Tmin, deg C)
      3: Relative Humidity (RH, %, in [0, 100])
      4: U wind (m/s)
      5: V wind (m/s)

    Args:
        members_phys: Tensor of shape [K, B, T_f, C, H, W] in physical units.

    Returns:
        Tuple of (clipped_members, repair_diagnostics).
    """
    clipped = members_phys.clone()
    num_elements = members_phys.shape[0] * members_phys.shape[1] * members_phys.shape[2] * members_phys.shape[4] * members_phys.shape[5]

    # 1. Precipitation non-negativity (Channel 0)
    p_raw = clipped[:, :, :, 0]
    p_neg_mask = p_raw < 0.0
    p_clipped_fraction = p_neg_mask.float().mean().item()
    p_clipped = torch.clamp(p_raw, min=0.0)
    
    # Calculate mass shift: difference between raw and clipped sum / clipped sum
    raw_mass = p_raw.sum().item()
    clipped_mass = p_clipped.sum().item()
    if abs(clipped_mass) > 1e-6:
        precip_mass_shift_pct = abs(clipped_mass - raw_mass) / clipped_mass * 100.0
    else:
        precip_mass_shift_pct = 0.0

    clipped[:, :, :, 0] = p_clipped

    # 2. Relative Humidity bounds [0, 100]% (Channel 3)
    rh_raw = clipped[:, :, :, 3]
    rh_viol_mask = (rh_raw < 0.0) | (rh_raw > 100.0)
    rh_clipped_fraction = rh_viol_mask.float().mean().item()
    clipped[:, :, :, 3] = torch.clamp(rh_raw, min=0.0, max=100.0)

    # 3. Thermodynamic Temperature Ordering: Tmax >= Tmin (Channels 1 and 2)
    tmax_raw = clipped[:, :, :, 1]
    tmin_raw = clipped[:, :, :, 2]
    temp_viol_mask = tmin_raw > tmax_raw
    tmin_gt_tmax_rate = temp_viol_mask.float().mean().item()

    if temp_viol_mask.any():
        tmax_repaired = torch.maximum(tmax_raw, tmin_raw)
        tmax_repair_delta = (tmax_repaired - tmax_raw)[temp_viol_mask].mean().item()
        clipped[:, :, :, 1] = tmax_repaired
    else:
        tmax_repair_delta = 0.0

    diagnostics = {
        "precip_clipped_fraction": p_clipped_fraction,
        "precip_mass_shift_pct": precip_mass_shift_pct,
        "rh_clipped_fraction": rh_clipped_fraction,
        "tmin_gt_tmax_rate": tmin_gt_tmax_rate,
        "tmax_repair_delta_mean": tmax_repair_delta,
    }

    return clipped, diagnostics


def compute_crps(
    members: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, bool]:
    """
    Computes Continuous Ranked Probability Score.
    If K >= 2: Evaluates unbiased Fair-CRPS with 2K(K-1) denominator (Ferro et al. 2008).
    If K == 1: Degenerates to ordinary deterministic CRPS = MAE (strictly flagged is_fair=False).

    Args:
        members: Tensor of shape [K, ...].
        targets: Tensor of shape [...] broadcastable to members[0].

    Returns:
        Tuple of (crps_value, is_fair_crps).
    """
    k = members.shape[0]
    if k == 1:
        # Deterministic point forecast: CRPS_det = MAE
        mae = torch.mean(torch.abs(members[0] - targets)).item()
        return mae, False

    # K >= 2: Unbiased Fair-CRPS
    # Term 1: (1 / K) * sum_k |y^(k) - y|
    term1 = torch.mean(torch.abs(members - targets.unsqueeze(0)), dim=0) # [...]

    # Term 2: (1 / (2 * K * (K - 1))) * sum_k sum_j |y^(k) - y^(j)|
    # Compute pairwise absolute differences
    diff_sum = torch.zeros_like(term1)
    for i in range(k):
        for j in range(k):
            diff_sum += torch.abs(members[i] - members[j])
    term2 = diff_sum / (2.0 * k * (k - 1))

    fair_crps = torch.mean(term1 - term2).item()
    return fair_crps, True


def compute_per_variable_crps(
    members_phys: torch.Tensor,
    targets_phys: torch.Tensor,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Computes per-variable Fair-CRPS (or deterministic MAE for K=1) in native physical units.

    Args:
        members_phys: Tensor of shape [K, B, T_f, C, H, W] in physical units.
        targets_phys: Tensor of shape [B, T_f, C, H, W] in physical units.
        channel_names: Optional list of channel names (defaults to CHANNEL_NAMES_6CH).

    Returns:
        Dict mapping variable name to dict with 'crps', 'unit', and 'is_fair'.
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES_6CH

    num_channels = members_phys.shape[3]
    results: Dict[str, Dict[str, Any]] = {}
    for c in range(min(num_channels, len(channel_names))):
        name = channel_names[c]
        unit = CHANNEL_UNITS_6CH.get(name, "unknown")
        c_members = members_phys[:, :, :, c]
        if targets_phys.ndim == members_phys.ndim:
            c_targets = targets_phys[:, :, :, c]
        else:
            c_targets = targets_phys[:, :, c]
        crps_val, is_fair = compute_crps(c_members, c_targets)
        results[name] = {
            "crps": crps_val,
            "unit": unit,
            "is_fair": is_fair,
        }
    return results


def compute_brier_score(
    predicted_probs: torch.Tensor,
    observed_events: torch.Tensor,
    clim_train_rate: Optional[float] = None,
    num_bins: int = 10,
) -> Dict[str, float]:
    """
    Computes Brier Score, Brier Skill Score against training climatology and sample climatology,
    and Murphy three-component decomposition (Reliability, Resolution, Uncertainty).

    Args:
        predicted_probs: Tensor of forecast probabilities in [0, 1].
        observed_events: Tensor of binary ground-truth occurrences in {0, 1}.
        clim_train_rate: Optional historical event frequency from training split.
        num_bins: Number of probability bins for calibration (default 10).

    Returns:
        Dict containing brier_score, bss_train, bss_sample, reliability, resolution, uncertainty.
    """
    p = predicted_probs.flatten().float()
    o = observed_events.flatten().float()
    n = p.shape[0]

    if n == 0:
        return {"brier_score": 0.0, "reliability": 0.0, "resolution": 0.0, "uncertainty": 0.0}

    # Brier Score
    bs = torch.mean((p - o) ** 2).item()

    # Sample Climatology & BSS
    sample_rate = torch.mean(o).item()
    bs_clim_sample = sample_rate * (1.0 - sample_rate)
    bss_sample = 1.0 - (bs / max(bs_clim_sample, 1e-6)) if bs_clim_sample > 1e-6 else 0.0

    # Training Climatology & BSS
    if clim_train_rate is not None and 0.0 < clim_train_rate < 1.0:
        bs_clim_train = clim_train_rate * (1.0 - clim_train_rate)
        bss_train = 1.0 - (bs / max(bs_clim_train, 1e-6))
    else:
        bss_train = bss_sample

    # Murphy Decomposition
    # Bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=p.device)
    reliability = 0.0
    resolution = 0.0

    for m in range(num_bins):
        low = bin_edges[m]
        high = bin_edges[m + 1]
        if m == num_bins - 1:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)

        n_m = mask.sum().item()
        if n_m > 0:
            p_bar_m = p[mask].mean().item()
            o_bar_m = o[mask].mean().item()
            reliability += (n_m / n) * ((p_bar_m - o_bar_m) ** 2)
            resolution += (n_m / n) * ((o_bar_m - sample_rate) ** 2)

    uncertainty = bs_clim_sample

    return {
        "brier_score": bs,
        "brier_skill_score_train": bss_train,
        "brier_skill_score_sample": bss_sample,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "sample_event_rate": sample_rate,
    }


def compute_pairwise_diversity(
    members: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes inter-member diversity diagnostics across ensemble members.

    Args:
        members: Tensor of shape [K, ...] or [K, B, T_f, C, H, W].

    Returns:
        Dict containing mean_pairwise_rmse, mean_pairwise_spatial_correlation,
        global_member_correlation, ensemble_variance, and num_pairs.
    """
    k = members.shape[0]
    if k <= 1:
        return {
            "mean_pairwise_rmse": 0.0,
            "mean_pairwise_spatial_correlation": 1.0,
            "global_member_correlation": 1.0,
            "mean_pairwise_correlation": 1.0,
            "ensemble_variance": 0.0,
            "num_pairs": 0,
        }

    # Flatten each member to 1D vector for global correlation
    flat_members = members.reshape(k, -1)
    d = flat_members.shape[1]

    # Ensemble Variance
    ens_var = torch.mean(torch.var(members, dim=0, unbiased=True)).item()

    # Pairwise RMSE and Global Pearson Correlation
    num_pairs = k * (k - 1) // 2
    rmse_sum = 0.0
    global_corr_sum = 0.0

    for i in range(k):
        for j in range(i + 1, k):
            diff = flat_members[i] - flat_members[j]
            rmse = torch.sqrt(torch.mean(diff ** 2)).item()
            rmse_sum += rmse

            fi = flat_members[i] - flat_members[i].mean()
            fj = flat_members[j] - flat_members[j].mean()
            norm_i = torch.norm(fi)
            norm_j = torch.norm(fj)
            if norm_i > 1e-6 and norm_j > 1e-6:
                corr = torch.dot(fi, fj) / (norm_i * norm_j)
                global_corr_sum += corr.item()
            else:
                global_corr_sum += 1.0

    # 2D Spatial Correlation if 6D tensor [K, B, T_f, C, H, W]
    if members.ndim == 6:
        K_dim, B_dim, T_dim, C_dim, H_dim, W_dim = members.shape
        spatial_grids = members.reshape(K_dim, B_dim * T_dim * C_dim, H_dim * W_dim)
        spatial_corr_sum = 0.0
        spatial_corr_count = 0
        for i in range(k):
            for j in range(i + 1, k):
                gi = spatial_grids[i]
                gj = spatial_grids[j]
                mi = gi - gi.mean(dim=-1, keepdim=True)
                mj = gj - gj.mean(dim=-1, keepdim=True)
                ni = torch.norm(mi, dim=-1)
                nj = torch.norm(mj, dim=-1)
                valid = (ni > 1e-6) & (nj > 1e-6)
                if valid.any():
                    dot = (mi[valid] * mj[valid]).sum(dim=-1)
                    r = dot / (ni[valid] * nj[valid])
                    spatial_corr_sum += r.sum().item()
                    spatial_corr_count += valid.sum().item()
                non_valid_count = (~valid).sum().item()
                if non_valid_count > 0:
                    spatial_corr_sum += float(non_valid_count)
                    spatial_corr_count += non_valid_count
        mean_spatial_corr = spatial_corr_sum / max(1, spatial_corr_count)
    else:
        mean_spatial_corr = global_corr_sum / num_pairs

    mean_global_corr = global_corr_sum / num_pairs

    return {
        "mean_pairwise_rmse": rmse_sum / num_pairs,
        "mean_pairwise_spatial_correlation": mean_spatial_corr,
        "global_member_correlation": mean_global_corr,
        "mean_pairwise_correlation": mean_global_corr,
        "ensemble_variance": ens_var,
        "num_pairs": num_pairs,
    }


def compute_per_variable_spread_skill(
    members_phys: torch.Tensor,
    targets_phys: torch.Tensor,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Computes Spread-Skill Ratio per meteorological channel in native physical units.
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES_6CH

    num_channels = members_phys.shape[3]
    results: Dict[str, Dict[str, float]] = {}
    for c in range(min(num_channels, len(channel_names))):
        name = channel_names[c]
        c_members = members_phys[:, :, :, c]
        if targets_phys.ndim == members_phys.ndim:
            c_targets = targets_phys[:, :, :, c]
        else:
            c_targets = targets_phys[:, :, c]
        ssr_dict = compute_spread_skill_ratio(c_members, c_targets)
        results[name] = ssr_dict
    return results


def compute_per_variable_prediction_interval(
    members_phys: torch.Tensor,
    targets_phys: torch.Tensor,
    nominal_levels: Tuple[float, ...] = (0.50, 0.80, 0.90),
    channel_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Computes empirical coverage and interval sharpness per meteorological channel.
    """
    if channel_names is None:
        channel_names = CHANNEL_NAMES_6CH

    num_channels = members_phys.shape[3]
    results: Dict[str, Dict[str, float]] = {}
    for c in range(min(num_channels, len(channel_names))):
        name = channel_names[c]
        c_members = members_phys[:, :, :, c]
        if targets_phys.ndim == members_phys.ndim:
            c_targets = targets_phys[:, :, :, c]
        else:
            c_targets = targets_phys[:, :, c]
        cov_dict = compute_prediction_interval_coverage(c_members, c_targets, nominal_levels)
        results[name] = cov_dict
    return results


def compute_spread_skill_ratio(
    members: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes Ensemble Spread, RMSE of Ensemble Mean, and Spread-Skill Ratio (SSR).

    Args:
        members: Tensor of shape [K, ...].
        targets: Tensor of shape [...] broadcastable to members[0].
    """
    k = members.shape[0]
    ens_mean = torch.mean(members, dim=0)
    rmse = torch.sqrt(torch.mean((ens_mean - targets) ** 2)).item()

    if k > 1:
        spread = torch.sqrt(torch.mean(torch.var(members, dim=0, unbiased=True))).item()
    else:
        spread = 0.0

    ssr = spread / max(rmse, 1e-6)
    return {
        "ensemble_spread": spread,
        "rmse_ensemble_mean": rmse,
        "spread_skill_ratio": ssr,
    }


def compute_prediction_interval_coverage(
    members: torch.Tensor,
    targets: torch.Tensor,
    nominal_levels: Tuple[float, ...] = (0.50, 0.80, 0.90),
) -> Dict[str, Dict[str, float]]:
    """
    Computes empirical coverage and interval sharpness for prediction intervals.
    """
    k = members.shape[0]
    results = {}
    if k < 2:
        for lvl in nominal_levels:
            results[f"coverage_{int(lvl*100)}"] = 0.0
            results[f"sharpness_{int(lvl*100)}"] = 0.0
        return results

    # Sort members along K
    sorted_members, _ = torch.sort(members, dim=0)
    for lvl in nominal_levels:
        alpha = 1.0 - lvl
        q_low_idx = max(0, int(round((alpha / 2.0) * (k - 1))))
        q_high_idx = min(k - 1, int(round((1.0 - alpha / 2.0) * (k - 1))))

        low_bound = sorted_members[q_low_idx]
        high_bound = sorted_members[q_high_idx]

        in_interval = (targets >= low_bound) & (targets <= high_bound)
        cov = in_interval.float().mean().item()
        sharp = (high_bound - low_bound).mean().item()

        results[f"coverage_{int(lvl*100)}"] = cov
        results[f"sharpness_{int(lvl*100)}"] = sharp

    return results


def compute_multivariate_energy_score(
    members: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Computes Multivariate Energy Score across channels (Gneiting & Raftery 2007).
    Expected shape: [K, B, T_f, C, H, W] and targets [B, T_f, C, H, W].
    """
    k = members.shape[0]
    # Reshape channels into feature vectors: [K, (B * T_f * H * W), C]
    b, t, c, h, w = members.shape[1], members.shape[2], members.shape[3], members.shape[4], members.shape[5]
    flat_members = members.permute(0, 1, 2, 4, 5, 3).reshape(k, -1, c)
    flat_targets = targets.permute(0, 1, 3, 4, 2).reshape(-1, c)

    # Term 1: (1/K) * sum_k ||y^(k) - y||_2
    dist1 = torch.norm(flat_members - flat_targets.unsqueeze(0), dim=-1).mean(dim=0) # [-1]
    term1 = dist1.mean().item()

    if k < 2:
        return term1

    # Term 2: 1 / (2 * K * (K - 1)) * sum_k sum_j ||y^(k) - y^(j)||_2
    diff_sum = torch.zeros(flat_members.shape[1], device=members.device)
    for i in range(k):
        for j in range(k):
            diff_sum += torch.norm(flat_members[i] - flat_members[j], dim=-1)
    term2 = (diff_sum / (2.0 * k * (k - 1))).mean().item()

    return term1 - term2


class EnsembleGenerator:
    """
    Reusable, memory-safe ensemble sampler supporting nested seeds and chunked execution.
    """

    def __init__(
        self,
        denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        alphas_cumprod: torch.Tensor,
        timesteps_desc: List[int],
        eta: float = 0.0,
        prediction_type: str = "v_prediction",
    ):
        self.denoise_fn = denoise_fn
        self.alphas_cumprod = alphas_cumprod
        self.timesteps_desc = timesteps_desc
        self.eta = eta
        self.prediction_type = prediction_type

    def generate(
        self,
        shape: Tuple[int, ...],
        num_members: int = 1,
        base_seed: int = 20260927,
        batch_idx: int = 0,
        chunk_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Generates num_members in chunks of size chunk_size.
        Returns:
            (members_tensor, diversity_diagnostics)
            members_tensor shape: [K, B, T_f, C, H, W]
        """
        if device is None:
            device = self.alphas_cumprod.device

        seeds = generate_nested_seeds(num_members, base_seed=base_seed, batch_idx=batch_idx)
        all_members = []

        chunk_size = max(1, min(chunk_size, num_members))
        num_chunks = math.ceil(num_members / chunk_size)

        for chunk_idx in range(num_chunks):
            start_k = chunk_idx * chunk_size
            end_k = min(start_k + chunk_size, num_members)
            cur_chunk_seeds = seeds[start_k:end_k]

            # Generate each member with its dedicated PRNG stream
            for seed in cur_chunk_seeds:
                gen = torch.Generator(device=device).manual_seed(seed)
                r_init = torch.randn(shape, generator=gen, device=device)

                # Reverse DDIM loop
                r_cur = r_init.clone()
                b = shape[0]

                for i, t_curr in enumerate(self.timesteps_desc):
                    t_tensor = torch.full((b,), t_curr, device=device, dtype=torch.long)
                    model_out = self.denoise_fn(r_cur, t_tensor)

                    alpha_bar_curr = self.alphas_cumprod[t_curr].view(1, 1, 1, 1, 1)

                    if self.prediction_type == "v_prediction":
                        # v_to_r0_and_eps
                        r_0_pred = torch.sqrt(alpha_bar_curr) * r_cur - torch.sqrt(1.0 - alpha_bar_curr) * model_out
                        eps_pred = torch.sqrt(1.0 - alpha_bar_curr) * r_cur + torch.sqrt(alpha_bar_curr) * model_out
                    else:
                        eps_pred = model_out
                        r_0_pred = (r_cur - torch.sqrt(1.0 - alpha_bar_curr) * eps_pred) / torch.sqrt(alpha_bar_curr)

                    if i < len(self.timesteps_desc) - 1:
                        t_prev = self.timesteps_desc[i + 1]
                        alpha_bar_prev = self.alphas_cumprod[t_prev].view(1, 1, 1, 1, 1)
                        if self.eta == 0.0:
                            r_cur = torch.sqrt(alpha_bar_prev) * r_0_pred + torch.sqrt(1.0 - alpha_bar_prev) * eps_pred
                        else:
                            sigma = self.eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_curr) * (1.0 - alpha_bar_curr / alpha_bar_prev))
                            step_noise = torch.randn(shape, generator=gen, device=device)
                            r_cur = torch.sqrt(alpha_bar_prev) * r_0_pred + torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)) * eps_pred + sigma * step_noise
                    else:
                        r_cur = r_0_pred

                all_members.append(r_cur)

        members_tensor = torch.stack(all_members, dim=0) # [K, B, T_f, C, H, W]
        diversity_diag = compute_pairwise_diversity(members_tensor)

        return members_tensor, diversity_diag
