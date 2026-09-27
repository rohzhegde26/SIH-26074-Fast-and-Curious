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


def compute_paired_bootstrap(
    metrics_a: Dict[str, Any],
    metrics_b: Optional[Dict[str, Any]] = None,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 20260927,
) -> Dict[str, Any]:
    """
    Computes case-level paired bootstrap confidence intervals with replacement.

    Parameters:
        metrics_a: Dictionary mapping metric name to list or 1D array of case-level values.
        metrics_b: Optional second condition dictionary for paired difference analysis (delta = A - B).
        n_resamples: Number of bootstrap resamples (default: 1000).
        confidence_level: Desired coverage level (default: 0.95).
        seed: Random seed for bootstrap index generation.

    Returns:
        Dictionary containing per-metric point estimates, bootstrap distributions,
        95% confidence intervals, and paired differences when metrics_b is provided.
    """
    import numpy as np

    if not metrics_a:
        raise ValueError("metrics_a cannot be empty")

    # Validate lengths
    n_cases = None
    processed_a: Dict[str, np.ndarray] = {}
    for k, v in metrics_a.items():
        arr = np.asarray(v, dtype=np.float64)
        if n_cases is None:
            n_cases = len(arr)
        elif len(arr) != n_cases:
            raise ValueError(f"Length mismatch in metrics_a for '{k}': {len(arr)} != {n_cases}")
        processed_a[k] = arr

    if n_cases is None or n_cases < 2:
        raise ValueError(f"Need at least 2 cases for bootstrap, got {n_cases}")

    processed_b: Optional[Dict[str, np.ndarray]] = None
    if metrics_b is not None:
        processed_b = {}
        for k, v in metrics_b.items():
            arr = np.asarray(v, dtype=np.float64)
            if len(arr) != n_cases:
                raise ValueError(f"Length mismatch in metrics_b for '{k}': {len(arr)} != {n_cases}")
            processed_b[k] = arr

    rng = np.random.default_rng(seed)
    # Generate B x N indices
    boot_indices = rng.integers(0, n_cases, size=(n_resamples, n_cases))

    alpha = (1.0 - confidence_level) / 2.0
    results: Dict[str, Any] = {
        "n_cases": n_cases,
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "metrics_a": {},
        "paired_differences": {},
    }

    # Evaluate metrics_a
    for k, arr in processed_a.items():
        if k.endswith("_hits") or k.endswith("_fps") or k.endswith("_fns"):
            continue
        resampled = arr[boot_indices] # [B, N]
        boot_means = np.mean(resampled, axis=1) # [B]
        point_est = float(np.mean(arr))
        ci_low = float(np.percentile(boot_means, 100.0 * alpha))
        ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
        std_err = float(np.std(boot_means, ddof=1))

        results["metrics_a"][k] = {
            "point_estimate": point_est,
            "bootstrap_mean": float(np.mean(boot_means)),
            "std_error": std_err,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
        }

    # Check for contingency table metrics in metrics_a (e.g. p30_hits, p30_fps, p30_fns)
    for prefix in ["p15", "p30", "p50"]:
        h_k = f"{prefix}_hits"
        f_k = f"{prefix}_fps"
        m_k = f"{prefix}_fns"
        if h_k in processed_a and f_k in processed_a and m_k in processed_a:
            h_arr = processed_a[h_k]
            f_arr = processed_a[f_k]
            m_arr = processed_a[m_k]
            tot_h = float(np.sum(h_arr))
            tot_f = float(np.sum(f_arr))
            tot_m = float(np.sum(m_arr))
            point_csi = float(tot_h / (tot_h + tot_f + tot_m)) if (tot_h + tot_f + tot_m) > 0 else 1.0

            boot_h = np.sum(h_arr[boot_indices], axis=1)
            boot_f = np.sum(f_arr[boot_indices], axis=1)
            boot_m = np.sum(m_arr[boot_indices], axis=1)
            den = boot_h + boot_f + boot_m
            boot_csi = np.where(den > 0, boot_h / np.maximum(1e-8, den), 1.0)

            ci_low = float(np.percentile(boot_csi, 100.0 * alpha))
            ci_high = float(np.percentile(boot_csi, 100.0 * (1.0 - alpha)))
            results["metrics_a"][f"csi_{prefix}"] = {
                "point_estimate": point_csi,
                "bootstrap_mean": float(np.mean(boot_csi)),
                "std_error": float(np.std(boot_csi, ddof=1)),
                "ci_lower": ci_low,
                "ci_upper": ci_high,
            }

    # Evaluate paired differences if metrics_b is provided
    if processed_b is not None:
        results["metrics_b"] = {}
        for k, arr_b in processed_b.items():
            if k.endswith("_hits") or k.endswith("_fps") or k.endswith("_fns"):
                continue
            resampled_b = arr_b[boot_indices]
            boot_means_b = np.mean(resampled_b, axis=1)
            point_b = float(np.mean(arr_b))
            ci_low_b = float(np.percentile(boot_means_b, 100.0 * alpha))
            ci_high_b = float(np.percentile(boot_means_b, 100.0 * (1.0 - alpha)))
            std_err_b = float(np.std(boot_means_b, ddof=1))

            results["metrics_b"][k] = {
                "point_estimate": point_b,
                "bootstrap_mean": float(np.mean(boot_means_b)),
                "std_error": std_err_b,
                "ci_lower": ci_low_b,
                "ci_upper": ci_high_b,
            }

            if k in processed_a:
                arr_a = processed_a[k]
                resampled_a = arr_a[boot_indices]
                boot_means_a = np.mean(resampled_a, axis=1)

                # Paired delta = A - B
                deltas = boot_means_a - boot_means_b
                delta_pt = float(np.mean(arr_a) - np.mean(arr_b))
                d_ci_low = float(np.percentile(deltas, 100.0 * alpha))
                d_ci_high = float(np.percentile(deltas, 100.0 * (1.0 - alpha)))
                d_std_err = float(np.std(deltas, ddof=1))

                # Empirical two-sided p-value
                p_val = 2.0 * min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0)))
                p_val = min(1.0, max(0.0, p_val))
                is_significant = bool(d_ci_low * d_ci_high > 0.0)

                results["paired_differences"][k] = {
                    "delta_point_estimate": delta_pt,
                    "delta_bootstrap_mean": float(np.mean(deltas)),
                    "delta_std_error": d_std_err,
                    "ci_lower": d_ci_low,
                    "ci_upper": d_ci_high,
                    "p_value": p_val,
                    "statistically_significant": is_significant,
                }

        # Contingency table paired CSI
        for prefix in ["p15", "p30", "p50"]:
            h_k = f"{prefix}_hits"
            f_k = f"{prefix}_fps"
            m_k = f"{prefix}_fns"
            if (h_k in processed_a and f_k in processed_a and m_k in processed_a and
                h_k in processed_b and f_k in processed_b and m_k in processed_b):
                ha, fa, ma = processed_a[h_k], processed_a[f_k], processed_a[m_k]
                hb, fb, mb = processed_b[h_k], processed_b[f_k], processed_b[m_k]

                boot_ha = np.sum(ha[boot_indices], axis=1)
                boot_fa = np.sum(fa[boot_indices], axis=1)
                boot_ma = np.sum(ma[boot_indices], axis=1)
                den_a = boot_ha + boot_fa + boot_ma
                csi_a = np.where(den_a > 0, boot_ha / np.maximum(1e-8, den_a), 1.0)

                boot_hb = np.sum(hb[boot_indices], axis=1)
                boot_fb = np.sum(fb[boot_indices], axis=1)
                boot_mb = np.sum(mb[boot_indices], axis=1)
                den_b = boot_hb + boot_fb + boot_mb
                csi_b = np.where(den_b > 0, boot_hb / np.maximum(1e-8, den_b), 1.0)

                tot_ha, tot_fa, tot_ma = float(np.sum(ha)), float(np.sum(fa)), float(np.sum(ma))
                tot_hb, tot_fb, tot_mb = float(np.sum(hb)), float(np.sum(fb)), float(np.sum(mb))
                pt_a = float(tot_ha / (tot_ha + tot_fa + tot_ma)) if (tot_ha + tot_fa + tot_ma) > 0 else 1.0
                pt_b = float(tot_hb / (tot_hb + tot_fb + tot_mb)) if (tot_hb + tot_fb + tot_mb) > 0 else 1.0

                ci_low_b = float(np.percentile(csi_b, 100.0 * alpha))
                ci_high_b = float(np.percentile(csi_b, 100.0 * (1.0 - alpha)))
                results["metrics_b"][f"csi_{prefix}"] = {
                    "point_estimate": pt_b,
                    "bootstrap_mean": float(np.mean(csi_b)),
                    "std_error": float(np.std(csi_b, ddof=1)),
                    "ci_lower": ci_low_b,
                    "ci_upper": ci_high_b,
                }

                deltas_csi = csi_a - csi_b
                d_ci_low = float(np.percentile(deltas_csi, 100.0 * alpha))
                d_ci_high = float(np.percentile(deltas_csi, 100.0 * (1.0 - alpha)))
                d_std_err = float(np.std(deltas_csi, ddof=1))
                p_val = 2.0 * min(float(np.mean(deltas_csi <= 0.0)), float(np.mean(deltas_csi >= 0.0)))
                p_val = min(1.0, max(0.0, p_val))
                is_sig = bool(d_ci_low * d_ci_high > 0.0)

                results["paired_differences"][f"csi_{prefix}"] = {
                    "delta_point_estimate": pt_a - pt_b,
                    "delta_bootstrap_mean": float(np.mean(deltas_csi)),
                    "delta_std_error": d_std_err,
                    "ci_lower": d_ci_low,
                    "ci_upper": d_ci_high,
                    "p_value": p_val,
                    "statistically_significant": is_sig,
                }

    return results

