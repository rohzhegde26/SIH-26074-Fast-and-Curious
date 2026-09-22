"""
src/data/tensor_builder.py

Pure functional tensor assembly and invertible normalization engine for Sprint 2.
Features:
  1. Generic H-parameterized history tensor assembly ([H, 6, H_c, W_c]).
  2. Multi-lead forecast tensor assembly ([L, 6, H_c, W_c]) and fine target tensor assembly ([L, 6, 80, 80]).
  3. Train-only statistical normalization with log1p precipitation scaling.
  4. Lossless round-trip inversion with physical clipping guards (precip >= 0, 0 <= RH <= 100).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

WEATHER_CHANNELS = ["precipitation", "tmax", "tmin", "rh", "wind_u", "wind_v"]


def assemble_history_tensor(
    history_dict: Dict[str, np.ndarray],
    history_window_days: int = 3,
) -> np.ndarray:
    """
    Assembles antecedent history daily arrays into a contiguous temporal tensor [H, 6, H_c, W_c].
    Sorts dates chronologically and strictly verifies that exactly H days are provided.
    """
    if len(history_dict) < history_window_days:
        raise ValueError(
            f"Insufficient history dates: expected at least {history_window_days}, "
            f"got {len(history_dict)} ({list(history_dict.keys())})"
        )

    # Sort dates chronologically and select the last H days
    sorted_dates = sorted(history_dict.keys())
    selected_dates = sorted_dates[-history_window_days:]

    arrays = [history_dict[d].astype(np.float32) for d in selected_dates]
    for i, arr in enumerate(arrays):
        if arr.ndim != 3 or arr.shape[0] != 6:
            raise ValueError(
                f"History array for date {selected_dates[i]} has invalid shape {arr.shape}. "
                f"Expected [6, H_c, W_c]"
            )

    history_tensor = np.stack(arrays, axis=0)  # [H, 6, H_c, W_c]
    return history_tensor


def assemble_forecast_tensor(
    forecast_dict: Dict[int, np.ndarray],
    lead_days: int = 7,
) -> np.ndarray:
    """
    Assembles forecast daily arrays into a contiguous lead tensor [L, 6, H_c, W_c].
    Enforces lead indices 0 through lead_days - 1.
    """
    for k in range(lead_days):
        if k not in forecast_dict:
            raise KeyError(f"Missing forecast lead index {k} in forecast_dict")

    arrays = [forecast_dict[k].astype(np.float32) for k in range(lead_days)]
    for k, arr in enumerate(arrays):
        if arr.ndim != 3 or arr.shape[0] != 6:
            raise ValueError(
                f"Forecast array for lead index {k} has invalid shape {arr.shape}. "
                f"Expected [6, H_c, W_c]"
            )

    forecast_tensor = np.stack(arrays, axis=0)  # [L, 6, H_c, W_c]
    return forecast_tensor


def assemble_target_tensor(
    target_dict: Dict[int, np.ndarray],
    lead_days: int = 7,
) -> np.ndarray:
    """
    Assembles fine reference targets into a contiguous temporal target tensor [L, 6, 80, 80].
    Enforces lead indices 0 through lead_days - 1.
    """
    for k in range(lead_days):
        if k not in target_dict:
            raise KeyError(f"Missing target lead index {k} in target_dict")

    arrays = [target_dict[k].astype(np.float32) for k in range(lead_days)]
    for k, arr in enumerate(arrays):
        if arr.ndim != 3 or arr.shape[0] != 6 or arr.shape[1] != 80 or arr.shape[2] != 80:
            raise ValueError(
                f"Target array for lead index {k} has invalid shape {arr.shape}. "
                f"Expected [6, 80, 80]"
            )

    target_tensor = np.stack(arrays, axis=0)  # [L, 6, 80, 80]
    return target_tensor


def fit_normalization_stats(
    train_array: np.ndarray,
    channel_names: Optional[List[str]] = None,
    epsilon: float = 1e-6,
) -> Dict[str, Any]:
    """
    Calculates per-channel normalization parameters strictly over training samples.
    Precipitation (channel 0) fits log1p-transformed z-score: y = log(1 + max(0, P)).
    Other channels fit standard z-score: (x - mean) / std.
    """
    channels = channel_names or WEATHER_CHANNELS
    n_channels = len(channels)

    # train_array can be [N, C, H, W] or [N, T, C, H, W]
    if train_array.ndim == 4:
        # [N, C, H, W] -> axis 1 is channel
        c_axis = 1
    elif train_array.ndim == 5:
        # [N, T, C, H, W] -> axis 2 is channel
        c_axis = 2
    else:
        raise ValueError(f"Unsupported train_array shape for normalization fit: {train_array.shape}")

    if train_array.shape[c_axis] != n_channels:
        raise ValueError(
            f"Expected {n_channels} channels along axis {c_axis}, got {train_array.shape[c_axis]}"
        )

    stats: Dict[str, Any] = {}

    for c_idx, ch_name in enumerate(channels):
        if c_axis == 1:
            ch_data = train_array[:, c_idx, :, :]
        else:
            ch_data = train_array[:, :, c_idx, :, :]

        valid_mask = np.isfinite(ch_data)
        valid_vals = ch_data[valid_mask]

        if len(valid_vals) == 0:
            raise ValueError(f"Channel '{ch_name}' contains zero finite values for normalization fitting")

        if ch_name == "precipitation" or c_idx == 0:
            # Non-linear log1p transform for heavy-tailed precipitation
            log_p = np.log1p(np.maximum(0.0, valid_vals))
            mean_val = float(np.mean(log_p))
            std_val = float(np.std(log_p)) + epsilon
            stats[ch_name] = {
                "transform": "log1p_zscore",
                "mean": mean_val,
                "std": std_val,
                "epsilon": epsilon,
                "min_physical": float(np.min(valid_vals)),
                "max_physical": float(np.max(valid_vals)),
            }
        else:
            mean_val = float(np.mean(valid_vals))
            std_val = float(np.std(valid_vals)) + epsilon
            stats[ch_name] = {
                "transform": "zscore",
                "mean": mean_val,
                "std": std_val,
                "epsilon": epsilon,
                "min_physical": float(np.min(valid_vals)),
                "max_physical": float(np.max(valid_vals)),
            }

    return stats


def apply_normalization(
    array: np.ndarray,
    stats: Dict[str, Any],
    channel_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Applies forward normalization:
      - Precipitation: y = (log1p(max(0, P)) - mean) / std
      - Others: y = (x - mean) / std
    Preserves input tensor shape: [6, H, W] or [T, 6, H, W].
    """
    channels = channel_names or WEATHER_CHANNELS
    normalized = array.copy().astype(np.float64)

    # Determine channel axis
    if normalized.ndim == 3 and normalized.shape[0] == len(channels):
        for c_idx, ch_name in enumerate(channels):
            st = stats[ch_name]
            mean = st["mean"]
            std = st["std"]
            if st.get("transform") == "log1p_zscore":
                log_val = np.log1p(np.maximum(0.0, normalized[c_idx]))
                normalized[c_idx] = (log_val - mean) / std
            else:
                normalized[c_idx] = (normalized[c_idx] - mean) / std
    elif normalized.ndim == 4 and normalized.shape[1] == len(channels):
        for c_idx, ch_name in enumerate(channels):
            st = stats[ch_name]
            mean = st["mean"]
            std = st["std"]
            if st.get("transform") == "log1p_zscore":
                log_val = np.log1p(np.maximum(0.0, normalized[:, c_idx]))
                normalized[:, c_idx] = (log_val - mean) / std
            else:
                normalized[:, c_idx] = (normalized[:, c_idx] - mean) / std
    else:
        raise ValueError(f"Array shape {array.shape} does not match channel count {len(channels)}")

    return normalized.astype(np.float32)


def invert_normalization(
    normalized_array: np.ndarray,
    stats: Dict[str, Any],
    channel_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Inverts statistical normalization back to physical units:
      - Precipitation: P = max(0, exp(y * std + mean) - 1)
      - RH: clipped to [0, 100]%
      - Others: x = y * std + mean
    """
    channels = channel_names or WEATHER_CHANNELS
    reconstructed = normalized_array.copy().astype(np.float64)

    if reconstructed.ndim == 3 and reconstructed.shape[0] == len(channels):
        for c_idx, ch_name in enumerate(channels):
            st = stats[ch_name]
            mean = st["mean"]
            std = st["std"]
            if st.get("transform") == "log1p_zscore":
                log_val = reconstructed[c_idx] * std + mean
                reconstructed[c_idx] = np.maximum(0.0, np.expm1(log_val))
            else:
                reconstructed[c_idx] = reconstructed[c_idx] * std + mean
                if ch_name == "rh":
                    reconstructed[c_idx] = np.clip(reconstructed[c_idx], 0.0, 100.0)
    elif reconstructed.ndim == 4 and reconstructed.shape[1] == len(channels):
        for c_idx, ch_name in enumerate(channels):
            st = stats[ch_name]
            mean = st["mean"]
            std = st["std"]
            if st.get("transform") == "log1p_zscore":
                log_val = reconstructed[:, c_idx] * std + mean
                reconstructed[:, c_idx] = np.maximum(0.0, np.expm1(log_val))
            else:
                reconstructed[:, c_idx] = reconstructed[:, c_idx] * std + mean
                if ch_name == "rh":
                    reconstructed[:, c_idx] = np.clip(reconstructed[:, c_idx], 0.0, 100.0)
    else:
        raise ValueError(f"Array shape {normalized_array.shape} does not match channel count {len(channels)}")

    return reconstructed.astype(np.float32)

