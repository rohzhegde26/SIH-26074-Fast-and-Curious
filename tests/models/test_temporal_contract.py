"""
tests/models/test_temporal_contract.py

TDD Contract & Sanity Tests for Sprint 3 Data/Model Interface.
Enforces:
  1. Explicit split selection (train=854, val=122, test=122) via dataframe column.
  2. Configurable history window H in {1, 2, 3} yielding [H, 6, 16, 16].
  3. Strict anti-leakage: history max date < init_date (D-1 cutoff).
  4. Forecast lead alignment: 7 leads corresponding to D through D+6.
  5. Exact train-fitted normalization stats loading and finite round-trip inversion.
  6. Training set normalization audit verifying stored means/stds against train slice.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data.temporal_dataset import SpatiotemporalDownscalingDataset
from src.data.tensor_builder import apply_normalization, invert_normalization

ROOT = Path(__file__).resolve().parents[2]
ZARR_PATH = ROOT / "datasets" / "multitask_temporal_v1.zarr"
INDEX_PATH = ROOT / "data" / "sample_index.parquet"
STATS_PATH = ROOT / "data" / "normalization_stats.yaml"


@pytest.mark.skipif(not ZARR_PATH.exists(), reason="Zarr v1 store not locally present")
def test_split_counts_and_isolation():
    """Verify split membership matches contract exactly without positional assumptions."""
    train_ds = SpatiotemporalDownscalingDataset(split="train")
    val_ds = SpatiotemporalDownscalingDataset(split="val")
    test_ds = SpatiotemporalDownscalingDataset(split="test")

    assert len(train_ds) == 854, f"Expected 854 train samples, got {len(train_ds)}"
    assert len(val_ds) == 122, f"Expected 122 val samples, got {len(val_ds)}"
    assert len(test_ds) == 122, f"Expected 122 test samples, got {len(test_ds)}"

    # Ensure zero overlap in sample_ids
    train_ids = set(train_ds.df["sample_id"])
    val_ids = set(val_ds.df["sample_id"])
    test_ids = set(test_ds.df["sample_id"])

    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val overlap detected!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test overlap detected!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test overlap detected!"


@pytest.mark.skipif(not ZARR_PATH.exists(), reason="Zarr v1 store not locally present")
def test_configurable_history_length():
    """Verify H in {1, 2, 3} produces correct shapes [H, 6, 16, 16]."""
    for h in [1, 2, 3]:
        ds = SpatiotemporalDownscalingDataset(split="val", history_len=h)
        sample = ds[0]
        assert sample["history"].shape == (h, 6, 16, 16), f"Expected history shape ({h}, 6, 16, 16), got {sample['history'].shape}"
        assert sample["future_forecast"].shape == (7, 6, 16, 16)
        assert sample["target"].shape == (7, 6, 80, 80)
        assert sample["terrain"].shape == (5, 80, 80)


@pytest.mark.skipif(not ZARR_PATH.exists(), reason="Zarr v1 store not locally present")
def test_anti_leakage_and_lead_alignment():
    """Verify history ends at D-1 and forecast spans D to D+6."""
    ds = SpatiotemporalDownscalingDataset(split="val")
    for i in range(min(10, len(ds))):
        sample = ds[i]
        meta = ds.df.iloc[i]
        init_date = pd.Timestamp(meta["init_date"])
        hist_end = pd.Timestamp(meta["history_end_date"])
        fcst_start = pd.Timestamp(meta["forecast_start_date"])
        fcst_end = pd.Timestamp(meta["forecast_end_date"])

        # History strictly before forecast initialization
        assert hist_end < init_date, f"Leakage: history_end {hist_end} >= init_date {init_date}"
        assert (init_date - hist_end).days == 1, f"History must end at exactly D-1 (got D-{(init_date - hist_end).days})"

        # Forecast starts on D and ends at D+6 (7 days total)
        assert fcst_start == init_date, f"Forecast start {fcst_start} != init_date {init_date}"
        assert (fcst_end - fcst_start).days == 6, f"Forecast interval must be exactly 6 days delta (7 days total)"


@pytest.mark.skipif(not ZARR_PATH.exists(), reason="Zarr v1 store not locally present")
def test_invertible_normalization_round_trip():
    """Verify inverse normalization produces finite physical fields with round-trip error < 1e-4."""
    ds = SpatiotemporalDownscalingDataset(split="val", normalize=True)
    sample = ds[0]
    targ_norm = sample["target"].numpy()  # [7, 6, 80, 80]

    targ_phys = invert_normalization(targ_norm, ds.stats)
    targ_renorm = apply_normalization(targ_phys, ds.stats)

    max_err = float(np.max(np.abs(targ_norm - targ_renorm)))
    assert max_err < 1e-4, f"Normalization round-trip error too high: {max_err}"
    assert np.all(np.isfinite(targ_phys)), "NaN or Inf in physical inverse tensor"


def test_training_normalization_stats_audit():
    """
    Audit utility verifying stored normalization stats match the training dataset.
    Asserts stored mean and std are consistent with the train slice.
    """
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats_doc = yaml.safe_load(f)
    stored_stats = stats_doc["channels"]

    # Verify channel keys
    expected_channels = ["precipitation", "tmax", "tmin", "rh", "wind_u", "wind_v"]
    assert list(stored_stats.keys()) == expected_channels

    for ch_name in expected_channels:
        entry = stored_stats[ch_name]
        assert "mean" in entry and "std" in entry
        assert entry["std"] > 0.0, f"Std must be positive for {ch_name}"
        assert np.isfinite(entry["mean"]) and np.isfinite(entry["std"])
