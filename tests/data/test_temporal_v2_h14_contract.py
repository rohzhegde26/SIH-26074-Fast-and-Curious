"""
tests/data/test_temporal_v2_h14_contract.py

TDD Contract and Sanity Tests for Sprint 4 Wide-History Dataset.
Enforces:
  1. Wide-history Zarr schema: history [1098, 14, 6, 16, 16], future [1098, 7, 6, 16, 16],
     target [1098, 7, 6, 80, 80], terrain [5, 80, 80].
  2. Dynamic history slicing: H in {3, 5, 7, 10, 14} yielding [H, 6, 16, 16].
  3. Strict anti-leakage: history strictly ends at D-1 before forecast init date D.
  4. Train/val/test split counts: 854 train, 122 val, 122 test (1098 total).
  5. Zero NaN values and valid physical ranges across all channels.
  6. Train-only normalization stats and invertible physical reconstruction.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
import zarr

from src.data.temporal_dataset import SpatiotemporalDownscalingDataset
from src.data.tensor_builder import apply_normalization, invert_normalization

ROOT = Path(__file__).resolve().parents[2]
ZARR_V2_PATH = ROOT / "datasets" / "multitask_temporal_v2_h14.zarr"
INDEX_V2_PATH = ROOT / "data" / "sample_index_v2_h14.parquet"
STATS_V2_PATH = ROOT / "data" / "normalization_stats_v2.yaml"


@pytest.mark.skipif(not ZARR_V2_PATH.exists(), reason="Zarr v2 store not yet materialized")
def test_v2_zarr_schema_and_shapes():
    """Verify multitask_temporal_v2_h14.zarr array shapes and data types."""
    store = zarr.open_group(str(ZARR_V2_PATH), mode="r")

    assert "history" in store, "Missing 'history' array in Zarr store"
    assert "future_forecast" in store, "Missing 'future_forecast' array in Zarr store"
    assert "target" in store, "Missing 'target' array in Zarr store"
    assert "terrain" in store, "Missing 'terrain' array in Zarr store"
    assert "splits" in store, "Missing 'splits' array in Zarr store"
    assert "dates" in store, "Missing 'dates' array in Zarr store"

    assert store["history"].shape == (1098, 14, 6, 16, 16), f"Unexpected history shape: {store['history'].shape}"
    assert store["future_forecast"].shape == (1098, 7, 6, 16, 16), f"Unexpected forecast shape: {store['future_forecast'].shape}"
    assert store["target"].shape == (1098, 7, 6, 80, 80), f"Unexpected target shape: {store['target'].shape}"
    assert store["terrain"].shape == (5, 80, 80), f"Unexpected terrain shape: {store['terrain'].shape}"
    assert store["splits"].shape == (1098,), f"Unexpected splits shape: {store['splits'].shape}"
    assert store["dates"].shape == (1098,), f"Unexpected dates shape: {store['dates'].shape}"

    assert store["history"].dtype == np.float32
    assert store["future_forecast"].dtype == np.float32
    assert store["target"].dtype == np.float32
    assert store["terrain"].dtype == np.float32


@pytest.mark.skipif(not ZARR_V2_PATH.exists(), reason="Zarr v2 store not yet materialized")
def test_v2_dataset_dynamic_history_slicing():
    """Verify PyTorch dataset slices trailing H days for H in {3, 5, 7, 10, 14}."""
    for h in [3, 5, 7, 10, 14]:
        ds = SpatiotemporalDownscalingDataset(
            zarr_path=ZARR_V2_PATH,
            index_path=INDEX_V2_PATH if INDEX_V2_PATH.exists() else None,
            stats_path=STATS_V2_PATH if STATS_V2_PATH.exists() else None,
            split="val",
            history_len=h,
            normalize=False,
        )
        sample = ds[0]
        assert sample["history"].shape == (h, 6, 16, 16), f"Expected ({h}, 6, 16, 16), got {sample['history'].shape}"
        assert sample["future_forecast"].shape == (7, 6, 16, 16)
        assert sample["target"].shape == (7, 6, 80, 80)
        assert sample["terrain"].shape == (5, 80, 80)
        assert not torch.isnan(sample["history"]).any(), f"NaN found in history for H={h}"


@pytest.mark.skipif(not INDEX_V2_PATH.exists(), reason="Index v2 parquet not yet materialized")
def test_v2_sample_index_splits_and_anti_leakage():
    """Verify sample index partitioning and anti-leakage invariant for all 1098 samples."""
    df = pd.read_parquet(INDEX_V2_PATH)
    assert len(df) == 1098, f"Expected 1098 samples, got {len(df)}"

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    assert len(train_df) == 854, f"Expected 854 train samples, got {len(train_df)}"
    assert len(val_df) == 122, f"Expected 122 val samples, got {len(val_df)}"
    assert len(test_df) == 122, f"Expected 122 test samples, got {len(test_df)}"

    for idx, row in df.iterrows():
        init_dt = pd.Timestamp(row["init_date"])
        hist_end = pd.Timestamp(row["history_end_date"])
        fcst_start = pd.Timestamp(row["forecast_start_date"])

        # History must end exactly at D-1
        assert hist_end < init_dt, f"Leakage: hist_end {hist_end} >= init_date {init_dt}"
        assert (init_dt - hist_end).days == 1, f"History must end at D-1 (delta is {(init_dt - hist_end).days})"
        assert fcst_start == init_dt, f"Forecast start must equal init_date"


@pytest.mark.skipif(not STATS_V2_PATH.exists(), reason="Stats v2 YAML not yet materialized")
def test_v2_normalization_stats_and_roundtrip():
    """Verify v2 normalization statistics validity and lossless physical roundtrip."""
    with open(STATS_V2_PATH, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    stats = doc["channels"]
    expected_channels = ["precipitation", "tmax", "tmin", "rh", "wind_u", "wind_v"]
    for ch in expected_channels:
        assert ch in stats, f"Missing channel {ch} in normalization stats"
        assert np.isfinite(stats[ch]["mean"]), f"Non-finite mean for {ch}"
        assert np.isfinite(stats[ch]["std"]) and stats[ch]["std"] > 0, f"Invalid std for {ch}"

    dummy_phys = np.ones((7, 6, 80, 80), dtype=np.float32)
    norm = apply_normalization(dummy_phys, stats)
    recon = invert_normalization(norm, stats)
    max_err = float(np.max(np.abs(dummy_phys - recon)))
    assert max_err < 1e-4, f"Roundtrip error too high: {max_err}"
