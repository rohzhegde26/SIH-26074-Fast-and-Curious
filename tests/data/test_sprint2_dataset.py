"""
tests/data/test_sprint2_dataset.py

TDD Test Suite for Sprint 2: Dataset Builder & Data QA Engine.
Enforces:
  1. Canonical preprocessing configuration contract and schemas.
  2. GFSMessageKey multi-step accumulation disambiguation.
  3. Dynamic H-parameterized tensor assembly and shape invariants.
  4. Train-only normalization fitting and lossless round-trip inversion.
  5. 2014 pre-operational quarantine and strict 2015-2023 forecast conditioning.
  6. Zero Open-Meteo fallback policy during dataset construction.
  7. Golden sample (2023-07-15 00Z) structural and physical verification.
"""

from datetime import date, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "data" / "preprocessing_config.yaml"


# ---------------------------------------------------------------------------
# Slice 1: Preprocessing Configuration Contract & Schemas
# ---------------------------------------------------------------------------

def test_preprocessing_config_exists_and_parses():
    assert CONFIG_PATH.exists(), f"Missing required configuration: {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)
    assert cfg["dataset_name"] == "multitask_temporal_v1"
    assert cfg["version"] == "1.0.0"
    assert cfg["temporal_convention"] == "calendar_day_00_24_utc"


def test_split_definitions_and_2014_quarantine():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    splits = cfg["splits"]
    train_years = splits["train_years"]
    val_years = splits["val_years"]
    test_years = splits["test_years"]

    assert train_years == [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    assert val_years == [2022]
    assert test_years == [2023]

    # Verify 2014 is explicitly quarantined
    assert 2014 not in train_years and 2014 not in val_years and 2014 not in test_years
    assert 2014 in splits.get("quarantined_pretrain_history_only_years", [])

    # Verify total samples arithmetic: 9 seasons * 122 days = 1098
    days_per_season = splits["season"]["days_per_season"]
    assert days_per_season == 122
    total_forecast_samples = len(train_years + val_years + test_years) * days_per_season
    assert total_forecast_samples == 1098
    assert splits["total_forecast_samples"] == 1098
    assert splits["expected_split_counts"]["train"] == 854
    assert splits["expected_split_counts"]["val"] == 122
    assert splits["expected_split_counts"]["test"] == 122


def test_channel_order_and_zero_fallback_policy():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    channels = cfg["channels"]
    expected_order = ["precipitation", "tmax", "tmin", "rh", "wind_u", "wind_v"]
    assert channels["weather_channels"] == expected_order

    # Verify zero-fallback contract
    policy = cfg["zero_fallback_policy"]
    assert policy["allow_openmeteo_construction_fallback"] is False
    assert policy["missing_data_action"] == "flag_and_exclude_sample"
    assert "ERA5" in policy["forbidden_sources_in_forecast"]
    assert "CHIRPS" in policy["forbidden_sources_in_forecast"]


def test_provenance_contract_in_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prov = cfg["provenance_contract"]
    assert prov["gfs_source"] == "NOAA_GFS"
    assert prov["era5_wind_source"] == "ECMWF_ERA5_OPENMETEO_NATIVE_UV"
    assert "ncar_rda_ds084_1" in prov["gfs_archive_tiers"]["ncar_tier_name"]
    assert "aws_open_data" in prov["gfs_archive_tiers"]["aws_tier_name"]


def test_daily_forecast_rules_in_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rules = cfg["daily_forecast_rules"]
    assert "max" in rules["tmax_rule"]
    assert "min" in rules["tmin_rule"]
    assert "mean" in rules["rh_rule"]
    assert "mean" in rules["wind_u_rule"]
    assert "mean" in rules["wind_v_rule"]
    assert "sum" in rules["precip_rule"]


def test_separate_era5_land_and_wind_sources():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sources = cfg["raw_sources"]
    assert "era5_land_thermo_file" in sources
    assert "era5_wind_file" in sources
    assert "era5_land" in sources["era5_land_thermo_file"]
    assert sources["era5_wind_file"] != sources["era5_land_thermo_file"]


def test_lead_mapping_table_integrity():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    leads = cfg["lead_mapping"]
    assert len(leads) == 7

    # Lead 0 (Day D): 0-24h
    lead0 = leads[0]
    assert lead0["day_offset"] == 0
    assert lead0["forecast_interval_hours"] == [0, 24]
    assert lead0["apcp_6h_buckets"] == [
        "0-6 hour acc fcst",
        "6-12 hour acc fcst",
        "12-18 hour acc fcst",
        "18-24 hour acc fcst",
    ]
    assert lead0["thermo_wind_lead_hours"] == [3, 6, 9, 12, 15, 18, 21, 24]

    # Lead 6 (Day D+6): 144-168h
    lead6 = leads[6]
    assert lead6["day_offset"] == 6
    assert lead6["forecast_interval_hours"] == [144, 168]
    assert lead6["apcp_6h_buckets"] == [
        "144-150 hour acc fcst",
        "150-156 hour acc fcst",
        "156-162 hour acc fcst",
        "162-168 hour acc fcst",
    ]
    assert lead6["thermo_wind_lead_hours"] == [147, 150, 153, 156, 159, 162, 165, 168]


def test_zarr_chunk_shapes():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    zl = cfg["zarr_layout"]
    assert zl["history_chunk"] == [1, 3, 6, 16, 16]
    assert zl["future_forecast_chunk"] == [1, 7, 6, 16, 16]
    assert zl["target_chunk"] == [1, 7, 6, 80, 80]
    assert zl["terrain_chunk"] == [5, 80, 80]


# ---------------------------------------------------------------------------
# Slice 2: GFSMessageKey & Multi-Step Index Disambiguation
# ---------------------------------------------------------------------------

def test_gfs_message_key_dataclass():
    from src.data.gfs_slice_downloader import GFSMessageKey

    key = GFSMessageKey(
        var_name="APCP",
        level="surface",
        forecast_step="0-6 hour acc fcst",
        byte_start=1000,
        byte_end=2000,
    )
    assert key.var_name == "APCP"
    assert key.level == "surface"
    assert key.forecast_step == "0-6 hour acc fcst"
    assert key.byte_start == 1000
    assert key.byte_end == 2000
    assert key.lookup_key == "APCP:surface:0-6 hour acc fcst"


def test_parse_gfs_idx_multi_step_disambiguation():
    from src.data.gfs_slice_downloader import parse_gfs_idx_multi_step

    sample_idx = (
        "1:0:d=2023071500:TMP:2 m above ground:3 hour fcst:\n"
        "2:1000000:d=2023071500:APCP:surface:0-6 hour acc fcst:\n"
        "3:2000000:d=2023071500:TMP:2 m above ground:6 hour fcst:\n"
        "4:3000000:d=2023071500:APCP:surface:6-12 hour acc fcst:\n"
        "5:4000000:d=2023071500:APCP:surface:12-18 hour acc fcst:\n"
        "6:5000000:d=2023071500:APCP:surface:18-24 hour acc fcst:\n"
        "7:6000000:d=2023071500:TMP:2 m above ground:24 hour fcst:\n"
    )

    msg_map = parse_gfs_idx_multi_step(sample_idx)
    assert len(msg_map) == 7

    # Check that repeated APCP:surface entries did not overwrite each other
    apcp_0_6 = msg_map["APCP:surface:0-6 hour acc fcst"]
    assert apcp_0_6.byte_start == 1000000
    assert apcp_0_6.byte_end == 1999999

    apcp_6_12 = msg_map["APCP:surface:6-12 hour acc fcst"]
    assert apcp_6_12.byte_start == 3000000
    assert apcp_6_12.byte_end == 3999999

    apcp_18_24 = msg_map["APCP:surface:18-24 hour acc fcst"]
    assert apcp_18_24.byte_start == 5000000
    assert apcp_18_24.byte_end == 5999999

    # Check repeated TMP entries
    tmp_3 = msg_map["TMP:2 m above ground:3 hour fcst"]
    assert tmp_3.byte_start == 0
    assert tmp_3.byte_end == 999999

    tmp_24 = msg_map["TMP:2 m above ground:24 hour fcst"]
    assert tmp_24.byte_start == 6000000
    assert tmp_24.byte_end is None


# ---------------------------------------------------------------------------
# Slice 3: Dynamic Tensor Assembly & Shape Invariants
# ---------------------------------------------------------------------------

def test_tensor_builder_generic_h_parameterization():
    from src.data.tensor_builder import assemble_history_tensor

    # For H = 3 days
    dummy_history_3 = {
        d: np.zeros((6, 16, 16), dtype=np.float32)
        for d in ["2023-07-12", "2023-07-13", "2023-07-14"]
    }
    t3 = assemble_history_tensor(dummy_history_3, history_window_days=3)
    assert t3.shape == (3, 6, 16, 16)
    assert t3.dtype == np.float32

    # For H = 1 day
    dummy_history_1 = {"2023-07-14": np.zeros((6, 16, 16), dtype=np.float32)}
    t1 = assemble_history_tensor(dummy_history_1, history_window_days=1)
    assert t1.shape == (1, 6, 16, 16)

    # For H = 7 days
    dummy_history_7 = {
        f"2023-07-{d:02d}": np.zeros((6, 16, 16), dtype=np.float32)
        for d in range(8, 15)
    }
    t7 = assemble_history_tensor(dummy_history_7, history_window_days=7)
    assert t7.shape == (7, 6, 16, 16)


def test_tensor_builder_forecast_and_target_shapes():
    from src.data.tensor_builder import assemble_forecast_tensor, assemble_target_tensor

    dummy_fcst = {k: np.zeros((6, 16, 16), dtype=np.float32) for k in range(7)}
    fcst_tensor = assemble_forecast_tensor(dummy_fcst, lead_days=7)
    assert fcst_tensor.shape == (7, 6, 16, 16)

    dummy_target = {k: np.zeros((6, 80, 80), dtype=np.float32) for k in range(7)}
    target_tensor = assemble_target_tensor(dummy_target, lead_days=7)
    assert target_tensor.shape == (7, 6, 80, 80)


# ---------------------------------------------------------------------------
# Slice 4: Train-Only Normalization & Inversion Fidelity
# ---------------------------------------------------------------------------

def test_normalization_round_trip_inversion_fidelity():
    from src.data.tensor_builder import fit_normalization_stats, apply_normalization, invert_normalization

    # Generate synthetic training targets: [100 samples, 6 channels, 80, 80]
    np.random.seed(42)
    n_samples = 50
    precip = np.random.exponential(scale=5.0, size=(n_samples, 1, 80, 80)).astype(np.float32)
    tmax = np.random.uniform(25.0, 35.0, size=(n_samples, 1, 80, 80)).astype(np.float32)
    tmin = tmax - np.random.uniform(2.0, 8.0, size=(n_samples, 1, 80, 80)).astype(np.float32)
    rh = np.random.uniform(40.0, 95.0, size=(n_samples, 1, 80, 80)).astype(np.float32)
    wind_u = np.random.normal(2.0, 5.0, size=(n_samples, 1, 80, 80)).astype(np.float32)
    wind_v = np.random.normal(-1.0, 4.0, size=(n_samples, 1, 80, 80)).astype(np.float32)

    raw_train = np.concatenate([precip, tmax, tmin, rh, wind_u, wind_v], axis=1)

    # Fit statistics
    stats = fit_normalization_stats(raw_train)
    assert "precipitation" in stats
    assert stats["precipitation"]["transform"] == "log1p_zscore"
    assert stats["precipitation"]["mean"] > 0.0
    assert stats["precipitation"]["std"] > 0.0

    # Pick a sample and test forward + backward transform
    sample = raw_train[0]  # [6, 80, 80]
    normalized = apply_normalization(sample, stats)
    reconstructed = invert_normalization(normalized, stats)

    # Inversion error must be negligible (< 1e-5)
    max_error = np.max(np.abs(sample - reconstructed))
    assert max_error < 1e-5, f"Excessive round-trip normalization inversion error: {max_error}"
    assert np.all(reconstructed[0] >= 0.0), "Precipitation must remain non-negative after inversion"


# ---------------------------------------------------------------------------
# Slice 5: Phase 1 Golden Sample Assembly & Invariant Verification
# ---------------------------------------------------------------------------

def test_golden_sample_assembly_and_invariants():
    from src.data.build_dataset import DatasetBuilder

    builder = DatasetBuilder()
    golden = builder.build_golden_sample()

    assert golden["sample_id"] == "20230715_00Z"
    assert golden["init_date"] == "2023-07-15"
    assert golden["split"] == "test"
    assert golden["history"].shape == (3, 6, 16, 16)
    assert golden["future_forecast"].shape == (7, 6, 16, 16)
    assert golden["terrain"].shape == (5, 80, 80)
    assert golden["target"].shape == (7, 6, 80, 80)

    # Anti-leakage: history strictly through D-1
    assert golden["history_end_date"] == "2023-07-14"
    assert golden["history_end_date"] < golden["init_date"]

    # Physical checks
    assert np.all(golden["target"][:, 0] >= 0.0), "Precipitation must be non-negative"
    assert np.all(golden["future_forecast"][:, 0] >= 0.0), "Forecast precipitation must be non-negative"
    assert np.all((golden["target"][:, 3] >= 0.0) & (golden["target"][:, 3] <= 100.0)), "RH out of bounds"
    assert np.all(golden["target"][:, 1] >= golden["target"][:, 2]), "Tmax must be >= Tmin in target"
    assert np.all(golden["future_forecast"][:, 1] >= golden["future_forecast"][:, 2]), "Tmax must be >= Tmin in forecast"

    # Zero NaNs
    assert not np.isnan(golden["history"]).any()
    assert not np.isnan(golden["future_forecast"]).any()
    assert not np.isnan(golden["terrain"]).any()
    assert not np.isnan(golden["target"]).any()
    assert golden["qa_status"] == "PASSED"


# ---------------------------------------------------------------------------
# Slice 6: Normalization Stats & Parquet Catalog Integrity
# ---------------------------------------------------------------------------

def test_normalization_stats_persisted_file():
    stats_path = ROOT / "data" / "normalization_stats.yaml"
    assert stats_path.exists(), f"Missing {stats_path}"

    with open(stats_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    assert doc["dataset_name"] == "multitask_temporal_v1"
    assert doc["computed_split"] == "train"
    assert doc["train_years"] == [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    assert doc["train_sample_count"] == 854
    assert doc["train_lead_days"] == 7
    assert doc["total_training_observations"] == 5978

    channels = doc["channels"]
    assert "precipitation" in channels
    assert channels["precipitation"]["transform"] == "log1p_zscore"
    for ch in ["precipitation", "tmax", "tmin", "rh", "wind_u", "wind_v"]:
        assert ch in channels
        assert np.isfinite(channels[ch]["mean"])
        assert np.isfinite(channels[ch]["std"])
        assert channels[ch]["std"] > 0.0


def test_sample_index_parquet_catalog():
    idx_path = ROOT / "data" / "sample_index.parquet"
    assert idx_path.exists(), f"Missing {idx_path}"

    df = pd.read_parquet(idx_path)
    assert len(df) == 1098, f"Expected 1,098 samples, got {len(df)}"

    # Check splits
    splits = df["split"].value_counts().to_dict()
    assert splits["train"] == 854
    assert splits["val"] == 122
    assert splits["test"] == 122

    # Check 2014 quarantine
    assert not any(d.startswith("2014") for d in df["init_date"]), "2014 sample leak detected"

    # Check zero leakage
    leakage_mask = df["history_end_date"] >= df["init_date"]
    assert not leakage_mask.any(), f"Found {leakage_mask.sum()} anti-leakage violations"

    # Check 100% QA pass
    qa_counts = df["qa_status"].value_counts().to_dict()
    assert qa_counts.get("PASSED") == 1098


# ---------------------------------------------------------------------------
# Slice 7: Zarr Store & Manifest Integrity
# ---------------------------------------------------------------------------

def test_zarr_multitask_temporal_v1_store():
    import zarr
    z_path = ROOT / "datasets" / "multitask_temporal_v1.zarr"
    assert z_path.exists(), f"Missing {z_path}"

    store = zarr.open_group(str(z_path), mode="r")
    assert "history" in store
    assert "future_forecast" in store
    assert "target" in store
    assert "terrain" in store
    assert "sample_ids" in store
    assert "dates" in store
    assert "splits" in store

    assert store["history"].shape == (1098, 3, 6, 16, 16)
    assert store["future_forecast"].shape == (1098, 7, 6, 16, 16)
    assert store["target"].shape == (1098, 7, 6, 80, 80)
    assert store["terrain"].shape == (5, 80, 80)

    # Chunk shapes
    assert store["history"].chunks == (1, 3, 6, 16, 16)
    assert store["future_forecast"].chunks == (1, 7, 6, 16, 16)
    assert store["target"].chunks == (1, 7, 6, 80, 80)
    assert store["terrain"].chunks == (5, 80, 80)


def test_dataset_qa_report_and_manifest():
    man_path = ROOT / "data" / "dataset_manifest.yaml"
    qa_path = ROOT / "data" / "dataset_qa_report.md"

    assert man_path.exists(), f"Missing {man_path}"
    assert qa_path.exists(), f"Missing {qa_path}"

    with open(man_path, "r", encoding="utf-8") as f:
        man = yaml.safe_load(f)
    assert man["total_samples"] == 1098
    assert man["splits"] == {"train": 854, "val": 122, "test": 122}

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_text = f.read()
    assert "[PASS] ALL GATES PASSED (100%)" in qa_text
    assert "\u2014" not in qa_text, "Em dash violation detected in QA report!"


# ---------------------------------------------------------------------------
# Slice 8: PyTorch Spatiotemporal Downscaling Dataset & Streaming Loaders
# ---------------------------------------------------------------------------

def test_pytorch_spatiotemporal_dataset_and_loaders():
    from src.data.temporal_dataset import SpatiotemporalDownscalingDataset, get_temporal_dataloaders

    train_ds = SpatiotemporalDownscalingDataset(split="train")
    val_ds = SpatiotemporalDownscalingDataset(split="val")
    test_ds = SpatiotemporalDownscalingDataset(split="test")

    assert len(train_ds) == 854
    assert len(val_ds) == 122
    assert len(test_ds) == 122

    sample = train_ds[0]
    assert sample["history"].shape == (3, 6, 16, 16)
    assert sample["future_forecast"].shape == (7, 6, 16, 16)
    assert sample["terrain"].shape == (5, 80, 80)
    assert sample["target"].shape == (7, 6, 80, 80)

    # Test DataLoader batch streaming
    train_loader, val_loader, test_loader = get_temporal_dataloaders(batch_size=4)
    batch = next(iter(train_loader))
    assert batch["history"].shape == (4, 3, 6, 16, 16)
    assert batch["future_forecast"].shape == (4, 7, 6, 16, 16)
    assert batch["terrain"].shape == (4, 5, 80, 80)
    assert batch["target"].shape == (4, 7, 6, 80, 80)

    # Test target unnormalization fidelity
    unnorm = train_ds.unnormalize_target(sample["target"])
    assert unnorm.shape == (7, 6, 80, 80)
    assert np.all(unnorm[:, 0] >= 0.0), "Unnormalized precipitation must be non-negative"


# ---------------------------------------------------------------------------
# Slice 9: Provenance Integrity, Dual Backends & Gate 7/8 Verification
# ---------------------------------------------------------------------------

def test_circular_wind_std_calculation():
    from src.data.era5_wind_ingestion import compute_circular_std_deg

    # Synthetic fixed 250 deg wind must yield ~0.0 deg variance
    s = np.array([5.0, 10.0, 15.0])
    theta = np.radians(250.0)
    u_fix = -s * np.sin(theta)
    v_fix = -s * np.cos(theta)
    assert compute_circular_std_deg(u_fix, v_fix) < 1.0

    # Authentic variable wind must yield > 15.0 deg variance
    u_var = np.array([5.0, -3.0, 8.0, -10.0, 2.0])
    v_var = np.array([2.0, 6.0, -4.0, 1.0, -8.0])
    assert compute_circular_std_deg(u_var, v_var) > 15.0


def test_fine_grid_coordinate_registration():
    import xarray as xr
    ch_path = ROOT / "data" / "raw" / "chirps" / "chirps_daily.nc"
    dem_path = ROOT / "data" / "raw" / "dem" / "glo30_mandya_terrain.nc"

    with xr.open_dataset(ch_path) as ch, xr.open_dataset(dem_path) as dem:
        assert ch.lat.shape == (80,)
        assert dem.lat.shape == (80,)
        # Exact DEM fine grid equality with CHIRPS fine grid
        max_dem_lat_diff = float(np.max(np.abs(dem.lat.values - ch.lat.values)))
        max_dem_lon_diff = float(np.max(np.abs(dem.lon.values - ch.lon.values)))
        assert max_dem_lat_diff < 1e-5, f"DEM lat mismatch: {max_dem_lat_diff}"
        assert max_dem_lon_diff < 1e-5, f"DEM lon mismatch: {max_dem_lon_diff}"


def test_dual_gfs_backends_routing():
    from src.data.gfs_slice_downloader import AWSGFSBackend, NCARGFSBackend

    aws_backend = AWSGFSBackend()
    ncar_backend = NCARGFSBackend()

    # AWS key formatting for 2023
    key_2023 = aws_backend.format_s3_key(date(2023, 7, 15), cycle_hour=0, forecast_hour=24)
    assert "gfs.20230715/00/atmos/gfs.t00z.pgrb2.0p25.f024" in key_2023

    # NCAR URL formatting for 2018
    url_2018 = ncar_backend.format_fileserver_url(date(2018, 7, 15), cycle_hour=0, forecast_hour=24)
    assert "thredds.rda.ucar.edu" in url_2018
    assert "2018071500.f024.grib2" in url_2018


def test_validator_gate7_and_gate8():
    from src.data.validate_dataset import DatasetValidator

    validator = DatasetValidator()
    results = validator.run_all_gates()

    hg = results["hard_gates"]
    assert "gate7_provenance_integrity" in hg
    assert "gate8_sample_count_completeness" in hg

    assert hg["gate7_provenance_integrity"]["passed"] is True, f"Gate 7 failed: {hg['gate7_provenance_integrity']['details']}"
    assert hg["gate8_sample_count_completeness"]["passed"] is True, f"Gate 8 failed: {hg['gate8_sample_count_completeness']['details']}"
    assert results["all_hard_gates_passed"] is True


def test_ncar_ncss_subset_extraction():
    from src.data.gfs_slice_downloader import NCARGFSBackend
    ncar = NCARGFSBackend()
    res = ncar.fetch_ncss_subset(date(2018, 7, 15), cycle_hour=0, forecast_hour=6)
    assert res["t2m"].shape == (16, 16)
    assert res["rh"].shape == (16, 16)
    assert res["u"].shape == (16, 16)
    assert res["v"].shape == (16, 16)
    assert res["p"] is not None and res["p"].shape == (16, 16)
    assert "thredds.rda.ucar.edu" in res["source_url"]


def test_gfs_3hourly_aggregation_math():
    from src.data.gfs_slice_downloader import extract_daily_gfs_lead_aggregated
    lead_arr, cum_p, files = extract_daily_gfs_lead_aggregated(date(2018, 7, 15), day_offset=0)
    assert lead_arr.shape == (6, 16, 16)
    tmax = lead_arr[1]
    tmin = lead_arr[2]
    assert np.all(tmax >= tmin), "Tmax must be >= Tmin across all grid cells"
    assert np.all(lead_arr[0] >= 0.0), "Daily precipitation must be non-negative"
    assert len(files) == 8, f"Expected 8 3-hourly step files, got {len(files)}"


def test_boundary_observations_all_channels():
    import xarray as xr
    ch = xr.open_dataset(ROOT / "data" / "raw" / "chirps" / "chirps_daily.nc")
    w = xr.open_dataset(ROOT / "data" / "raw" / "era5" / "era5_wind_daily.nc")

    boundary_dates = ["2023-05-28", "2023-05-31", "2023-10-01", "2023-10-07"]
    times_ch = [str(t)[:10] for t in ch.time.values]
    times_w = [str(t)[:10] for t in w.time.values]

    for d in boundary_dates:
        assert d in times_ch, f"{d} missing in CHIRPS"
        assert d in times_w, f"{d} missing in ERA5 wind"


def test_gate7_rejects_synthetic_cache():
    from src.data.validate_dataset import DatasetValidator
    validator = DatasetValidator()

    fake_file = ROOT / "data" / "raw" / "forecast" / "gfs" / "gfs_fake_synthetic_16x16.npz"
    try:
        np.savez_compressed(
            fake_file,
            forecast=np.zeros((7, 6, 16, 16), dtype=np.float32),
            is_synthetic=True,
            grib_magic_verified=False,
            source_files=[],
        )
        passed, msg, details = validator._check_source_and_provenance_integrity()
        assert passed is False, "Gate 7 must reject synthetic GFS cache"
        assert any("fake_synthetic" in s for s in details["synthetic_or_invalid_gfs_caches"])
    finally:
        if fake_file.exists():
            fake_file.unlink()

