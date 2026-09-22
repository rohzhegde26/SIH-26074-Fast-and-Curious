"""
tests/data/test_sprint1_sources.py

TDD test suite for Sprint 1: Data Audit & Source Finalization.
Enforces:
  1. Source manifest schema, licensing, and provenance taxonomy.
  2. Domain configuration geometry and scalability from M (4°x4°) to 2.5M (10°x10°).
  3. Strict anti-leakage 00Z forecast-history boundary.
  4. Exact 6-variable GFS forecast derivation and accumulation rules.
  5. 2015–2023 forecast archive boundary enforcement (excluding 2014 from forecast conditioning).
  6. NOAA station independent sanity-check layer constraints.
"""

from datetime import datetime, date
from pathlib import Path
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Slice 1: Source Manifest Schema & Provenance Taxonomy
# ---------------------------------------------------------------------------

def test_source_manifest_exists_and_parses():
    manifest_path = ROOT / "data" / "source_manifest.yaml"
    assert manifest_path.exists(), f"Missing required deliverable: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    assert isinstance(manifest, dict), "source_manifest.yaml must be a valid YAML mapping"
    assert "streams" in manifest, "Manifest must declare top-level 'streams' mapping"


def test_source_manifest_provenance_and_licensing():
    manifest_path = ROOT / "data" / "source_manifest.yaml"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    valid_provenance_classes = {
        "reanalysis",
        "satellite_gauge_product",
        "numerical_weather_prediction",
        "terrain_dsm",
        "direct_observation",
        "mixed",
    }

    streams = manifest.get("streams", {})
    required_streams = ["history", "forecast", "target", "prior", "station_validation"]
    for s in required_streams:
        assert s in streams, f"Required stream '{s}' missing in source_manifest.yaml"

    # Verify provenance classes
    for stream_name, stream_cfg in streams.items():
        prov = stream_cfg.get("provenance_class")
        assert prov in valid_provenance_classes, (
            f"Stream '{stream_name}' has invalid provenance_class '{prov}'. "
            f"Must be one of {valid_provenance_classes}"
        )
        assert "license" in stream_cfg, f"Stream '{stream_name}' must document dataset license"
        assert "access_terms" in stream_cfg, f"Stream '{stream_name}' must document API access terms"


def test_temporal_convention_binding():
    manifest_path = ROOT / "data" / "source_manifest.yaml"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    domain_path = ROOT / "data" / "domain_config.yaml"
    with open(domain_path, "r", encoding="utf-8") as f:
        domain = yaml.safe_load(f)

    for doc_name, doc in [("source_manifest.yaml", manifest), ("domain_config.yaml", domain)]:
        assert "temporal_convention" in doc, f"Missing 'temporal_convention' in {doc_name}"
        tc = doc["temporal_convention"]
        assert tc.get("selected") == "calendar_day_00_24_utc", (
            f"Expected selected: 'calendar_day_00_24_utc' in {doc_name}, got {tc.get('selected')}"
        )
        assert tc.get("status") == "binding", (
            f"Expected status: 'binding' in {doc_name}, got {tc.get('status')}"
        )


def test_no_ground_truth_claims_in_manifest():
    manifest_path = ROOT / "data" / "source_manifest.yaml"
    content = manifest_path.read_text(encoding="utf-8").lower()
    assert "ground truth" not in content, (
        "Manifest must not claim 'ground truth' for satellite/reanalysis target products. "
        "Use 'supervision target' or 'reference target'."
    )


# ---------------------------------------------------------------------------
# Slice 2: Domain Configuration Geometry & Scaling (M to 2.5M)
# ---------------------------------------------------------------------------

def test_domain_config_exists_and_parses():
    domain_path = ROOT / "data" / "domain_config.yaml"
    assert domain_path.exists(), f"Missing required deliverable: {domain_path}"
    with open(domain_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), "domain_config.yaml must be a valid YAML mapping"


def test_domain_geometry_and_scaling_ratios():
    domain_path = ROOT / "data" / "domain_config.yaml"
    with open(domain_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    core = cfg["core_target_domain"]
    assert core["lat_bounds"] == [11.0, 15.0]
    assert core["lon_bounds"] == [74.0, 78.0]
    assert core["fine_grid"]["shape"] == [80, 80]
    assert core["fine_grid"]["resolution_deg"] == 0.05
    assert core["coarse_grid"]["shape"] == [16, 16]
    assert core["coarse_grid"]["resolution_deg"] == 0.25
    assert core["scale_factor"] == 5

    # Check candidate ratios
    ratios = cfg["candidate_context_ratios"]
    assert ratios == [1.0, 1.25, 1.5, 1.75, 2.0, 2.5], "Candidate ratios must include 1.0x through 2.5x"

    # Maximum context (2.5M)
    max_ctx = cfg["max_context_domain"]
    assert max_ctx["lat_bounds"] == [8.0, 18.0], "2.5M lat bounds must span 10.0 degrees"
    assert max_ctx["lon_bounds"] == [71.0, 81.0], "2.5M lon bounds must span 10.0 degrees"
    assert max_ctx["shape"] == [40, 40], "40 cells at 0.25 deg = 10.0 deg"


# ---------------------------------------------------------------------------
# Slice 3: Strict 00Z Forecast-History Boundary & Anti-Leakage
# ---------------------------------------------------------------------------

def test_forecast_history_leakage_boundary():
    from src.data.source_auditor import verify_temporal_leakage_boundary, DataLeakageError

    # Scenario 1: Valid 00Z forecast on Day D (e.g. 2023-07-15 00:00 UTC)
    # Completed history is strictly through Day D-1 (2023-07-14 24:00 / 2023-07-15 00:00 UTC)
    forecast_init = datetime(2023, 7, 15, 0, 0)
    valid_history_end = datetime(2023, 7, 15, 0, 0)
    assert verify_temporal_leakage_boundary(forecast_init, valid_history_end) is True

    # Scenario 2: Earlier history (e.g. up to D-1 18:00 UTC) is also safe
    earlier_history = datetime(2023, 7, 14, 18, 0)
    assert verify_temporal_leakage_boundary(forecast_init, earlier_history) is True

    # Scenario 3: Leakage! Attempting to include full Day D observation (e.g. 2023-07-15 23:59 UTC)
    # Day D observation has not happened yet at 00Z initialization on Day D!
    leaked_history = datetime(2023, 7, 15, 23, 59)
    with pytest.raises(DataLeakageError):
        verify_temporal_leakage_boundary(forecast_init, leaked_history)


# ---------------------------------------------------------------------------
# Slice 4: GFS 6-Variable Derivation & Accumulation Rules
# ---------------------------------------------------------------------------

def test_gfs_6_variable_mapping_in_manifest():
    manifest_path = ROOT / "data" / "source_manifest.yaml"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    forecast_vars = manifest["streams"]["forecast"]["variables"]
    expected_vars = ["rain", "tmax", "tmin", "rh", "wind_u", "wind_v"]
    for v in expected_vars:
        assert v in forecast_vars, f"Variable '{v}' missing in GFS forecast mapping"
        cfg = forecast_vars[v]
        assert "grib_key" in cfg, f"'{v}' must declare 'grib_key'"
        assert "level" in cfg, f"'{v}' must declare 'level'"
        assert "accumulation_or_step_rule" in cfg, f"'{v}' must declare 'accumulation_or_step_rule'"
        assert "daily_aggregation_rule" in cfg, f"'{v}' must declare 'daily_aggregation_rule'"
        assert "target_unit" in cfg, f"'{v}' must declare 'target_unit'"


def test_gfs_accumulation_and_thermo_derivation_logic():
    from src.data.source_auditor import (
        derive_gfs_daily_precipitation,
        derive_gfs_daily_temperatures,
        derive_gfs_daily_wind,
    )

    # 1. Precipitation accumulation: 6-hour interval steps [0-6h, 6-12h, 12-18h, 18-24h]
    # In NOAA GFS, APCP is either 6-hour buckets or de-accumulated steps
    apcp_buckets = np.array([5.2, 10.4, 3.1, 8.5])  # mm in each 6-hour bucket
    daily_p = derive_gfs_daily_precipitation(apcp_buckets, is_bucket=True)
    assert np.isclose(daily_p, 27.2), f"Expected 27.2 mm, got {daily_p}"
    assert daily_p >= 0.0

    # 2. Daily Tmax / Tmin derivation from 3-hourly temperature sequence (Kelvin -> Celsius)
    # Sequence of 8 3-hourly forecast values across day D
    tmp_kelvin = np.array([295.15, 298.15, 303.15, 306.15, 304.15, 301.15, 297.15, 294.15])
    tmax, tmin = derive_gfs_daily_temperatures(tmp_kelvin)
    assert np.isclose(tmax, 33.0), f"Expected Tmax 33.0 C, got {tmax}"
    assert np.isclose(tmin, 21.0), f"Expected Tmin 21.0 C, got {tmin}"
    assert tmax >= tmin, "Physical invariant violated: Tmax must be >= Tmin"

    # 3. Wind speed from vector components
    u_vals = np.array([3.0, 4.0, 5.0, 4.0])
    v_vals = np.array([4.0, 3.0, 0.0, -3.0])
    mean_u, mean_v, scalar_speed = derive_gfs_daily_wind(u_vals, v_vals)
    assert np.isclose(mean_u, 4.0)
    assert np.isclose(mean_v, 1.0)
    assert scalar_speed >= 0.0


# ---------------------------------------------------------------------------
# Slice 5: Forecast Archive Coverage Boundary (2015–2023 vs 2014)
# ---------------------------------------------------------------------------

def test_forecast_archive_date_validation():
    from src.data.source_auditor import validate_forecast_archive_year

    # 2015 through 2023 must be supported for forecast conditioning
    for y in range(2015, 2024):
        assert validate_forecast_archive_year(y, mode="forecast_conditioned") is True

    # 2014 must fail for forecast conditioning (no authentic GFS 0.25° AWS archive in 2014)
    with pytest.raises(ValueError) as exc:
        validate_forecast_archive_year(2014, mode="forecast_conditioned")
    assert "2015-2023" in str(exc.value)

    # But 2014 is valid in history_only or target_only mode
    assert validate_forecast_archive_year(2014, mode="history_only") is True
    assert validate_forecast_archive_year(2014, mode="target_only") is True


# ---------------------------------------------------------------------------
# Slice 6: NOAA Station Sanity-Check Constraints
# ---------------------------------------------------------------------------

def test_noaa_station_sanity_layer_rules():
    manifest_path = ROOT / "data" / "source_manifest.yaml"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    stn = manifest["streams"]["station_validation"]
    assert stn["role"] == "independent_point_validation"
    assert stn["spatially_interpolated"] is False, (
        "Station data must NOT be gridded or spatially interpolated into pseudo-observations"
    )
    assert "supported_variables" in stn
    assert len(stn["supported_variables"]) < 6, (
        "Station layer must explicitly document that it does NOT provide the full 6-variable set"
    )


def test_peninsular_stations_catalog_integrity():
    from src.data.noaa_station_auditor import PENINSULAR_STATIONS
    assert len(PENINSULAR_STATIONS) >= 5
    for stn in PENINSULAR_STATIONS:
        assert 11.0 <= stn["lat"] <= 15.0, f"Station {stn['name']} outside target latitude bounds"
        assert 74.0 <= stn["lon"] <= 78.0, f"Station {stn['name']} outside target longitude bounds"
        assert stn["elev_m"] > 0.0


def test_gfs_s3_key_formatter_and_idx_parser():
    from src.data.gfs_archive_auditor import format_gfs_s3_key, parse_gfs_idx_byte_ranges

    key_2023 = format_gfs_s3_key(date(2023, 7, 15), cycle_hour=0, forecast_hour=24)
    assert key_2023 == "gfs.20230715/00/atmos/gfs.t00z.pgrb2.0p25.f024"

    key_2018 = format_gfs_s3_key(date(2018, 7, 15), cycle_hour=0, forecast_hour=24)
    assert key_2018 == "gfs.20180715/00/gfs.t00z.pgrb2.0p25.f024"

    sample_idx = (
        "1:0:d=2023071500:TMP:2 m above ground:24 hour fcst:\n"
        "2:2048500:d=2023071500:RH:2 m above ground:24 hour fcst:\n"
        "3:4105000:d=2023071500:UGRD:10 m above ground:24 hour fcst:\n"
        "4:6200000:d=2023071500:VGRD:10 m above ground:24 hour fcst:\n"
        "5:8300000:d=2023071500:APCP:surface:18-24 hour acc fcst:\n"
    )
    ranges = parse_gfs_idx_byte_ranges(sample_idx)
    assert "TMP:2 m above ground" in ranges
    assert ranges["TMP:2 m above ground"] == (0, 2048499)
    assert ranges["RH:2 m above ground"] == (2048500, 4104999)
    assert ranges["APCP:surface"] == (8300000, None)


def test_meteorological_wind_vector_conversion():
    from src.data.source_auditor import derive_vector_wind_from_speed_direction

    speed = 10.0  # m/s

    # 1. North wind (0°): blowing towards south => U = 0, V = -10
    u, v = derive_vector_wind_from_speed_direction(speed, 0.0)
    assert np.isclose(u, 0.0, atol=1e-6)
    assert np.isclose(v, -10.0, atol=1e-6)

    # 2. East wind (90°): blowing towards west => U = -10, V = 0
    u, v = derive_vector_wind_from_speed_direction(speed, 90.0)
    assert np.isclose(u, -10.0, atol=1e-6)
    assert np.isclose(v, 0.0, atol=1e-6)

    # 3. South wind (180°): blowing towards north => U = 0, V = +10
    u, v = derive_vector_wind_from_speed_direction(speed, 180.0)
    assert np.isclose(u, 0.0, atol=1e-6)
    assert np.isclose(v, 10.0, atol=1e-6)

    # 4. West wind (270°): blowing towards east => U = +10, V = 0
    u, v = derive_vector_wind_from_speed_direction(speed, 270.0)
    assert np.isclose(u, 10.0, atol=1e-6)
    assert np.isclose(v, 0.0, atol=1e-6)

    # 5. Calm (0 m/s): U = 0, V = 0 regardless of angle
    u, v = derive_vector_wind_from_speed_direction(0.0, 45.0)
    assert np.isclose(u, 0.0, atol=1e-6)
    assert np.isclose(v, 0.0, atol=1e-6)

    # 6. Vectorized numpy array operation
    speeds = np.array([10.0, 5.0, 0.0])
    angles = np.array([0.0, 180.0, 90.0])
    u_arr, v_arr = derive_vector_wind_from_speed_direction(speeds, angles)
    assert len(u_arr) == 3 and len(v_arr) == 3
    assert np.isclose(u_arr[0], 0.0) and np.isclose(v_arr[0], -10.0)
    assert np.isclose(u_arr[1], 0.0) and np.isclose(v_arr[1], 5.0)


def test_era5_hourly_to_daily_derivation():
    from src.data.source_auditor import derive_era5_daily_from_hourly

    # Create 24 hours of synthetic hourly data
    times = [f"2023-07-15T{h:02d}:00" for h in range(24)]
    # Diurnal temperature cycle: min 20°C at dawn, max 32°C at afternoon
    temps = [20.0 + 12.0 * np.sin(np.pi * h / 24.0) for h in range(24)]
    rhs = [85.0 - 30.0 * np.sin(np.pi * h / 24.0) for h in range(24)]
    speeds = [4.0] * 24
    dirs = [270.0] * 24  # Pure westerly wind

    hourly_dict = {
        "time": times,
        "temperature_2m": temps,
        "relative_humidity_2m": rhs,
        "wind_speed_10m": speeds,
        "wind_direction_10m": dirs,
    }

    daily = derive_era5_daily_from_hourly(hourly_dict)
    assert np.isclose(daily["tmax"], 32.0, atol=0.1)
    assert np.isclose(daily["tmin"], 20.0, atol=0.1)
    assert daily["tmax"] >= daily["tmin"]
    assert 0.0 <= daily["rh"] <= 100.0
    assert np.isclose(daily["wind_u"], 4.0, atol=1e-4)  # West wind blowing east => +U
    assert np.isclose(daily["wind_v"], 0.0, atol=1e-4)

    # Check error on non-multiple of 24
    invalid_dict = {k: v[:20] for k, v in hourly_dict.items()}
    with pytest.raises(ValueError):
        derive_era5_daily_from_hourly(invalid_dict)

    # Check error on NaNs
    nan_dict = {k: list(v) for k, v in hourly_dict.items()}
    nan_dict["temperature_2m"][5] = np.nan
    with pytest.raises(ValueError):
        derive_era5_daily_from_hourly(nan_dict)


def test_on_disk_spatial_registration():
    from src.data.source_auditor import audit_dataset_spatial_registration

    res = audit_dataset_spatial_registration(root_dir=ROOT)
    assert res["all_datasets_registered"] is True, f"Spatial registration failed: {res}"
    assert res["pixel_convention"] == "Pixel-Is-Area"
    assert res["scale_factor"] == 5

    files = res["files"]
    for key in ["glo30_dem", "chirps_daily", "era5_land_daily"]:
        assert key in files
        f = files[key]
        assert f["exists"] is True
        assert f["verified"] is True
        assert f["fine_dims"] == [80, 80]
        # Pixel-Is-Area cell-center check: [11.025, 14.975] and [74.025, 77.975]
        assert np.isclose(f["lat_range"][0], 11.025, atol=1e-3)
        assert np.isclose(f["lat_range"][1], 14.975, atol=1e-3)
        assert np.isclose(f["lon_range"][0], 74.025, atol=1e-3)
        assert np.isclose(f["lon_range"][1], 77.975, atol=1e-3)


def test_manifest_wind_sources_declare_era5():
    manifest_path = ROOT / "data" / "source_manifest.yaml"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    hist_vars = manifest["streams"]["history"]["variables"]
    assert "ECMWF ERA5" in hist_vars["wind_u"]["source"], "wind_u source must be ECMWF ERA5"
    assert "Land" not in hist_vars["wind_u"]["source"], "wind_u must not claim ERA5-Land (Open-Meteo lacks Land wind)"
    assert hist_vars["wind_u"]["native_resolution_deg"] == 0.25

    assert "ECMWF ERA5" in hist_vars["wind_v"]["source"], "wind_v source must be ECMWF ERA5"
    assert "Land" not in hist_vars["wind_v"]["source"], "wind_v must not claim ERA5-Land"
    assert hist_vars["wind_v"]["native_resolution_deg"] == 0.25

    # Check target subcomponents
    target_sub = manifest["streams"]["target"]["subcomponents"]
    assert "wind" in target_sub, "target stream must explicitly declare 'wind' subcomponent from ERA5"
    assert "ECMWF ERA5" in target_sub["wind"]["source"]
    assert target_sub["wind"]["variables"] == ["wind_u", "wind_v"]
    assert target_sub["thermodynamics"]["variables"] == ["tmax", "tmin", "rh"]


def test_gfs_archive_tier_routing():
    from src.data.gfs_archive_auditor import audit_gfs_archive_source

    # 2023 -> AWS Open Data tier (date-specific file)
    res_2023 = audit_gfs_archive_source(2023, month=7, day=15)
    assert res_2023["tier"] == "aws_open_data"
    assert "noaa-gfs-bdp-pds" in res_2023["repository"]
    assert res_2023["available"] is True

    # 2018 -> NCAR RDA ds084.1 tier (date-specific THREDDS catalog)
    res_2018 = audit_gfs_archive_source(2018, month=7, day=15)
    assert res_2018["tier"] == "ncar_rda_ds084_1"
    assert "NCAR RDA ds084.1" in res_2018["repository"]
    assert "2018/20180715/catalog.xml" in res_2018["catalog_url"]
    assert res_2018["available"] is True
    assert "gfs.0p25.2018071500.f024.grib2" in res_2018["target_file"]

    # 2014 -> Pre-operational (unsupported)
    res_2014 = audit_gfs_archive_source(2014, month=7, day=15)
    assert res_2014["available"] is False
    assert res_2014["tier"] == "pre_operational"


