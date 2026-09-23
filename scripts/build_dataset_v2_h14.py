"""
scripts/build_dataset_v2_h14.py

Materializes datasets/multitask_temporal_v2_h14.zarr with 14-day antecedent history:
  - history: [1098, 14, 6, 16, 16] float32
  - future_forecast: [1098, 7, 6, 16, 16] float32 (reused from v1 for exact parity)
  - target: [1098, 7, 6, 80, 80] float32 (reused from v1 for exact parity)
  - terrain: [5, 80, 80] float32 (reused from v1)
  - dates: [1098] string
  - splits: [1098] string (854 train, 122 val, 122 test)

Also updates data/sample_index_v2_h14.parquet and fits data/normalization_stats_v2.yaml
strictly over the 854 training samples.
"""

from datetime import datetime, timedelta
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.tensor_builder import fit_normalization_stats

V1_ZARR_PATH = ROOT / "datasets" / "multitask_temporal_v1.zarr"
V2_ZARR_PATH = ROOT / "datasets" / "multitask_temporal_v2_h14.zarr"
EXTENSION_PATH = ROOT / "data" / "raw" / "antecedent_extension_2015_2023.npz"
RAW_CHIRPS = ROOT / "data" / "raw" / "chirps" / "chirps_daily.nc"
RAW_ERA5_LAND = ROOT / "data" / "raw" / "era5_land" / "era5_land_daily.nc"
RAW_ERA5_WIND = ROOT / "data" / "raw" / "era5" / "era5_wind_daily.nc"
INDEX_V1_PATH = ROOT / "data" / "sample_index.parquet"
INDEX_V2_PATH = ROOT / "data" / "sample_index_v2_h14.parquet"
STATS_V2_PATH = ROOT / "data" / "normalization_stats_v2.yaml"


def build_v2_dataset() -> None:
    print("[*] Starting build of multitask_temporal_v2_h14.zarr...", flush=True)

    # 1. Open source datasets
    print("  [1/5] Loading v1 Zarr and raw NetCDF sources...", flush=True)
    v1_store = zarr.open_group(str(V1_ZARR_PATH), mode="r")
    num_samples = v1_store["dates"].shape[0]  # 1098
    dates = [str(d) for d in v1_store["dates"][:]]
    splits = [str(s) for s in v1_store["splits"][:]]

    ext_data = np.load(EXTENSION_PATH)
    ext_dates = set(ext_data["dates"].tolist())
    ext_date_to_idx = {d: i for i, d in enumerate(ext_data["dates"].tolist())}

    chirps_ds = xr.open_dataset(RAW_CHIRPS)
    era5_ds = xr.open_dataset(RAW_ERA5_LAND)
    wind_ds = xr.open_dataset(RAW_ERA5_WIND)

    chirps_times = pd.to_datetime(chirps_ds.time.values)
    netcdf_date_to_idx = {t.strftime("%Y-%m-%d"): i for i, t in enumerate(chirps_times)}

    def get_slice(date_str: str) -> np.ndarray:
        if date_str in ext_date_to_idx:
            idx = ext_date_to_idx[date_str]
            arr = np.stack([
                ext_data["coarse_precip"][idx],
                ext_data["coarse_tmax"][idx],
                ext_data["coarse_tmin"][idx],
                ext_data["coarse_rh"][idx],
                ext_data["coarse_wind_u"][idx],
                ext_data["coarse_wind_v"][idx],
            ], axis=0).astype(np.float32)
            return arr
        elif date_str in netcdf_date_to_idx:
            idx = netcdf_date_to_idx[date_str]
            arr = np.stack([
                chirps_ds["coarse_precip"][idx].values,
                era5_ds["coarse_tmax"][idx].values,
                era5_ds["coarse_tmin"][idx].values,
                era5_ds["coarse_rh"][idx].values,
                wind_ds["coarse_wind_u"][idx].values,
                wind_ds["coarse_wind_v"][idx].values,
            ], axis=0).astype(np.float32)
            return arr
        else:
            raise KeyError(f"Date {date_str} not found in extension or raw NetCDFs!")

    # 2. Build wide history array [1098, 14, 6, 16, 16]
    print("  [2/5] Assembling 14-day antecedent history for all 1098 samples...", flush=True)
    history_h14 = np.zeros((num_samples, 14, 6, 16, 16), dtype=np.float32)

    for i in range(num_samples):
        init_str = dates[i]
        init_dt = datetime.strptime(init_str, "%Y-%m-%d").date()

        # 14 antecedent days ending at D-1: [D-14, D-13, ..., D-1]
        for step_idx, days_back in enumerate(range(14, 0, -1)):
            target_dt = init_dt - timedelta(days=days_back)
            target_str = target_dt.strftime("%Y-%m-%d")
            assert target_str < init_str, f"Anti-leakage violated: {target_str} >= {init_str}"
            history_h14[i, step_idx] = get_slice(target_str)

        if (i + 1) % 200 == 0 or i == num_samples - 1:
            print(f"    - Processed {i + 1}/{num_samples} samples", flush=True)

    assert not np.isnan(history_h14).any(), "NaN found in assembled 14-day history!"
    print(f"  [+] History shape: {history_h14.shape}, mean: {history_h14.mean():.3f}")

    # 3. Create v2 Zarr store
    print("  [3/5] Serializing to datasets/multitask_temporal_v2_h14.zarr...", flush=True)
    if V2_ZARR_PATH.exists():
        shutil.rmtree(V2_ZARR_PATH)

    v2_store = zarr.open_group(str(V2_ZARR_PATH), mode="w")

    # Arrays
    arr_hist = v2_store.create_array(
        "history",
        shape=(num_samples, 14, 6, 16, 16),
        chunks=(16, 14, 6, 16, 16),
        dtype="float32",
    )
    arr_hist[:] = history_h14

    arr_fcst = v2_store.create_array(
        "future_forecast",
        shape=(num_samples, 7, 6, 16, 16),
        chunks=(16, 7, 6, 16, 16),
        dtype="float32",
    )
    arr_fcst[:] = v1_store["future_forecast"][:]

    arr_targ = v2_store.create_array(
        "target",
        shape=(num_samples, 7, 6, 80, 80),
        chunks=(16, 7, 6, 80, 80),
        dtype="float32",
    )
    arr_targ[:] = v1_store["target"][:]

    arr_terr = v2_store.create_array(
        "terrain",
        shape=(5, 80, 80),
        chunks=(5, 80, 80),
        dtype="float32",
    )
    arr_terr[:] = v1_store["terrain"][:]

    arr_dates = v2_store.create_array(
        "dates",
        shape=(num_samples,),
        chunks=(122,),
        dtype=str,
    )
    arr_dates[:] = np.array(dates, dtype=str)

    arr_splits = v2_store.create_array(
        "splits",
        shape=(num_samples,),
        chunks=(122,),
        dtype=str,
    )
    arr_splits[:] = np.array(splits, dtype=str)
    v2_store.attrs["version"] = "2.0.0"
    v2_store.attrs["history_len"] = 14
    v2_store.attrs["antecedent_policy"] = "authentic_ecmwf_era5_and_chirps"

    print(f"  [+] Materialized Zarr v2 store: {V2_ZARR_PATH}", flush=True)

    # 4. Generate updated sample index parquet
    print("  [4/5] Updating sample index parquet with 14-day history start dates...", flush=True)
    df_v1 = pd.read_parquet(INDEX_V1_PATH)
    df_v2 = df_v1.copy()

    # Recalculate history_start_date as D-14
    init_dts = pd.to_datetime(df_v2["init_date"])
    df_v2["history_start_date"] = (init_dts - pd.Timedelta(days=14)).dt.strftime("%Y-%m-%d")
    df_v2["history_end_date"] = (init_dts - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    df_v2["history_window_days"] = 14
    df_v2.to_parquet(INDEX_V2_PATH, index=False)
    print(f"  [+] Saved v2 sample index: {INDEX_V2_PATH}", flush=True)

    # 5. Fit train-only normalization stats
    print("  [5/5] Fitting normalization stats strictly over 854 training samples...", flush=True)
    train_mask = np.array(splits) == "train"
    train_targets = v1_store["target"][:][train_mask]  # [854, 7, 6, 80, 80]

    channels = ["precipitation", "tmax", "tmin", "rh", "wind_u", "wind_v"]
    stats_dict = {}

    for c_idx, ch_name in enumerate(channels):
        ch_vals = train_targets[:, :, c_idx, :, :].ravel()
        ch_mean = float(np.mean(ch_vals))
        ch_std = float(np.std(ch_vals))
        stats_dict[ch_name] = {
            "mean": ch_mean,
            "std": max(ch_std, 1e-4),
            "min": float(np.min(ch_vals)),
            "max": float(np.max(ch_vals)),
            "unit": "mm" if ch_name == "precipitation" else ("degC" if "t" in ch_name else ("%" if ch_name == "rh" else "m/s")),
        }

    stats_doc = {
        "dataset_name": "multitask_temporal_v2_h14",
        "fitted_partition": "train_only (2015-2021, 854 samples)",
        "channels": stats_dict,
    }

    with open(STATS_V2_PATH, "w", encoding="utf-8") as f:
        yaml.dump(stats_doc, f, sort_keys=False, indent=2)

    print(f"  [+] Saved v2 normalization stats: {STATS_V2_PATH}", flush=True)
    print("[+] All v2 dataset artifacts generated successfully!", flush=True)


if __name__ == "__main__":
    build_v2_dataset()
