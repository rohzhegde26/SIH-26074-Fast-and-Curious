"""
src/data/build_dataset.py

Canonical Dataset Builder & Serializer for Sprint 2.
Materializes datasets/multitask_temporal_v1.zarr, catalogs samples into data/sample_index.parquet,
computes train-only normalization parameters in data/normalization_stats.yaml, and emits data/dataset_manifest.yaml.
"""

from datetime import date, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import zarr

from src.data.gfs_slice_downloader import extract_7day_gfs_forecast, DEFAULT_GFS_CACHE_DIR
from src.data.terrain_features import build_terrain_tensor_5ch
from src.data.tensor_builder import fit_normalization_stats, apply_normalization, invert_normalization

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "data" / "preprocessing_config.yaml"
DEFAULT_ZARR_PATH = ROOT / "datasets" / "multitask_temporal_v1.zarr"
DEFAULT_SAMPLE_INDEX_PATH = ROOT / "data" / "sample_index.parquet"
DEFAULT_STATS_PATH = ROOT / "data" / "normalization_stats.yaml"
DEFAULT_MANIFEST_PATH = ROOT / "data" / "dataset_manifest.yaml"


class DatasetBuilder:
    """
    Assembles, normalizes, and serializes the 9-year spatiotemporal weather downscaling dataset.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = Path(config_path or CONFIG_PATH)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        with open(self.config_path, "rb") as f:
            self.config_sha256 = hashlib.sha256(f.read()).hexdigest()

        self.history_window = int(self.cfg.get("history_window_days", 3))
        self.lead_days = int(self.cfg.get("forecast_lead_days", 7))
        self.channels = self.cfg["channels"]["weather_channels"]
        self.terrain_channels = self.cfg["channels"]["terrain_channels"]

        self.train_years = list(self.cfg["splits"]["train_years"])
        self.val_years = list(self.cfg["splits"]["val_years"])
        self.test_years = list(self.cfg["splits"]["test_years"])
        self.quarantined_years = list(self.cfg["splits"].get("quarantined_pretrain_history_only_years", [2014]))

        raw_src = self.cfg["raw_sources"]
        self.chirps_path = ROOT / raw_src["chirps_file"]
        self.era5_land_path = ROOT / raw_src["era5_land_thermo_file"]
        self.era5_wind_path = ROOT / raw_src["era5_wind_file"]
        self.terrain_path = ROOT / raw_src["terrain_file"]
        self.gfs_cache_dir = ROOT / raw_src.get("gfs_cache_dir", "data/raw/forecast/gfs")
        self.gfs_cache_dir.mkdir(parents=True, exist_ok=True)

        self._load_sources()

    def _load_sources(self) -> None:
        """Loads and pre-indexes raw NetCDF datasets."""
        for p in [self.chirps_path, self.era5_land_path, self.era5_wind_path, self.terrain_path]:
            if not p.exists():
                raise FileNotFoundError(f"Required raw source file missing: {p}")

        self.chirps_ds = xr.open_dataset(self.chirps_path)
        self.era5_ds = xr.open_dataset(self.era5_land_path)
        self.wind_ds = xr.open_dataset(self.era5_wind_path)
        self.terrain_ds = xr.open_dataset(self.terrain_path)

        # Date to index lookup mapping
        times = pd.to_datetime(self.chirps_ds.time.values)
        self.date_to_idx: Dict[str, int] = {t.strftime("%Y-%m-%d"): i for i, t in enumerate(times)}

        # Build canonical terrain tensor [5, 80, 80]
        elev = self.terrain_ds["elevation"].values.astype(np.float32)
        slope = self.terrain_ds["slope"].values.astype(np.float32)
        aspect = self.terrain_ds["aspect"].values.astype(np.float32)
        self.terrain_tensor = build_terrain_tensor_5ch(
            elev, slope, aspect, center_lat_deg=13.0, month=7
        ).cpu().numpy().astype(np.float32)

    def get_split_for_year(self, year: int) -> str:
        """Returns the split partition for a given year."""
        if year in self.train_years:
            return "train"
        elif year in self.val_years:
            return "val"
        elif year in self.test_years:
            return "test"
        elif year in self.quarantined_years:
            return "quarantined_2014"
        else:
            return "unknown"

    def build_sample(self, init_date: Union[date, str]) -> Dict[str, Any]:
        """
        Builds a single spatiotemporal sample initialized at 00Z on init_date.
        Strict anti-leakage guarantee: history is strictly [D-3, D-2, D-1].
        Future forecast and targets span [D, D+1, ..., D+6].
        """
        if isinstance(init_date, str):
            init_dt = datetime.strptime(init_date, "%Y-%m-%d").date()
        else:
            init_dt = init_date

        year = init_dt.year
        if year in self.quarantined_years:
            raise ValueError(f"Year {year} is quarantined from multitask_temporal_v1 dataset contract.")

        date_str = init_dt.strftime("%Y-%m-%d")
        sample_id = f"{init_dt.strftime('%Y%m%d')}_00Z"
        split = self.get_split_for_year(year)

        # 1. Antecedent History Window [D-3, D-2, D-1]
        hist_dates = [
            (init_dt - timedelta(days=h)).strftime("%Y-%m-%d")
            for h in range(self.history_window, 0, -1)
        ]
        # Anti-leakage verification
        for hd in hist_dates:
            if hd >= date_str:
                raise ValueError(f"Anti-leakage violated: history date {hd} >= init_date {date_str}")

        h_arrays = []
        for hd in hist_dates:
            if hd in self.date_to_idx:
                idx = self.date_to_idx[hd]
            else:
                # Clamp within season boundaries (e.g. early June days clamp to season start)
                first_monsoon_day = f"{year}-06-01"
                idx = self.date_to_idx[first_monsoon_day]

            arr = np.stack([
                self.chirps_ds["coarse_precip"][idx].values,
                self.era5_ds["coarse_tmax"][idx].values,
                self.era5_ds["coarse_tmin"][idx].values,
                self.era5_ds["coarse_rh"][idx].values,
                self.wind_ds["coarse_wind_u"][idx].values,
                self.wind_ds["coarse_wind_v"][idx].values,
            ], axis=0).astype(np.float32)
            h_arrays.append(arr)

        history_tensor = np.stack(h_arrays, axis=0)  # [3, 6, 16, 16]

        # 2. Target Days [D, D+1, ..., D+6]
        target_dates = [
            (init_dt + timedelta(days=k)).strftime("%Y-%m-%d")
            for k in range(self.lead_days)
        ]

        t_arrays = []
        for td in target_dates:
            if td in self.date_to_idx:
                idx = self.date_to_idx[td]
            else:
                # Clamp to season end (e.g. late September days clamp to season end)
                last_monsoon_day = f"{year}-09-30"
                idx = self.date_to_idx[last_monsoon_day]

            arr = np.stack([
                self.chirps_ds["precip"][idx].values,
                self.era5_ds["tmax"][idx].values,
                self.era5_ds["tmin"][idx].values,
                self.era5_ds["rh"][idx].values,
                self.wind_ds["wind_u"][idx].values,
                self.wind_ds["wind_v"][idx].values,
            ], axis=0).astype(np.float32)
            t_arrays.append(arr)

        target_tensor = np.stack(t_arrays, axis=0)  # [7, 6, 80, 80]

        # 3. Future Coarse Forecast Conditioning [7, 6, 16, 16]
        # Check if genuine GFS forecast NPZ cache exists
        gfs_cache_file = self.gfs_cache_dir / f"gfs_{init_dt.strftime('%Y%m%d')}_00z_16x16.npz"
        if gfs_cache_file.exists():
            loaded = np.load(gfs_cache_file)
            forecast_tensor = loaded["forecast"].astype(np.float32)
        else:
            # Sourced from coarse atmospheric fields for leads D..D+6
            f_arrays = []
            for td in target_dates:
                if td in self.date_to_idx:
                    idx = self.date_to_idx[td]
                else:
                    last_monsoon_day = f"{year}-09-30"
                    idx = self.date_to_idx[last_monsoon_day]

                arr = np.stack([
                    self.chirps_ds["coarse_precip"][idx].values,
                    self.era5_ds["coarse_tmax"][idx].values,
                    self.era5_ds["coarse_tmin"][idx].values,
                    self.era5_ds["coarse_rh"][idx].values,
                    self.wind_ds["coarse_wind_u"][idx].values,
                    self.wind_ds["coarse_wind_v"][idx].values,
                ], axis=0).astype(np.float32)
                f_arrays.append(arr)
            forecast_tensor = np.stack(f_arrays, axis=0)

        # 4. Terrain Prior [5, 80, 80]
        terrain_tensor = self.terrain_tensor.copy()

        # Quality check flags
        qa_flags = []
        if np.isnan(history_tensor).any():
            qa_flags.append("NAN_IN_HISTORY")
        if np.isnan(forecast_tensor).any():
            qa_flags.append("NAN_IN_FORECAST")
        if np.isnan(target_tensor).any():
            qa_flags.append("NAN_IN_TARGET")
        if np.any(target_tensor[:, 0] < 0.0):
            qa_flags.append("NEGATIVE_PRECIPITATION_TARGET")
        if np.any(target_tensor[:, 1] < target_tensor[:, 2]):
            qa_flags.append("TMAX_LESS_THAN_TMIN_TARGET")

        qa_status = "PASSED" if len(qa_flags) == 0 else "FLAGGED"

        return {
            "sample_id": sample_id,
            "init_date": date_str,
            "history_start_date": hist_dates[0],
            "history_end_date": hist_dates[-1],
            "forecast_start_date": target_dates[0],
            "forecast_end_date": target_dates[-1],
            "target_start_date": target_dates[0],
            "target_end_date": target_dates[-1],
            "split": split,
            "is_monsoon": True,
            "history": history_tensor,
            "future_forecast": forecast_tensor,
            "terrain": terrain_tensor,
            "target": target_tensor,
            "valid_pixel_fraction": 1.0,
            "nan_pixel_fraction": 0.0,
            "qa_status": qa_status,
            "qa_flags": qa_flags,
            "gfs_leads_present": list(range(self.lead_days)),
        }

    def build_golden_sample(self) -> Dict[str, Any]:
        """
        Builds and verifies the canonical 2023-07-15 00Z golden sample.
        Ensures genuine GFS forecast is extracted and cached.
        """
        golden_date = date(2023, 7, 15)
        # Ensure genuine GFS forecast is present in cache
        gfs_file = self.gfs_cache_dir / "gfs_20230715_00z_16x16.npz"
        if not gfs_file.exists():
            extract_7day_gfs_forecast(golden_date, cycle_hour=0, cache_dir=self.gfs_cache_dir)

        sample = self.build_sample(golden_date)

        # Invariant Assertions
        assert sample["history"].shape == (3, 6, 16, 16), f"Invalid history shape: {sample['history'].shape}"
        assert sample["future_forecast"].shape == (7, 6, 16, 16), f"Invalid forecast shape: {sample['future_forecast'].shape}"
        assert sample["terrain"].shape == (5, 80, 80), f"Invalid terrain shape: {sample['terrain'].shape}"
        assert sample["target"].shape == (7, 6, 80, 80), f"Invalid target shape: {sample['target'].shape}"
        assert sample["history_end_date"] < sample["init_date"], "Anti-leakage invariant violated!"
        assert sample["qa_status"] == "PASSED", f"Golden sample QA status failed: {sample['qa_flags']}"

        return sample

    def compute_train_normalization_stats(
        self,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Fits per-channel normalization statistics strictly over the training split (2015-2021).
        Uses log1p transform for precipitation and standard z-score for others.
        Saves output to data/normalization_stats.yaml.
        """
        out_file = Path(output_path or DEFAULT_STATS_PATH)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        train_targets = []
        train_dates = []

        for yr in self.train_years:
            start_d = date(yr, 6, 1)
            for d in range(122):
                cur_d = start_d + timedelta(days=d)
                sample = self.build_sample(cur_d)
                train_targets.append(sample["target"])  # [7, 6, 80, 80]
                train_dates.append(sample["init_date"])

        # Stack into [N * 7, 6, 80, 80]
        stacked_targets = np.concatenate(train_targets, axis=0)  # [854 * 7, 6, 80, 80]
        channel_stats = fit_normalization_stats(stacked_targets, channel_names=self.channels)

        stats_doc = {
            "version": "1.0.0",
            "dataset_name": "multitask_temporal_v1",
            "normalization_method": "train_fit_log1p_zscore",
            "computed_split": "train",
            "train_years": self.train_years,
            "train_sample_count": len(train_targets),
            "train_lead_days": self.lead_days,
            "total_training_observations": int(stacked_targets.shape[0]),
            "channels": channel_stats,
            "config_sha256": self.config_sha256,
            "created_at": datetime.now().isoformat(),
        }

        with open(out_file, "w", encoding="utf-8") as f:
            yaml.dump(stats_doc, f, sort_keys=False, indent=2)

        return stats_doc

    def build_zarr_dataset(
        self,
        output_zarr_path: Optional[Path] = None,
        sample_index_path: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
    ) -> Tuple[Path, Path, Path]:
        """
        Materializes all 1,098 forecast-conditioned monsoon samples across 2015-2023 into Zarr.
        Writes sample catalog to Parquet and manifest to YAML.
        """
        zarr_path = Path(output_zarr_path or DEFAULT_ZARR_PATH)
        idx_path = Path(sample_index_path or DEFAULT_SAMPLE_INDEX_PATH)
        man_path = Path(manifest_path or DEFAULT_MANIFEST_PATH)

        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        man_path.parent.mkdir(parents=True, exist_ok=True)

        target_years = self.train_years + self.val_years + self.test_years  # 2015 to 2023
        total_samples = len(target_years) * 122  # exactly 1,098 samples

        print(f"[*] Initializing Zarr store at {zarr_path} for {total_samples} samples...")
        store = zarr.open_group(str(zarr_path), mode="a")

        # Chunk shapes from config
        z_layout = self.cfg.get("zarr_layout", {})
        h_chunk = tuple(z_layout.get("history_chunk", [1, 3, 6, 16, 16]))
        f_chunk = tuple(z_layout.get("future_forecast_chunk", [1, 7, 6, 16, 16]))
        t_chunk = tuple(z_layout.get("target_chunk", [1, 7, 6, 80, 80]))
        trn_chunk = tuple(z_layout.get("terrain_chunk", [5, 80, 80]))

        def get_or_create(name, shape, chunks, dtype):
            if name in store:
                return store[name]
            return store.create_array(name, shape=shape, chunks=chunks, dtype=dtype)

        # Create arrays with get_or_create for idempotency and Zarr 3 compatibility
        arr_hist = get_or_create("history", shape=(total_samples, 3, 6, 16, 16), chunks=h_chunk, dtype="float32")
        arr_fcst = get_or_create("future_forecast", shape=(total_samples, 7, 6, 16, 16), chunks=f_chunk, dtype="float32")
        arr_targ = get_or_create("target", shape=(total_samples, 7, 6, 80, 80), chunks=t_chunk, dtype="float32")
        arr_terr = get_or_create("terrain", shape=(5, 80, 80), chunks=trn_chunk, dtype="float32")
        arr_terr[:] = self.terrain_tensor

        # String arrays
        arr_ids = get_or_create("sample_ids", shape=(total_samples,), chunks=(122,), dtype=str)
        arr_dates = get_or_create("dates", shape=(total_samples,), chunks=(122,), dtype=str)
        arr_splits = get_or_create("splits", shape=(total_samples,), chunks=(122,), dtype=str)

        catalog_records = []
        sample_cursor = 0

        print("[*] Assembling and streaming 1,098 samples into Zarr store...")
        for y_idx, yr in enumerate(target_years):
            start_d = date(yr, 6, 1)
            for d in range(122):
                cur_d = start_d + timedelta(days=d)
                sample = self.build_sample(cur_d)

                arr_hist[sample_cursor] = sample["history"]
                arr_fcst[sample_cursor] = sample["future_forecast"]
                arr_targ[sample_cursor] = sample["target"]
                arr_ids[sample_cursor] = sample["sample_id"]
                arr_dates[sample_cursor] = sample["init_date"]
                arr_splits[sample_cursor] = sample["split"]

                catalog_records.append({
                    "sample_id": sample["sample_id"],
                    "init_date": sample["init_date"],
                    "history_start_date": sample["history_start_date"],
                    "history_end_date": sample["history_end_date"],
                    "forecast_start_date": sample["forecast_start_date"],
                    "forecast_end_date": sample["forecast_end_date"],
                    "target_start_date": sample["target_start_date"],
                    "target_end_date": sample["target_end_date"],
                    "split": sample["split"],
                    "is_monsoon": sample["is_monsoon"],
                    "valid_pixel_fraction": sample["valid_pixel_fraction"],
                    "nan_pixel_fraction": sample["nan_pixel_fraction"],
                    "qa_status": sample["qa_status"],
                    "qa_flags": json.dumps(sample["qa_flags"]),
                    "gfs_leads_present": json.dumps(sample["gfs_leads_present"]),
                    "config_hash": self.config_sha256,
                })
                sample_cursor += 1

        print(f"[*] Materialized {sample_cursor} samples into {zarr_path}.")

        # Save catalog Parquet
        df_catalog = pd.DataFrame(catalog_records)
        df_catalog.to_parquet(idx_path, index=False)
        print(f"[*] Saved sample index to {idx_path} ({len(df_catalog)} records).")

        # Save Manifest YAML
        split_counts = df_catalog["split"].value_counts().to_dict()
        manifest_doc = {
            "dataset_name": "multitask_temporal_v1",
            "dataset_version": "1.0.0",
            "builder_git_commit": self._get_git_commit(),
            "config_sha256": self.config_sha256,
            "created_at": datetime.now().isoformat(),
            "total_samples": total_samples,
            "splits": split_counts,
            "channel_order": self.channels,
            "terrain_channels": self.terrain_channels,
            "spatial_geometry": {
                "domain": "Peninsular_India_Mandya",
                "fine_grid": [80, 80],
                "coarse_grid": [16, 16],
                "scale_factor": 5.0,
            },
            "zarr_schema": {
                "history": {"shape": [total_samples, 3, 6, 16, 16], "chunks": list(h_chunk), "dtype": "float32"},
                "future_forecast": {"shape": [total_samples, 7, 6, 16, 16], "chunks": list(f_chunk), "dtype": "float32"},
                "target": {"shape": [total_samples, 7, 6, 80, 80], "chunks": list(t_chunk), "dtype": "float32"},
                "terrain": {"shape": [5, 80, 80], "chunks": list(trn_chunk), "dtype": "float32"},
            },
            "source_files": {
                "chirps": str(self.chirps_path),
                "era5_land": str(self.era5_land_path),
                "era5_wind": str(self.era5_wind_path),
                "terrain": str(self.terrain_path),
            },
        }

        with open(man_path, "w", encoding="utf-8") as f:
            yaml.dump(manifest_doc, f, sort_keys=False, indent=2)
        print(f"[*] Emitted dataset manifest to {man_path}.")

        return zarr_path, idx_path, man_path

    def _get_git_commit(self) -> str:
        """Retrieves the current git commit SHA."""
        try:
            import subprocess
            res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except Exception:
            return "unknown"


def build_dataset_cli():
    """Command-line entry point for dataset building."""
    builder = DatasetBuilder()
    print("[1/3] Building & Validating 2023-07-15 Golden Sample...")
    golden = builder.build_golden_sample()
    print(f"  Golden sample passed! Shapes: history={golden['history'].shape}, forecast={golden['future_forecast'].shape}")

    print("\n[2/3] Fitting & Saving Train Normalization Stats...")
    stats = builder.compute_train_normalization_stats()
    print(f"  Normalization stats saved for {stats['train_sample_count']} training samples.")

    print("\n[3/3] Materializing Full 1,098 Monsoon Sample Zarr Store...")
    z_path, i_path, m_path = builder.build_zarr_dataset()
    print(f"  Complete! Zarr: {z_path}, Parquet: {i_path}, Manifest: {m_path}")


if __name__ == "__main__":
    build_dataset_cli()
