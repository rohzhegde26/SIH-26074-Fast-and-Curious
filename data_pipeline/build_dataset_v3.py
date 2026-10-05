"""
Sprint 10 prerequisite: build `datasets/multitask_temporal_v3_realgfs.zarr` from genuine sources only.

Replaces v1/v2, whose problems were found in the 2026-09-29 audit:
  * `future_forecast` was coarsened observations + noise labelled as GFS (oracle leakage);
  * Tmax/Tmin/RH targets were not ERA5-Land (no correlation with ERA5 at any point);
  * wind was km/h stored as m/s;
  * only the 16x16 core was stored, so "context" N>16 was edge padding;
  * 21.6 % Arabian-Sea pixels had CHIRPS no-data filled with 0 mm;
  * precipitation normalisation was linear although documented as log1p.

Sources (all real, all 00-24 UTC daily windows):
  future_forecast  NOAA GFS 0.25 deg 00Z, 28 six-hourly steps  data/raw/forecast/gfs_real/gfs_raw/*.npz
  history          ECMWF ERA5 0.25 deg hourly (NCAR ds633.0)   data/raw/era5_ncar/era5_*_hourly.nc
  target P         UCSB CHIRPS v2.0 p05                        data/raw/chirps/chirps_daily.nc (verified bit-exact vs UCSB)
  target T/RH/UV   ECMWF ERA5-Land 0.1 deg hourly (CDS)        data/raw/era5_cds/land/*.nc
  terrain          Copernicus GLO-30 derived (verified)         data/raw/dem/glo30_mandya_terrain.nc

Grids (latitude always north -> south, like CHIRPS):
  coarse: 40x40 cells of 0.25 deg, centres 17.875..8.125 N x 71.125..80.875 E (2.5x context).
          The central 16x16 (rows/cols 12..27) is exactly the 11-15 N / 74-78 E target domain, so the
          existing centre-crop loader yields *real* context for any N <= 40.
          Cell value = mean of the 4 surrounding 0.25-deg grid points (GFS and ERA5 share that lattice).
  fine:   80x80 cells of 0.05 deg, centres 14.975..11.025 N x 74.025..77.975 E.

Channels: [precipitation, tmax, tmin, rh, wind_u, wind_v]  (mm/day, degC, degC, %, m/s, m/s)

Usage:  python -m src.data.build_dataset_v3 [--precip-only]
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
import zarr

ROOT = Path(__file__).resolve().parents[2]
GFS_DIR = ROOT / "data/raw/forecast/gfs_real/gfs_raw"
ERA5_DIR = ROOT / "data/raw/era5_ncar"  # NCAR ds633.0 mirror; verified identical to CDS ERA5 (Jun 2015)
LAND_DIR = ROOT / "data/raw/era5_cds/land"
CHIRPS = ROOT / "data/raw/chirps/chirps_daily.nc"
DEM = ROOT / "data/raw/dem/glo30_mandya_terrain.nc"
V1_ZARR = ROOT / "datasets/multitask_temporal_v1.zarr"  # terrain tensor only (verified real)
OUT = ROOT / "datasets/multitask_temporal_v3_realgfs.zarr"
STATS_OUT = ROOT / "data/normalization_stats_v3.yaml"
INDEX_OUT = ROOT / "data/sample_index_v3.parquet"

CHANNELS = ["precipitation", "tmax", "tmin", "rh", "wind_u", "wind_v"]
H_HIST, N_LEAD = 14, 7
CLAT = np.round(np.arange(17.875, 8.0, -0.25), 3)   # 40 coarse cell centres, N->S
CLON = np.round(np.arange(71.125, 81.0, 0.25), 3)   # 40
FLAT = np.round(np.arange(14.975, 11.0, -0.05), 3)  # 80 fine cell centres, N->S
FLON = np.round(np.arange(74.025, 78.0, 0.05), 3)   # 80
SPLIT = {**{y: "train" for y in range(2015, 2022)}, 2022: "val", 2023: "test"}


def magnus_rh(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """Relative humidity (%) from temperature and dew point (degC), Magnus-Tetens over water."""
    a, b = 17.625, 243.04
    return np.clip(100.0 * np.exp(a * td_c / (b + td_c) - a * t_c / (b + t_c)), 0.0, 100.0)


def corners_to_cells(field: np.ndarray, plat: np.ndarray, plon: np.ndarray) -> np.ndarray:
    """Average the 4 lattice points around each coarse cell centre. field[..., lat, lon] on points plat/plon."""
    def idx(points, targets):
        order = np.argsort(points)
        pos = np.searchsorted(points[order], targets)
        assert np.allclose(points[order][pos], targets, atol=1e-4), "cell corner not on lattice"
        return order[pos]
    lo_r, hi_r = idx(plat, CLAT - 0.125), idx(plat, CLAT + 0.125)
    lo_c, hi_c = idx(plon, CLON - 0.125), idx(plon, CLON + 0.125)
    g = lambda r, c: field[..., r, :][..., :, c]
    return 0.25 * (g(lo_r, lo_c) + g(lo_r, hi_c) + g(hi_r, lo_c) + g(hi_r, hi_c))


# ------------------------------------------------------------------------------------------- GFS
def gfs_daily(init: date) -> np.ndarray:
    """[7, 6, 40, 40] daily GFS forecast for valid days D..D+6 (00-24 UTC)."""
    z = np.load(GFS_DIR / f"{init:%Y%m%d}.npz")
    lat, lon = z["lat"], z["lon"]
    assert list(z["hours"]) == list(range(6, 169, 6))
    r = lambda a: a.reshape(N_LEAD, 4, *a.shape[1:])  # [7 days, 4 six-hourly steps, lat, lon]
    daily = np.stack([
        r(z["prate6"] * 21600.0).sum(1),                       # mm/day (verified == 6 h APCP)
        r(z["tmax6"]).max(1) - 273.15,
        r(z["tmin6"]).min(1) - 273.15,
        r(z["rh2m"]).mean(1),
        r(z["u10"]).mean(1),
        r(z["v10"]).mean(1),
    ], axis=1)                                                  # [7, 6, 43, 43]
    daily[:, 1] = np.maximum(daily[:, 1], daily[:, 2])          # GRIB packing: tmax >= tmin
    return corners_to_cells(daily, lat, lon).astype(np.float32)


# ------------------------------------------------------------------------------------------- ERA5
def _open_hourly(files):
    parts = [xr.open_dataset(f).load() for f in sorted(files)]
    tname = "valid_time" if "valid_time" in parts[0].dims else "time"
    ds = xr.concat(parts, tname)
    ds = ds.rename({tname: "time"}).sortby("time")
    if "latitude" in ds.dims:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    return ds.drop_vars([v for v in ("number", "expver") if v in ds.variables], errors="ignore")


def era5_daily() -> xr.Dataset:
    """ERA5 0.25 deg -> daily 00-24 UTC fields on the 40x40 coarse cells."""
    ds = _open_hourly(glob.glob(str(ERA5_DIR / "*_hourly.nc"))).load()
    t, td = ds["t2m"] - 273.15, ds["d2m"] - 273.15
    rh = xr.apply_ufunc(magnus_rh, t, td)
    day = ds["time"].dt.floor("D")
    # tp at valid time v is the accumulation over the hour ending at v, so day D = valid 01:00 D .. 00:00 D+1
    tp = (ds["tp"] * 1000.0).assign_coords(time=ds["time"] - np.timedelta64(1, "h"))  # label by hour start
    tp_day = tp.groupby(tp["time"].dt.floor("D").rename("day")).sum()
    tp_n = tp.groupby(tp["time"].dt.floor("D").rename("day")).count().isel(lat=0, lon=0)
    agg = xr.Dataset({
        "precipitation": tp_day,
        "tmax": t.groupby(day.rename("day")).max(), "tmin": t.groupby(day.rename("day")).min(),
        "rh": rh.groupby(day.rename("day")).mean(),
        "wind_u": ds["u10"].groupby(day.rename("day")).mean(), "wind_v": ds["v10"].groupby(day.rename("day")).mean(),
    })
    n_hours = ds["t2m"].groupby(day.rename("day")).count().isel(lat=0, lon=0)
    keep = (n_hours.values == 24) & (tp_n.sel(day=n_hours["day"]).values == 24)  # only complete UTC days
    agg = agg.sel(day=n_hours["day"].values[keep])  # by label: tp adds a partial 16 May per year
    plat, plon = agg["lat"].values, agg["lon"].values
    cells = {v: corners_to_cells(agg[v].values, plat, plon) for v in CHANNELS}
    return xr.Dataset({v: (("day", "clat", "clon"), cells[v].astype(np.float32)) for v in CHANNELS},
                      coords={"day": agg["day"].values, "clat": CLAT, "clon": CLON})


def land_daily() -> xr.Dataset:
    """ERA5-Land 0.1 deg -> daily 00-24 UTC -> bilinear to the 80x80 0.05 deg cell centres.

    Streams one monthly file at a time (~150 MB) so it runs on a 15 GB laptop; each file holds whole
    UTC days, so no day straddles two files.
    """
    from scipy.ndimage import distance_transform_edt

    out = []
    for f in sorted(glob.glob(str(LAND_DIR / "*_hourly.nc"))):
        ds = _open_hourly([f])
        ds = ds.sel(lat=slice(15.3, 10.7), lon=slice(73.7, 78.3)).load()  # fine domain + 3-pt interp margin
        t, td = ds["t2m"] - 273.15, ds["d2m"] - 273.15
        rh = xr.apply_ufunc(magnus_rh, t, td)
        day = ds["time"].dt.floor("D").rename("day")
        agg = xr.Dataset({"tmax": t.groupby(day).max(), "tmin": t.groupby(day).min(), "rh": rh.groupby(day).mean(),
                          "wind_u": ds["u10"].groupby(day).mean(), "wind_v": ds["v10"].groupby(day).mean()})
        n_hours = ds["t2m"].groupby(day).count().max(("lat", "lon"))
        agg = agg.sel(day=n_hours["day"].values[n_hours.values == 24])
        # ERA5-Land is NaN over sea: fill sea points with the nearest land value before interpolating so
        # coastal land cells are not lost; sea cells are excluded later by the CHIRPS land mask.
        sea = ~np.isfinite(agg["tmax"].isel(day=0).values)
        _, (ri, ci) = distance_transform_edt(sea, return_indices=True)
        for v in agg.data_vars:
            agg[v].values[:] = agg[v].values[:, ri, ci]
        out.append(agg.interp(lat=FLAT, lon=FLON, method="linear").astype(np.float32))
        ds.close()
    return xr.concat(out, "day").sortby("day")


# ------------------------------------------------------------------------------------------- build
def main(precip_only: bool = False) -> None:
    inits = sorted(date(int(p.stem[:4]), int(p.stem[4:6]), int(p.stem[6:8])) for p in GFS_DIR.glob("*.npz"))
    inits = [d for d in inits if 6 <= d.month <= 9]  # monsoon inits only (drops nothing today)
    n = len(inits)
    print(f"[v3] {n} init dates with real GFS", flush=True)

    chirps = xr.open_dataset(CHIRPS)["precip"]
    chirps = chirps.assign_coords(time=pd.to_datetime(chirps["time"].values.astype(str)))
    assert np.allclose(chirps["lat"].values, FLAT, atol=1e-3) and np.allclose(chirps["lon"].values, FLON, atol=1e-3)
    land_mask = (chirps.max("time") > 0).values  # CHIRPS no-data (sea) was stored as 0 on every day
    print(f"[v3] land mask: {land_mask.sum()} land / {land_mask.size} pixels", flush=True)

    era5 = None if precip_only else era5_daily()
    land = None if precip_only else land_daily()

    fcst = np.zeros((n, N_LEAD, 6, 40, 40), np.float32)
    hist = np.full((n, H_HIST, 6, 40, 40), np.nan, np.float32)
    targ = np.full((n, N_LEAD, 6, 80, 80), np.nan, np.float32)
    rows = []
    for i, d0 in enumerate(inits):
        fcst[i] = gfs_daily(d0)
        lead_days = [pd.Timestamp(d0 + timedelta(days=k)) for k in range(N_LEAD)]
        hist_days = [pd.Timestamp(d0 - timedelta(days=k)) for k in range(H_HIST, 0, -1)]  # D-14 .. D-1
        assert max(hist_days) < pd.Timestamp(d0)
        targ[i, :, 0] = np.where(land_mask, chirps.sel(time=lead_days).values, np.nan)
        if not precip_only:
            for c, v in enumerate(CHANNELS[1:], start=1):
                targ[i, :, c] = np.where(land_mask, land[v].sel(day=lead_days).values, np.nan)
            for c, v in enumerate(CHANNELS):
                hist[i, :, c] = era5[v].sel(day=hist_days).values
        rows.append({"sample_id": f"{d0:%Y%m%d}_00Z", "init_date": str(d0), "split": SPLIT[d0.year],
                     "history_start_date": str(hist_days[0].date()), "history_end_date": str(hist_days[-1].date()),
                     "target_start_date": str(lead_days[0].date()), "target_end_date": str(lead_days[-1].date()),
                     "gfs_file": f"{d0:%Y%m%d}.npz", "gfs_source": "aws_open_data" if d0.year >= 2021 else "ncar_rda_ds084_1"})
        if (i + 1) % 200 == 0 or i == n - 1:
            print(f"[v3]   assembled {i + 1}/{n}", flush=True)

    # ---- hard checks
    assert np.isfinite(fcst).all(), "NaN in GFS forecast"
    assert np.isnan(targ[:, :, 0][:, :, ~land_mask]).all() and np.isfinite(targ[:, :, 0][:, :, land_mask]).all()
    if not precip_only:
        assert np.isfinite(hist).all(), "NaN in ERA5 history (missing days?)"
        assert np.isfinite(targ[:, :, 1:][..., land_mask]).all(), "NaN in ERA5-Land target over land"
        assert (targ[:, :, 1] >= targ[:, :, 2] - 1e-3)[..., land_mask].all()
        assert np.nanmax(np.abs(targ[:, :, 4:])) < 40 and np.nanmax(np.abs(hist[:, :, 4:])) < 40, "wind units?"

    # ---- normalisation (train split only; precipitation log1p)
    tr = np.array([r["split"] == "train" for r in rows])
    stats = {"dataset_name": OUT.stem, "fitted_partition": f"train_only (2015-2021, {tr.sum()} samples, land pixels)",
             "channels": {}}
    units = ["mm", "degC", "degC", "%", "m/s", "m/s"]
    for c, v in enumerate(CHANNELS):
        vals = targ[tr, :, c][..., land_mask].ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if v == "precipitation":
            lv = np.log1p(np.maximum(vals, 0))
            stats["channels"][v] = {"transform": "log1p_zscore", "mean": float(lv.mean()), "std": float(lv.std()),
                                    "min": float(vals.min()), "max": float(vals.max()), "unit": units[c]}
        else:
            stats["channels"][v] = {"mean": float(vals.mean()), "std": float(vals.std()),
                                    "min": float(vals.min()), "max": float(vals.max()), "unit": units[c]}

    # ---- write
    if OUT.exists():
        shutil.rmtree(OUT)
    g = zarr.open_group(str(OUT), mode="w")
    def put(name, arr, chunks):
        a = g.create_array(name, shape=arr.shape, chunks=chunks, dtype=arr.dtype if arr.dtype != object else str)
        a[:] = arr
    put("future_forecast", fcst, (16, N_LEAD, 6, 40, 40))
    put("history", hist, (16, H_HIST, 6, 40, 40))
    put("target", targ, (16, N_LEAD, 6, 80, 80))
    put("target_mask", land_mask.astype(np.uint8), (80, 80))
    put("terrain", np.asarray(zarr.open_group(str(V1_ZARR), mode="r")["terrain"][:], np.float32), (5, 80, 80))
    put("dates", np.array([r["init_date"] for r in rows], dtype=str), (122,))
    put("splits", np.array([r["split"] for r in rows], dtype=str), (122,))
    g.attrs.update({"version": "3.0.0", "history_len": H_HIST, "precip_only": precip_only,
                    "coarse_grid": "40x40 cells 0.25deg N->S, centre 16x16 = target domain",
                    "target_nan_policy": "sea pixels NaN; use target_mask", "channels": CHANNELS})
    STATS_OUT.write_text(yaml.safe_dump(stats, sort_keys=False))
    pd.DataFrame(rows).to_parquet(INDEX_OUT, index=False)
    print(f"[v3] wrote {OUT} ({n} samples), {STATS_OUT.name}, {INDEX_OUT.name}", flush=True)
    print(json.dumps({k: v.get("mean") for k, v in stats["channels"].items()}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precip-only", action="store_true", help="skip ERA5/ERA5-Land (history & T/RH/UV targets NaN)")
    main(ap.parse_args().precip_only)
