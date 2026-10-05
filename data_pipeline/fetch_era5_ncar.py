"""
Fetch ECMWF ERA5 0.25 deg hourly fields for the 2.5x context domain from the NCAR RDA ds633.0
mirror (much faster than the CDS queue), and write one merged file per year:

    data/raw/era5_ncar/era5_{year}_hourly.nc   dims (time, lat, lon); vars t2m, d2m, u10, v10 (K, K, m/s)
                                                and tp (m, accumulation over the hour ENDING at `time`)

Instantaneous analysis fields come from e5.oper.an.sfc monthly files via THREDDS NCSS.
Precipitation is lsp + cp from e5.oper.fc.sfc.accumu (06/18 UTC forecasts, hours 1..12, each value the
accumulation over that forecast hour), read via OPeNDAP because NCSS cannot serve their 2-D time axis;
valid time = forecast_initial_time + forecast_hour.

Covers 17 May .. 8 Oct each year (history D-14 for 1 Jun inits; targets to D+6 for 30 Sep inits).
Resumable: raw pieces are cached in data/raw/era5_ncar/raw/. Prints a progress bar + ETA.
"""
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "era5_ncar"
RAW = OUT / "raw"
YEARS = range(2014, 2024)
BBOX = dict(north=18.5, south=7.5, east=81.5, west=70.5)
THREDDS = "https://thredds.rda.ucar.edu/thredds"
AN = {"t2m": ("128_167_2t", "VAR_2T"), "d2m": ("128_168_2d", "VAR_2D"),
      "u10": ("128_165_10u", "VAR_10U"), "v10": ("128_166_10v", "VAR_10V")}
FC = {"lsp": ("128_142_lsp", "LSP"), "cp": ("128_143_cp", "CP")}
S = requests.Session()


def month_end(y, m):
    return (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)).day


def an_task(y, m, var):
    code, name = AN[var]
    out = RAW / f"an_{var}_{y}{m:02d}.nc"
    if out.exists():
        return out
    f = f"e5.oper.an.sfc.{code}.ll025sc.{y}{m:02d}0100_{y}{m:02d}{month_end(y, m):02d}23.nc"
    url = (f"{THREDDS}/ncss/grid/files/g/d633000/e5.oper.an.sfc/{y}{m:02d}/{f}?var={name}"
           f"&north={BBOX['north']}&south={BBOX['south']}&east={BBOX['east']}&west={BBOX['west']}"
           "&temporal=all&accept=netcdf")
    for k in range(6):
        try:
            r = S.get(url, timeout=600)
            if r.status_code == 200 and r.content[:3] == b"CDF":
                ds = xr.open_dataset(io.BytesIO(r.content), engine="scipy").load()
                assert ds.sizes["time"] == 24 * month_end(y, m), ds.sizes
                ds.to_netcdf(out.with_suffix(".part"), engine="scipy")
                out.with_suffix(".part").replace(out)
                return out
            err = f"HTTP {r.status_code} {r.text[:120]}"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        time.sleep(10 * (k + 1))
    raise RuntimeError(f"{out.name}: {err}")


def fc_task(y, m, half, var):
    code, name = FC[var]
    out = RAW / f"fc_{var}_{y}{m:02d}_{half}.nc"
    if out.exists():
        return out
    if half == 1:
        span = f"{y}{m:02d}0106_{y}{m:02d}1606"
    else:
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        span = f"{y}{m:02d}1606_{ny}{nm:02d}0106"
    url = f"{THREDDS}/dodsC/files/g/d633000/e5.oper.fc.sfc.accumu/{y}{m:02d}/e5.oper.fc.sfc.accumu.{code}.ll025sc.{span}.nc"
    for k in range(6):
        try:
            d = xr.open_dataset(url, engine="netcdf4")
            lat, lon = d.latitude.values, d.longitude.values
            r = np.where((lat <= BBOX["north"]) & (lat >= BBOX["south"]))[0]
            c = np.where((lon >= BBOX["west"]) & (lon <= BBOX["east"]))[0]
            x = d[name].isel(latitude=slice(r[0], r[-1] + 1), longitude=slice(c[0], c[-1] + 1)).load()
            d.close()
            x.to_dataset(name=var).to_netcdf(out.with_suffix(".part"))
            out.with_suffix(".part").replace(out)
            return out
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"{out.name}: {err}")


def merge_year(y):
    out = OUT / f"era5_{y}_hourly.nc"
    t0, t1 = pd.Timestamp(y, 5, 17), pd.Timestamp(y, 10, 8, 23)
    an = []
    for var in AN:
        ds = xr.concat([xr.open_dataset(RAW / f"an_{var}_{y}{m:02d}.nc", engine="scipy").load()
                        for m in range(5, 11)], "time")
        an.append(ds[AN[var][1]].rename(var))
    an = xr.merge(an).sel(time=slice(t0, t1))
    parts = []
    for var in FC:
        for m in range(4, 11):
            for half in (1, 2):
                p = RAW / f"fc_{var}_{y}{m:02d}_{half}.nc"
                if not p.exists():
                    continue
                x = xr.open_dataset(p)[var].load()
                valid = x.forecast_initial_time.values[:, None] + x.forecast_hour.values[None, :].astype("timedelta64[h]")
                s = xr.DataArray(x.values.reshape(-1, *x.shape[2:]), dims=("time", "latitude", "longitude"),
                                 coords={"time": valid.ravel(), "latitude": x.latitude, "longitude": x.longitude}, name=var)
                parts.append(s)
    fc = xr.merge([xr.concat([p for p in parts if p.name == v], "time").drop_duplicates("time").sortby("time")
                   for v in FC])
    tp = (fc["lsp"] + fc["cp"]).rename("tp").sel(time=slice(t0, t1 + pd.Timedelta(hours=1)))
    ds = xr.merge([an, tp.reindex(time=an.time)]).rename({"latitude": "lat", "longitude": "lon"})
    n_expected = int((t1 - t0) / pd.Timedelta(hours=1)) + 1
    assert ds.sizes["time"] == n_expected, (ds.sizes, n_expected)
    for v in ds.data_vars:
        assert np.isfinite(ds[v].values).all(), f"{y} {v} has gaps"
    ds.attrs.update(source="ECMWF ERA5 via NCAR RDA ds633.0", tp_convention="accumulation over hour ending at time")
    ds.to_netcdf(out)
    return out


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [(an_task, (y, m, v)) for y in YEARS for m in range(5, 11) for v in AN]
    # fc files are keyed by initial-time month; mid-April..mid-October covers valid 17 May .. 8 Oct
    jobs += [(fc_task, (y, m, h, v)) for y in YEARS for (m, h) in [(5, 1), (5, 2), (6, 1), (6, 2), (7, 1), (7, 2),
                                                                 (8, 1), (8, 2), (9, 1), (9, 2), (10, 1)] for v in FC]
    total, done, failed, t0 = len(jobs), 0, [], time.time()
    print(f"{total} NCAR requests -> {RAW}", flush=True)
    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(fn, *a) for fn, a in jobs]
        for fu in as_completed(futs):
            done += 1
            try:
                fu.result()
            except Exception as e:
                failed.append(str(e)[:200])
            el = time.time() - t0
            eta = el / done * (total - done)
            print(f"[{'#' * int(done / total * 40):<40}] {done}/{total} | elapsed {el / 60:.1f} min | "
                  f"ETA {eta / 60:.1f} min | failed {len(failed)}", flush=True)
    if failed:
        print("FAILED:", *failed, sep="\n  ")
        sys.exit(1)
    for y in YEARS:
        print("merged", merge_year(y).name, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
