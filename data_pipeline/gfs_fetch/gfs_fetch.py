"""
Real NOAA GFS 0.25 deg forecast fetcher for SIH-26074 (replaces the synthetic
"GFS" caches that were built from observations).

One output file per 00Z init date: gfs_raw/{YYYYMMDD}.npz containing, for each
of the 28 six-hourly forecast steps f006..f168, the fields below on the exact
43x43 GFS point grid lat 7.75..18.25 N, lon 70.75..81.25 E (covers the 2.5x
context domain; cell-centre regridding is done later by the dataset builder).

  t2m, rh2m, u10, v10        instantaneous at the step            (K, %, m/s)
  tmax6, tmin6               max/min over the preceding 6 h       (K)
  prate6                     mean precip rate over preceding 6 h  (kg m-2 s-1)
  apcp6                      precip accumulated over preceding 6h (kg m-2 == mm)
                             (NaN where the archive only exposes mixed intervals
                              and no 6 h bucket can be isolated; use prate6*21600)

MODE = "aws"  : 2021-2023 via AWS Open Data byte ranges + eccodes decode.
MODE = "ncar" : 2015-2020 via NCAR RDA ds084.1 THREDDS NetCDF subset service,
                plus two 2022 overlap dates for a cross-source consistency check.
"""
import io
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests

MODE = "__MODE__"
YEARS = __YEARS__  # used by MODE == "ncar"
TOPUP_DATES = __TOPUP__  # explicit init dates (ISO) for a refetch job, else []
OUT = Path("/kaggle/working/gfs_raw") if Path("/kaggle").exists() else Path(os.environ.get("GFS_OUT", "gfs_raw"))
OUT.mkdir(parents=True, exist_ok=True)
HOURS = list(range(6, 169, 6))
LATS = np.round(np.arange(7.75, 18.25 + 1e-6, 0.25), 2)   # ascending, 43
LONS = np.round(np.arange(70.75, 81.25 + 1e-6, 0.25), 2)  # 43
FIELDS = ["t2m", "rh2m", "u10", "v10", "tmax6", "tmin6", "prate6", "apcp6"]

SESSION = requests.Session()
SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=64, max_retries=0))


def monsoon_inits(years):
    out = []
    for y in years:
        d = date(y, 6, 1)
        while d <= date(y, 9, 30):
            out.append(d)
            d += timedelta(days=1)
    return out


def get(url, headers=None, timeout=90, tries=6):
    """GET with exponential backoff on network errors / 429 / 5xx. Returns response (may be 4xx)."""
    for k in range(tries):
        try:
            r = SESSION.get(url, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}")
            return r
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(min(60, 2 ** k + np.random.rand()))


# ----------------------------------------------------------------------------- AWS
AWS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d}/00/atmos/gfs.t00z.pgrb2.0p25.f{h:03d}"
WIN = re.compile(r":(\d+)-(\d+) hour (acc|max|min|ave) fcst:")


def aws_pick(idx_lines, h):
    """Map field -> (start, end) byte range for forecast hour h."""
    want = {}
    for i, l in enumerate(idx_lines):
        p = l.split(":")
        if len(p) < 7:
            continue
        var, lev, stamp = p[3], p[4], ":" + ":".join(p[5:])
        m = WIN.search(stamp)
        six = bool(m) and int(m.group(2)) == h and int(m.group(2)) - int(m.group(1)) == 6
        inst = stamp.startswith(f":{h} hour fcst")
        key = None
        if lev == "2 m above ground" and inst and var == "TMP": key = "t2m"
        elif lev == "2 m above ground" and inst and var == "RH": key = "rh2m"
        elif lev == "10 m above ground" and inst and var == "UGRD": key = "u10"
        elif lev == "10 m above ground" and inst and var == "VGRD": key = "v10"
        elif lev == "2 m above ground" and six and var == "TMAX": key = "tmax6"
        elif lev == "2 m above ground" and six and var == "TMIN": key = "tmin6"
        elif lev == "surface" and six and var == "PRATE": key = "prate6"
        elif lev == "surface" and six and var == "APCP": key = "apcp6"
        if key and key not in want:
            start = int(p[1])
            end = int(idx_lines[i + 1].split(":")[1]) - 1 if i + 1 < len(idx_lines) else None
            want[key] = (start, end)
    return want


def aws_step(d, h):
    import eccodes
    url = AWS.format(d=d.strftime("%Y%m%d"), h=h)
    r = get(url + ".idx", timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"idx HTTP {r.status_code} {url}")
    rng = aws_pick(r.text.splitlines(), h)
    missing = [f for f in FIELDS if f not in rng]
    if missing:
        raise RuntimeError(f"fields missing in idx f{h:03d}: {missing}")
    out = {}
    for f, (s, e) in rng.items():
        rr = get(url, headers={"Range": f"bytes={s}-{'' if e is None else e}"}, timeout=60)
        if rr.status_code not in (200, 206):
            raise RuntimeError(f"range HTTP {rr.status_code}")
        gid = eccodes.codes_new_from_message(rr.content)
        try:
            geo = {k: eccodes.codes_get(gid, k) for k in ("Ni", "Nj", "latitudeOfFirstGridPointInDegrees",
                   "longitudeOfFirstGridPointInDegrees", "iDirectionIncrementInDegrees", "jScansPositively")}
            if (geo["Ni"], geo["Nj"], round(geo["latitudeOfFirstGridPointInDegrees"], 3),
                    round(geo["longitudeOfFirstGridPointInDegrees"], 3), round(geo["iDirectionIncrementInDegrees"], 3),
                    geo["jScansPositively"]) != (1440, 721, 90.0, 0.0, 0.25, 0):
                raise RuntimeError(f"unexpected GRIB grid {geo}")
            ni, nj = geo["Ni"], geo["Nj"]
            vals = eccodes.codes_get_values(gid).reshape(nj, ni)  # lat 90 -> -90, lon 0 -> 359.75
        finally:
            eccodes.codes_release(gid)
        rows = np.round((90.0 - LATS) / 0.25).astype(int)
        cols = np.round(LONS / 0.25).astype(int)
        out[f] = vals[np.ix_(rows, cols)].astype(np.float32)
    return out, {"url": url}


# ----------------------------------------------------------------------------- NCAR
NCSS = "https://thredds.rda.ucar.edu/thredds/ncss/grid/files/g/d084001/{y}/{d}/gfs.0p25.{d}00.f{h:03d}.grib2"
BASE_VARS = ["Temperature_height_above_ground", "Relative_humidity_height_above_ground",
             "u-component_of_wind_height_above_ground", "v-component_of_wind_height_above_ground",
             "Maximum_temperature_height_above_ground_6_Hour_Maximum",
             "Minimum_temperature_height_above_ground_6_Hour_Minimum",
             "Precipitation_rate_surface_6_Hour_Average"]
APCP_6H = "Total_precipitation_surface_6_Hour_Accumulation"
# NCAR's NCSS cannot serve the FV3-era "Mixed_intervals" APCP (HTTP 400 "Index out of range"),
# so from GFS v15 (first 00Z FV3 init: 2019-06-13) only f006 carries a 6 h APCP bucket there.
# Precipitation for every date/source is therefore taken from prate6 * 21600 downstream;
# apcp6 is kept where available as an independent cross-check.
FV3_START = date(2019, 6, 13)
BBOX = "&north=18.5&south=7.5&east=81.5&west=70.5&accept=netcdf"


def _level(da, value):
    for dim in da.dims:
        if dim.startswith("height_above_ground"):
            return da.sel({dim: value}, method="nearest")
    return da


def _crop(da):
    da = da.squeeze(drop=True)
    latn = "lat" if "lat" in da.dims else "latitude"
    lonn = "lon" if "lon" in da.dims else "longitude"
    da = da.sel({latn: LATS, lonn: LONS}, method="nearest", tolerance=0.01)
    return np.asarray(da.transpose(latn, lonn).values, dtype=np.float32)


def ncar_step(d, h):
    import xarray as xr
    ds_str = d.strftime("%Y%m%d")
    base = NCSS.format(y=d.year, d=ds_str, h=h)
    last = None
    cands = [APCP_6H, None] if (d < FV3_START or h == 6) else [None]
    for apcp in cands:
        vars_ = BASE_VARS + ([apcp] if apcp else [])
        url = base + "?" + "&".join("var=" + v for v in vars_) + BBOX
        r = get(url, timeout=120)
        if r.status_code == 200:
            break
        last = f"HTTP {r.status_code}: {r.text[:200]}"
    else:
        raise RuntimeError(f"NCSS failed f{h:03d}: {last}")
    ds = xr.open_dataset(io.BytesIO(r.content), engine="scipy")
    valid = np.datetime64(d.isoformat()) + np.timedelta64(h, "h")
    for v in list(ds.data_vars):
        tdim = [x for x in ds[v].dims if x.startswith("time")]
        if not tdim:
            continue
        tv = np.asarray(ds[tdim[0]].values)
        bn = next((b for b in ds.variables if b == tdim[0] + "_bounds"), None)
        if bn is not None:
            bb = np.asarray(ds[bn].values).reshape(-1, 2)
            if not (np.all(bb[:, 1] == valid) and np.all(bb[:, 1] - bb[:, 0] == np.timedelta64(6, "h"))):
                raise RuntimeError(f"{v} window {bb.tolist()} != 6 h ending f{h:03d}")
        elif not np.all(tv == valid):
            raise RuntimeError(f"{v} valid time {tv.tolist()} != f{h:03d}")
    out = {
        "t2m": _crop(_level(ds["Temperature_height_above_ground"], 2)),
        "rh2m": _crop(_level(ds["Relative_humidity_height_above_ground"], 2)),
        "u10": _crop(_level(ds["u-component_of_wind_height_above_ground"], 10)),
        "v10": _crop(_level(ds["v-component_of_wind_height_above_ground"], 10)),
        "tmax6": _crop(_level(ds["Maximum_temperature_height_above_ground_6_Hour_Maximum"], 2)),
        "tmin6": _crop(_level(ds["Minimum_temperature_height_above_ground_6_Hour_Minimum"], 2)),
        "prate6": _crop(ds["Precipitation_rate_surface_6_Hour_Average"]),
    }
    note = {"url": base, "apcp_var": apcp}
    out["apcp6"] = np.full((len(LATS), len(LONS)), np.nan, np.float32)
    if apcp and apcp in ds:
        da = ds[apcp]
        tdim = [x for x in da.dims if x.startswith("time")]
        if tdim and da.sizes[tdim[0]] != 1:
            raise RuntimeError(f"unexpected APCP time length {da.sizes[tdim[0]]} f{h:03d}")
        out["apcp6"] = _crop(da)
    ds.close()
    return out, note


# ----------------------------------------------------------------------------- driver
def sanity(arrs, tag):
    c = arrs
    checks = {
        "t2m K in [270,325]": np.nanmin(c["t2m"]) > 270 and np.nanmax(c["t2m"]) < 325,
        "rh2m in [0,100.5]": np.nanmin(c["rh2m"]) >= 0 and np.nanmax(c["rh2m"]) <= 100.5,
        "|wind|<60": np.nanmax(np.abs(c["u10"])) < 60 and np.nanmax(np.abs(c["v10"])) < 60,
        "tmax6>=tmin6 (0.2 K packing tol)": np.nanmin(c["tmax6"] - c["tmin6"]) >= -0.2,
        "prate6>=0": np.nanmin(c["prate6"]) >= 0,
        "apcp6>=0 or nan": np.all(np.isnan(c["apcp6"]) | (c["apcp6"] >= -1e-3)),
    }
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise RuntimeError(f"SANITY FAIL {tag}: {bad}")


def fetch_init(d, workers, step_fn):
    arr = {f: np.full((len(HOURS), len(LATS), len(LONS)), np.nan, np.float32) for f in FIELDS}
    notes = [None] * len(HOURS)
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(step_fn, d, h): i for i, h in enumerate(HOURS)}
        for fu, i in futs.items():
            fields, note = fu.result()
            for f in FIELDS:
                arr[f][i] = fields[f]
            notes[i] = note
    for i in range(len(HOURS)):
        sanity({f: arr[f][i] for f in FIELDS}, f"{d} f{HOURS[i]:03d}")
    np.savez_compressed(OUT / f"{d.strftime('%Y%m%d')}.npz", lat=LATS, lon=LONS, hours=np.array(HOURS),
                        notes=json.dumps(notes), source=MODE, **arr)


def fmt(sec):
    sec = int(max(0, sec))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def main():
    if MODE == "aws":
        for pkgs in ([], ["eccodes"], ["eccodes", "eccodeslib"], ["eccodes", "ecmwflibs"]):
            if pkgs:
                os.system(f"{sys.executable} -m pip install -q {' '.join(pkgs)} 2>&1 | tail -2")
            try:
                import importlib, eccodes
                importlib.reload(eccodes)
                print("eccodes", eccodes.codes_get_api_version(), flush=True)
                break
            except Exception as e:
                print("eccodes not usable yet:", type(e).__name__, e, flush=True)
        else:
            raise SystemExit("could not install eccodes")
        inits, workers, step_fn = monsoon_inits([2021, 2022, 2023]), 28, aws_step
    elif MODE == "ncar":
        inits = monsoon_inits(YEARS)
        if TOPUP_DATES:
            inits = [date.fromisoformat(x) for x in TOPUP_DATES]
        elif 2020 in YEARS:
            inits = [date(2022, 7, 15), date(2022, 7, 16)] + inits  # AWS-overlap cross-check
        workers, step_fn = 2, ncar_step
    else:
        raise SystemExit("set MODE")

    # Resume: reuse files from a previous version attached as input, if any.
    for p in Path("/kaggle/input").glob("**/gfs_raw/*.npz") if Path("/kaggle/input").exists() else []:
        if not (OUT / p.name).exists():
            (OUT / p.name).write_bytes(p.read_bytes())
    todo = [d for d in inits if not (OUT / f"{d.strftime('%Y%m%d')}.npz").exists()]
    total = len(inits)
    done0 = total - len(todo)
    print(f"[{MODE} {YEARS}] {total} init dates, {done0} already present, {len(todo)} to fetch, "
          f"{len(todo) * len(HOURS)} forecast files, {workers} parallel requests", flush=True)

    failed, t0, done, recent_err = {}, time.time(), 0, []
    for d in todo + [None]:
        if d is None:
            break
        try:
            fetch_init(d, workers, step_fn)
            recent_err.append(0)
        except Exception as e:
            failed[d] = f"{type(e).__name__}: {e}"
            recent_err.append(1)
            print(f"  ! FAILED {d}: {failed[d][:300]}", flush=True)
            parse_bug = "SANITY" in str(e) or "window" in str(e) or isinstance(e, (KeyError, ValueError, IndexError))
            if done == 0 and parse_bug:
                traceback.print_exc()
                raise  # anything wrong on the very first date: stop now, do not burn hours
        done += 1
        recent_err = recent_err[-20:]
        if MODE == "ncar" and sum(recent_err) >= 4 and workers > 1:
            workers -= 1
            recent_err = []  # judge the new setting on fresh evidence only
            print(f"  ! error rate rising, dropping to {workers} parallel requests", flush=True)
        el = time.time() - t0
        rate = done / el
        eta = (len(todo) - done) / rate if rate > 0 else 0
        frac = (done0 + done) / total
        bar = "#" * int(frac * 40) + "-" * (40 - int(frac * 40))
        print(f"[{bar}] {done0 + done}/{total} ({frac:6.1%}) | {d} | {rate * 60:5.1f} dates/min | "
              f"elapsed {fmt(el)} | ETA {fmt(eta)} | failed {len(failed)}", flush=True)

    # One retry pass for failures, slower.
    if failed:
        print(f"Retrying {len(failed)} failed dates with 2 parallel requests...", flush=True)
        for d in list(failed):
            try:
                fetch_init(d, 2, step_fn)
                failed.pop(d)
            except Exception as e:
                failed[d] = f"{type(e).__name__}: {e}"
    have = sorted(p.stem for p in OUT.glob("*.npz"))
    manifest = {"mode": MODE, "expected": total, "present": len(have), "failed": {str(k): v for k, v in failed.items()},
                "hours": HOURS, "lat": LATS.tolist(), "lon": LONS.tolist(), "fields": FIELDS,
                "wall_seconds": time.time() - t0}
    (OUT / f"manifest_{MODE}_{'topup' if TOPUP_DATES else ('_'.join(map(str, YEARS)) if MODE == 'ncar' else 'all')}.json").write_text(json.dumps(manifest, indent=1))
    print(f"DONE [{MODE}] present {len(have)}/{total}, failed {len(failed)}, wall {fmt(time.time() - t0)}", flush=True)
    for k, v in list(failed.items())[:20]:
        print("  FAILED", k, v, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        MODE = sys.argv[2]
        fn = ncar_step if MODE == "ncar" else aws_step
        d = date.fromisoformat(sys.argv[3])
        t = time.time()
        fetch_init(d, 4, fn)
        z = np.load(OUT / f"{d.strftime('%Y%m%d')}.npz")
        print("ok", round(time.time() - t, 1), "s", {f: (float(np.nanmin(z[f])), float(np.nanmax(z[f])), int(np.isnan(z[f]).sum())) for f in FIELDS})
        print("notes f012:", json.loads(str(z["notes"]))[1])
    else:
        main()
