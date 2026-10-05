"""
Fetch genuine ECMWF hourly reanalysis from the Copernicus CDS (replaces the corrupt
data/raw/era5_land/era5_land_daily.nc and the km/h-labelled-as-m/s era5_wind_daily.nc).

  ERA5-Land (0.1 deg, land only)  -> fine thermodynamic/wind targets
  ERA5      (0.25 deg, land+sea)  -> coarse history + wider spatial context

Hourly fields 00..23 UTC are fetched per month; daily 00-24 UTC max/min/mean are computed
later by the dataset builder (matching CHIRPS and the real-GFS daily windows). Area covers
the 2.5x context domain. The ERA5-Land and ERA5 queues are independent on CDS, so both run
concurrently. Existing files are skipped, so re-running resumes.

(The CDS "derived daily statistics" products were measured at ~3 min/file with 6 in
flight, ~18 h total; hourly retrieval is ~15 min per month with several in flight.)

Usage:  python scripts/fetch_era5_cds.py   (needs ~/.cdsapirc)
"""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cdsapi

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "era5_cds"
AREA = [18.5, 70.5, 7.5, 81.5]  # N, W, S, E
YEARS = range(2014, 2024)
MONTH_DAYS = {"05": range(18, 32), "06": range(1, 31), "07": range(1, 32),
              "08": range(1, 32), "09": range(1, 31), "10": range(1, 8)}
HOURS = [f"{h:02d}:00" for h in range(24)]
DATASETS = {
    "land": ("reanalysis-era5-land", 3,
             ["2m_temperature", "2m_dewpoint_temperature",
              "10m_u_component_of_wind", "10m_v_component_of_wind"]),
}
# ERA5 0.25 deg is fetched from the NCAR RDA ds633.0 mirror instead (scripts/fetch_era5_ncar.py):
# ~8 s per month-variable there versus ~15 min per month in the CDS queue.

lock = threading.Lock()
state = {"done": 0, "failed": [], "t0": time.time()}


def fetch(short, ds, vars_, y, m, days, total):
    out = OUT / short / f"{short}_{y}_{m}_hourly.nc"
    status = "skip"
    if not (out.exists() and out.stat().st_size > 0):
        out.parent.mkdir(parents=True, exist_ok=True)
        req = {"product_type": ["reanalysis"], "variable": vars_, "year": str(y), "month": m,
               "day": [f"{d:02d}" for d in days], "time": HOURS, "area": AREA,
               "data_format": "netcdf", "download_format": "unarchived"}
        tmp = out.with_suffix(".part")
        status = "FAILED"
        for attempt in range(12):
            try:
                cdsapi.Client(quiet=True, progress=False).retrieve(ds, req).download(str(tmp))
                tmp.replace(out)
                status = "ok"
                break
            except Exception as e:
                status = f"FAILED {type(e).__name__}: {str(e)[:200]}"
                time.sleep(min(600, 60 * (attempt + 1)))  # includes 'queue temporarily limited' rejections
    with lock:
        state["done"] += 1
        if status.startswith("FAILED"):
            state["failed"].append((out.name, status))
        el = time.time() - state["t0"]
        d = state["done"]
        eta = el / d * (total - d)
        bar = "#" * int(d / total * 40)
        print(f"[{bar:<40}] {d}/{total} | {out.name} {status} | elapsed {el / 60:.0f} min | "
              f"ETA {eta / 60:.0f} min | failed {len(state['failed'])}", flush=True)


def main():
    work = {k: [(y, m, days) for y in YEARS for m, days in MONTH_DAYS.items()] for k in DATASETS}
    total = sum(len(v) for v in work.values())
    print(f"{total} monthly CDS requests -> {OUT}", flush=True)
    pools = []
    for short, (ds, conc, vars_) in DATASETS.items():
        ex = ThreadPoolExecutor(conc)
        for y, m, days in work[short]:
            ex.submit(fetch, short, ds, vars_, y, m, days, total)
        pools.append(ex)
    for ex in pools:
        ex.shutdown(wait=True)
    print("DONE", total - len(state["failed"]), "ok,", len(state["failed"]), "failed", flush=True)
    for f in state["failed"]:
        print("  ", f, flush=True)
    sys.exit(1 if state["failed"] else 0)


if __name__ == "__main__":
    main()
