"""Collect all real-GFS fetch outputs into one archive + a completeness/consistency report."""
import glob, json, os, tarfile
from datetime import date, timedelta
import numpy as np
files = {}
for p in glob.glob("/kaggle/input/**/gfs_raw/*.npz", recursive=True):
    n = os.path.basename(p)
    if n not in files or "topup" in p:          # top-up outputs win over earlier failures
        files[n] = p
expected = []
for y in range(2015, 2024):
    d = date(y, 6, 1)
    while d <= date(y, 9, 30):
        expected.append(d.strftime("%Y%m%d")); d += timedelta(days=1)
have = {n[:-4] for n in files}
missing = sorted(set(expected) - have)
print(f"present {len(set(expected) & have)}/{len(expected)} ; missing {len(missing)}: {missing}", flush=True)
# full re-validation of every file
bad = []
for n, p in sorted(files.items()):
    z = np.load(p)
    try:
        assert z["t2m"].shape == (28, 43, 43)
        for f in ["t2m", "rh2m", "u10", "v10", "tmax6", "tmin6", "prate6"]:
            assert np.isfinite(z[f]).all(), f"nan in {f}"
        assert (z["prate6"] >= 0).all() and 270 < z["t2m"].min() and z["t2m"].max() < 325
        assert (z["tmax6"] - z["tmin6"]).min() >= -0.2
        assert np.allclose(z["lat"], np.arange(7.75, 18.26, 0.25)) and np.allclose(z["lon"], np.arange(70.75, 81.26, 0.25))
    except AssertionError as e:
        bad.append((n, str(e)))
print("files failing re-validation:", bad, flush=True)
# AWS vs NCAR overlap check (2022-07-15/16 fetched by both)
over = {}
for n in ["20220715.npz", "20220716.npz"]:
    srcs = [p for p in glob.glob(f"/kaggle/input/**/gfs_raw/{n}", recursive=True)]
    arrs = {str(np.load(p)["source"]): np.load(p) for p in srcs}
    if {"aws", "ncar"} <= set(arrs):
        a, b = arrs["aws"], arrs["ncar"]
        over[n] = {f: float(np.nanmax(np.abs(a[f] - b[f]))) for f in ["t2m", "rh2m", "u10", "v10", "tmax6", "tmin6", "prate6"]}
        over[n]["prate6_x21600_mm"] = over[n].pop("prate6") * 21600
print("AWS vs NCAR max abs diff on overlap dates:", json.dumps(over, indent=1), flush=True)
os.makedirs("/kaggle/working", exist_ok=True)
with tarfile.open("/kaggle/working/gfs_raw_all.tar", "w") as t:   # npz already compressed
    for n, p in sorted(files.items()):
        t.add(p, arcname=f"gfs_raw/{n}")
report = {"present": len(set(expected) & have), "expected": len(expected), "missing": missing, "bad": bad, "overlap": over}
json.dump(report, open("/kaggle/working/gfs_bundle_report.json", "w"), indent=1)
print("tar MB", os.path.getsize("/kaggle/working/gfs_raw_all.tar") / 1e6, flush=True)
