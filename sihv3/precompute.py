"""
Precompute gram-panchayat (GP) forecasts from the shipped bundle for the web demo (demo/), one season at a time.

For every init date of --year and every requested mode, runs FinalDownscaler.predict and aggregates the 80x80 fields
onto the 234 Mandya GPs with the area weights in gp_cells.json (demo/backend/gp_weights.py). Per GP and lead day:

  rain, rain_lo, rain_hi   ensemble mean and 5-95 % range of the members' GP-mean rain (FAST: single value)
  p15, p30, p64            fraction of members whose GP-mean rain >= 15 / 30 / 64.5 mm (diffusion modes)
  rain_pm                  probability-matched rain (diffusion modes; heavy-rain maps)
  tmax, tmin, rh, wind     ensemble mean (wind = speed km/h, averaged over members)
and, once per date (mode-independent): the observed values (targets) and raw GFS bilinear-interpolated rain/Tmax.

  python -m sihv3.precompute --bundle_glob '**/modes.yaml' --gp_cells '**/gp_cells.json' --year 2023 \
      --modes FAST BALANCED --tag demo23_a
Writes <out>/<tag>/gp_<MODE>.npz (+ gp_obs.npz with --with_obs): arrays [n_dates, n_gp, 7], dates, gp codes.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sihv3.data import V3Data
from sihv3.predict import FinalDownscaler
from sihv3.train import resolve

FIELDS = ("rain", "rain_lo", "rain_hi", "p15", "p30", "p64", "rain_pm", "tmax", "tmin", "rh", "wind")


def weight_matrix(gp_cells: dict):
    codes = sorted(gp_cells)
    W = np.zeros((len(codes), 80 * 80), np.float32)
    for i, c in enumerate(codes):
        for r, col, w in gp_cells[c]["cells"]:
            W[i, r * 80 + col] = w
    return codes, W


def gp(field, W):
    """field [..., 80, 80] -> [..., n_gp]"""
    return field.reshape(*field.shape[:-2], 6400) @ W.T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle_glob", required=True, help="glob of the bundle's modes.yaml")
    p.add_argument("--gp_cells", required=True)
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--modes", nargs="+", default=["FAST", "BALANCED", "ACCURATE", "ENSEMBLE"])
    p.add_argument("--with_obs", action="store_true")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--max_dates", type=int, default=0)
    p.add_argument("--tag", default="demo")
    p.add_argument("--out", default=os.environ.get("SIH_OUT", "/kaggle/working/out" if Path("/kaggle").exists() else "out"))
    a = p.parse_args()
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    bundle = Path(resolve(a.bundle_glob)).parent
    model = FinalDownscaler(bundle)
    H, N = model.args.H, model.args.N
    data = V3Data(history_len=H, context=N, fc_history=getattr(model.args, "fc_history", False))
    codes, W = weight_matrix(json.load(open(resolve(a.gp_cells), encoding="utf-8")))
    years = np.array([int(d[:4]) for d in data.dates])
    rows = np.where(years == a.year)[0]
    if a.max_dates:
        rows = rows[:a.max_dates]
    data.idx["_y"] = rows
    dates = data.dates[rows]
    print(f"[{a.tag}] bundle {bundle} | {len(rows)} init dates of {a.year} | {len(codes)} GPs | modes {a.modes}", flush=True)

    if a.with_obs:
        inv = lambda x: data.norm.inv(x, axis=2)
        T = inv(data.targ[rows])                                           # [n,7,6,80,80]
        lo = (N - 16) // 2
        up = F.interpolate(torch.from_numpy(data.fcst[rows][:, :, :, lo:lo + 16, lo:lo + 16]).flatten(0, 1), size=(80, 80),
                           mode="bilinear", align_corners=False).view(-1, 7, 6, 80, 80).numpy()
        G = inv(up)
        obs = {"rain": gp(T[:, :, 0], W), "tmax": gp(T[:, :, 1], W), "tmin": gp(T[:, :, 2], W), "rh": gp(T[:, :, 3], W),
               "wind": gp(np.hypot(T[:, :, 4], T[:, :, 5]) * 3.6, W),
               "gfs_rain": gp(np.maximum(G[:, :, 0], 0), W), "gfs_tmax": gp(G[:, :, 1], W)}
        np.savez_compressed(out / "gp_obs.npz", dates=dates, codes=np.array(codes),
                            **{k: np.transpose(v, (0, 2, 1)).astype(np.float32) for k, v in obs.items()})
        print(f"[{a.tag}] observed + raw GFS saved", flush=True)

    for mode in a.modes:
        t0 = time.time()
        acc = {f: [] for f in FIELDS}
        for b in data.batches("_y", a.bs, False, model.dev):
            o = model.predict(b["history"], b["forecast"], mode=mode, seed=int(b["index"][0]))
            if "members" in o:
                E = o["members"]                                              # [K,B,7,6,80,80]
                er = gp(E[:, :, :, 0], W)                                     # [K,B,7,G]
                acc["rain"].append(er.mean(0)); acc["rain_lo"].append(np.percentile(er, 5, axis=0))
                acc["rain_hi"].append(np.percentile(er, 95, axis=0))
                for f, t in (("p15", 15.0), ("p30", 30.0), ("p64", 64.5)):
                    acc[f].append((er >= t).mean(0))
                acc["rain_pm"].append(gp(o["rain_pm"], W))
                acc["tmax"].append(gp(E[:, :, :, 1], W).mean(0)); acc["tmin"].append(gp(E[:, :, :, 2], W).mean(0))
                acc["rh"].append(gp(E[:, :, :, 3], W).mean(0))
                acc["wind"].append(gp(np.hypot(E[:, :, :, 4], E[:, :, :, 5]) * 3.6, W).mean(0))
            else:
                Mn = o["mean"]                                                # [B,7,6,80,80]
                r = gp(Mn[:, :, 0], W)
                for f in ("rain", "rain_lo", "rain_hi", "rain_pm"):
                    acc[f].append(r)
                for f in ("p15", "p30", "p64"):
                    acc[f].append(np.full_like(r, np.nan))
                acc["tmax"].append(gp(Mn[:, :, 1], W)); acc["tmin"].append(gp(Mn[:, :, 2], W)); acc["rh"].append(gp(Mn[:, :, 3], W))
                acc["wind"].append(gp(np.hypot(Mn[:, :, 4], Mn[:, :, 5]) * 3.6, W))
        arrs = {f: np.transpose(np.concatenate(v), (0, 2, 1)).astype(np.float32) for f, v in acc.items()}  # [n,G,7]
        np.savez_compressed(out / f"gp_{mode}.npz", dates=dates, codes=np.array(codes), **arrs)
        print(f"[{a.tag}] {mode}: {len(rows)} dates in {time.time() - t0:.0f} s | GP rain mean {np.nanmean(arrs['rain']):.2f} mm, "
              f"max {np.nanmax(arrs['rain']):.0f} | tmax {np.nanmean(arrs['tmax']):.1f} C", flush=True)
    print(f"[{a.tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
