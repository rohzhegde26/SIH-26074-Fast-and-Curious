"""
Sprint 7 (denoising steps x sampler) and Sprint 8 (ensemble size vs steps at matched compute) evaluation
of a trained residual-diffusion model. Validation split only (decisions); test reported for the winners.

  python -m sihv3.eval_diff --diff_ckpt .../best.pt --det_ckpt .../best.pt --study s7 --tag s7_eval
  python -m sihv3.eval_diff ... --study s8 --tag s8_eval
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from sihv3.data import V3Data
from sihv3.metrics import fair_crps
from sihv3.train import build, det_predict, evaluate, resolve, sample

BINS = [(0, 1), (1, 15), (15, 30), (30, 64.5), (64.5, 1e9)]

S7_GRID = [(s, sm) for sm in ("ddim", "ddim_eta1", "dpmpp2m") for s in (4, 8, 12, 16, 24, 32)]
S8_GRID = [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2),          # 32 NFE
           (2, 32), (4, 16), (8, 8), (16, 4), (32, 2),         # 64 NFE
           (4, 32), (8, 16), (16, 8), (32, 4)]                 # 128 NFE (K<=32 keeps RAM < 6 GB)


def load(path, dev, args_over=None):
    ck = torch.load(resolve(path), map_location=dev, weights_only=False)
    a = argparse.Namespace(**{**ck["args"], **(args_over or {})})
    m = build(a, diffusion=a.mode == "diff", size=a.size).to(dev)
    m.load_state_dict(ck["ema"])
    return m.eval().requires_grad_(False), ck


def by_bins(ens_phys, targ_phys, mask):
    """CRPS and ensemble-mean MAE of precipitation, per observed-intensity bin."""
    E = ens_phys[:, :, :, 0]  # [n,K,7,80,80]
    T, M = targ_phys[:, :, 0], mask[:, :, 0].astype(bool)
    e = E.transpose(1, 0, 2, 3, 4)[:, M]
    o = T[M]
    crps = fair_crps(e, o) if e.shape[0] > 1 else np.abs(e[0] - o)
    mae = np.abs(e.mean(0) - o)
    return {f"{lo:g}-{hi:g}": {"n": int(((o >= lo) & (o < hi)).sum()),
                               "crps": float(crps[(o >= lo) & (o < hi)].mean()) if ((o >= lo) & (o < hi)).any() else None,
                               "mae": float(mae[(o >= lo) & (o < hi)].mean()) if ((o >= lo) & (o < hi)).any() else None}
            for lo, hi in BINS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diff_ckpt", required=True)
    p.add_argument("--det_ckpt", required=True)
    p.add_argument("--study", choices=["s7", "s8"], required=True)
    p.add_argument("--members", type=int, default=8, help="ensemble size for the s7 sweep")
    p.add_argument("--part", type=int, default=0, help="split the grid across GPU lanes")
    p.add_argument("--nparts", type=int, default=1)
    p.add_argument("--tag", default="eval")
    p.add_argument("--out", default=os.environ.get("SIH_OUT", "/kaggle/working/out" if Path("/kaggle").exists() else "out"))
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    diff, ck = load(a.diff_ckpt, dev)
    det, _ = load(a.det_ckpt, dev)
    sigma = torch.tensor(ck["sigma"], device=dev).view(1, 1, 6, 1, 1)
    data = V3Data(history_len=ck["args"]["H"], context=ck["args"]["N"])
    grid = S7_GRID if a.study == "s7" else S8_GRID
    grid = grid[a.part::a.nparts]
    rows = []
    for item in grid:
        if a.study == "s7":
            steps, smp = item
            K = a.members
        else:
            K, steps = item
            smp = "ddim"
        eta = 1.0 if smp == "ddim_eta1" else 0.0
        smp_name = "ddim" if smp.startswith("ddim") else smp
        g = torch.Generator(device=dev).manual_seed(11)
        store = {}

        def pf(b):
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                yd = det_predict(det, b).float()
                ens = sample(diff, b, yd, sigma, steps=steps, members=K, sampler=smp_name, eta=eta, gen=g)
            return ens

        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        acc = {"e": [], "t": [], "m": []}

        def hook(pp, tt, mm):  # keep precipitation only: [K,B,7,80,80]
            acc["e"].append(pp[:, :, :, 0]); acc["t"].append(tt[:, :, 0]); acc["m"].append(mm[:, :, 0])

        r = evaluate(data, "val", pf, dev, hook=hook)
        E = np.concatenate(acc["e"], 1)[:, :, :, None].transpose(1, 0, 2, 3, 4, 5)  # [n,K,7,1,80,80]
        bins = by_bins(E, np.concatenate(acc["t"])[:, :, None], np.concatenate(acc["m"])[:, :, None])
        del acc, E
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        row = {"sampler": smp, "steps": steps, "members": K, "nfe": steps * K,
               "sec_per_sample": dt / len(data.idx["val"]), "css_ensmean": r["css"],
               "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None,
               "aggregate": r["aggregate"], "per_lead": r["per_lead"], "precip_by_intensity": bins}
        rows.append(row)
        print(f"[{a.tag}] {smp} S={steps} K={K} NFE={steps * K} css {r['css']:.4f} "
              f"crps_p {r['aggregate'].get('precip_crps', float('nan')):.3f} ssr_p {r['aggregate'].get('precip_ssr', float('nan')):.2f} "
              f"cov90_p {r['aggregate'].get('precip_cov90', float('nan')):.2f} {dt / len(data.idx['val']):.2f}s/sample", flush=True)
        json.dump(rows, open(out / "rows.json", "w"), indent=1, default=float)
    print(f"[{a.tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
