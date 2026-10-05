"""Latency of the four operating modes for ONE forecast (batch 1) on the current device: members sampled sequentially
vs batched into one pass (sample(..., batch_members=True), same initial noise -> same samples).

  python -m sihv3.bench --det_ckpt '**/s10f_fin_det_s0/best.pt' --diff_ckpt '**/s10f_diff_oof_S_s0/best.pt'
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
from sihv3.modes import load
from sihv3.train import det_predict, sample

MODES = {"FAST": (0, 1), "BALANCED": (16, 8), "ACCURATE": (32, 8), "ENSEMBLE": (24, 16)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--det_ckpt", required=True)
    p.add_argument("--diff_ckpt", required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--tag", default="bench")
    p.add_argument("--out", default=os.environ.get("SIH_OUT", "/kaggle/working/out" if Path("/kaggle").exists() else "out"))
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det, _, da = load(a.det_ckpt, dev, False)
    diff, sigma, _ = load(a.diff_ckpt, dev, True)
    data = V3Data(history_len=da.H, context=da.N, fc_history=getattr(da, "fc_history", False))
    years = np.array([int(d[:4]) for d in data.dates])
    data.idx["_one"] = np.where(years == 2023)[0][:a.n + 1]
    bs = list(data.batches("_one", 1, False, dev))
    amp = dict(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda")
    res = {"device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "modes": {}}
    for mode, (S, K) in MODES.items():
        for batched in ([False] if K == 1 else [False, True]):
            def run(b):
                with torch.no_grad(), torch.autocast(**amp):
                    yd = det_predict(det, b).float()
                    if K > 1:
                        sample(diff, b, yd, sigma, steps=S, members=K, sampler="dpmpp2m",
                               gen=torch.Generator(device=dev).manual_seed(0), batch_members=batched)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            run(bs[0])  # warm-up
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for b in bs[1:]:
                run(b)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            lat = (time.time() - t0) / a.n
            vram = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None
            key = f"{mode} ({'batched' if batched else 'sequential'})"
            res["modes"][key] = {"steps": S, "members": K, "latency_s": lat, "peak_vram_gb": vram}
            print(f"[{a.tag}] {key:24s} S={S:2d} K={K:2d}: {lat:.3f} s/forecast, {vram or 0:.2f} GB", flush=True)
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "bench.json", "w"), indent=1)


if __name__ == "__main__":
    main()
