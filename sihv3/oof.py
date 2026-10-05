"""
Precompute the deterministic prediction y_det (normalised space) for every sample, for cross-fitted residual
diffusion (the diffusion stage then learns *out-of-sample* residuals, which are wider and wetter than the
in-sample residuals that caused the under-dispersed, dry ensembles in Sprint 6).

  --mode oof      : sample in a fold's held-out years -> that fold's model; 2023 -> the base (final) model
  --mode insample : every sample -> the base model (control: the standard, non-cross-fitted recipe)

  python -m sihv3.oof --mode oof --base '**/fin_det_s0/best.pt' \
      --folds '**/cf_f0/best.pt=2015,2016' '**/cf_f1/best.pt=2017,2018' ... --tag oof
Writes <out>/<tag>/ydet.npz with ydet [N,7,6,80,80] float16, dates [N], kind.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from sihv3.data import V3Data
from sihv3.train import build, det_predict, parse_years, resolve


def load(path, dev):
    ck = torch.load(resolve(path), map_location=dev, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    m = build(a, diffusion=False, size=a.size).to(dev)
    m.load_state_dict(ck["ema"])
    return m.eval().requires_grad_(False), a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["oof", "insample"], required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--folds", nargs="*", default=[], help="ckpt_pattern=held_out_years")
    p.add_argument("--tag", default="oof")
    p.add_argument("--out", default=os.environ.get("SIH_OUT", "/kaggle/working/out" if Path("/kaggle").exists() else "out"))
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, ba = load(a.base, dev)
    data = V3Data(history_len=ba.H, context=ba.N, fc_history=getattr(ba, "fc_history", False))
    years = np.array([int(d[:4]) for d in data.dates])
    owner = {}  # sample row -> model
    if a.mode == "oof":
        for spec in a.folds:
            pat, ys = spec.split("=")
            m, fa = load(pat, dev)
            assert (fa.H, fa.N) == (ba.H, ba.N), "fold and base models must share H/N"
            held = parse_years(ys)
            assert not set(held) & set(parse_years(fa.train_years)), f"{pat}: held-out years were in its training set"
            for i in np.where(np.isin(years, held))[0]:
                assert i not in owner, "a sample is held out by two folds"
                owner[i] = m
    for i in range(len(years)):
        owner.setdefault(i, base)
    if a.mode == "oof":
        assert all(owner[i] is base for i in np.where(years == 2023)[0])
        uncovered = sorted(set(years[[i for i in owner if owner[i] is base]]) - {2023})
        assert not uncovered, f"training years without an out-of-fold model: {uncovered}"
    ydet = np.zeros(data.targ.shape, np.float16)
    data.idx["all"] = np.arange(len(years))
    for b in data.batches("all", 16, False, dev):
        rows = b["index"]
        for m in {id(owner[i]): owner[i] for i in rows}.values():
            sel = np.array([owner[i] is m for i in rows])
            sub = {k: (v[torch.from_numpy(sel).to(v.device)] if torch.is_tensor(v) and v.shape[0] == len(rows) else v)
                   for k, v in b.items() if k != "index"}
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                ydet[rows[sel]] = det_predict(m, sub).float().cpu().numpy().astype(np.float16)
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "ydet.npz", ydet=ydet, dates=data.dates, kind=np.array(a.mode))
    if a.mode == "oof":  # out-of-fold skill on the training seasons = clean multi-season model-selection score
        import json
        import torch.nn.functional as F
        from sihv3.metrics import composite_skill, det_metrics
        res = {}
        for name, yrs in [("all_oof", sorted(set(years) - {2023}))] + [(str(y), [y]) for y in sorted(set(years) - {2023})]:
            rows = np.where(np.isin(years, yrs))[0]
            P = data.norm.inv(ydet[rows].astype(np.float32), axis=2)
            T = data.norm.inv(data.targ[rows], axis=2)
            M = data.mask[rows].astype(np.float32)
            lo = (data.N - 16) // 2
            up = F.interpolate(torch.from_numpy(data.fcst[rows][:, :, :, lo:lo + 16, lo:lo + 16]).flatten(0, 1),
                               size=(80, 80), mode="bilinear", align_corners=False).view(-1, 7, 6, 80, 80).numpy()
            R = data.norm.inv(up, axis=2)
            ma, ra = det_metrics(P, T, M)["aggregate"], det_metrics(R, T, M)["aggregate"]
            res[name] = {"css": composite_skill(ma, ra), "aggregate": ma}
        json.dump(res, open(out / "oof_metrics.json", "w"), indent=1, default=float)
        print(f"[{a.tag}] OOF CSS all seasons {res['all_oof']['css']:.4f} | per season "
              + " ".join(f"{k}:{v['css']:.3f}" for k, v in res.items() if k != "all_oof"), flush=True)
    by = {int(y): ("base" if owner[int(np.where(years == y)[0][0])] is base else "fold") for y in sorted(set(years))}
    print(f"[{a.tag}] wrote ydet {ydet.shape} mode={a.mode} source by year: {by}", flush=True)


if __name__ == "__main__":
    main()
