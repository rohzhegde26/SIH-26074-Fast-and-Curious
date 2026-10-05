"""
Post-hoc ensemble calibration study (Sprint 10 open problem: dry, under-dispersed rain ensembles).

Generates K-member residual-diffusion ensembles (DPM-Solver++), fits calibration on a FIT season and scores it on
an EVAL season the calibration never saw:
  raw      : as sampled
  spread   : x' = mean + alpha (x - mean), alpha per (lead, variable) minimising fair CRPS on FIT
  rainqm   : precipitation members quantile-mapped per lead (pooled over land pixels and members) to the observed
             FIT distribution
  both     : rainqm, then spread re-fitted
Optionally applies the FIT-season parameters to the ensembles of a second model (--apply_ckpt), e.g. the final
model trained on all seasons, for which no clean FIT season exists.

  python -m sihv3.calibrate --diff_ckpt '**/s10_diff_oof_S_cal_s0/best.pt' --det_pred_file '**/oof/ydet.npz' \
      --fit_years 2022 --eval_years 2023 --apply_ckpt '**/s10_diff_oof_S_s0/best.pt' --tag cal
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from sihv3.data import V3Data
from sihv3.metrics import composite_skill, det_metrics, fair_crps, prob_metrics
from sihv3.train import build, parse_years, resolve, sample

ALPHAS = np.round(np.arange(0.8, 4.01, 0.1), 2)


def load_diff(path, dev):
    ck = torch.load(resolve(path), map_location=dev, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    m = build(a, diffusion=True, size=a.size).to(dev)
    m.load_state_dict(ck["ema"])
    return m.eval().requires_grad_(False), torch.tensor(ck["sigma"], device=dev).view(1, 1, 6, 1, 1), a


def ensembles(model, sigma, data, ydet, rows, dev, K, steps, seed):
    """-> ens [n,K,7,6,80,80], targ, mask, ref (GFS-bilinear) in physical units."""
    g = torch.Generator(device=dev).manual_seed(seed)
    E, T, M, R = [], [], [], []
    data.idx["_sel"] = rows
    for b in data.batches("_sel", 8, False, dev):
        yd = ydet[b["index"]].to(dev)
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            ens = sample(model, b, yd, sigma, steps=steps, members=K, sampler="dpmpp2m", gen=g)
        E.append(data.norm.inv(ens.float(), axis=3).cpu().numpy().transpose(1, 0, 2, 3, 4, 5))
        T.append(data.norm.inv(b["target"], axis=2).cpu().numpy())
        M.append(b["mask"].cpu().numpy())
        lo = (b["forecast"].shape[-1] - 16) // 2
        up = torch.nn.functional.interpolate(b["forecast"][:, :, :, lo:lo + 16, lo:lo + 16].flatten(0, 1),
                                             size=(80, 80), mode="bilinear", align_corners=False).view(-1, 7, 6, 80, 80)
        R.append(data.norm.inv(up, axis=2).cpu().numpy())
    return np.concatenate(E), np.concatenate(T), np.concatenate(M), np.concatenate(R)


def inflate_rain(e, a, axis):
    """Mean-preserving spread inflation for non-negative rain: inflate, clip at 0, rescale to the original mean."""
    m = e.mean(axis, keepdims=True)
    x = np.maximum(m + a * (e - m), 0)
    mx = x.mean(axis, keepdims=True)
    return np.where(mx > 1e-6, x * (m / np.maximum(mx, 1e-6)), x)


def apply_spread(E, alpha):
    m = E.mean(1, keepdims=True)
    out = m + alpha[None, None] * (E - m)          # alpha [7,6,1,1]
    for l in range(7):
        out[:, :, l, 0] = inflate_rain(E[:, :, l, 0], alpha[l, 0, 0, 0], axis=1)
    out[:, :, :, 1] = np.maximum(out[:, :, :, 1], out[:, :, :, 2])
    return out


def fit_spread(E, T, M):
    alpha = np.ones((7, 6, 1, 1), np.float32)
    for l in range(7):
        for c in range(6):
            m = M[:, l, c].astype(bool)
            o = T[:, l, c][m]
            e = E[:, :, l, c].transpose(1, 0, 2, 3)[:, m]
            mu = e.mean(0)
            best = min(ALPHAS, key=lambda a: fair_crps(inflate_rain(e, a, axis=0) if c == 0 else mu + a * (e - mu), o).mean())
            alpha[l, c] = best
    return alpha


def fit_rainqm(E, T, M, nq=200):
    qs = np.concatenate([np.linspace(0, 0.99, nq - 20), np.linspace(0.99, 0.9999, 20)])
    out = []
    for l in range(7):
        m = M[:, l, 0].astype(bool)
        e = E[:, :, l, 0].transpose(1, 0, 2, 3)[:, m].ravel()
        out.append((np.quantile(e, qs), np.quantile(T[:, l, 0][m], qs)))
    return out


def apply_rainqm(E, qm):
    E = E.copy()
    for l, (pq, oq) in enumerate(qm):
        x = E[:, :, l, 0]
        y = np.interp(x, pq, oq)
        top = x > pq[-1]
        y[top] = x[top] * oq[-1] / max(pq[-1], 1e-3)
        E[:, :, l, 0] = y
    return E


def score(E, T, M, R):
    ref = det_metrics(R, T, M)["aggregate"]
    r = prob_metrics(E, T, M)
    a = r["aggregate"]
    return {"css_ensmean": composite_skill(a, ref), **{k: a[k] for k in a if any(s in k for s in (
        "crps", "ssr", "cov90", "brier", "bias_ratio", "csi15", "csi30", "wet_mae", "tmax_mae", "rh_mae", "wind"))}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diff_ckpt", required=True)
    p.add_argument("--det_pred_file", required=True)
    p.add_argument("--fit_years", required=True)
    p.add_argument("--eval_years", required=True)
    p.add_argument("--apply_ckpt", default=None)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--tag", default="cal")
    p.add_argument("--out", default=os.environ.get("SIH_OUT", "/kaggle/working/out" if Path("/kaggle").exists() else "out"))
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    model, sigma, ma = load_diff(a.diff_ckpt, dev)
    fy, ey = parse_years(a.fit_years), parse_years(a.eval_years)
    assert not set(fy) & set(ey), "fit and eval seasons must differ"
    assert not set(fy) & set(parse_years(ma.train_years) or []), "calibration FIT season was in the diffusion training set"
    data = V3Data(history_len=ma.H, context=ma.N, fc_history=getattr(ma, "fc_history", False))
    z = np.load(resolve(a.det_pred_file))
    assert list(z["dates"].astype(str)) == list(data.dates)
    ydet = torch.from_numpy(z["ydet"].astype(np.float32))
    years = np.array([int(d[:4]) for d in data.dates])
    rf, re_ = np.where(np.isin(years, fy))[0], np.where(np.isin(years, ey))[0]
    Ef, Tf, Mf, _ = ensembles(model, sigma, data, ydet, rf, dev, a.members, a.steps, 1)
    Ee, Te, Me, Re = ensembles(model, sigma, data, ydet, re_, dev, a.members, a.steps, 2)
    alpha = fit_spread(Ef, Tf, Mf)
    qm = fit_rainqm(Ef, Tf, Mf)
    alpha2 = fit_spread(apply_rainqm(Ef, qm), Tf, Mf)
    res = {"fit_years": fy, "eval_years": ey, "alpha_spread": alpha[:, :, 0, 0].tolist(),
           "alpha_after_rainqm": alpha2[:, :, 0, 0].tolist(), "model": {}, "applied": {}}
    variants = {"raw": lambda E: E, "spread": lambda E: apply_spread(E, alpha), "rainqm": lambda E: apply_rainqm(E, qm),
                "both": lambda E: apply_spread(apply_rainqm(E, qm), alpha2)}
    for k, f in variants.items():
        res["model"][k] = score(f(Ee), Te, Me, Re)
        print(f"[{a.tag}] {ma.tag} eval {ey} {k:7s}: " + " ".join(f"{kk} {v:.4f}" for kk, v in res["model"][k].items()
                                                                  if kk in ("css_ensmean", "precip_crps", "precip_ssr", "precip_cov90", "precip_bias_ratio", "brier30", "tmax_crps")), flush=True)
    if a.apply_ckpt:
        m2, s2, a2 = load_diff(a.apply_ckpt, dev)
        assert (a2.H, a2.N) == (ma.H, ma.N)
        E2, T2, M2, R2 = ensembles(m2, s2, data, ydet, re_, dev, a.members, a.steps, 3)
        for k, f in variants.items():
            res["applied"][k] = score(f(E2), T2, M2, R2)
            print(f"[{a.tag}] {a2.tag} eval {ey} {k:7s} (params from {ma.tag}): " + " ".join(
                f"{kk} {v:.4f}" for kk, v in res["applied"][k].items()
                if kk in ("css_ensmean", "precip_crps", "precip_ssr", "precip_cov90", "precip_bias_ratio", "brier30", "tmax_crps")), flush=True)
    json.dump(res, open(out / "calibration.json", "w"), indent=1, default=float)
    print(f"[{a.tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
