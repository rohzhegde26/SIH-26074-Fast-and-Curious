"""
Sprint 11 post-training improvements, scored on one season without retraining anything:

  backbone  : y_det from ONE seed's model vs the AVERAGE of several seeds' models (taken from their OOF files, whose
              rows for the evaluated season come from models that never saw it: fold models for 2022, the final
              models for 2023)
  FAST      : y_det raw and with rain quantile mapping fitted on the other training seasons of the same OOF files
  diffusion : the same initial noise for both backbones (paired), DPM-Solver++ S steps, K members;
              ensemble mean vs PROBABILITY-MATCHED mean for rain (keeps the members' rain distribution, so extremes
              are not averaged away); raw vs spread-calibrated (alpha from sihv3.calibrate)

  python -m sihv3.post --det_files '**/oof/ydet.npz' '**/oof_s1.npz' '**/oof_s2.npz' \
      --diff_ckpt '**/s10f_diff_oof_S_cal_s0/best.pt' --alpha_file '**/calf/calibration.json' --eval_year 2022 --tag post22
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sihv3.calibrate import apply_spread
from sihv3.data import V3Data
from sihv3.metrics import composite_skill, det_metrics, prob_metrics
from sihv3.modes import apply_qm, fit_qm_arrays, load
from sihv3.train import resolve, sample


def pm_mean(E, M):
    """Probability-matched ensemble mean of rain. E [n,K,7,6,80,80] physical, M mask [n,7,6,80,80].
    Per sample and lead: rank the land pixels by the ensemble mean, then give them the n quantiles (every K-th
    value) of the pooled members' rain values. Other variables keep the plain mean."""
    out = E.mean(1)
    K = E.shape[1]
    for i in range(E.shape[0]):
        for l in range(7):
            land = M[i, l, 0].astype(bool)
            n = int(land.sum())
            if n == 0:
                continue
            pooled = np.sort(E[i, :, l, 0][:, land].ravel())          # K*n values
            q = pooled[K // 2::K][:n]                                  # n matched quantiles, ascending
            order = np.argsort(out[i, l, 0][land], kind="stable")
            vals = np.empty(n, np.float32)
            vals[order] = q
            field = out[i, l, 0]
            field[land] = vals
    return out


def tail_cap(targ_phys, mask, block=5, factor=2.0, ceiling=1.2):
    """Rain cap per pixel = min(factor x the largest observed daily rain in its 5x5 block, ceiling x the largest in
    the whole domain), from the given (training) rows only. factor 2 / ceiling 1.2 were chosen by leave-one-season-out
    over 2015-2022: no held-out season ever exceeds the cap (factor 1 clipped 0.014 % of real pixel-days)."""
    P = np.where(mask[:, :, 0], targ_phys[:, :, 0], 0.0)              # [n,7,80,80]
    bmax = P.reshape(P.shape[0], 7, 80 // block, block, 80 // block, block).max((0, 1, 3, 5))
    return np.minimum(factor * np.kron(bmax, np.ones((block, block), np.float32)), ceiling * P.max())


def apply_cap(E, cap):
    """Cap every member's rain at the training block maximum (other variables unchanged)."""
    E = E.copy()
    E[:, :, :, 0] = np.minimum(E[:, :, :, 0], cap)
    return E


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--det_files", nargs="+", required=True, help="OOF ydet files of several seeds; the first is 'single'")
    p.add_argument("--diff_ckpt", required=True)
    p.add_argument("--alpha_file", required=True)
    p.add_argument("--eval_year", type=int, required=True)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--members", type=int, default=16)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--backbones", nargs="+", default=["single", "avg"], choices=["single", "avg"])
    p.add_argument("--tag", default="post")
    p.add_argument("--out", default=os.environ.get("SIH_OUT", "/kaggle/working/out" if Path("/kaggle").exists() else "out"))
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    diff, sigma, fa = load(a.diff_ckpt, dev, True)
    assert a.eval_year not in fa_years(fa), f"the diffusion model trained on {a.eval_year}"
    data = V3Data(history_len=fa.H, context=fa.N)
    files = [np.load(resolve(f)) for f in a.det_files]
    for z in files:
        assert list(z["dates"].astype(str)) == list(data.dates) and str(z["kind"]) == "oof"
    single = files[0]["ydet"].astype(np.float32)
    avg = np.mean([z["ydet"].astype(np.float32) for z in files], 0)
    years = np.array([int(d[:4]) for d in data.dates])
    ev = np.where(years == a.eval_year)[0]
    fit = np.where((years != a.eval_year) & (years != 2023))[0]          # QM fit seasons (all out-of-fold rows)
    inv = lambda x, ax: data.norm.inv(x, axis=ax)
    T = inv(data.targ[ev], 2)
    M = data.mask[ev].astype(np.float32)
    lo = (data.N - 16) // 2
    up = F.interpolate(torch.from_numpy(data.fcst[ev][:, :, :, lo:lo + 16, lo:lo + 16]).flatten(0, 1),
                       size=(80, 80), mode="bilinear", align_corners=False).view(-1, 7, 6, 80, 80).numpy()
    ref = det_metrics(inv(up, 2), T, M)["aggregate"]
    cal = json.load(open(resolve(a.alpha_file)))
    alpha = np.array(cal["alpha_spread"], np.float32).reshape(7, 6, 1, 1)
    res = {"eval_year": a.eval_year, "n_seeds": len(files), "rows": []}
    cap = tail_cap(inv(data.targ[fit], 2), data.mask[fit])          # training seasons only (never the evaluated one)
    print(f"[{a.tag}] rain tail cap from {len(fit)} training forecasts: block max range {cap.min():.0f}-{cap.max():.0f} mm/day; "
          f"evaluated-season observed max {np.where(M[:, :, 0] > 0, T[:, :, 0], 0).max():.0f}", flush=True)

    def row(name, agg, extra=None):
        r = {"variant": name, "css": composite_skill(agg, ref), "aggregate": agg, **(extra or {})}
        res["rows"].append(r)
        print(f"[{a.tag}] {name:58s} CSS {r['css']:.4f} | csi30 {agg['precip_csi30']:.3f} csi64.5 {agg['precip_csi64.5']:.3f} "
              f"pod64.5 {agg['precip_pod64.5']:.2f} bias {agg['precip_bias_ratio']:.2f}"
              + (f" | crps_p {agg['precip_crps']:.3f} ssr_p {agg['precip_ssr']:.2f} brier30 {agg['brier30']:.4f}" if "precip_crps" in agg else ""),
              flush=True)

    # FAST
    for nm, Y in (("single seed", single), (f"{len(files)}-seed average", avg)):
        P = inv(Y[ev], 2)
        qm = fit_qm_arrays(inv(Y[fit], 2), inv(data.targ[fit], 2), data.mask[fit])
        row(f"FAST {nm} raw", det_metrics(P, T, M)["aggregate"])
        row(f"FAST {nm} + rain QM", det_metrics(apply_qm(P, qm), T, M)["aggregate"])

    # diffusion, paired noise across backbones
    data.idx["_e"] = ev
    bb = {"single": ("single seed", single), "avg": (f"{len(files)}-seed average", avg)}
    for nm, Y in (bb[k] for k in a.backbones):
        g = torch.Generator(device=dev).manual_seed(1234)
        Yt = torch.from_numpy(Y)
        E = []
        for b in data.batches("_e", a.bs, False, dev):
            yd = Yt[b["index"]].to(dev)
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                s = sample(diff, b, yd, sigma, steps=a.steps, members=a.members, sampler="dpmpp2m", gen=g, batch_members=True)
            E.append(inv(s.float(), 3).cpu().numpy().transpose(1, 0, 2, 3, 4, 5))
        E = np.concatenate(E)
        lm = M[:, None, :, 0].astype(bool)
        x = E[:, :, :, 0][np.broadcast_to(lm, E[:, :, :, 0].shape)]
        res.setdefault("member_tails", {})[nm] = {q: float(np.quantile(x, float(q))) for q in ("0.99", "0.999", "0.9999")}
        res["member_tails"][nm]["max"] = float(x.max())
        res["member_tails"][nm]["frac_above_cap"] = float((E[:, :, :, 0] > cap)[np.broadcast_to(lm, E[:, :, :, 0].shape)].mean())
        print(f"[{a.tag}] {nm} member rain tails (land): {res['member_tails'][nm]}", flush=True)
        for K in sorted({8, a.members}):
            sc = apply_spread(E[:, :K], alpha)
            for cname, X in (("raw", E[:, :K]), ("spread-cal", sc), ("spread-cal + tail cap", apply_cap(sc, cap))):
                agg = prob_metrics(X, T, M)["aggregate"]
                row(f"diff {nm} K={K} {cname} | ens mean", agg)
                pm = det_metrics(pm_mean(X, M), T, M)["aggregate"]
                for k in [k for k in agg if any(s in k for s in ("crps", "ssr", "cov90", "brier"))]:
                    pm[k] = agg[k]                                          # probabilistic scores are unchanged
                row(f"diff {nm} K={K} {cname} | PM mean", pm)
        json.dump(res, open(out / "post.json", "w"), indent=1, default=float)
    json.dump(res, open(out / "post.json", "w"), indent=1, default=float)
    print(f"[{a.tag}] DONE", flush=True)


def fa_years(fa):
    from sihv3.train import parse_years
    return set(parse_years(getattr(fa, "train_years", None)) or [])


if __name__ == "__main__":
    main()
