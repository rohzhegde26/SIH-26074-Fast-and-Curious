"""
Sprint 10 operating modes + test-time-compute matrix for the final system, scored on the 2023 test season.

  FAST      : deterministic backbone (+ rain quantile mapping fitted on the multi-season out-of-fold predictions)
  diffusion : DPM-Solver++ steps x members grid, raw and with mean-preserving spread calibration (--alpha, fitted
              on 2022 by sihv3.calibrate with a model that never trained on 2022)
Each configuration: CSS (ensemble mean), probabilistic scores, multivariate consistency, latency for ONE forecast
(batch 1, det + diffusion, the operational unit) and per-sample throughput at batch 8, peak VRAM.

  python -m sihv3.modes --det_ckpt '**/s10f_fin_det_s0/best.pt' --diff_ckpt '**/s10f_diff_oof_S_s0/best.pt' \
      --oof_file '**/oof/ydet.npz' --alpha "a1,...,a42" --members 4 16 --tag modes_a
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

from sihv3.calibrate import apply_spread
from sihv3.consistency import consistency
from sihv3.data import V3Data
from sihv3.metrics import composite_skill, det_metrics, prob_metrics
from sihv3.train import build, det_predict, resolve, sample


def load(path, dev, diffusion):
    ck = torch.load(resolve(path), map_location=dev, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    m = build(a, diffusion=diffusion, size=a.size).to(dev)
    m.load_state_dict(ck["ema"])
    sig = torch.tensor(ck["sigma"], device=dev).view(1, 1, 6, 1, 1) if diffusion else None
    return m.eval().requires_grad_(False), sig, a


def fit_qm_arrays(P, T, M, nq=100, block=5):
    """Per lead, per 5x5 block precipitation quantile map P -> T (physical, land only)."""
    qs = np.concatenate([np.linspace(0, 0.98, nq - 10), np.linspace(0.985, 0.9995, 10)])
    nb = 80 // block
    pq, oq = np.zeros((7, nb, nb, nq), np.float32), np.zeros((7, nb, nb, nq), np.float32)
    for l in range(7):
        for i in range(nb):
            for j in range(nb):
                sl = (slice(None), l, 0, slice(i * block, (i + 1) * block), slice(j * block, (j + 1) * block))
                m = M[sl].astype(bool)
                if m.sum() < 50:
                    pq[l, i, j] = oq[l, i, j] = qs * 100
                    continue
                pq[l, i, j], oq[l, i, j] = np.quantile(P[sl][m], qs), np.quantile(T[sl][m], qs)
    return pq, oq


def apply_qm(X, qm, block=5):
    """X physical [..., 7, 6, 80, 80] -> precipitation quantile-mapped copy."""
    pq, oq = qm
    X = X.copy()
    nb = pq.shape[1]
    for l in range(7):
        for i in range(nb):
            for j in range(nb):
                sl = (Ellipsis, l, 0, slice(i * block, (i + 1) * block), slice(j * block, (j + 1) * block))
                x = X[sl]
                y = np.interp(x, pq[l, i, j], oq[l, i, j])
                top = x > pq[l, i, j, -1]
                y[top] = x[top] * oq[l, i, j, -1] / max(pq[l, i, j, -1], 1e-3)
                X[sl] = y
    return X


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--det_ckpt", required=True)
    p.add_argument("--diff_ckpt", required=True)
    p.add_argument("--oof_file", required=True)
    p.add_argument("--alpha", default=None, help="42 comma-separated spread factors [lead-major 7 x 6]")
    p.add_argument("--alpha_file", default=None, help="calibration.json from sihv3.calibrate (alpha_spread 7 x 6)")
    p.add_argument("--steps", type=int, nargs="+", default=[8, 16, 24, 32])
    p.add_argument("--members", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--with_fast", action="store_true")
    p.add_argument("--eval_year", type=int, default=2023,
                   help="season to score; != 2023 is for selection (e.g. 2022 with the fold det + calibration diffusion)")
    p.add_argument("--tag", default="modes")
    p.add_argument("--out", default=os.environ.get("SIH_OUT", "/kaggle/working/out" if Path("/kaggle").exists() else "out"))
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    det, _, da = load(a.det_ckpt, dev, False)
    diff, sigma, fa = load(a.diff_ckpt, dev, True)
    assert (da.H, da.N) == (fa.H, fa.N)
    data = V3Data(history_len=da.H, context=da.N, fc_history=getattr(da, "fc_history", False))
    z = np.load(resolve(a.oof_file))
    assert list(z["dates"].astype(str)) == list(data.dates) and str(z["kind"]) == "oof"
    ydet_all = z["ydet"]
    years = np.array([int(d[:4]) for d in data.dates])
    assert not (a.with_fast and a.eval_year != 2023), "FAST/QM evaluation is defined on the 2023 test season only"
    test = np.where(years == a.eval_year)[0]
    assert (a.alpha is None) != (a.alpha_file is None), "give exactly one of --alpha / --alpha_file"
    if a.alpha_file:
        cal = json.load(open(resolve(a.alpha_file)))
        alpha = np.array(cal["alpha_spread"], np.float32).reshape(7, 6, 1, 1)
        print(f"[{a.tag}] spread factors from {resolve(a.alpha_file)} (fit {cal['fit_years']} -> eval {cal['eval_years']})", flush=True)
    else:
        alpha = np.array([float(x) for x in a.alpha.split(",")], np.float32).reshape(7, 6, 1, 1)
    inv = lambda x, ax: data.norm.inv(x, axis=ax)
    T = inv(data.targ[test], 2)
    M = data.mask[test].astype(np.float32)
    lo = (data.N - 16) // 2
    up = F.interpolate(torch.from_numpy(data.fcst[test][:, :, :, lo:lo + 16, lo:lo + 16]).flatten(0, 1),
                       size=(80, 80), mode="bilinear", align_corners=False).view(-1, 7, 6, 80, 80).numpy()
    ref = det_metrics(inv(up, 2), T, M)["aggregate"]
    obs_cons = consistency(T, T, M)["observed"]
    res = {"observed_consistency": obs_cons, "configs": []}
    data.idx["_t"] = test

    def latency(fn, n=3):
        """seconds for ONE forecast (batch 1), mean of n samples after a warm-up."""
        data.idx["_one"] = test[:n + 1]
        bs = list(data.batches("_one", 1, False, dev))
        fn(bs[0])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for b in bs[1:]:
            fn(b)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / n

    if a.with_fast:
        def fast(b):
            with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                return det_predict(det, b).float()
        Pn = np.concatenate([fast(b).cpu().numpy() for b in data.batches("_t", 8, False, dev)])
        # the OOF file's 2023 rows come from this same final model: consistency check of the artefacts
        res["fast_vs_oof_file_maxdiff"] = float(np.abs(Pn - ydet_all[test].astype(np.float32)).max())
        P = inv(Pn, 2)
        train = np.where(years != 2023)[0]
        qm = fit_qm_arrays(inv(ydet_all[train].astype(np.float32), 2), inv(data.targ[train], 2), data.mask[train])
        np.savez(out / "calibration_params.npz", alpha=alpha.ravel(), qm_pq=qm[0], qm_oq=qm[1])  # for the final bundle
        lat = latency(fast)
        for name, X in (("FAST raw", P), ("FAST + rain QM (multi-season OOF fit)", apply_qm(P, qm))):
            agg = det_metrics(X, T, M)["aggregate"]
            res["configs"].append({"mode": name, "steps": 0, "members": 1, "css": composite_skill(agg, ref), "aggregate": agg,
                                   "consistency": consistency(X, T, M), "latency_s_one_forecast": lat})
            print(f"[{a.tag}] {name}: CSS {composite_skill(agg, ref):.4f} bias {agg['precip_bias_ratio']:.2f} "
                  f"csi30 {agg['precip_csi30']:.3f} Tmax {agg['tmax_mae']:.3f} | {lat:.3f} s/forecast", flush=True)

    ydet_t = torch.from_numpy(ydet_all.astype(np.float32))
    for K in a.members:
        for S in a.steps:
            g = torch.Generator(device=dev).manual_seed(100 + K * 7 + S)

            def run(b, K=K, S=S, g=g):
                with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    yd = det_predict(det, b).float()  # operational path: det model output, not the file
                    return sample(diff, b, yd, sigma, steps=S, members=K, sampler="dpmpp2m", gen=g)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            E = np.concatenate([inv(run(b).float(), 3).cpu().numpy().transpose(1, 0, 2, 3, 4, 5)
                                for b in data.batches("_t", 8, False, dev)])
            thr = (time.time() - t0) / len(test)
            vram = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else None
            lat = latency(run)
            for cal, X in (("raw", E), ("spread-calibrated", apply_spread(E, alpha))):
                r = prob_metrics(X, T, M)
                agg = r["aggregate"]
                row = {"mode": f"diffusion K={K} S={S} {cal}", "steps": S, "members": K, "calibration": cal,
                       "css": composite_skill(agg, ref), "aggregate": agg, "consistency": consistency(X, T, M),
                       "latency_s_one_forecast": lat, "throughput_s_per_sample_batch8": thr, "peak_vram_gb": vram}
                res["configs"].append(row)
                print(f"[{a.tag}] K={K:2d} S={S:2d} {cal:17s}: CSS {row['css']:.4f} crps_p {agg['precip_crps']:.3f} "
                      f"ssr_p {agg.get('precip_ssr', float('nan')):.2f} brier30 {agg['brier30']:.4f} tmax_crps {agg['tmax_crps']:.3f} "
                      f"| {lat:.2f} s/forecast, {vram or 0:.1f} GB", flush=True)
            json.dump(res, open(out / "modes.json", "w"), indent=1, default=float)
    json.dump(res, open(out / "modes.json", "w"), indent=1, default=float)
    print(f"[{a.tag}] DONE", flush=True)


if __name__ == "__main__":
    main()
