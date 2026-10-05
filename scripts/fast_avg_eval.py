"""FAST (seed-averaged backbone + rain QM fitted on the other out-of-fold seasons) scored on one season, from OOF
files only (CPU). Same protocol as the FAST rows of sihv3.post.

  python scripts/fast_avg_eval.py --files a.npz b.npz c.npz --eval_year 2022 [--H 5 --fc_history]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sihv3.data import V3Data  # noqa: E402
from sihv3.metrics import composite_skill, det_metrics  # noqa: E402
from sihv3.modes import apply_qm, fit_qm_arrays  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--eval_year", type=int, nargs="+", required=True, help="one or more seasons")
    a = p.parse_args()
    data = V3Data(history_len=3, context=40)  # inputs are not used: only targets, masks and the GFS reference
    Y = None
    for f in a.files:
        z = np.load(f)
        assert list(z["dates"].astype(str)) == list(data.dates) and str(z["kind"]) == "oof", f
        y = z["ydet"].astype(np.float32)
        Y = y if Y is None else Y + y
        del y
    Y /= len(a.files)
    years = np.array([int(d[:4]) for d in data.dates])
    inv = lambda x: data.norm.inv(x, axis=2)
    for ey in a.eval_year:
        ev, fit = np.where(years == ey)[0], np.where((years != ey) & (years != 2023))[0]
        T, M = inv(data.targ[ev]), data.mask[ev].astype(np.float32)
        up = F.interpolate(torch.from_numpy(data.fcst[ev][:, :, :, 12:28, 12:28]).flatten(0, 1), size=(80, 80), mode="bilinear",
                           align_corners=False).view(-1, 7, 6, 80, 80).numpy()
        ref = det_metrics(inv(up), T, M)["aggregate"]
        qm = fit_qm_arrays(inv(Y[fit]), inv(data.targ[fit]), data.mask[fit])
        for name, P in (("raw", inv(Y[ev])), ("+ rain QM", apply_qm(inv(Y[ev]), qm))):
            agg = det_metrics(P, T, M)["aggregate"]
            print(f"FAST {len(a.files)}-file average {name:10s} {ey}: CSS {composite_skill(agg, ref):.4f} "
                  f"bias {agg['precip_bias_ratio']:.2f} csi30 {agg['precip_csi30']:.3f} Tmax MAE {agg['tmax_mae']:.3f}", flush=True)
