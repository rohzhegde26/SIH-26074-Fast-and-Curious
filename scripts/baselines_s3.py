"""
Sprint 3 non-learned baselines on the v3 dataset (CPU, chunked, < 2 GB RAM):
  1. GFS-bilinear          : real GFS interpolated to 0.05 deg (the CSS reference)
  2. GFS-QM                : + per-lead, per-pixel empirical quantile mapping fitted on 2015-2021
                             (precipitation and temperatures/RH; wind gets mean-bias correction)
Writes results/baselines_s3.json with det_metrics + CSS for val and test.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sihv3.data import find_store  # noqa: E402
from sihv3.metrics import composite_skill, det_metrics  # noqa: E402

Q = np.linspace(0.01, 0.99, 50)


def upsample(fc):  # [n,7,6,40,40] -> centre 16 -> [n,7,6,80,80]
    c = torch.from_numpy(fc[:, :, :, 12:28, 12:28].astype(np.float32))
    return F.interpolate(c.flatten(0, 1), size=(80, 80), mode="bilinear", align_corners=False).view(-1, 7, 6, 80, 80).numpy()


def main():
    g = zarr.open_group(str(find_store()), mode="r")
    splits = np.asarray(g["splits"][:]).astype(str)
    idx = {s: np.where(splits == s)[0] for s in ("train", "val", "test")}
    # quantiles per (lead, var, coarse-cell 16x16) for forecast and target-at-coarse-scale (land mean)
    tr = idx["train"]
    fq = np.zeros((7, 6, len(Q), 80, 80), np.float32)
    tq = np.zeros_like(fq)
    for l in range(7):
        f = upsample(np.asarray(g["future_forecast"].get_orthogonal_selection((tr, slice(l, l + 1)))).repeat(7, 1)[:, :1])[:, 0] \
            if False else None
        fl = np.asarray(g["future_forecast"].get_orthogonal_selection((tr, slice(l, l + 1))))  # [n,1,6,40,40]
        up = F.interpolate(torch.from_numpy(fl[:, 0, :, 12:28, 12:28]), size=(80, 80), mode="bilinear",
                           align_corners=False).numpy()  # [n,6,80,80]
        tl = np.asarray(g["target"].get_orthogonal_selection((tr, slice(l, l + 1))))[:, 0]  # [n,6,80,80]
        fq[l] = np.quantile(up, Q, axis=0).transpose(1, 0, 2, 3)
        tq[l] = np.nanquantile(tl, Q, axis=0).transpose(1, 0, 2, 3)
        print("fitted lead", l, flush=True)
    out = {}
    for split in ("val", "test"):
        ids = idx[split]
        fc = np.asarray(g["future_forecast"].get_orthogonal_selection((ids,)))
        T = np.asarray(g["target"].get_orthogonal_selection((ids,)))
        M = np.isfinite(T).astype(np.float32)
        T = np.nan_to_num(T)
        R = upsample(fc)
        QM = R.copy()
        for l in range(7):
            for c in range(6):
                fqq, tqq = fq[l, c], np.nan_to_num(tq[l, c])
                x = R[:, l, c]
                if c >= 4:  # wind: mean bias correction
                    QM[:, l, c] = x + (tqq.mean(0) - fqq.mean(0))[None]
                    continue
                # per-pixel interpolation of x through (fq -> tq); linear extrapolation clipped at ends
                rank = (x[:, None] > fqq[None]).sum(1)  # [n,80,80] in 0..len(Q)
                lo = np.clip(rank - 1, 0, len(Q) - 2)
                ii = lo[:, None]
                f0 = np.take_along_axis(fqq[None].repeat(len(x), 0), ii, 1)[:, 0]
                f1 = np.take_along_axis(fqq[None].repeat(len(x), 0), ii + 1, 1)[:, 0]
                t0 = np.take_along_axis(tqq[None].repeat(len(x), 0), ii, 1)[:, 0]
                t1 = np.take_along_axis(tqq[None].repeat(len(x), 0), ii + 1, 1)[:, 0]
                w = np.clip((x - f0) / np.maximum(f1 - f0, 1e-6), 0, 1)
                QM[:, l, c] = t0 + w * (t1 - t0)
        QM[:, :, 0] = np.maximum(QM[:, :, 0], 0)
        QM[:, :, 1] = np.maximum(QM[:, :, 1], QM[:, :, 2])
        ref = det_metrics(R, T, M)
        qm = det_metrics(QM, T, M)
        out[split] = {"gfs_bilinear": ref, "gfs_qm": qm, "gfs_qm_css": composite_skill(qm["aggregate"], ref["aggregate"])}
        a, b = ref["aggregate"], qm["aggregate"]
        print(f"{split}: CSS(GFS-QM)={out[split]['gfs_qm_css']:.4f}")
        for k in ("precip_mae", "precip_wet_mae", "precip_bias_ratio", "precip_csi5", "precip_csi15", "precip_csi30",
                  "precip_fss15", "tmax_mae", "tmin_mae", "rh_mae", "wind_vec_rmse"):
            print(f"   {k:18s} GFS-bilinear {a[k]:7.3f}   GFS-QM {b[k]:7.3f}")
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/baselines_s3.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
