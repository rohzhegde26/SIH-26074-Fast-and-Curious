"""Real gridded fields over Mandya for the demo's proof plots, for every 2023 init date and lead day:
raw GFS 0.25 deg (shown as its 0.25 deg blocks on the 0.05 deg grid), the v3 FAST forecast (0.05 deg) and the observed
rain (CHIRPS, 0.05 deg). Runs the shipped bundle on CPU (FAST mode, ~1 s per forecast).

  python -m demo.backend.precompute_fields      (needs the v3 dataset: SIH_DATA)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr

from sihv3.data import Normalizer, find, find_store
from sihv3.predict import FinalDownscaler

DEMO = Path(__file__).resolve().parents[1]
REPO = DEMO.parent
R0, R1, C0, C1 = 36, 58, 44, 70          # fine-grid window around Mandya (rows N->S, cols W->E); GP cells span 38-55 x 46-66
FLAT = np.round(np.arange(14.975, 11.0, -0.05), 3)
FLON = np.round(np.arange(74.025, 78.0, 0.05), 3)
CLAT = np.round(np.arange(17.875, 8.0, -0.25), 3)
CLON = np.round(np.arange(71.125, 81.0, 0.25), 3)


def main():
    m = FinalDownscaler(REPO / "models" / "final", device="cpu")
    H, N = m.args.H, m.args.N
    g = zarr.open_group(str(find_store()), mode="r")
    dates = np.asarray(g["dates"][:]).astype(str)
    rows = np.where(np.char.startswith(dates, "2023"))[0]
    nm = Normalizer(find("normalization_stats_v3.yaml"))
    lo = (40 - N) // 2
    sl = slice(lo, lo + N)
    # which coarse (0.25 deg) cell contains each fine cell of the window (context is the full 40 x 40 when N=40)
    ri = np.array([int(np.argmin(np.abs(CLAT - la))) for la in FLAT[R0:R1]]) - lo
    ci = np.array([int(np.argmin(np.abs(CLON - lo_))) for lo_ in FLON[C0:C1]]) - lo
    v3, gfs, obs = [], [], []
    for k in range(0, len(rows), 8):
        idx = rows[k:k + 8]
        h = nm.fwd(np.asarray(g["history"].get_orthogonal_selection((idx, slice(14 - H, 14), slice(None), sl, sl))), axis=2)
        f_raw = np.asarray(g["future_forecast"].get_orthogonal_selection((idx, slice(None), slice(None), sl, sl)))
        out = m.predict(torch.from_numpy(h), torch.from_numpy(nm.fwd(f_raw, axis=2)), mode="FAST")
        v3.append(out["mean"][:, :, 0, R0:R1, C0:C1])
        gfs.append(np.maximum(f_raw[:, :, 0][:, :, ri][:, :, :, ci], 0))
        t = np.asarray(g["target"].get_orthogonal_selection((idx, slice(None), slice(0, 1))))[:, :, 0, R0:R1, C0:C1]
        obs.append(t)
        print(f"{k + len(idx)}/{len(rows)}", flush=True)
    path = DEMO / "data" / "serving" / "precomputed" / "fields_FAST.npz"
    np.savez_compressed(path, dates=dates[rows], v3=np.concatenate(v3).astype(np.float16),
                        gfs=np.concatenate(gfs).astype(np.float16), obs=np.concatenate(obs).astype(np.float16),
                        window=np.array([R0, R1, C0, C1]), lat=FLAT[R0:R1], lon=FLON[C0:C1])
    print("written", path)


if __name__ == "__main__":
    main()
