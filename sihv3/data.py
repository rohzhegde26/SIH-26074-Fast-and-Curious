"""
In-memory loader for the v3 dataset (real GFS forecasts, ERA5 history, CHIRPS + ERA5-Land targets).

Store layout (datasets/multitask_temporal_v3_realgfs.zarr, built by data_pipeline/build_dataset_v3.py):
  future_forecast [N,7,6,40,40]  history [N,14,6,40,40]  target [N,7,6,80,80] (NaN over sea)
  target_mask [80,80]  terrain [5,80,80]  dates [N]  splits [N]
Coarse grids are 40x40 cells (0.25 deg) whose central 16x16 is the target domain, so a context of
N cells is the central N x N crop (N/M = N/16).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import torch
import yaml

CHANNELS = ["precipitation", "tmax", "tmin", "rh", "wind_u", "wind_v"]


def find(name: str) -> Path:
    """Locate a dataset file locally or under /kaggle/input (any depth)."""
    roots = [os.environ.get("SIH_DATA", ""), "/kaggle/input", str(Path(__file__).resolve().parents[1] / "data")]
    for r in roots:
        if not r:
            continue
        hits = glob.glob(os.path.join(r, "**", name), recursive=True)
        if hits:
            return Path(sorted(hits, key=len)[0])
    raise FileNotFoundError(name)


def find_store() -> Path:
    """Locate the v3 zarr store by content (a zarr.json next to future_forecast/), whatever the mount layout."""
    roots = [os.environ.get("SIH_DATA", ""), "/kaggle/input", "/kaggle/tmp/data", str(Path(__file__).resolve().parents[1] / "data")]
    for r in roots:
        if not r:
            continue
        for zj in sorted(glob.glob(os.path.join(r, "**", "zarr.json"), recursive=True), key=len):
            d = Path(zj).parent
            if (d / "future_forecast").is_dir() and (d / "target").is_dir():
                return d
    raise FileNotFoundError("v3 zarr store (zarr.json + future_forecast/) not found")


class Normalizer:
    def __init__(self, stats_path: Path):
        ch = yaml.safe_load(open(stats_path))["channels"]
        self.mean = np.array([ch[c]["mean"] for c in CHANNELS], np.float32)
        self.std = np.array([ch[c]["std"] for c in CHANNELS], np.float32)
        assert ch["precipitation"].get("transform") == "log1p_zscore"

    def fwd(self, x: np.ndarray, axis: int) -> np.ndarray:
        """x[..., 6 at `axis`, ...] physical -> normalised (precip log1p z-score)."""
        x = np.array(x, dtype=np.float32, copy=True)
        sl = [slice(None)] * x.ndim
        sl[axis] = 0
        x[tuple(sl)] = np.log1p(np.maximum(x[tuple(sl)], 0))
        shape = [1] * x.ndim
        shape[axis] = 6
        return (x - self.mean.reshape(shape)) / self.std.reshape(shape)

    def inv(self, x, axis: int):
        """normalised -> physical; works on numpy or torch."""
        shape = [1] * x.ndim
        shape[axis] = 6
        if isinstance(x, torch.Tensor):
            m, s = torch.tensor(self.mean, device=x.device).view(shape), torch.tensor(self.std, device=x.device).view(shape)
            y = x * s + m
            p = y.narrow(axis, 0, 1).clamp(max=7.0)  # log1p(1096 mm) guard
            return torch.cat([torch.expm1(p).clamp_min(0), y.narrow(axis, 1, 5)], dim=axis)
        y = x * self.std.reshape(shape) + self.mean.reshape(shape)
        sl = [slice(None)] * y.ndim
        sl[axis] = 0
        y[tuple(sl)] = np.clip(np.expm1(np.minimum(y[tuple(sl)], 7.0)), 0, None)
        return y


class V3Data:
    """Holds the whole (normalised) dataset in memory; `batches()` yields cropped tensors."""

    def __init__(self, history_len: int = 14, context: int = 16, val_year: int | None = None,
                 train_years: list | None = None, val_years: list | None = None, fc_history: bool = False):
        import zarr

        assert 1 <= history_len <= 14 and 16 <= context <= 40 and context % 2 == 0
        self.H, self.N = history_len, context
        zp = find_store()
        g = zarr.open_group(str(zp), mode="r")
        self.norm = Normalizer(find("normalization_stats_v3.yaml"))
        lo = (40 - context) // 2
        sl = slice(lo, lo + context)
        splits_all = np.asarray(g["splits"][:]).astype(str)
        rows = np.arange(len(splits_all))
        if os.environ.get("SIH_SMOKE"):  # tiny subset for local tests
            rows = np.concatenate([np.where(splits_all == s)[0][:n] for s, n in (("train", 16), ("val", 8), ("test", 8))])
        self.fcst = self.norm.fwd(np.asarray(g["future_forecast"].get_orthogonal_selection((rows, slice(None), slice(None), sl, sl))), axis=2)
        self.hist = self.norm.fwd(np.asarray(g["history"].get_orthogonal_selection((rows, slice(14 - history_len, 14), slice(None), sl, sl))), axis=2)
        targ_phys = np.asarray(g["target"].get_orthogonal_selection((rows,)))
        self.mask = np.isfinite(targ_phys)                                # [N,7,6,80,80] bool
        self.targ = np.nan_to_num(self.norm.fwd(targ_phys, axis=2), nan=0.0)
        del targ_phys
        self.land = np.asarray(g["target_mask"][:]).astype(np.float32)  # [80,80]
        terr = np.asarray(g["terrain"][:]).astype(np.float32)
        terr = (terr - terr.mean((1, 2), keepdims=True)) / (terr.std((1, 2), keepdims=True) + 1e-6)
        self.static = np.concatenate([terr, self.land[None]], 0)       # [6,80,80]
        self.dates = np.asarray(g["dates"][:]).astype(str)[rows]
        self.splits = splits_all[rows]
        self.fc_history = fc_history
        if fc_history:
            # Forecast history: for each history day t (D-H .. D-1), the GFS forecast valid on t from the 00Z run issued
            # on t itself (lead 0 = day t; that run exists before the new forecast at 00Z D). Paired with the observed
            # history it shows how wrong GFS has been over the last H days. Days whose run is not in the store (the
            # first days of each June) fall back to the observation, i.e. zero apparent error.
            from datetime import date, timedelta
            all_dates = np.asarray(g["dates"][:]).astype(str)
            pos = {d: i for i, d in enumerate(all_dates)}
            lead0 = self.norm.fwd(np.asarray(g["future_forecast"].get_orthogonal_selection(
                (slice(None), slice(0, 1), slice(None), sl, sl)))[:, 0], axis=1)  # [all,6,N,N]
            self.hist_fc = self.hist.copy()
            hit = 0
            for r, d in enumerate(self.dates):
                d0 = date.fromisoformat(str(d))
                for j in range(history_len):  # hist[:, j] is day D-(H-j)
                    t = (d0 - timedelta(days=history_len - j)).isoformat()
                    if t in pos:
                        assert t < str(d)
                        self.hist_fc[r, j] = lead0[pos[t]]
                        hit += 1
            self.fc_history_coverage = hit / max(1, len(self.dates) * history_len)
            del lead0
        if val_year is not None:  # alternative validation season: that year -> val, 2022 -> train, 2023 stays test
            years = np.array([int(d[:4]) for d in self.dates])
            assert val_year not in (2023,) and (years == val_year).any(), val_year
            self.splits = np.where(years == val_year, "val", np.where(self.splits == "test", "test", "train"))
        if train_years is not None:  # explicit year-based split; 2023 is always the test season
            years = np.array([int(d[:4]) for d in self.dates])
            assert 2023 not in set(train_years) | set(val_years or []), "2023 is the held-out test season"
            assert not set(train_years) & set(val_years or []), "train/val years overlap"
            self.splits = np.where(np.isin(years, train_years), "train",
                                   np.where(np.isin(years, val_years or []), "val",
                                            np.where(years == 2023, "test", "unused")))
        self.idx = {s: np.where(self.splits == s)[0] for s in ("train", "val", "test", "unused")}

    def batches(self, split: str, batch_size: int, shuffle: bool, device, rng=None):
        ids = self.idx[split].copy()
        if shuffle:
            (rng or np.random).shuffle(ids)
        st = torch.from_numpy(self.static).to(device)
        for i in range(0, len(ids), batch_size):
            b = np.sort(ids[i:i + batch_size])
            h = np.concatenate([self.hist[b], self.hist_fc[b]], 2) if self.fc_history else self.hist[b]  # [B,H,6|12,N,N]
            yield {
                "history": torch.from_numpy(h).to(device, non_blocking=True),
                "forecast": torch.from_numpy(self.fcst[b]).to(device, non_blocking=True),
                "static": st[None].expand(len(b), -1, -1, -1),
                "target": torch.from_numpy(self.targ[b]).to(device, non_blocking=True),
                "mask": torch.from_numpy(self.mask[b]).to(device, non_blocking=True).float(),
                "index": b,
            }
