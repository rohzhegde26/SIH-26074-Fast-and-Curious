"""
Masked verification metrics (physical units, land pixels only) and the composite selection score.

All error metrics are computed per lead day over (samples x land pixels); aggregate = mean over leads.
Reference forecast for skill scores = GFS bilinearly interpolated to 0.05 deg ("GFS-bilinear"), so every
score answers "how much better than simply interpolating the operational forecast?".
"""
from __future__ import annotations

import numpy as np

THRESH = (5.0, 15.0, 30.0, 64.5)  # mm/day: rain day, moderate, heavy, IMD "heavy rainfall" class
FSS_THRESH, FSS_WINDOW = 15.0, 25   # 25 fine pixels = 125 km neighbourhood

# Composite Skill Score weights (sum = 1). Precipitation is the primary advisory variable (0.50),
# thermodynamics drive crop/heat advisories (0.35), wind 0.15.
CSS_WEIGHTS = {
    "precip_wet_mae": 0.10, "precip_rmse": 0.05, "precip_csi5": 0.05, "precip_csi15": 0.08,
    "precip_csi30": 0.08, "precip_csi64.5": 0.04, "precip_fss15": 0.05, "precip_logbias": 0.05,
    "tmax_mae": 0.12, "tmin_mae": 0.12, "rh_mae": 0.11, "wind_vec_rmse": 0.15,
}
LOWER_BETTER = {"precip_wet_mae", "precip_rmse", "precip_logbias", "tmax_mae", "tmin_mae", "rh_mae",
                "wind_vec_rmse", "precip_crps", "tmax_crps", "tmin_crps", "rh_crps"}


def _box(x, w):
    """Mean filter with window w over the last two axes (zero padded), via cumulative sums."""
    p = w // 2
    xp = np.pad(x, [(0, 0)] * (x.ndim - 2) + [(p + 1, p), (p + 1, p)])
    c = xp.cumsum(-1).cumsum(-2)
    return (c[..., w:, w:] - c[..., :-w, w:] - c[..., w:, :-w] + c[..., :-w, :-w]) / (w * w)


def det_metrics(pred, targ, mask):
    """pred/targ: [n,7,6,80,80] physical; mask: [n,7,6,80,80] (1 = observed). Returns per-lead + aggregate."""
    per = []
    for l in range(pred.shape[1]):
        m = mask[:, l].astype(bool)
        P, T = pred[:, l], targ[:, l]
        pm, tm = P[:, 0][m[:, 0]], T[:, 0][m[:, 0]]
        r = {}
        wet = tm > 1.0
        r["precip_mae"] = float(np.abs(pm - tm).mean())
        r["precip_wet_mae"] = float(np.abs(pm[wet] - tm[wet]).mean()) if wet.any() else np.nan
        r["precip_rmse"] = float(np.sqrt(((pm - tm) ** 2).mean()))
        r["precip_bias_ratio"] = float(pm.sum() / max(tm.sum(), 1e-6))
        r["precip_logbias"] = float(abs(np.log(max(r["precip_bias_ratio"], 1e-3))))
        for th in THRESH:
            hit = ((pm >= th) & (tm >= th)).sum()
            fa = ((pm >= th) & (tm < th)).sum()
            mi = ((pm < th) & (tm >= th)).sum()
            r[f"precip_csi{th:g}"] = float(hit / max(hit + fa + mi, 1))
            r[f"precip_pod{th:g}"] = float(hit / max(hit + mi, 1))
            r[f"precip_far{th:g}"] = float(fa / max(hit + fa, 1))
        land = m[:, 0]
        fp = _box(((P[:, 0] >= FSS_THRESH) & land).astype(np.float32), FSS_WINDOW)
        ft = _box(((T[:, 0] >= FSS_THRESH) & land).astype(np.float32), FSS_WINDOW)
        num, den = ((fp - ft) ** 2).sum(), (fp ** 2 + ft ** 2).sum()
        r["precip_fss15"] = float(1 - num / den) if den > 0 else np.nan
        ok = land.reshape(land.shape[0], -1)
        pf, tf = P[:, 0].reshape(len(P), -1), T[:, 0].reshape(len(T), -1)
        cors = [np.corrcoef(pf[i][ok[i]], tf[i][ok[i]])[0, 1] for i in range(len(P))
                if tf[i][ok[i]].std() > 0 and pf[i][ok[i]].std() > 0]
        r["precip_spatial_corr"] = float(np.mean(cors)) if cors else np.nan
        for c, name in ((1, "tmax"), (2, "tmin"), (3, "rh")):
            e = P[:, c][m[:, c]] - T[:, c][m[:, c]]
            r[f"{name}_mae"], r[f"{name}_rmse"], r[f"{name}_bias"] = float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean())), float(e.mean())
        du = (P[:, 4] - T[:, 4])[m[:, 4]]
        dv = (P[:, 5] - T[:, 5])[m[:, 5]]
        r["wind_vec_rmse"] = float(np.sqrt((du ** 2 + dv ** 2).mean()))
        r["tmax_lt_tmin_rate"] = float((P[:, 1] < P[:, 2] - 0.05)[m[:, 1]].mean())
        per.append(r)
    agg = {k: float(np.nanmean([p[k] for p in per])) for k in per[0]}
    return {"per_lead": per, "aggregate": agg}


def fair_crps(ens, obs):
    """ens [K, ...], obs [...] -> fair CRPS per element (Ferro 2008), via sorted-ensemble identity."""
    k = ens.shape[0]
    e = np.sort(ens, axis=0)
    t1 = np.abs(e - obs[None]).mean(0)
    w = (2 * np.arange(1, k + 1) - k - 1).reshape((k,) + (1,) * obs.ndim)
    t2 = (w * e).sum(0) * 2 / (k * (k - 1)) if k > 1 else 0.0
    return t1 - 0.5 * t2


def prob_metrics(ens, targ, mask):
    """ens [n,K,7,6,80,80] physical. Ensemble-mean det metrics + CRPS, spread/skill, coverage, Brier."""
    k = ens.shape[1]
    mean = ens.mean(1)
    out = det_metrics(mean, targ, mask)
    per = out["per_lead"]
    for l in range(ens.shape[2]):
        m = mask[:, l].astype(bool)
        for c, name in ((0, "precip"), (1, "tmax"), (2, "tmin"), (3, "rh")):
            E = ens[:, :, l, c].transpose(1, 0, 2, 3)[:, m[:, c]]  # [K, pts]
            o = targ[:, l, c][m[:, c]]
            if k > 1:
                per[l][f"{name}_crps"] = float(fair_crps(E, o).mean())
                spread = np.sqrt(E.var(0, ddof=1).mean() * (k + 1) / k)
                rmse = np.sqrt(((E.mean(0) - o) ** 2).mean())
                per[l][f"{name}_ssr"] = float(spread / max(rmse, 1e-9))
                lo, hi = np.quantile(E, [0.05, 0.95], axis=0)
                per[l][f"{name}_cov90"] = float(((o >= lo) & (o <= hi)).mean())
            else:
                per[l][f"{name}_crps"] = float(np.abs(E[0] - o).mean())
        E = ens[:, :, l, 0].transpose(1, 0, 2, 3)[:, m[:, 0]]
        o = targ[:, l, 0][m[:, 0]]
        for th in (15.0, 30.0):
            pr = (E >= th).mean(0)
            per[l][f"brier{th:g}"] = float(((pr - (o >= th)) ** 2).mean())
    out["aggregate"] = {kk: float(np.nanmean([p[kk] for p in per if kk in p])) for kk in per[0]}
    return out


def skill(model: dict, ref: dict) -> dict:
    """Per-component skill vs reference aggregate: 1 - e/e_ref (errors) or (s - s_ref)/(1 - s_ref) (scores)."""
    out = {}
    for k in CSS_WEIGHTS:
        a, b = model.get(k, np.nan), ref.get(k, np.nan)
        if k == "precip_logbias":
            # bounded: exp(-|log bias|) in (0,1], skill = difference (a ratio explodes when GFS is nearly unbiased)
            out[k] = float(np.exp(-a) - np.exp(-b)) if np.isfinite(a) and np.isfinite(b) else np.nan
            continue
        if k in LOWER_BETTER:
            out[k] = 1 - a / b if b and np.isfinite(b) and b > 0 else np.nan
        else:
            out[k] = (a - b) / (1 - b) if np.isfinite(b) and b < 1 else np.nan
    return out


def composite_skill(model: dict, ref: dict) -> float:
    """CSS in (-inf, 1]; 0 = as good as GFS-bilinear, 1 = perfect. Weighted mean of component skills."""
    s = skill(model, ref)
    w = {k: v for k, v in CSS_WEIGHTS.items() if np.isfinite(s[k])}
    return float(sum(CSS_WEIGHTS[k] * s[k] for k in w) / sum(w.values()))
