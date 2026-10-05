"""
Multivariate physical consistency (Sprint 10 "multivariate consistency" requirement).

For a set of fields X [n,(K,)7,6,80,80] in physical units (each ensemble member is scored on its own, because a
generative model must produce physically coherent *individual* scenarios, not only a coherent mean), compute the
same cross-variable statistics on the observations and report model vs observed:

  rain-RH    : Pearson corr(P, RH) over land pixels; mean RH on wet (P > 5 mm) minus dry (P < 0.5 mm) pixels
  rain-Tmax  : corr(P, Tmax); mean Tmax wet minus dry (rain cools the day)
  wind-rain  : corr(|wind|, P)
  Tmax-Tmin  : corr(Tmax, Tmin); mean diurnal range Tmax - Tmin; rate of Tmax < Tmin
Gap = |model - observed| for each statistic; small gaps = the model reproduces observed inter-variable structure.
"""
from __future__ import annotations

import numpy as np

WET, DRY = 5.0, 0.5


def _stats(F, M):
    """F [n,7,6,80,80] physical, M land mask same shape -> dict of cross-variable statistics."""
    m = M[:, :, 0].astype(bool) & M[:, :, 1].astype(bool) & M[:, :, 3].astype(bool)
    P, TX, TN, RH = F[:, :, 0][m], F[:, :, 1][m], F[:, :, 2][m], F[:, :, 3][m]
    W = np.hypot(F[:, :, 4], F[:, :, 5])[m]
    wet, dry = P > WET, P < DRY
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    return {
        "corr_rain_rh": corr(P, RH),
        "rh_wet_minus_dry": float(RH[wet].mean() - RH[dry].mean()) if wet.any() and dry.any() else np.nan,
        "corr_rain_tmax": corr(P, TX),
        "tmax_wet_minus_dry": float(TX[wet].mean() - TX[dry].mean()) if wet.any() and dry.any() else np.nan,
        "corr_wind_rain": corr(W, P),
        "corr_tmax_tmin": corr(TX, TN),
        "diurnal_range": float((TX - TN).mean()),
        "tmax_lt_tmin_rate": float((TX < TN - 0.05).mean()),
    }


def consistency(pred, targ, mask):
    """pred [n,7,6,80,80] or [n,K,7,6,80,80]; returns model stats (mean over members), observed stats, gaps."""
    obs = _stats(targ, mask)
    members = [pred] if pred.ndim == 5 else [pred[:, k] for k in range(pred.shape[1])]
    per = [_stats(p, mask) for p in members]
    mod = {k: float(np.nanmean([s[k] for s in per])) for k in obs}
    gap = {k: abs(mod[k] - obs[k]) for k in obs}
    return {"model": mod, "observed": obs, "gap": gap}


if __name__ == "__main__":  # self-test: perfect prediction -> zero gaps; destroying cross-variable structure -> gaps
    rng = np.random.default_rng(0)
    n = 6
    P = rng.gamma(0.6, 10, (n, 7, 80, 80))
    RH = 70 + 2.0 * np.log1p(P) + rng.normal(0, 3, P.shape)
    TX = 32 - 0.15 * P + rng.normal(0, 1, P.shape)
    TN = TX - 8 + rng.normal(0, 1, P.shape)
    U, V = 2 + 0.05 * P, rng.normal(0, 1, P.shape)
    T = np.stack([P, TX, TN, RH, U, V], 2)
    M = np.ones_like(T)
    r = consistency(T, T, M)
    assert max(r["gap"].values()) < 1e-9, r["gap"]
    shuffled = T.copy()
    shuffled[:, :, 3] = rng.permutation(T[:, :, 3].ravel()).reshape(T[:, :, 3].shape)  # break rain-RH link
    r2 = consistency(shuffled, T, M)
    print("perfect -> max gap", max(r["gap"].values()))
    print("RH shuffled -> rain-RH corr gap", round(r2["gap"]["corr_rain_rh"], 3), "| wet-dry RH gap", round(r2["gap"]["rh_wet_minus_dry"], 2),
          "| unrelated gaps (Tmax-Tmin corr)", round(r2["gap"]["corr_tmax_tmin"], 4))
    print("observed stats:", {k: round(v, 3) for k, v in r["observed"].items()})
