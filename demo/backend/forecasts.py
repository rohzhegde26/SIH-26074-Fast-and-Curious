"""Forecast data layer for the v3 demo: per-GP 7-day forecasts precomputed from the shipped bundle
(sihv3.precompute) for every init date of the 2023 monsoon test season, in all four operating modes, plus the observed
values and raw GFS for verification. Builds the records the Mandya PWA expects (schemas.ForecastResponse) with real
day-by-day values from the model (the original demo extrapolated days 2-7 from day 1).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from .advisory.engine import DAY_NAMES_KN, build_advisory, evaluate_lookahead_risk, rainfall_band_name

DEMO = Path(__file__).resolve().parents[1]
PRE = DEMO / "data" / "serving" / "precomputed"
MODES = ("FAST", "BALANCED", "ACCURATE", "ENSEMBLE")
MODE_INFO = {
    "FAST": "Deterministic spatiotemporal transformer (average of 3 seeds) + rain quantile mapping. ~0.2 s per forecast on a T4.",
    "BALANCED": "Transformer + cross-fitted residual diffusion, DPM-Solver++ 24 steps x 8 members, spread-calibrated, rain tail cap. ~4 s.",
    "ACCURATE": "As BALANCED with 32 denoising steps x 8 members. ~5.3 s.",
    "ENSEMBLE": "As BALANCED with 16 members (24 steps): best threshold probabilities for advisories. ~8.3 s.",
}


def crop_stages(d: date) -> dict:
    """Simple Mandya kharif calendar (demo assumption, stated in demo/README.md)."""
    m = d.month
    ragi = "sowing" if m <= 6 else "vegetative" if m <= 8 else "flowering"
    paddy = "sowing" if m <= 7 else "vegetative" if m == 8 else "flowering"
    return {"ragi": ragi, "paddy": paddy, "sugarcane": "grand_growth"}


@lru_cache(maxsize=1)
def gp_meta() -> dict:
    cells = json.load(open(DEMO / "data" / "serving" / "gp_cells.json", encoding="utf-8"))
    cent = json.load(open(DEMO / "data" / "serving" / "mandya_centroids.json", encoding="utf-8"))
    for k, v in cells.items():
        v["elevation_m"] = cent.get(k, {}).get("elevation_m", 660.0)
    return cells


@lru_cache(maxsize=None)
def load(mode: str) -> dict:
    z = np.load(PRE / f"gp_{mode}.npz")
    return {k: z[k] for k in z.files}


@lru_cache(maxsize=1)
def obs() -> dict:
    z = np.load(PRE / "gp_obs.npz")
    return {k: z[k] for k in z.files}


def available_modes() -> list[str]:
    return [m for m in MODES if (PRE / f"gp_{m}.npz").exists()]


def dates() -> list[str]:
    return [str(d) for d in load(available_modes()[0])["dates"]]


@lru_cache(maxsize=1)
def default_date() -> str:
    """The init date with the wettest observed district-mean week: shows the advisories under real rain."""
    o = obs()
    wk = np.nanmean(o["rain"], axis=1).sum(1)
    return str(o["dates"][int(np.argmax(wk))])


def _r(x, n=1):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def day_items(mode: str, di: int, gi: int, init: date) -> list[dict]:
    a, o = load(mode), obs()
    items = []
    for d in range(7):
        cur = init + timedelta(days=d)
        exp, lo, hi = float(a["rain"][di, gi, d]), float(a["rain_lo"][di, gi, d]), float(a["rain_hi"][di, gi, d])
        nxt_exp = float(a["rain"][di, gi, d + 1]) if d < 6 else 0.0
        nxt_hi = float(a["rain_hi"][di, gi, d + 1]) if d < 6 else 0.0
        look = evaluate_lookahead_risk(exp, nxt_exp, nxt_hi)
        band = rainfall_band_name(exp)
        if band == "dry" and not look["has_washoff_hazard"]:
            s_en, s_kn = ("Safe weather window. Ideal for field labour, spraying, and weeding.",
                          "ಅನುಕೂಲಕರ ಒಣ ಹವೆ. ಕಳೆ ಕೀಳಲು ಮತ್ತು ಸಿಂಪರಣೆಗೆ ಸೂಕ್ತ ದಿನ.")
        elif look["has_washoff_hazard"]:
            s_en = f"Clear today, but {nxt_exp:.1f} mm rain tomorrow! Withhold fertilizer top-dressing."
            s_kn = f"ಇಂದು ಒಣಹವೆ, ಆದರೆ ನಾಳೆ {nxt_exp:.1f} ಮಿಮೀ ಮಳೆ! ರಸಗೊಬ್ಬರ ಹಾಕಬೇಡಿ."
        elif band == "light":
            s_en, s_kn = ("Light showers expected. Field operations can safely proceed with care.",
                          "ಸಾಧಾರಣ ತುಂತುರು ಮಳೆ. ಎಚ್ಚರಿಕೆಯಿಂದ ಕೃಷಿ ಕೆಲಸ ಮುಂದುವರಿಸಿ.")
        else:
            s_en = f"Heavy rain risk ({exp:.1f} mm). Postpone fertilizer and clear drainage."
            s_kn = f"ಭಾರಿ ಮಳೆ ಸಂಭವ ({exp:.1f} ಮಿಮೀ). ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ ಹಾಗೂ ಚರಂಡಿ ಸ್ವಚ್ಛಗೊಳಿಸಿ."
        tmax, tmin, rh = float(a["tmax"][di, gi, d]), float(a["tmin"][di, gi, d]), float(a["rh"][di, gi, d])
        wind = float(a["wind"][di, gi, d])
        items.append({
            "date": cur.isoformat(), "day_offset": d,
            "day_label_en": "Today" if d == 0 else "Tomorrow" if d == 1 else cur.strftime("%a %d"),
            "day_label_kn": "ಇಂದು" if d == 0 else "ನಾಳೆ" if d == 1 else f"{DAY_NAMES_KN[cur.weekday()]} {cur.strftime('%d')}",
            "expected_mm": _r(exp), "likely_min_mm": _r(lo), "likely_max_mm": _r(hi),
            "tmax_c": _r(tmax), "tmin_c": _r(tmin), "rh_pct": _r(min(rh, 100.0)), "wind_kph": _r(wind),
            "heat_stress_level": "SEVERE" if tmax >= 38 else "MODERATE" if tmax >= 35 else "NONE",
            "disease_risk_flag": bool(rh >= 85.0 and 20.0 <= tmax <= 30.0),
            "rainfall_band": band, "spray_window": look["spray_window"], "harvest_window": look["harvest_window"],
            "irrigation_window": look["irrigation_window"],
            "lookahead_warning_en": look["warning_en"], "lookahead_warning_kn": look["warning_kn"],
            "advisory_summary_en": s_en, "advisory_summary_kn": s_kn,
            "provenance": f"SIH26074_V3_{mode}",
            "parcels": [],
            # v3 additions (ignored by older clients)
            "prob_ge_15mm": _r(a["p15"][di, gi, d], 2), "prob_ge_30mm": _r(a["p30"][di, gi, d], 2),
            "prob_ge_64_5mm": _r(a["p64"][di, gi, d], 2), "heavy_rain_pm_mm": _r(a["rain_pm"][di, gi, d]),
            "gfs_raw_mm": _r(o["gfs_rain"][di, gi, d]), "observed_mm": _r(o["rain"][di, gi, d]),
            "observed_tmax_c": _r(o["tmax"][di, gi, d]),
        })
    return items


def record(mode: str, init_date: str, lgd: str) -> dict | None:
    a = load(mode)
    ds = [str(d) for d in a["dates"]]
    codes = [str(c) for c in a["codes"]]
    if init_date not in ds or lgd not in codes:
        return None
    di, gi = ds.index(init_date), codes.index(lgd)
    init = date.fromisoformat(init_date)
    meta = gp_meta()[lgd]
    days = day_items(mode, di, gi, init)
    d0 = days[0]
    st = crop_stages(init)
    adv = {c: build_advisory(c, st[c], d0["expected_mm"], d0["likely_max_mm"], wind_kph=d0["wind_kph"], rh_pct=d0["rh_pct"],
                             tmax_c=d0["tmax_c"], tmin_c=d0["tmin_c"]) for c in ("ragi", "paddy", "sugarcane")}
    cov = coverage(mode)
    return {
        "lgd_code": lgd, "panchayat_name": meta["name"], "taluk": meta["taluk"], "district": "MANDYA",
        "forecast_date": init_date, "timestamp_utc": datetime(init.year, init.month, init.day, tzinfo=timezone.utc).isoformat(),
        "cycle_age_days": 0, "mode": mode,
        "rainfall": {"expected": d0["expected_mm"], "likely_min": d0["likely_min_mm"], "likely_max": d0["likely_max_mm"],
                     "empirical_coverage": "deterministic (no range in FAST)" if mode == "FAST" else
                     f"5-95 % ensemble range; covered {cov:.0%} of 2023 GP-days"},
        "advisory": adv, "days": days, "stages": st,
        "agromet": {"temp_c": _r((d0["tmax_c"] + d0["tmin_c"]) / 2), "rh_pct": d0["rh_pct"], "wind_kph": d0["wind_kph"],
                    "tmax_c": d0["tmax_c"], "tmin_c": d0["tmin_c"]},
        "meta": meta,
    }


@lru_cache(maxsize=None)
def coverage(mode: str) -> float:
    a, o = load(mode), obs()
    y = o["rain"]
    return float(np.mean((y >= a["rain_lo"] - 0.05) & (y <= a["rain_hi"] + 0.05)))


@lru_cache(maxsize=None)
def skill(mode: str) -> dict:
    """GP-level skill of a mode on the 2023 test season (all init dates x 234 GPs x 7 leads), vs observed and vs raw GFS."""
    a, o = load(mode), obs()
    y, f, g = o["rain"], a["rain"], o["gfs_rain"]
    wet = y > 2.5
    out = {
        "rain_mae_mm": float(np.mean(np.abs(f - y))), "gfs_rain_mae_mm": float(np.mean(np.abs(g - y))),
        "wet_day_mae_mm": float(np.mean(np.abs(f - y)[wet])), "gfs_wet_day_mae_mm": float(np.mean(np.abs(g - y)[wet])),
        "rain_bias_ratio": float(f.mean() / y.mean()), "gfs_rain_bias_ratio": float(g.mean() / y.mean()),
        "tmax_mae_c": float(np.mean(np.abs(a["tmax"] - o["tmax"]))), "gfs_tmax_mae_c": float(np.mean(np.abs(o["gfs_tmax"] - o["tmax"]))),
        "rh_mae_pct": float(np.mean(np.abs(a["rh"] - o["rh"]))),
        "n_gp_days": int(y.size),
    }
    if mode != "FAST":
        out["range_coverage_5_95"] = coverage(mode)
        for k, t in (("p15", 15.0), ("p30", 30.0)):
            out[f"brier_ge_{int(t)}mm"] = float(np.mean((a[k] - (y >= t)) ** 2))
            out[f"climatology_brier_ge_{int(t)}mm"] = float(np.mean(((y >= t).mean() - (y >= t)) ** 2))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
