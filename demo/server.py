"""
SIH-26074 v3 web demo: the Mandya panchayat advisory PWA (frontend/, from the original project) served on top of the
v3 model (models/final: spatiotemporal transformer + cross-fitted residual diffusion, four operating modes).

  pip install fastapi uvicorn
  python -m uvicorn demo.server:app --port 8000        (from the repo root)   ->  http://localhost:8000

Forecasts for every init date of the 2023 monsoon test season (held out from all training and selection) are
precomputed per gram panchayat for all four modes by `sihv3.precompute` on a GPU (demo/data/serving/precomputed/).
/api/v1/infer runs the FAST mode live from the real GFS + ERA5 inputs when the v3 dataset is available locally.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from demo.backend import feedback_store
from demo.backend import forecasts as fc
from demo.backend.schemas import (AdvisorySet, AgrometVariables, CropAdvisory, DailyForecastItem, FinancialRiskSchema,
                                  ForecastResponse, NandiniStatsResponse, NandiniValidationRequest,
                                  NandiniValidationResponse, Rainfall, VirtualARGResponse, VirtualARGUncertainty)

DEMO = Path(__file__).resolve().parent
REPO = DEMO.parent
FRONTEND = DEMO / "frontend"
FEEDBACK_DB = DEMO / "data" / "serving" / "nandini_feedback.db"

app = FastAPI(title="SIH-26074 v3 Mandya Downscaling Demo", version="3.0",
              description="0.25 deg GFS -> 0.05 deg, 7 days x 6 variables, four operating modes, served per gram panchayat.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])


@app.middleware("http")
async def no_cache(request, call_next):
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.endswith((".html", ".js", ".css", ".webmanifest")):
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.on_event("startup")
def _startup():
    feedback_store.init_db(FEEDBACK_DB, DEMO / "data" / "serving" / "no_seed.json")  # starts empty: no seeded validations


def _mode(mode: str) -> str:
    m = (mode or "BALANCED").upper()
    if m not in fc.available_modes():
        raise HTTPException(404, f"mode {m} not available; have {fc.available_modes()}")
    return m


def _date(d: str | None) -> str:
    d = d or fc.default_date()
    if d not in fc.dates():
        raise HTTPException(404, f"init date {d} not in the precomputed season ({fc.dates()[0]} .. {fc.dates()[-1]})")
    return d


def _adv(a) -> CropAdvisory:
    fin = FinancialRiskSchema(**a.financial_risk.__dict__) if a.financial_risk else None
    return CropAdvisory(stage=a.stage, action_en=a.action_en, action_kn=a.action_kn, financial_risk=fin)


def _response(rec: dict) -> ForecastResponse:
    d0, ag = rec["days"][0], rec["agromet"]
    return ForecastResponse(
        lgd_code=rec["lgd_code"], panchayat_name=rec["panchayat_name"], taluk=rec["taluk"], district=rec["district"],
        forecast_date=rec["forecast_date"], timestamp_utc=rec["timestamp_utc"], cycle_age_days=0,
        rainfall_mm=Rainfall(expected=d0["expected_mm"], likely_min=d0["likely_min_mm"], likely_max=d0["likely_max_mm"],
                             empirical_coverage=rec["rainfall"]["empirical_coverage"]),
        advisory=AdvisorySet(**{c: _adv(a) for c, a in rec["advisory"].items()}),
        agromet_context=AgrometVariables(
            temp_c=ag["temp_c"], rh_pct=ag["rh_pct"], wind_kph=ag["wind_kph"], tmax_c=ag["tmax_c"], tmin_c=ag["tmin_c"],
            spray_drift_risk="HIGH" if ag["wind_kph"] >= 15 else "LOW",
            fungal_disease_risk="HIGH" if (ag["rh_pct"] >= 85 and 20 <= ag["tmax_c"] <= 30) else "LOW",
            heat_stress_level=d0["heat_stress_level"], source=f"v3 {rec['mode']} downscaling (all 6 variables)",
            provenance=f"SIH26074_V3_{rec['mode']}"),
        multi_day_forecast=[DailyForecastItem(**d) for d in rec["days"]],
        mode=rec["mode"], crop_stages=rec["stages"])


@app.get("/api/v1/dates", tags=["demo"])
def dates():
    return {"dates": fc.dates(), "default": fc.default_date(), "modes": fc.available_modes(), "mode_info": fc.MODE_INFO,
            "season": "2023 monsoon (Jun-Sep) - held-out test season, never used for training or model selection"}


@app.get("/api/forecasts", response_model=list[ForecastResponse], tags=["forecast"])
def forecasts(mode: str = Query("BALANCED"), date: str | None = Query(None)):
    m, d = _mode(mode), _date(date)
    return [_response(fc.record(m, d, code)) for code in sorted(fc.gp_meta())]


@app.get("/api/forecast/{lgd_code}", response_model=ForecastResponse, tags=["forecast"])
def forecast(lgd_code: str, mode: str = Query("BALANCED"), date: str | None = Query(None)):
    r = fc.record(_mode(mode), _date(date), lgd_code)
    if r is None:
        raise HTTPException(404, "Panchayat not found")
    return _response(r)


@app.get("/api/v1/model-card", tags=["demo"])
def model_card(mode: str = Query("BALANCED")):
    m = _mode(mode)
    prov = json.load(open(REPO / "models" / "final" / "provenance.json")) if (REPO / "models" / "final" / "provenance.json").exists() else {}
    return {"mode": m, "description": fc.MODE_INFO[m], "skill_2023_gp_level": fc.skill(m),
            "skill_2023_gp_level_all_modes": {k: fc.skill(k) for k in fc.available_modes()},
            "bundle": prov, "grid": "0.25 deg GFS (40x40 context) -> 0.05 deg (80x80), 11-15 N x 74-78 E; 234 Mandya GPs area-weighted",
            "report": "reports/final_system_report.md, docs/results_s11.md"}


@app.get("/api/v1/virtual-arg/{lgd_code}", response_model=VirtualARGResponse, tags=["panchayat-feed"])
def virtual_arg(lgd_code: str, mode: str = Query("BALANCED"), date: str | None = Query(None)):
    r = fc.record(_mode(mode), _date(date), lgd_code)
    if r is None:
        raise HTTPException(404, "Panchayat not found")
    d0, meta = r["days"][0], r["meta"]
    return VirtualARGResponse(
        station_id=f"VARG_KA_MAN_{lgd_code}", station_name=f"{r['panchayat_name']} Virtual ARG (v3 forecast)",
        lgd_code=lgd_code, latitude=meta["lat"], longitude=meta["lon"], elevation_m=meta["elevation_m"],
        observation_datetime_utc=f"{r['forecast_date']}T00:00:00Z", observation_datetime_ist=f"{r['forecast_date']} 05:30:00 IST",
        rainfall_24h_mm=d0["expected_mm"],
        uncertainty_range_90pct=VirtualARGUncertainty(lower_bound_mm=d0["likely_min_mm"], upper_bound_mm=d0["likely_max_mm"],
                                                      confidence=r["rainfall"]["empirical_coverage"]),
        qc_status="FORECAST_NOT_OBSERVATION", data_type="V3_DOWNSCALED_FORECAST", provenance=f"SIH26074_V3_{r['mode']}")


@app.post("/api/v1/validation/nandini", response_model=NandiniValidationResponse, tags=["validation"])
async def nandini(req: NandiniValidationRequest, mode: str = Query("BALANCED"), date: str | None = Query(None)):
    r = fc.record(_mode(mode), _date(date), req.lgd_code)
    predicted = bool(r and (r["days"][0]["expected_mm"] >= 2.5 or r["days"][0]["likely_max_mm"] >= 5.0))
    disc = predicted != req.rained_bool
    vid, at = str(uuid.uuid4())[:8], datetime.now(timezone.utc).isoformat()
    entry = {"validation_id": vid, "lgd_code": req.lgd_code, "panchayat_name": req.panchayat_name, "rained_bool": req.rained_bool,
             "observer_role": req.observer_role, "milk_center_id": req.milk_center_id or f"KMF_KA_MAN_{req.lgd_code[:4]}",
             "observation_period": req.observation_period, "recorded_at_utc": at, "recalibration_flagged": disc,
             "source": "field_submission"}
    await run_in_threadpool(feedback_store.insert_feedback, entry, FEEDBACK_DB)
    return NandiniValidationResponse(status="success", validation_id=vid, lgd_code=req.lgd_code, recorded_at_utc=at,
                                     recalibration_flagged=disc,
                                     message="Validation recorded: field report agrees with the forecast." if not disc else
                                     "Validation recorded: field report disagrees with the forecast (logged for review).")


@app.get("/api/v1/validation/stats", response_model=NandiniStatsResponse, tags=["validation"])
async def nandini_stats():
    return NandiniStatsResponse(**await run_in_threadpool(feedback_store.get_feedback_stats, FEEDBACK_DB))


@app.get("/api/v1/fields", tags=["demo"])
def fields(date: str | None = Query(None), day: int = Query(0, ge=0, le=6), mode: str = Query("BALANCED")):
    """Real gridded rain over Mandya for the proof plots: raw GFS (0.25 deg blocks), v3 FAST (0.05 deg), observed."""
    import numpy as np
    from datetime import date as _d, timedelta
    d = _date(date)
    f = _fields()
    if f is None:
        raise HTTPException(404, "fields_FAST.npz not precomputed (python -m demo.backend.precompute_fields)")
    i = [str(x) for x in f["dates"]].index(d)
    r0, r1, c0, c1 = [int(x) for x in f["window"]]
    gp_cells = sorted({(r - r0, c - c0) for v in fc.gp_meta().values() for r, c, _ in v["cells"]})
    to = lambda a: np.round(np.asarray(a[i, day], np.float32), 1).tolist()
    return {"init_date": d, "day": day, "valid_date": (_d.fromisoformat(d) + timedelta(days=day)).isoformat(),
            "gfs": to(f["gfs"]), "v3": to(f["v3"]), "obs": to(f["obs"]), "gp_cells": gp_cells,
            "lat": [float(x) for x in f["lat"]], "lon": [float(x) for x in f["lon"]], "v3_mode_shown": "FAST"}


_fields_cache = {}


def _fields():
    import numpy as np
    if "f" not in _fields_cache:
        p = fc.PRE / "fields_FAST.npz"
        _fields_cache["f"] = dict(np.load(p)) if p.exists() else None
    return _fields_cache["f"]


# ---------------- live FAST inference (needs the v3 dataset locally: SIH_DATA or ./data) ----------------
_live = {}


def _live_model():
    if "m" not in _live:
        import numpy as np  # noqa: F401
        from sihv3.predict import FinalDownscaler
        _live["m"] = FinalDownscaler(REPO / "models" / "final", device=os.environ.get("SIH_DEMO_DEVICE"))
    return _live["m"]


def _inputs(init_date: str, H: int, N: int):
    import numpy as np
    import torch
    import zarr
    from sihv3.data import Normalizer, find, find_store
    g = zarr.open_group(str(find_store()), mode="r")
    ds = np.asarray(g["dates"][:]).astype(str)
    i = int(np.where(ds == init_date)[0][0])
    lo = (40 - N) // 2
    sl = slice(lo, lo + N)
    nm = Normalizer(find("normalization_stats_v3.yaml"))
    h = nm.fwd(np.asarray(g["history"].get_orthogonal_selection(([i], slice(14 - H, 14), slice(None), sl, sl))), axis=2)
    f = nm.fwd(np.asarray(g["future_forecast"].get_orthogonal_selection(([i], slice(None), slice(None), sl, sl))), axis=2)
    return torch.from_numpy(h), torch.from_numpy(f), nm.inv(np.asarray(f), axis=2)


class InferReq(BaseModel):
    date: str | None = None
    mode: str = "FAST"


@app.post("/api/v1/infer", tags=["inference"])
def infer(req: InferReq | None = None):
    import numpy as np
    req = req or InferReq()
    d, mode = _date(req.date), (req.mode or "FAST").upper()
    if mode != "FAST":
        raise HTTPException(400, "live inference on this server runs FAST; the diffusion modes are precomputed on a GPU")
    try:
        model = _live_model()
        h, f, f_phys = _inputs(d, model.args.H, model.args.N)
    except Exception as e:  # dataset or checkpoints not available on this machine
        raise HTTPException(503, f"live inference unavailable here ({type(e).__name__}: {e}); precomputed forecasts still served")
    t0 = time.time()
    out = model.predict(h, f, mode="FAST")
    ms = (time.time() - t0) * 1000
    meta = fc.gp_meta()
    codes = sorted(meta)
    W = np.zeros((len(codes), 6400), np.float32)
    for k, c in enumerate(codes):
        for r, col, w in meta[c]["cells"]:
            W[k, r * 80 + col] = w
    rain = out["mean"][0, 0, 0].reshape(6400) @ W.T
    pre = fc.load("FAST")
    di = [str(x) for x in pre["dates"]].index(d)
    gfs_mean = float(np.mean(fc.obs()["gfs_rain"][di, :, 0]))
    order = np.argsort(-rain)
    summ = lambda k: {"lgd_code": codes[k], "panchayat_name": meta[codes[k]]["name"], "rainfall_expected_mm": round(float(rain[k]), 1)}
    return {"status": "success", "model_name": "SIH26074 v3 FAST (3-seed spatiotemporal transformer + rain QM)", "init_date": d,
            "lead_days": 7, "input_shape": [1, model.args.H, 6, model.args.N, model.args.N] + [1, 7, 6, model.args.N, model.args.N],
            "output_shape": list(out["mean"].shape), "execution_time_ms": round(ms, 1),
            "coarse_mean_mm": round(gfs_mean, 2), "downscaled_mean_mm": round(float(rain.mean()), 2),
            "gfs_correction_pct": round(100 * (float(rain.mean()) - gfs_mean) / max(gfs_mean, 1e-3), 1),
            "matches_precomputed_max_abs_mm": round(float(np.max(np.abs(rain - pre["rain"][di, :, 0]))), 3),
            "total_panchayats_mapped": len(codes),
            "top_wettest_panchayats": [summ(k) for k in order[:5]], "driest_panchayats": [summ(k) for k in order[-5:]],
            "multivariate_fields": ["rain", "tmax", "tmin", "rh", "u10", "v10"]}


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="pwa")
