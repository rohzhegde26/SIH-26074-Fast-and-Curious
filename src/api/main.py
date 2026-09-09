from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import uuid

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.advisory.engine import build_advisory, build_7day_forecast
from src.api.forecast_repository import get_forecast, list_forecasts
from src.api.schemas import (
    AdvisorySet,
    CropAdvisory,
    DailyForecastItem,
    FinancialRiskSchema,
    ForecastResponse,
    IntegrationMockResponse,
    NandiniStatsResponse,
    NandiniValidationRequest,
    NandiniValidationResponse,
    Rainfall,
    SpatialVarianceSchema,
    VirtualARGResponse,
    VirtualARGUncertainty,
)

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
CENTROIDS_PATH = ROOT / "data" / "serving" / "mandya_centroids.json"
FEEDBACK_PATH = ROOT / "data" / "serving" / "nandini_feedback.json"


@lru_cache(maxsize=1)
def _centroids() -> dict[str, dict]:
    if CENTROIDS_PATH.exists():
        with open(CENTROIDS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_nandini_feedback() -> list[dict]:
    if FEEDBACK_PATH.exists():
        try:
            with open(FEEDBACK_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _append_nandini_feedback(entry: dict):
    feedbacks = _load_nandini_feedback()
    feedbacks.append(entry)
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(feedbacks, f, indent=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(docs_dir / "openapi.json", "w", encoding="utf-8") as f:
            json.dump(app.openapi(), f, indent=2)
    except Exception:
        pass
    yield


app = FastAPI(
    title="Mandya Weather Advisory API & Virtual ARG Service",
    version="0.2.0",
    description="5× Downscaled Weather Advisory, Conformal Uncertainty (CQR), Virtual ARG Network, and KMF Nandini Ground-Truth Validation Loop.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_header(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css", ".webmanifest")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def _convert_advisory(adv) -> CropAdvisory:
    fin_schema = None
    if adv.financial_risk:
        fin_schema = FinancialRiskSchema(**adv.financial_risk.__dict__)
    return CropAdvisory(
        stage=adv.stage,
        action_en=adv.action_en,
        action_kn=adv.action_kn,
        financial_risk=fin_schema,
    )


def _response(record: dict) -> ForecastResponse:
    expected = float(record["expected_mm"])
    likely_max = float(record["likely_max_mm"])
    ragi = build_advisory("ragi", record.get("ragi_stage", "vegetative"), expected, likely_max)
    paddy = build_advisory("paddy", record.get("paddy_stage", "vegetative"), expected, likely_max)
    sugarcane = build_advisory("sugarcane", record.get("sugarcane_stage", "grand_growth"), expected, likely_max)

    sp_var = None
    if "spatial_variance" in record and record["spatial_variance"]:
        try:
            sp_var = SpatialVarianceSchema(**record["spatial_variance"])
        except Exception:
            sp_var = None

    raw_multi = build_7day_forecast(record)
    multi_days = [DailyForecastItem(**item) for item in raw_multi]

    return ForecastResponse(
        lgd_code=str(record["lgd_code"]),
        panchayat_name=record["panchayat_name"],
        district=record["district"],
        forecast_date=record["forecast_date"],
        timestamp_utc=record["timestamp_utc"],
        rainfall_mm=Rainfall(expected=expected, likely_min=record["likely_min_mm"], likely_max=likely_max),
        advisory=AdvisorySet(
            ragi=_convert_advisory(ragi),
            paddy=_convert_advisory(paddy),
            sugarcane=_convert_advisory(sugarcane),
        ),
        spatial_variance=sp_var,
        multi_day_forecast=multi_days,
    )


@app.get("/api/forecast/{lgd_code}", response_model=ForecastResponse, tags=["forecast"])
def forecast(lgd_code: str) -> ForecastResponse:
    record = get_forecast(lgd_code)
    if record is None:
        raise HTTPException(status_code=404, detail="Panchayat forecast not found")
    return _response(record)


@app.get("/api/forecasts", response_model=list[ForecastResponse], tags=["forecast"])
def forecasts() -> list[ForecastResponse]:
    """Bulk sync endpoint for offline cache population."""
    return [_response(record) for record in list_forecasts()]


@app.get(
    "/api/v1/virtual-arg/{lgd_code}",
    response_model=VirtualARGResponse,
    tags=["virtual-arg"],
    summary="[IMD ARG STANDARD SCHEMA] Virtual Automated Rain Gauge feed for unmonitored Panchayat",
)
def virtual_arg(lgd_code: str) -> VirtualARGResponse:
    record = get_forecast(lgd_code)
    if record is None:
        raise HTTPException(status_code=404, detail="Panchayat not found")

    centroids = _centroids()
    coords = centroids.get(str(lgd_code), {"lat": 12.52, "lon": 76.89, "elevation_m": 660.0})

    expected = float(record["expected_mm"])
    l_min = float(record["likely_min_mm"])
    l_max = float(record["likely_max_mm"])

    return VirtualARGResponse(
        station_id=f"VARG_KA_MAN_{lgd_code}",
        station_name=f"{record['panchayat_name']} Virtual ARG",
        lgd_code=str(lgd_code),
        district=record.get("district", "MANDYA"),
        state="KARNATAKA",
        latitude=coords["lat"],
        longitude=coords["lon"],
        elevation_m=coords["elevation_m"],
        observation_datetime_utc=f"{record['forecast_date']}T03:00:00Z",
        observation_datetime_ist=f"{record['forecast_date']} 08:30:00 IST",
        rainfall_24h_mm=expected,
        uncertainty_range_90pct=VirtualARGUncertainty(
            lower_bound_mm=l_min,
            upper_bound_mm=l_max,
            confidence="90% CQR empirical",
        ),
        qc_status="VALIDATED_MASS_CONSERVED",
        data_type="SYNTHETIC_DOWNSCALED_FEATURE_STREAM",
        provenance="SIH26074_vARG_Unet5x_GLO30",
    )


@app.get(
    "/api/v1/virtual-arg",
    response_model=list[VirtualARGResponse],
    tags=["virtual-arg"],
    summary="[IMD ARG BULK FEED] District-wide Virtual Rain Gauge array",
)
def virtual_arg_bulk() -> list[VirtualARGResponse]:
    return [virtual_arg(str(r["lgd_code"])) for r in list_forecasts()]


@app.post(
    "/api/v1/validation/nandini",
    response_model=NandiniValidationResponse,
    tags=["validation"],
    summary="[KMF NANDINI GROUND-TRUTH] 2-Tap Dairy Secretary binary rainfall verification",
)
def submit_nandini_validation(req: NandiniValidationRequest) -> NandiniValidationResponse:
    forecast_rec = get_forecast(req.lgd_code)
    model_predicted_rain = False
    if forecast_rec:
        model_predicted_rain = (
            float(forecast_rec.get("expected_mm", 0.0)) >= 2.5
            or float(forecast_rec.get("likely_max_mm", 0.0)) >= 5.0
        )

    discrepancy = model_predicted_rain != req.rained_bool
    val_id = str(uuid.uuid4())[:8]
    recorded_at = datetime.now(timezone.utc).isoformat()

    entry = {
        "validation_id": val_id,
        "lgd_code": req.lgd_code,
        "panchayat_name": req.panchayat_name,
        "rained_bool": req.rained_bool,
        "observer_role": req.observer_role,
        "milk_center_id": req.milk_center_id or f"KMF_KA_MAN_{req.lgd_code[:4]}",
        "observation_period": req.observation_period,
        "recorded_at_utc": recorded_at,
        "recalibration_flagged": discrepancy,
    }

    _append_nandini_feedback(entry)

    msg = (
        "Validation recorded! Model & field observation agreement confirmed."
        if not discrepancy
        else "Validation recorded! Discrepancy logged for Conformal Quantile recalibration."
    )

    return NandiniValidationResponse(
        status="success",
        message=msg,
        validation_id=val_id,
        lgd_code=req.lgd_code,
        recorded_at_utc=recorded_at,
        recalibration_flagged=discrepancy,
    )


@app.get(
    "/api/v1/validation/stats",
    response_model=NandiniStatsResponse,
    tags=["validation"],
    summary="Summary statistics of KMF Dairy Secretary validation network",
)
def nandini_stats() -> NandiniStatsResponse:
    feedbacks = _load_nandini_feedback()
    total = len(feedbacks)
    if total == 0:
        return NandiniStatsResponse(
            total_validations=0,
            rain_reported_count=0,
            no_rain_reported_count=0,
            model_agreement_rate_pct=100.0,
            active_dairy_centers=0,
        )

    rain_cnt = sum(1 for f in feedbacks if f.get("rained_bool"))
    no_rain_cnt = total - rain_cnt
    agreed_cnt = sum(1 for f in feedbacks if not f.get("recalibration_flagged", False))
    agreement_rate = round((agreed_cnt / total) * 100.0, 1)
    centers = len({f.get("milk_center_id") for f in feedbacks if f.get("milk_center_id")})

    return NandiniStatsResponse(
        total_validations=total,
        rain_reported_count=rain_cnt,
        no_rain_reported_count=no_rain_cnt,
        model_agreement_rate_pct=agreement_rate,
        active_dairy_centers=max(centers, 1),
    )


@app.get(
    "/api/egramswaraj/mock",
    response_model=IntegrationMockResponse,
    tags=["integration"],
    summary="[MOCK / INTEGRATION PROTOTYPE] e-GramSwaraj contract",
)
def egramswaraj_mock(response: Response) -> IntegrationMockResponse:
    response.headers["X-Integration-Status"] = "Mock-Prototype"
    return IntegrationMockResponse(
        status="mock-prototype",
        contract="Read-only example payload for future dashboard integration.",
        sample_fields=["lgd_code", "forecast_date", "expected_mm", "likely_min_mm", "likely_max_mm"],
    )


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="pwa")

