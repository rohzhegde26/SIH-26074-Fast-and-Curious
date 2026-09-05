from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.advisory.engine import build_advisory
from src.api.forecast_repository import get_forecast, list_forecasts
from src.api.schemas import AdvisorySet, CropAdvisory, ForecastResponse, IntegrationMockResponse, Rainfall

import json

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(docs_dir / "openapi.json", "w", encoding="utf-8") as f:
            json.dump(app.openapi(), f, indent=2)
    except Exception as e:
        pass
    yield


app = FastAPI(title="Mandya Weather Advisory API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _response(record: dict) -> ForecastResponse:
    expected = float(record["expected_mm"])
    likely_max = float(record["likely_max_mm"])
    ragi = build_advisory("ragi", record.get("ragi_stage", "vegetative"), expected, likely_max)
    paddy = build_advisory("paddy", record.get("paddy_stage", "vegetative"), expected, likely_max)
    sugarcane = build_advisory("sugarcane", record.get("sugarcane_stage", "grand_growth"), expected, likely_max)
    return ForecastResponse(
        lgd_code=str(record["lgd_code"]),
        panchayat_name=record["panchayat_name"],
        district=record["district"],
        forecast_date=record["forecast_date"],
        timestamp_utc=record["timestamp_utc"],
        rainfall_mm=Rainfall(expected=expected, likely_min=record["likely_min_mm"], likely_max=likely_max),
        advisory=AdvisorySet(
            ragi=CropAdvisory(**ragi.__dict__),
            paddy=CropAdvisory(**paddy.__dict__),
            sugarcane=CropAdvisory(**sugarcane.__dict__),
        ),
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
    "/api/egramswaraj/mock",
    response_model=IntegrationMockResponse,
    tags=["integration"],
    summary="[MOCK / INTEGRATION PROTOTYPE] e-GramSwaraj contract",
)
def egramswaraj_mock(response: Response) -> IntegrationMockResponse:
    """[MOCK / INTEGRATION PROTOTYPE] No government system is contacted or written to."""
    response.headers["X-Integration-Status"] = "Mock-Prototype"
    return IntegrationMockResponse(
        status="mock-prototype",
        contract="Read-only example payload for future dashboard integration.",
        sample_fields=["lgd_code", "forecast_date", "expected_mm", "likely_min_mm", "likely_max_mm"],
    )


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="pwa")
