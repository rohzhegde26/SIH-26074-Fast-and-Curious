from datetime import date, datetime

from pydantic import BaseModel, Field


class Rainfall(BaseModel):
    expected: float = Field(ge=0)
    likely_min: float = Field(ge=0)
    likely_max: float = Field(ge=0)
    empirical_coverage: str = "90% calibrated (test 2023)"


class CropAdvisory(BaseModel):
    stage: str
    action_en: str
    action_kn: str


class AdvisorySet(BaseModel):
    ragi: CropAdvisory
    paddy: CropAdvisory
    sugarcane: CropAdvisory | None = None


class ForecastResponse(BaseModel):
    lgd_code: str
    panchayat_name: str
    district: str
    forecast_date: date
    timestamp_utc: datetime
    rainfall_mm: Rainfall
    advisory: AdvisorySet
    is_cached: bool = False


class IntegrationMockResponse(BaseModel):
    status: str
    contract: str
    sample_fields: list[str]
