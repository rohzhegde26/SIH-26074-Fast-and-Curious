from datetime import date, datetime

from pydantic import BaseModel, Field


class Rainfall(BaseModel):
    expected: float = Field(ge=0)
    likely_min: float = Field(ge=0)
    likely_max: float = Field(ge=0)
    empirical_coverage: str = "90% calibrated (test 2023)"


class FinancialRiskSchema(BaseModel):
    risk_level: str
    cost_estimate_inr: int
    impact_title_en: str
    impact_title_kn: str
    impact_desc_en: str
    impact_desc_kn: str


class CropAdvisory(BaseModel):
    stage: str
    action_en: str
    action_kn: str
    financial_risk: FinancialRiskSchema | None = None


class AdvisorySet(BaseModel):
    ragi: CropAdvisory
    paddy: CropAdvisory
    sugarcane: CropAdvisory | None = None


class ParcelDetailSchema(BaseModel):
    parcel_id: str
    name_en: str
    name_kn: str
    centroid: list[float]  # [lat, lon]
    area_share_pct: float
    expected_mm: float
    likely_min_mm: float
    likely_max_mm: float


class ConstituentCellSchema(BaseModel):
    cardinal_dir_en: str
    cardinal_dir_kn: str
    lat: float
    lon: float
    rainfall_mm: float
    weight_pct: float
    leach_risk: str


class SpatialVarianceSchema(BaseModel):
    has_exclaves: bool = False
    exclave_count: int = 1
    max_exclave_span_km: float = 0.0
    is_high_variance: bool = False
    spatial_variance_mm: float = 0.0
    min_mm: float = 0.0
    max_mm: float = 0.0
    cell_count: int = 0
    parcels: list[ParcelDetailSchema] = []
    constituent_cells: list[ConstituentCellSchema] = []


class DailyForecastItem(BaseModel):
    date: str
    day_offset: int
    day_label_en: str
    day_label_kn: str
    expected_mm: float
    likely_min_mm: float
    likely_max_mm: float
    rainfall_band: str = "dry"
    spray_window: str = "SAFE"
    harvest_window: str = "SAFE"
    irrigation_window: str = "IRRIGATE"
    lookahead_warning_en: str | None = None
    lookahead_warning_kn: str | None = None
    advisory_summary_en: str = ""
    advisory_summary_kn: str = ""


class AgrometVariables(BaseModel):
    temp_c: float = Field(default=29.0, description="Block NWP 2m Air Temperature (°C)")
    rh_pct: float = Field(default=68.0, description="Block NWP Relative Humidity (%)")
    wind_kph: float = Field(default=8.0, description="Block NWP 10m Wind Speed (km/h)")
    spray_drift_risk: str = Field(default="LOW", description="Foliar chemical spray drift indicator")
    fungal_disease_risk: str = Field(default="LOW", description="Fungal infection humidity risk indicator")
    source: str = "Block NWP Coarse Coupling"


class ForecastResponse(BaseModel):
    lgd_code: str
    panchayat_name: str
    taluk: str | None = None
    district: str
    forecast_date: date
    timestamp_utc: datetime
    rainfall_mm: Rainfall
    advisory: AdvisorySet
    agromet_context: AgrometVariables | None = None
    is_cached: bool = False
    spatial_variance: SpatialVarianceSchema | None = None
    multi_day_forecast: list[DailyForecastItem] = []


class IntegrationMockResponse(BaseModel):
    status: str
    contract: str
    sample_fields: list[str]


class VirtualARGUncertainty(BaseModel):
    lower_bound_mm: float
    upper_bound_mm: float
    confidence: str = "90% CQR empirical"


class VirtualARGResponse(BaseModel):
    station_id: str
    station_name: str
    lgd_code: str
    district: str = "MANDYA"
    state: str = "KARNATAKA"
    latitude: float
    longitude: float
    elevation_m: float
    observation_datetime_utc: str
    observation_datetime_ist: str
    rainfall_24h_mm: float
    uncertainty_range_90pct: VirtualARGUncertainty
    qc_status: str = "VALIDATED_MASS_CONSERVED"
    data_type: str = "SYNTHETIC_DOWNSCALED_FEATURE_STREAM"
    provenance: str = "SIH26074_vARG_Unet5x_GLO30"


class NandiniValidationRequest(BaseModel):
    lgd_code: str
    panchayat_name: str
    rained_bool: bool
    observer_role: str = "DAIRY_SECRETARY"
    milk_center_id: str | None = None
    observation_period: str = "LAST_12_HOURS"


class NandiniValidationResponse(BaseModel):
    status: str
    message: str
    validation_id: str
    lgd_code: str
    recorded_at_utc: str
    recalibration_flagged: bool


class NandiniStatsResponse(BaseModel):
    total_validations: int
    rain_reported_count: int
    no_rain_reported_count: int
    model_agreement_rate_pct: float
    active_dairy_centers: int


class InferenceRequest(BaseModel):
    coarse_grid: list[list[float]] | None = Field(
        default=None,
        description="Optional 16x16 2D array of coarse precipitation (mm). If omitted, uses default Mandya test grid.",
    )


class GPInferenceSummary(BaseModel):
    lgd_code: str
    panchayat_name: str
    rainfall_expected_mm: float
    rainfall_likely_min_mm: float
    rainfall_likely_max_mm: float
    spray_recommendation: str


class InferenceResponse(BaseModel):
    status: str = "success"
    model_name: str = "UNet5x-SuperRes"
    input_shape: list[int] = [1, 1, 16, 16]
    output_shape: list[int] = [1, 1, 80, 80]
    coarse_mean_mm: float
    downscaled_mean_mm: float
    mass_conservation_error_pct: float = 0.000
    execution_time_ms: float
    total_panchayats_mapped: int
    top_wettest_panchayats: list[GPInferenceSummary]
    driest_panchayats: list[GPInferenceSummary]
    sample_downscaled_grid: list[list[float]] | None = None

