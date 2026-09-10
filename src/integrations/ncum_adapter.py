"""
src/integrations/ncum_adapter.py

Typed abstract interface for NCMRWF Unified Model (NCUM) numerical weather prediction feeds.
Defines contracts for 3-hourly NWP atmospheric forecast tensor ingestion.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class NCUMAdapterInterface(ABC):
    """
    Abstract interface for NCMRWF NCUM operational global and regional model feeds.
    Ingests 3-hourly forecasted atmospheric parameters on a 12 km grid.
    """

    @abstractmethod
    def fetch_3hourly_tensors(
        self,
        forecast_init_date: str,
        lead_hours: int = 72,
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Polls and ingests 3-hourly NCUM numerical atmospheric tensors.

        Args:
            forecast_init_date: Model cycle initialization date (YYYY-MM-DD).
            lead_hours: Forecast lead horizon in hours (default 72h).
            bounding_box: (min_lat, min_lon, max_lat, max_lon) in EPSG:4326.

        Returns:
            Dict containing 4D tensors [T, C, H, W] for:
                - precipitation_flux_kg_m2_s
                - air_temperature_2m_k
                - specific_humidity_2m_kg_kg
                - wind_u_10m_mps
                - wind_v_10m_mps
                - surface_pressure_pa
        """
        raise NotImplementedError("Stage-2: requires NCMRWF/NSIDC sandbox credentials")

    @abstractmethod
    def interpolate_to_block_mesh(
        self,
        raw_tensors: Dict[str, Any],
        target_resolution_deg: float = 0.25,
    ) -> Dict[str, Any]:
        """
        Spatially aligns NCUM 12 km curvilinear grid to standard 0.25° IMD coarse grid mesh.
        """
        raise NotImplementedError("Stage-2: requires NCMRWF/NSIDC sandbox credentials")


class NCUMAdapter(NCUMAdapterInterface):
    """Production client adapter for NCMRWF Unified Model data distribution nodes."""

    def __init__(self, endpoint_url: str = "https://ncmrwf.gov.in/opendap/ncum_global"):
        self.endpoint_url = endpoint_url

    def fetch_3hourly_tensors(
        self,
        forecast_init_date: str,
        lead_hours: int = 72,
        bounding_box: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Stage-2: requires NCMRWF/NSIDC sandbox credentials")

    def interpolate_to_block_mesh(
        self,
        raw_tensors: Dict[str, Any],
        target_resolution_deg: float = 0.25,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Stage-2: requires NCMRWF/NSIDC sandbox credentials")
