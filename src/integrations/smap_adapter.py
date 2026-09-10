"""
src/integrations/smap_adapter.py

Typed abstract interface for NASA/ISRO SMAP 9 km satellite soil moisture observations.
Defines contracts for antecedent soil moisture raster ingestion and API index calculation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class SMAPAdapterInterface(ABC):
    """
    Abstract interface for NASA/ISRO SMAP L3 Enhanced 9 km Radiometer Soil Moisture data.
    Ingests surface volumetric soil moisture (0-5 cm) to condition runoff risk models.
    """

    @abstractmethod
    def fetch_soil_moisture_raster(
        self,
        observation_date: str,
        bounding_box: Tuple[float, float, float, float],
    ) -> Dict[str, Any]:
        """
        Retrieves gridded SMAP L3 Enhanced NetCDF product for the target bounding box.

        Args:
            observation_date: Satellite overpass date (YYYY-MM-DD).
            bounding_box: (min_lat, min_lon, max_lat, max_lon) in EPSG:4326.

        Returns:
            Dict containing:
                - soil_moisture_volumetric_cm3_cm3: [H, W] ndarray
                - retrieval_qual_flag: [H, W] bitmask
                - surface_temperature_k: [H, W] ndarray
                - spatial_resolution_km: float (9.0)
        """
        raise NotImplementedError("Stage-2: requires NCMRWF/NSIDC sandbox credentials")

    @abstractmethod
    def compute_antecedent_precipitation_index(
        self,
        soil_moisture_grid: Any,
        decay_constant: float = 0.85,
    ) -> Any:
        """
        Computes Antecedent Precipitation Index (API) proxy from multi-day satellite moisture.
        """
        raise NotImplementedError("Stage-2: requires NCMRWF/NSIDC sandbox credentials")


class SMAPAdapter(SMAPAdapterInterface):
    """Production client adapter for NASA Earthdata / NSIDC DAAC SMAP repository."""

    def __init__(self, endpoint_url: str = "https://n5eil01u.ecs.nsidc.org/SMAP/SPL3SMP_E.005/"):
        self.endpoint_url = endpoint_url

    def fetch_soil_moisture_raster(
        self,
        observation_date: str,
        bounding_box: Tuple[float, float, float, float],
    ) -> Dict[str, Any]:
        raise NotImplementedError("Stage-2: requires NCMRWF/NSIDC sandbox credentials")

    def compute_antecedent_precipitation_index(
        self,
        soil_moisture_grid: Any,
        decay_constant: float = 0.85,
    ) -> Any:
        raise NotImplementedError("Stage-2: requires NCMRWF/NSIDC sandbox credentials")
