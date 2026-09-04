"""
src/data/zonal_aggregation.py

Clean Grid-to-Polygon Zonal Aggregation with Analytic Spherical Area Weighting.

Mathematical Formulation:
    For each panchayat P:
        1. Fractional coverage: f_i in [0, 1] for HR cell i.
        2. Analytic spherical cell area:
           A_i = R^2 * d_phi * d_lambda * cos(lat_i)
           where R = 6371008.8 m, d_phi and d_lambda in radians.
        3. Effective weight: w_i = f_i * A_i.
        4. Aggregated rainfall:
           Rain_P = sum_i(HR_i * w_i) / sum_i(w_i)

Critical Guardrails:
    - Input polygons must strictly load from data/processed/mandya_full.geojson (never simplified TopoJSON).
    - Avoid CRS double-counting: never multiply an equal-area projected area by cos(lat).
    - pyproj.Geod is used for rigorous ellipsoidal geodesic area audits.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import geopandas as gpd
import numpy as np
import pyproj
from shapely.geometry import Polygon, box


# Earth radius in meters (WGS84 authalic sphere)
EARTH_RADIUS_M = 6371008.8
HR_RES_DEG = 0.05
RAD_PER_DEG = np.pi / 180.0


def compute_spherical_cell_area_m2(lat_deg: float, d_lat_deg: float = 0.05, d_lon_deg: float = 0.05) -> float:
    """
    Analytic spherical area of a grid cell on Earth:
    A = R^2 * d_phi * d_lambda * cos(lat)
    """
    d_phi = d_lat_deg * RAD_PER_DEG
    d_lambda = d_lon_deg * RAD_PER_DEG
    lat_rad = lat_deg * RAD_PER_DEG
    return float((EARTH_RADIUS_M ** 2) * d_phi * d_lambda * np.cos(lat_rad))


def compute_geod_polygon_area_m2(polygon: Polygon) -> float:
    """Compute absolute spherical geodesic area in m^2 using authalic sphere R=6371008.8m."""
    geod = pyproj.Geod(a=EARTH_RADIUS_M, b=EARTH_RADIUS_M)
    area, _ = geod.geometry_area_perimeter(polygon)
    return float(abs(area))


class ZonalAggregator:
    """
    Zonal aggregation engine mapping 0.05° HR gridded rainfall to panchayat vectors.
    Precomputes spatial intersection weights for fast inference.
    """

    def __init__(
        self,
        geojson_path: str = "data/processed/mandya_full.geojson",
        hr_step: float = 0.05,
    ):
        self.geojson_path = Path(geojson_path)
        assert self.geojson_path.exists(), f"Polygon file {self.geojson_path} not found"
        # Guardrail: Never use simplified topojson for zonal aggregation
        assert not str(self.geojson_path).endswith(".topojson"), (
            "Guardrail Violation: TopoJSON cannot be used for zonal aggregation area math!"
        )

        self.gdf = gpd.read_file(self.geojson_path)
        self.hr_step = hr_step
        self.geod = pyproj.Geod(a=EARTH_RADIUS_M, b=EARTH_RADIUS_M)
        self.weights_cache: Dict[str, Dict] = {}
        self._precompute_polygon_weights()

    def _precompute_polygon_weights(self):
        """Precompute cell fractional intersections and spherical weights for all polygons."""
        for idx, row in self.gdf.iterrows():
            gpcode = str(row.get("gpcode", idx))
            poly = row.geometry
            poly_bounds = poly.bounds  # minx, miny, maxx, maxy

            # Determine overlapping HR cells
            lon_min = np.floor(poly_bounds[0] / self.hr_step) * self.hr_step
            lon_max = np.ceil(poly_bounds[2] / self.hr_step) * self.hr_step
            lat_min = np.floor(poly_bounds[1] / self.hr_step) * self.hr_step
            lat_max = np.ceil(poly_bounds[3] / self.hr_step) * self.hr_step

            lons = np.arange(lon_min, lon_max + 1e-6, self.hr_step)
            lats = np.arange(lat_min, lat_max + 1e-6, self.hr_step)

            cell_coords = []
            weights = []
            areas = []

            for lat in lats:
                for lon in lons:
                    cell_box = box(lon, lat, lon + self.hr_step, lat + self.hr_step)
                    if poly.intersects(cell_box):
                        inter = poly.intersection(cell_box)
                        if not inter.is_empty:
                            cell_area = compute_spherical_cell_area_m2(
                                lat + self.hr_step / 2.0, self.hr_step, self.hr_step
                            )
                            # Fractional coverage
                            frac = inter.area / cell_box.area
                            w = frac * cell_area
                            cell_coords.append((float(lat + self.hr_step / 2.0), float(lon + self.hr_step / 2.0)))
                            weights.append(w)
                            areas.append(inter.area)

            total_weight = sum(weights) if weights else 1.0
            norm_weights = [w / total_weight for w in weights] if weights else []

            self.weights_cache[gpcode] = {
                "gpname": row.get("gpname", ""),
                "cell_coords": cell_coords,
                "weights": weights,
                "normalized_weights": norm_weights,
                "total_weight": total_weight,
                "geod_area_m2": compute_geod_polygon_area_m2(poly),
            }

    def aggregate_constant(self, value: float = 1.0) -> Dict[str, float]:
        """Aggregate a uniform constant field across all panchayats for testing."""
        result = {}
        for gpcode, data in self.weights_cache.items():
            if data["normalized_weights"]:
                result[gpcode] = float(sum(value * w for w in data["normalized_weights"]))
            else:
                result[gpcode] = value
        return result

    def aggregate_grid(
        self,
        hr_grid: Union[np.ndarray, "torch.Tensor"],
        hr_lats: Union[np.ndarray, List[float]],
        hr_lons: Union[np.ndarray, List[float]],
    ) -> Dict[str, float]:
        """
        Aggregate high-resolution 0.05° grid rainfall to panchayat polygons.
        Formula:
            Rain_P = sum(HR_i * w_i) / sum(w_i) where w_i = f_i * A_i
        """
        if hasattr(hr_grid, "detach"):
            hr_grid = hr_grid.detach().cpu().numpy()
        if hr_grid.ndim == 3:
            hr_grid = hr_grid[0]
        elif hr_grid.ndim == 4:
            hr_grid = hr_grid[0, 0]

        hr_lats = np.asarray(hr_lats, dtype=np.float32)
        hr_lons = np.asarray(hr_lons, dtype=np.float32)

        results = {}
        for gpcode, data in self.weights_cache.items():
            cell_coords = data["cell_coords"]
            weights = data["weights"]
            total_weight = data["total_weight"]

            if not cell_coords or total_weight <= 0:
                results[gpcode] = 0.0
                continue

            weighted_sum = 0.0
            for (c_lat, c_lon), w in zip(cell_coords, weights):
                r_idx = int(np.argmin(np.abs(hr_lats - c_lat)))
                c_idx = int(np.argmin(np.abs(hr_lons - c_lon)))
                pixel_val = float(hr_grid[r_idx, c_idx])
                weighted_sum += pixel_val * w

            results[gpcode] = float(weighted_sum / total_weight)

        return results

    def validate_interior_cell_partition(self) -> Dict[str, Union[bool, int, float, List[Dict]]]:
        """
        Validates validation gate 2:
        For cells strictly interior to the district, verify sum_P(f_i) ≈ 1.0.
        Report boundary cells separately (do not fail them).
        """
        district_poly = self.gdf.geometry.unary_union
        bounds = district_poly.bounds

        lon_min = np.floor(bounds[0] / self.hr_step) * self.hr_step
        lon_max = np.ceil(bounds[2] / self.hr_step) * self.hr_step
        lat_min = np.floor(bounds[1] / self.hr_step) * self.hr_step
        lat_max = np.ceil(bounds[3] / self.hr_step) * self.hr_step

        lons = np.arange(lon_min, lon_max + 1e-6, self.hr_step)
        lats = np.arange(lat_min, lat_max + 1e-6, self.hr_step)

        interior_reports = []
        boundary_reports = []

        for lat in lats:
            for lon in lons:
                cell = box(lon, lat, lon + self.hr_step, lat + self.hr_step)
                if not cell.intersects(district_poly):
                    continue

                sum_frac = 0.0
                for poly in self.gdf.geometry:
                    if poly.intersects(cell):
                        inter = poly.intersection(cell)
                        if not inter.is_empty:
                            sum_frac += inter.area / cell.area

                # Interior check: cell buffer completely within district boundary
                is_interior = cell.buffer(-1e-5).within(district_poly)

                cell_info = {
                    "lat": float(lat + self.hr_step / 2.0),
                    "lon": float(lon + self.hr_step / 2.0),
                    "sum_fraction": float(sum_frac),
                    "is_interior": is_interior,
                }

                if is_interior:
                    interior_reports.append(cell_info)
                else:
                    boundary_reports.append(cell_info)

        interior_sums = [c["sum_fraction"] for c in interior_reports]
        boundary_sums = [c["sum_fraction"] for c in boundary_reports]

        interior_mean = float(np.mean(interior_sums)) if interior_sums else 1.0
        boundary_mean = float(np.mean(boundary_sums)) if boundary_sums else 0.5
        all_interior_passed = all(abs(s - 1.0) <= 0.02 for s in interior_sums)

        return {
            "passed": all_interior_passed,
            "interior_count": len(interior_reports),
            "interior_mean_sum": interior_mean,
            "boundary_count": len(boundary_reports),
            "boundary_mean_sum": boundary_mean,
            "interior_reports": interior_reports,
            "boundary_reports": boundary_reports,
        }

    def compute_discrepancy_statistics(
        self,
        hr_grid: np.ndarray,
        lr_grid: np.ndarray,
        hr_lats: np.ndarray,
        hr_lons: np.ndarray,
        lr_lats: np.ndarray,
        lr_lons: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Log discrepancy statistic |mean_HR(P) - mean_LR(P_cell)| for each panchayat.
        """
        hr_agg = self.aggregate_grid(hr_grid, hr_lats, hr_lons)
        results = {}

        for gpcode, data in self.weights_cache.items():
            cell_coords = data["cell_coords"]
            hr_val = hr_agg.get(gpcode, 0.0)

            if cell_coords:
                center_lat = np.mean([c[0] for c in cell_coords])
                center_lon = np.mean([c[1] for c in cell_coords])
                r_lr = int(np.argmin(np.abs(lr_lats - center_lat)))
                c_lr = int(np.argmin(np.abs(lr_lons - center_lon)))
                lr_val = float(lr_grid[r_lr, c_lr])
            else:
                lr_val = 0.0

            discrepancy = abs(hr_val - lr_val)
            results[gpcode] = {
                "gpname": data["gpname"],
                "hr_panchayat_mm": hr_val,
                "lr_cell_mm": lr_val,
                "discrepancy_mm": discrepancy,
                "rel_discrepancy": discrepancy / max(lr_val, 1.0),
            }

        return results

    def audit_area_closure(self, relative_tol: float = 1e-3) -> Tuple[bool, List[Dict]]:
        """
        Audit polygon area closure:
        sum_cells(inter_area) == area(P) within relative tolerance 10^-3.
        """
        reports = []
        all_passed = True
        for gpcode, data in self.weights_cache.items():
            spherical_area = data["total_weight"]
            geod_area = data["geod_area_m2"]
            rel_diff = abs(spherical_area - geod_area) / max(geod_area, 1.0)
            passed = rel_diff <= relative_tol
            if not passed:
                all_passed = False
            reports.append({
                "gpcode": gpcode,
                "gpname": data["gpname"],
                "spherical_area_m2": spherical_area,
                "geod_area_m2": geod_area,
                "rel_diff": rel_diff,
                "passed": passed,
            })
        return all_passed, reports
