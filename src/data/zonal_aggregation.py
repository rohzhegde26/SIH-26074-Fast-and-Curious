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

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import geopandas as gpd
import numpy as np
import pyproj
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union


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


def compute_cardinal_bearing(c_lat: float, c_lon: float, cent_lat: float, cent_lon: float) -> Tuple[str, str]:
    """Compute 8-point compass bearing and Kannada translation relative to reference centroid."""
    d_lat = c_lat - cent_lat
    d_lon = c_lon - cent_lon
    dist_km = math.hypot(d_lat, d_lon) * 111.0
    if dist_km < 1.8:
        return "Central", "ಕೇಂದ್ರ ಭಾಗ"
    angle = (math.degrees(math.atan2(d_lon, d_lat)) + 360.0) % 360.0
    sectors = [
        ("North", "ಉತ್ತರ"),
        ("North-East", "ಈಶಾನ್ಯ"),
        ("East", "ಪೂರ್ವ"),
        ("South-East", "ಆಗ್ನೇಯ"),
        ("South", "ದಕ್ಷಿಣ"),
        ("South-West", "ನೈಋತ್ಯ"),
        ("West", "ಪಶ್ಚಿಮ"),
        ("North-West", "ವಾಯುವ್ಯ"),
    ]
    idx = int((angle + 22.5) // 45.0) % 8
    return sectors[idx]


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

            # Centroid & cell directions relative to GP centroid
            cent_lat = float(poly.centroid.y)
            cent_lon = float(poly.centroid.x)
            cell_directions = [
                compute_cardinal_bearing(c_lat, c_lon, cent_lat, cent_lon)
                for (c_lat, c_lon) in cell_coords
            ]

            # Multi-parcel exclave detection (disconnected clusters > 3.0 km apart)
            parcels_data = []
            has_exclaves = False
            max_exclave_span_km = 0.0

            if poly.geom_type == "MultiPolygon" and len(poly.geoms) > 1:
                parts = list(poly.geoms)
                # Sort parts by area descending: part 0 is always the largest (Main Cluster)
                parts.sort(key=lambda p: p.area, reverse=True)

                main_p = parts[0]
                main_lat = float(main_p.centroid.y)
                main_lon = float(main_p.centroid.x)

                total_poly_area = poly.area if poly.area > 0 else 1.0
                K = len(parts)

                # Rule C3: If K > 4, top 4 by area + one summary parcel for remaining
                selected_parcels = []
                if K <= 4:
                    for i, p in enumerate(parts):
                        selected_parcels.append((i, p, False))
                else:
                    for i in range(4):
                        selected_parcels.append((i, parts[i], False))
                    remaining_union = unary_union(parts[4:])
                    selected_parcels.append((4, remaining_union, True))

                has_exclaves = True
                max_span = 0.0

                for i, p_geom, is_summary in selected_parcels:
                    p_area = p_geom.area
                    area_pct = round((p_area / total_poly_area) * 100.0, 1)
                    c_lat = float(p_geom.centroid.y)
                    c_lon = float(p_geom.centroid.x)
                    span = float(math.hypot(c_lat - main_lat, c_lon - main_lon) * 111.0)
                    if span > max_span:
                        max_span = span

                    if i == 0:
                        label_en = "Main Cluster"
                        label_kn = "ಮುಖ್ಯ ಭಾಗ"
                    elif is_summary:
                        rem_count = K - 4
                        label_en = f"+{rem_count} More Parcels"
                        label_kn = f"+{rem_count} ಇತರೆ ಭಾಗಗಳು"
                    else:
                        b_en, b_kn = compute_cardinal_bearing(c_lat, c_lon, main_lat, main_lon)
                        dist_str = f"{span:.1f}" if span < 10 else f"{span:.0f}"
                        label_en = f"{b_en} Parcel ({dist_str} km {b_en})"
                        label_kn = f"{b_kn} ಪ್ರತ್ಯೇಕ ಭಾಗ ({dist_str} ಕಿ.ಮೀ {b_kn})"

                    # Compute grid cells intersecting p_geom
                    p_bounds = p_geom.bounds
                    c_lon_min = np.floor(p_bounds[0] / self.hr_step) * self.hr_step
                    c_lon_max = np.ceil(p_bounds[2] / self.hr_step) * self.hr_step
                    c_lat_min = np.floor(p_bounds[1] / self.hr_step) * self.hr_step
                    c_lat_max = np.ceil(p_bounds[3] / self.hr_step) * self.hr_step

                    c_lons = np.arange(c_lon_min, c_lon_max + 1e-6, self.hr_step)
                    c_lats = np.arange(c_lat_min, c_lat_max + 1e-6, self.hr_step)

                    p_cell_coords = []
                    p_weights = []
                    for lat in c_lats:
                        for lon in c_lons:
                            box_elem = box(lon, lat, lon + self.hr_step, lat + self.hr_step)
                            if p_geom.intersects(box_elem):
                                c_inter = p_geom.intersection(box_elem)
                                if not c_inter.is_empty:
                                    c_area_m2 = compute_spherical_cell_area_m2(
                                        lat + self.hr_step / 2.0, self.hr_step, self.hr_step
                                    )
                                    w = (c_inter.area / box_elem.area) * c_area_m2
                                    p_cell_coords.append((float(lat + self.hr_step / 2.0), float(lon + self.hr_step / 2.0)))
                                    p_weights.append(w)

                    if not p_cell_coords:
                        r_lat = np.round(c_lat / self.hr_step) * self.hr_step
                        r_lon = np.round(c_lon / self.hr_step) * self.hr_step
                        p_cell_coords.append((float(r_lat), float(r_lon)))
                        p_weights.append(1.0)

                    p_total_w = sum(p_weights) if p_weights else 1.0
                    parcels_data.append({
                        "parcel_id": f"{gpcode}_p{i}",
                        "name_en": label_en,
                        "name_kn": label_kn,
                        "centroid": [round(c_lat, 4), round(c_lon, 4)],
                        "area_share_pct": area_pct,
                        "cell_coords": p_cell_coords,
                        "weights": p_weights,
                        "total_weight": p_total_w,
                    })

                # Normalize area_share_pct to strictly sum to 100.0
                sum_pct = sum(p["area_share_pct"] for p in parcels_data)
                if sum_pct > 0:
                    for p in parcels_data:
                        p["area_share_pct"] = round((p["area_share_pct"] / sum_pct) * 100.0, 1)

                max_exclave_span_km = round(max_span, 1)

            self.weights_cache[gpcode] = {
                "gpname": row.get("gpname", ""),
                "centroid": (cent_lat, cent_lon),
                "cell_coords": cell_coords,
                "cell_directions": cell_directions,
                "weights": weights,
                "normalized_weights": norm_weights,
                "total_weight": total_weight,
                "geod_area_m2": compute_geod_polygon_area_m2(poly),
                "has_exclaves": has_exclaves,
                "max_exclave_span_km": max_exclave_span_km,
                "parcels": parcels_data,
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

    def aggregate_grid_detailed(
        self,
        hr_mean: Union[np.ndarray, "torch.Tensor"],
        hr_lo: Union[np.ndarray, "torch.Tensor"],
        hr_hi: Union[np.ndarray, "torch.Tensor"],
        hr_lats: Union[np.ndarray, List[float]],
        hr_lons: Union[np.ndarray, List[float]],
    ) -> Dict[str, Dict]:
        """
        Aggregate high-resolution 0.05° grid rainfall with detailed intra-panchayat
        variance analysis, constituent boundary cell dispersion, and disconnected exclave detection.
        """
        def _to_np(g):
            if hasattr(g, "detach"):
                g = g.detach().cpu().numpy()
            if g.ndim == 3:
                g = g[0]
            elif g.ndim == 4:
                g = g[0, 0]
            return g

        hr_mean = _to_np(hr_mean)
        hr_lo = _to_np(hr_lo)
        hr_hi = _to_np(hr_hi)

        hr_lats = np.asarray(hr_lats, dtype=np.float32)
        hr_lons = np.asarray(hr_lons, dtype=np.float32)

        detailed_results = {}
        for gpcode, data in self.weights_cache.items():
            cell_coords = data["cell_coords"]
            weights = data["weights"]
            total_weight = data["total_weight"]
            cell_dirs = data.get("cell_directions", [])
            has_exclaves = data.get("has_exclaves", False)
            max_span = data.get("max_exclave_span_km", 0.0)
            parcels = data.get("parcels", [])

            if not cell_coords or total_weight <= 0:
                detailed_results[gpcode] = {
                    "has_exclaves": False,
                    "exclave_count": 0,
                    "max_exclave_span_km": 0.0,
                    "is_high_variance": False,
                    "spatial_variance_mm": 0.0,
                    "min_mm": 0.0,
                    "max_mm": 0.0,
                    "cell_count": 0,
                    "parcels": [],
                    "constituent_cells": [],
                }
                continue

            constituent_cells = []
            cell_vals = []
            for (c_lat, c_lon), w, (b_en, b_kn) in zip(cell_coords, weights, cell_dirs):
                r_idx = int(np.argmin(np.abs(hr_lats - c_lat)))
                c_idx = int(np.argmin(np.abs(hr_lons - c_lon)))
                pixel_val = round(float(hr_mean[r_idx, c_idx]), 1)
                cell_vals.append(pixel_val)
                w_pct = round((w / total_weight) * 100.0, 1)

                if pixel_val >= 15.0:
                    leach = "High"
                elif pixel_val >= 5.0:
                    leach = "Moderate"
                else:
                    leach = "Low"

                constituent_cells.append({
                    "cardinal_dir_en": b_en,
                    "cardinal_dir_kn": b_kn,
                    "lat": round(float(c_lat), 4),
                    "lon": round(float(c_lon), 4),
                    "rainfall_mm": pixel_val,
                    "weight_pct": w_pct,
                    "leach_risk": leach,
                })

            min_mm = min(cell_vals) if cell_vals else 0.0
            max_mm = max(cell_vals) if cell_vals else 0.0
            spatial_delta = round(max_mm - min_mm, 1)

            parcel_results = []
            if has_exclaves and parcels:
                for p in parcels:
                    p_cells = p["cell_coords"]
                    p_weights = p["weights"]
                    p_total_w = p["total_weight"]
                    if not p_cells or p_total_w <= 0:
                        continue

                    p_weighted_mean = 0.0
                    p_weighted_lo = 0.0
                    p_weighted_hi = 0.0
                    for (c_lat, c_lon), w in zip(p_cells, p_weights):
                        r_idx = int(np.argmin(np.abs(hr_lats - c_lat)))
                        c_idx = int(np.argmin(np.abs(hr_lons - c_lon)))
                        p_weighted_mean += float(hr_mean[r_idx, c_idx]) * w
                        p_weighted_lo += float(hr_lo[r_idx, c_idx]) * w
                        p_weighted_hi += float(hr_hi[r_idx, c_idx]) * w

                    p_mean = round(p_weighted_mean / p_total_w, 1)
                    p_lo = round(p_weighted_lo / p_total_w, 1)
                    p_hi = round(p_weighted_hi / p_total_w, 1)
                    p_lo = min(p_lo, p_mean)
                    p_hi = max(p_hi, p_mean)

                    parcel_results.append({
                        "parcel_id": p["parcel_id"],
                        "name_en": p["name_en"],
                        "name_kn": p["name_kn"],
                        "centroid": p["centroid"],
                        "area_share_pct": p["area_share_pct"],
                        "expected_mm": p_mean,
                        "likely_min_mm": p_lo,
                        "likely_max_mm": p_hi,
                    })

            # High variance trigger: spread >= 4.0 mm with max >= 5.0 mm,
            # or multi-parcel exclaves with difference >= 2.0 mm
            is_high_var = (spatial_delta >= 4.0 and max_mm >= 5.0) or (has_exclaves and spatial_delta >= 2.0)

            detailed_results[gpcode] = {
                "has_exclaves": has_exclaves,
                "exclave_count": len(parcel_results) if has_exclaves else 1,
                "max_exclave_span_km": max_span,
                "is_high_variance": bool(is_high_var),
                "spatial_variance_mm": spatial_delta,
                "min_mm": round(min_mm, 1),
                "max_mm": round(max_mm, 1),
                "cell_count": len(constituent_cells),
                "parcels": parcel_results,
                "constituent_cells": constituent_cells,
            }

        return detailed_results

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
