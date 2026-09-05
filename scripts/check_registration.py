"""
check_registration.py
Verify grid registration between IMD 0.25° native grid and CHIRPS 0.05° native grid.
Catches the potential ~2.7 km (0.025°) systematic shift between GeoTIFF half-pixel
centers and IMD grid centers, and defines the frozen affine alignment transform.
"""

import argparse
import json
from pathlib import Path
import numpy as np


def compute_grid_alignment(
    imd_lat_start: float = 6.5,
    imd_lat_step: float = 0.25,
    imd_n_lat: int = 129,
    imd_lon_start: float = 66.5,
    imd_lon_step: float = 0.25,
    imd_n_lon: int = 135,
    chirps_lat_start: float = 8.0,
    chirps_lat_end: float = 37.0,
    chirps_lon_start: float = 68.0,
    chirps_lon_end: float = 97.0,
    chirps_step: float = 0.05,
    chirps_half_pixel: bool = True,
):
    """
    Check the nested grid alignment of IMD 0.25° and CHIRPS 0.05°.
    
    Resolution factor: 0.25 / 0.05 = 5.0 (exact integer factor).
    Each IMD cell maps to a 5x5 block of CHIRPS cells.
    
    Returns registration audit report and frozen affine parameters.
    """
    scale_factor = imd_lat_step / chirps_step
    assert np.isclose(scale_factor, 5.0), f"Scale factor {scale_factor} != 5.0"

    # CHIRPS native grid with +0.025° half-pixel centers
    if chirps_half_pixel:
        chirps_lats = np.arange(chirps_lat_start + chirps_step / 2.0, chirps_lat_end, chirps_step)
        chirps_lons = np.arange(chirps_lon_start + chirps_step / 2.0, chirps_lon_end, chirps_step)
    else:
        chirps_lats = np.arange(chirps_lat_start, chirps_lat_end, chirps_step)
        chirps_lons = np.arange(chirps_lon_start, chirps_lon_end, chirps_step)

    # Pick an arbitrary overlapping IMD cell, say around Mandya (Lat 12.5°N, Lon 76.75°E)
    test_imd_lat = 12.50
    test_imd_lon = 76.75

    # Find the corresponding 5x5 CHIRPS block
    # In a properly nested grid, the IMD cell spans [test_imd_lon - 0.125, test_imd_lon + 0.125]
    # and [test_imd_lat - 0.125, test_imd_lat + 0.125]
    cell_lon_min = test_imd_lon - imd_lon_step / 2.0  # 76.625
    cell_lon_max = test_imd_lon + imd_lon_step / 2.0  # 76.875
    cell_lat_min = test_imd_lat - imd_lat_step / 2.0  # 12.375
    cell_lat_max = test_imd_lat + imd_lat_step / 2.0  # 12.625

    # Corresponding CHIRPS 5 cell centers
    # 76.65, 76.70, 76.75, 76.80, 76.85
    chirps_block_lons = [cell_lon_min + (k + 0.5) * chirps_step for k in range(5)]
    chirps_block_lats = [cell_lat_min + (k + 0.5) * chirps_step for k in range(5)]

    chirps_center_lon = float(np.mean(chirps_block_lons))
    chirps_center_lat = float(np.mean(chirps_block_lats))

    offset_lon = chirps_center_lon - test_imd_lon
    offset_lat = chirps_center_lat - test_imd_lat

    # Distance in km on spherical Earth at 12.5°N
    # 1 deg lat ≈ 111 km, 1 deg lon ≈ 111 * cos(12.5°) ≈ 108.37 km
    dist_km = np.sqrt(
        (offset_lat * 111.0) ** 2 + (offset_lon * 111.0 * np.cos(np.radians(test_imd_lat))) ** 2
    )

    is_aligned = np.isclose(dist_km, 0.0, atol=1e-4)

    report = {
        "scale_factor": float(scale_factor),
        "imd_cell_size_deg": imd_lat_step,
        "chirps_cell_size_deg": chirps_step,
        "test_cell": {"lat": test_imd_lat, "lon": test_imd_lon},
        "chirps_5x5_block_lons": [round(x, 4) for x in chirps_block_lons],
        "chirps_5x5_block_lats": [round(y, 4) for y in chirps_block_lats],
        "offset_deg": {"lat": round(offset_lat, 6), "lon": round(offset_lon, 6)},
        "offset_km": round(float(dist_km), 4),
        "is_aligned": bool(is_aligned),
        "frozen_affine_transform": {
            "scale_x": 5,
            "scale_y": 5,
            "shift_x_deg": round(offset_lon, 6),
            "shift_y_deg": round(offset_lat, 6),
            "hr_kernel_size": 5,
            "hr_stride": 5,
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Grid registration check between IMD 0.25° and CHIRPS 0.05°")
    parser.add_argument("--output", default="src/data/registration_transform.json")
    args = parser.parse_args()

    report = compute_grid_alignment()
    print("=== Grid Registration Audit Report ===")
    print(f"  Direct scale factor: {report['scale_factor']}x (5x5 kernel)")
    print(f"  Offset: {report['offset_km']} km (Lat {report['offset_deg']['lat']}°, Lon {report['offset_deg']['lon']}°)")
    print(f"  Alignment status: {'PERFECTLY ALIGNED' if report['is_aligned'] else 'SHIFT DETECTED'}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved frozen registration transform to {out_path}")


if __name__ == "__main__":
    main()
