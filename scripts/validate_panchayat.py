"""
validate_panchayat.py
Ingest panchayat boundaries from LGD parquet using DuckDB pushdown,
validate topology with shapely, compute spatial holdout buffer, and
export dual representations: full GeoJSON and simplified TopoJSON.
"""

import argparse
import json
import os
from pathlib import Path
import duckdb
import geopandas as gpd
from shapely import wkt
from shapely.ops import unary_union
from shapely.validation import make_valid
import topojson as tp


def process_district(
    parquet_path: str = "data/raw/geodata/LGD_panchayats.parquet",
    district: str = "MANDYA",
    state: str = "KARNATAKA",
    buffer_deg: float = 0.5,
    output_dir: str = "data/processed",
    pinned_json_path: str = "src/data/pinned_district.json",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Querying LGD parquet via DuckDB pushdown for {district}, {state}...")
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    query = f"""
        SELECT 
            gpcode,
            gpname,
            dtname,
            stname,
            sdtname,
            blkname,
            ST_AsText(geometry) as geom_wkt
        FROM '{parquet_path}'
        WHERE upper(stname) = upper('{state}')
          AND upper(dtname) = upper('{district}')
    """
    df = con.execute(query).fetchdf()
    print(f"  Retrieved {len(df)} raw polygon records.")

    if len(df) == 0:
        raise ValueError(f"No records found for district {district} in {state}.")

    print("[2/5] Parsing geometries and validating topology...")
    geoms = [make_valid(wkt.loads(w)) for w in df["geom_wkt"]]
    gdf_raw = gpd.GeoDataFrame(df.drop(columns=["geom_wkt"]), geometry=geoms, crs="EPSG:4326")

    # Dissolve multi-part rows by gpcode
    print("  Dissolving multi-part polygons by gpcode...")
    gdf = gdf_raw.dissolve(
        by="gpcode",
        aggfunc={
            "gpname": "first",
            "dtname": "first",
            "stname": "first",
            "sdtname": "first",
            "blkname": "first",
        },
        as_index=False,
    )

    gp_count = len(gdf)
    valid_count = sum(gdf.geometry.is_valid)
    valid_fraction = valid_count / gp_count if gp_count > 0 else 0.0

    print(f"  GP Count: {gp_count}")
    print(f"  Valid Geometries: {valid_count}/{gp_count} ({valid_fraction*100:.2f}%)")

    # Validation gates
    assert 80 <= gp_count <= 300, f"GP count {gp_count} outside expected range [80, 300]"
    assert valid_fraction >= 0.98, f"Valid geometry fraction {valid_fraction:.4f} < 0.98"

    # Compute district boundary & spatial holdout buffer
    print(f"[3/5] Computing district union and spatial holdout buffer ({buffer_deg}°)...")
    district_union = unary_union(gdf.geometry)
    holdout_buffer = district_union.buffer(buffer_deg)

    bounds = district_union.bounds  # minx, miny, maxx, maxy
    buffer_bounds = holdout_buffer.bounds

    print(f"  District Bounding Box: Lon [{bounds[0]:.4f}, {bounds[2]:.4f}], Lat [{bounds[1]:.4f}, {bounds[3]:.4f}]")
    print(f"  Holdout Buffer BBox:   Lon [{buffer_bounds[0]:.4f}, {buffer_bounds[2]:.4f}], Lat [{buffer_bounds[1]:.4f}, {buffer_bounds[3]:.4f}]")

    # Export Full GeoJSON
    full_geojson_path = out_dir / f"{district.lower()}_full.geojson"
    print(f"[4/5] Exporting full precision GeoJSON: {full_geojson_path}...")
    gdf.to_file(full_geojson_path, driver="GeoJSON")
    full_size_kb = os.path.getsize(full_geojson_path) / 1024
    print(f"  Full GeoJSON size: {full_size_kb:.1f} KB")

    # Export Simplified TopoJSON (< 400 KB for PWA)
    simplified_topojson_path = out_dir / f"{district.lower()}_simplified.topojson"
    print(f"[5/5] Generating simplified TopoJSON: {simplified_topojson_path}...")

    # Simplify geometries slightly (approx 10-20m tolerance in degrees: ~0.0001)
    gdf_simplified = gdf.copy()
    gdf_simplified["geometry"] = gdf_simplified.geometry.simplify(tolerance=0.0002, preserve_topology=True)

    # Convert to TopoJSON
    topo = tp.Topology(gdf_simplified, topology=True, prequantize=True, object_name="panchayats")
    topo_dict = topo.to_dict()

    with open(simplified_topojson_path, "w", encoding="utf-8") as f:
        json.dump(topo_dict, f, separators=(",", ":"))

    topo_size_kb = os.path.getsize(simplified_topojson_path) / 1024
    print(f"  Simplified TopoJSON size: {topo_size_kb:.1f} KB")
    assert topo_size_kb < 400.0, f"Simplified TopoJSON size {topo_size_kb:.1f} KB exceeds 400 KB limit!"

    # Save holdout geometry to processed
    holdout_geojson_path = out_dir / f"{district.lower()}_holdout_buffer.geojson"
    gpd.GeoDataFrame([{"holdout": True}], geometry=[holdout_buffer], crs="EPSG:4326").to_file(
        holdout_geojson_path, driver="GeoJSON"
    )

    # Update pinned_district.json
    pinned_path = Path(pinned_json_path)
    if pinned_path.exists():
        with open(pinned_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {}

    config.update(
        {
            "primary": district.upper(),
            "gp_count": gp_count,
            "valid_fraction": valid_fraction,
            "buffer_deg": buffer_deg,
            "district_bbox": {
                "min_lon": bounds[0],
                "min_lat": bounds[1],
                "max_lon": bounds[2],
                "max_lat": bounds[3],
            },
            "holdout_buffer_bbox": {
                "min_lon": buffer_bounds[0],
                "min_lat": buffer_bounds[1],
                "max_lon": buffer_bounds[2],
                "max_lat": buffer_bounds[3],
            },
            "full_geojson": str(full_geojson_path.as_posix()),
            "simplified_topojson": str(simplified_topojson_path.as_posix()),
            "holdout_geojson": str(holdout_geojson_path.as_posix()),
        }
    )

    with open(pinned_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"  Updated {pinned_path}")
    print(f"[SUCCESS] District validation and dual export for {district} complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Panchayat geometries and export dual formats")
    parser.add_argument("--parquet", default="data/raw/geodata/LGD_panchayats.parquet", help="Path to LGD parquet")
    parser.add_argument("--district", default="MANDYA", help="Target district")
    parser.add_argument("--state", default="KARNATAKA", help="Target state")
    parser.add_argument("--buffer", type=float, default=0.5, help="Spatial holdout buffer in degrees")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    args = parser.parse_args()

    process_district(
        parquet_path=args.parquet,
        district=args.district,
        state=args.state,
        buffer_deg=args.buffer,
        output_dir=args.output_dir,
    )
