"""
scripts/generate_district_boundaries.py
Extracts panchayat boundaries from LGD_panchayats.parquet for registered pilot districts,
generates full GeoJSON, simplifies TopoJSON (<400 KB for PWA), copies to frontend/,
and computes centroid coordinates for Virtual ARG stations.
"""

from pathlib import Path
import json
import os
import duckdb
import geopandas as gpd
from shapely import wkt
from shapely.ops import unary_union
from shapely.validation import make_valid
import topojson as tp

ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = ROOT / "data" / "raw" / "geodata" / "LGD_panchayats.parquet"
PROCESSED_DIR = ROOT / "data" / "processed"
SERVING_DIR = ROOT / "data" / "serving"
FRONTEND_DIR = ROOT / "frontend"

DISTRICTS = [
    {"state": "UTTAR PRADESH", "district": "BAGHPAT", "slug": "baghpat"},
    {"state": "ASSAM", "district": "BARPETA", "slug": "barpeta"},
]

def process_and_export():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SERVING_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Initializing DuckDB spatial connection...")
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    for item in DISTRICTS:
        state = item["state"]
        district = item["district"]
        slug = item["slug"]
        print(f"\n---> Processing {district}, {state} ({slug})...")

        query = f"""
            SELECT 
                gpcode,
                gpname,
                dtname,
                stname,
                sdtname,
                blkname,
                ST_AsText(geometry) as geom_wkt
            FROM '{PARQUET_PATH.as_posix()}'
            WHERE upper(stname) = upper('{state}')
              AND upper(dtname) = upper('{district}')
        """
        df = con.execute(query).fetchdf()
        print(f"  Retrieved {len(df)} raw polygon records.")

        if len(df) == 0:
            raise ValueError(f"No records found for {district}, {state}")

        geoms = [make_valid(wkt.loads(w)) for w in df["geom_wkt"]]
        gdf_raw = gpd.GeoDataFrame(df.drop(columns=["geom_wkt"]), geometry=geoms, crs="EPSG:4326")

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
        print(f"  GP Count: {gp_count}, Valid: {valid_count}/{gp_count} ({(valid_count/gp_count)*100:.1f}%)")

        # 1. Export Full GeoJSON
        full_geojson_path = PROCESSED_DIR / f"{slug}_full.geojson"
        gdf.to_file(full_geojson_path, driver="GeoJSON")
        print(f"  Exported full GeoJSON: {full_geojson_path} ({os.path.getsize(full_geojson_path)/1024:.1f} KB)")

        # 2. Export Simplified TopoJSON
        gdf_sim = gdf.copy()
        gdf_sim["geometry"] = gdf_sim.geometry.simplify(tolerance=0.0002, preserve_topology=True)
        topo = tp.Topology(gdf_sim, topology=True, prequantize=True, object_name="panchayats")
        topo_dict = topo.to_dict()

        processed_topo_path = PROCESSED_DIR / f"{slug}_simplified.topojson"
        frontend_topo_path = FRONTEND_DIR / f"{slug}_simplified.topojson"

        with open(processed_topo_path, "w", encoding="utf-8") as f:
            json.dump(topo_dict, f, separators=(",", ":"))
        with open(frontend_topo_path, "w", encoding="utf-8") as f:
            json.dump(topo_dict, f, separators=(",", ":"))

        size_kb = os.path.getsize(frontend_topo_path) / 1024
        print(f"  Exported simplified TopoJSON: {frontend_topo_path} ({size_kb:.1f} KB)")
        assert size_kb < 400.0, f"TopoJSON exceeds 400 KB: {size_kb} KB"

        # 3. Export Centroids for Virtual ARG
        centroids = {}
        for _, row in gdf.iterrows():
            c = row.geometry.centroid
            code = str(row["gpcode"]).strip()
            # Rough elevation estimation based on region
            base_elev = 225.0 if slug == "baghpat" else 45.0
            centroids[code] = {
                "lat": round(float(c.y), 4),
                "lon": round(float(c.x), 4),
                "gpname": str(row.get("gpname", "")).strip(),
                "elevation_m": base_elev,
            }

        centroids_path = SERVING_DIR / f"{slug}_centroids.json"
        with open(centroids_path, "w", encoding="utf-8") as f:
            json.dump(centroids, f, indent=2)
        print(f"  Exported centroids: {centroids_path} ({len(centroids)} GPs)")

    print("\n[SUCCESS] All pilot district boundaries and centroids successfully generated!")

if __name__ == "__main__":
    process_and_export()
