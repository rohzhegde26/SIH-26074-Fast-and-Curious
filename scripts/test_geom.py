import duckdb
from shapely import wkb, wkt

con = duckdb.connect()
# Load spatial extension if available, or fetch geometry
try:
    con.execute("INSTALL spatial; LOAD spatial;")
    print("DuckDB spatial extension loaded.")
except Exception as e:
    print("Could not load DuckDB spatial extension:", e)

row = con.execute("SELECT gpcode, gpname, ST_AsText(geometry) as wkt_geom FROM 'data/raw/geodata/LGD_panchayats.parquet' WHERE dtname='Mandya' LIMIT 1").fetchone()
print("GP:", row[0], row[1])
print("WKT preview:", str(row[2])[:80])
