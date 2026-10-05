"""Area weights mapping each Mandya gram panchayat (GP) polygon onto the v3 model's 0.05 deg fine grid.

v3 fine grid: 80 x 80 cells, centres 14.975 .. 11.025 N (row 0 = north) x 74.025 .. 77.975 E (col 0 = west).
Each GP gets the list of (row, col, weight) of the cells its polygon overlaps, weights = overlap area share
(GPs with several polygons are merged). Written once to demo/data/serving/gp_cells.json.

  python demo/backend/gp_weights.py
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

DEMO = Path(__file__).resolve().parents[1]
RES, LAT0, LON0, N = 0.05, 15.0, 74.0, 80   # cell (r, c) spans lat [15 - 0.05(r+1), 15 - 0.05r], lon [74 + 0.05c, ...]


def cell_box(r: int, c: int):
    return box(LON0 + RES * c, LAT0 - RES * (r + 1), LON0 + RES * (c + 1), LAT0 - RES * r)


def main():
    g = gpd.read_file(DEMO / "data" / "geo" / "mandya_full.geojson")
    g = g[g["gpcode"].astype(str).str.len() > 0]
    out = {}
    for code, grp in g.groupby("gpcode"):
        poly = unary_union(list(grp.geometry))
        minx, miny, maxx, maxy = poly.bounds
        r0, r1 = int((LAT0 - maxy) // RES), int((LAT0 - miny) // RES)
        c0, c1 = int((minx - LON0) // RES), int((maxx - LON0) // RES)
        cells = []
        for r in range(max(r0, 0), min(r1, N - 1) + 1):
            for c in range(max(c0, 0), min(c1, N - 1) + 1):
                a = poly.intersection(cell_box(r, c)).area
                if a > 0:
                    cells.append([r, c, a])
        tot = sum(x[2] for x in cells)
        cells = [[r, c, round(a / tot, 5)] for r, c, a in cells]
        ctr = poly.representative_point()
        out[str(code)] = {"name": str(grp.iloc[0]["gpname"]), "taluk": str(grp.iloc[0]["sdtname"]), "n_parts": int(len(grp)),
                          "lat": round(ctr.y, 4), "lon": round(ctr.x, 4), "area_km2": round(poly.area * 111.0 * 111.0 * np.cos(np.radians(ctr.y)), 1),
                          "cells": cells}
    path = DEMO / "data" / "serving" / "gp_cells.json"
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    n = np.array([len(v["cells"]) for v in out.values()])
    print(f"{len(out)} GPs -> {path.name}; cells per GP min {n.min()} median {np.median(n):.0f} max {n.max()}; "
          f"rows {min(c[0] for v in out.values() for c in v['cells'])}-{max(c[0] for v in out.values() for c in v['cells'])}, "
          f"cols {min(c[1] for v in out.values() for c in v['cells'])}-{max(c[1] for v in out.values() for c in v['cells'])}")


if __name__ == "__main__":
    main()
