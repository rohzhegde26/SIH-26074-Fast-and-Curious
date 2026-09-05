# PWA map data

Copy Sprint 3's `data/processed/mandya_simplified.topojson` to this directory before
the map test:

```powershell
Copy-Item <sprint-3-repo>\data\processed\mandya_simplified.topojson .\frontend\mandya_simplified.topojson
```

The browser intentionally never uses `mandya_full.geojson`; that file is for backend
zonal aggregation only.
