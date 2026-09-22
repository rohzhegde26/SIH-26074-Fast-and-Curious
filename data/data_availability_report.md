# Sprint 1 Deliverable: Data Availability, Licensing & Storage Report

**Project**: SIH-26074 Multivariate Spatiotemporal Diffusion Weather Downscaler  
**Audit Executed**: 2026-09-22T15:30:33.236234  
**Mode**: quick  

---

## 1. Provider Protocols, Observed Rate Limits & Request Constraints

| Provider / Archive | Protocol | Observed Latency | Observed Rate Limits / Fair-Use Terms | Recommended Ingestion Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **UCSB CHC (CHIRPS p05)** | HTTPS Direct | ~1.62s per tile | No API key required. High-volume concurrent scraping subject to IP rate throttling. | 2–4 parallel download threads with persistent HTTP session and local file caching. |
| **Open-Meteo (ERA5-Land)** | REST JSON API | ~1.20s per request | Free tier fair use: ~10,000 daily API calls, 1 concurrent connection per client IP. Observed headers: `{}`. | Batch temporal ranges into single multi-year requests; cache hourly NetCDF directly. |
| **NOAA AWS GFS Archive** | S3 / HTTPS Direct | ~3.73s per slice | Public AWS Open Data Registry. Zero egress charges; no API key or AWS credentials required. | Fetch 15 KB `.idx` file first; use HTTP `Range` headers to download only required variables (~2 MB vs 500 MB). |
| **NCAR RDA (ds084.1 GFS)**| HTTPS / OPeNDAP | ~0.45s per index | Free research access. Bulk subsetting requests queue via NCAR RDA batch service. | Pre-stage 2015-2020 GFS cycles via NCAR RDA subsetting API during dataset build phase. |
| **NOAA NCEI (GSOD)** | HTTPS Direct | ~0.40s per station | Public open archive. Fast response on annual CSV downloads (~50 KB per station-year). | Cache station CSVs locally in `data/raw/stations/noaa_gsod/`. |
| **Copernicus (GLO-30)** | S3 / Open Access | N/A (Pre-cached) | Free and open Copernicus WorldCover / DEM policy. | Static mosaic cached in `data/raw/dem/glo30_mandya_terrain.nc`. |

---

## 2. Licensing & Acceptable Use Matrix

| Product | Copyright Holder | License / Terms | Commercial Use? | Attribution Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **CHIRPS v2.0** | UC Santa Barbara Climate Hazards Center | Public Domain / Open Access | Yes | Cite Funk et al. (2015), Scientific Data |
| **ERA5-Land / ERA5** | ECMWF / Copernicus Climate Change Service | Creative Commons Attribution 4.0 (CC-BY-4.0) | Yes | "Generated using Copernicus Climate Change Service information [2026]" |
| **NOAA GFS** | NOAA / National Weather Service | Public Domain (U.S. Federal Government) | Yes | Standard public citation of NCEP/NOAA |
| **NOAA GSOD** | NOAA National Centers for Environmental Information | Public Domain (U.S. Federal Government) | Yes | Standard public citation of NOAA NCEI |
| **Copernicus GLO-30**| European Space Agency (ESA) / Airbus | Copernicus Open Access Policy | Yes | "Copernicus WorldCover / DEM data [2020]" |

---

## 3. Storage Budget & Format Sizing (Core Target $M$ vs. Max Context $2.5M$)

### A. Raw NetCDF & GeoTIFF Storage
- **CHIRPS v2.0 p05 (2014–2023, 3,652 days)**:
  - Bounding box $4^\circ \times 4^\circ$ ($80 \times 80$): ~35 MB total.
  - Full India context $10^\circ \times 10^\circ$ ($200 \times 200$): ~220 MB total.
- **ERA5-Land Hourly Thermodynamics (2014–2023)**:
  - Core domain $16 \times 16$ coarse: ~150 MB (daily aggregated) / ~3.6 GB (hourly raw).
  - Maximum context $40 \times 40$ coarse: ~750 MB (daily aggregated) / ~18 GB (hourly raw).
- **NOAA GFS 0.25° Forecast Slices (2015–2023, 7-day lead sequence)**:
  - Using `.idx` byte-range regional extraction: ~2 MB per daily forecast run $\times$ 3,285 days = ~6.5 GB.
- **Copernicus GLO-30 DSM**:
  - Core target: ~1.2 MB.
  - Maximum context ($10^\circ \times 10^\circ$): ~7.5 MB.

### B. Processed Tensor Cache (Sprint 2 Output Estimate)
- Zarr / NPZ compressed format:
  - 9 years (2015–2023) forecast-conditioned samples: ~3,285 samples $\times$ 180 KB/sample = **~590 MB**.
  - Fits comfortably within local SSD and Kaggle accelerator staging storage.

---

## 4. Zero-Mock Policy & Scientific Error Handling Contract

1. **No Synthetic Fallback**: If an API returns HTTP 404/429/500, or a file is corrupted, the pipeline raises an explicit `DataIngestionError` or `FileNotFoundError`.
2. **Missing Sample Exclusion**: Missing dates are recorded in `data/source_coverage_report.md` and explicitly skipped from training batches rather than filled with synthetic Gaussian noise or linear interpolation.
3. **Audit Verification**: Passing `scripts/audit_sprint1_sources.py` is a mandatory prerequisite for running Sprint 2 dataset builders.
