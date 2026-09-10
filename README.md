# SIH 2026 — Problem Statement 26074: Downscaling Weather Forecast from Block to Panchayat Level

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Inferring high-resolution plots, data, and information from low-resolution meteorological variables for hyper-local agro-meteorological advisory services.**

---

## 1. Problem Statement Mapping

| Attribute | Specification | Project Implementation |
| :--- | :--- | :--- |
| **Problem ID** | **26074** | Downscaling of weather forecast from Block level to Panchayat level |
| **Ministry / Dept** | **Ministry of Earth Sciences (MoES)** / **India Meteorological Department (IMD)** | Adheres to IMD 0.25° NWP conventions, AWS standard schemas, and DAMU agromet advisory standards |
| **Category & Theme** | **Software** / **Agriculture, FoodTech & Rural Development** | Production-ready FastAPI service + Offline-capable Field PWA + Stage-dependent crop advisory engine |
| **Input Grid** | Block-level forecast ($0.25^\circ \approx 27\text{ km}$) | IMD GFS / NCUM numerical weather prediction grid slices ($16\times 16$) |
| **Output Grid** | Panchayat-level forecast ($0.05^\circ \approx 5.5\text{ km}$) | 5× super-resolution ($80\times 80$) mapped to 234 active Gram Panchayats in the current Mandya pilot dataset with official Census/LGD codes |
| **End Use** | Agro-meteorological advisory services | Multi-variable decision engine (learned precipitation + physics-based thermodynamic refinement) for Paddy, Ragi, and Sugarcane |

---

## 2. Mathematical Formulation

The operational model bridges the $27\text{ km} \to 5.5\text{ km}$ spatial scale through a **5× Super-Resolution U-Net (`UNet5x`)** with Group Normalization ($G=8$) and local parent-grid precipitation-volume consistency.

```
+------------------------------------+          +-----------------------------------------+
|  Coarse IMD Block Grid [B,1,16,16]  |  ----->  |   Encoder-Decoder Backbone with 5x Head |
+------------------------------------+          +-----------------------------------------+
                                                                     |
                                                                     v
                                                +-----------------------------------------+
                                                |   Raw Super-Resolved HR [B, 1, 80, 80]  |
                                                +-----------------------------------------+
                                                |
                                                v
                                                +-----------------------------------------+
                                                |  Local 5x5 Block Mass Renormalization   |
                                                |  0.000% Error in Every 27 km Coarse Cell|
                                                +-----------------------------------------+
                                                |
                                                v
                                                +-----------------------------------------+
                                                | 234 Active Gram Panchayats in Mandya    |
                                                +-----------------------------------------+
```

### Local $5\times 5$ Parent-Cell Precipitation Consistency
Unlike global scaling which moves precipitation across distant macro-boxes, our projection preserves parent-cell precipitation volume **within each individual $27\text{ km}$ grid cell independently**:

$$\hat{y}_{\text{coarse}} = \frac{\text{avg\_pool2d}(y_{\text{HR}} \cdot \cos\phi, k=5, s=5)}{\text{avg\_pool2d}(\cos\phi, k=5, s=5)}$$

$$\text{scale} = \frac{x_{\text{coarse}}}{\text{clamp}(\hat{y}_{\text{coarse}}, \min=10^{-6})}$$

$$y_{\text{conserved}} = y_{\text{HR}} \times \text{repeat\_interleave}(\text{scale}, 5)$$

* **Parent-Cell Consistency Guarantee:** Integrated coarse-cell rainfall equals integrated fine-cell rainfall to $0.000\%$ mathematical precision.
* **Loss Objective:** $\mathcal{L}_{\text{total}} = \mathcal{L}_{1}(\log(1+\hat{y}), \log(1+y)) + 0.1 \cdot \mathcal{L}_{\text{cons}}(\hat{y}, x)$.

---

## 3. Agro-Meteorological Advisory Engine

DAMU (District Agro-Met Unit) and KVK agromet bulletins require multi-variable meteorological context. Our advisory engine couples **5× downscaled precipitation** with **coarse Block NWP thermodynamic variables**:

1. **Parameters Evaluated:**
   - **Precipitation (mm):** 5× Downscaled ($0.05^\circ$ Panchayat-level).
   - **Maximum Temperature (°C):** Block NWP ($0.25^\circ$).
   - **Relative Humidity (%):** Block NWP ($0.25^\circ$).
   - **Wind Speed (km/h):** Block NWP ($0.25^\circ$).

2. **Crop Decision Matrix:**
   - **Paddy (Vegetative / Grain Filling / Harvest):**
     - *Drainage Alert:* Expected Rain $>30\text{ mm}$ $\implies$ "Clear drainage outlets immediately; prevent standing water logging."
     - *Chemical Spray Window:* Rain $<2.5\text{ mm}$ **AND** Wind $<15\text{ km/h}$ **AND** $\text{RH} < 80\%$ $\implies$ "Safe for brown planthopper pesticide application."
     - *Drift Warning:* Wind $\ge 15\text{ km/h}$ $\implies$ "Withhold foliar spray: chemical drift risk exceeds threshold."
   - **Finger Millet (Ragi - Sowing / Tillering / Harvest):**
     - *Harvest Protection:* Expected Rain $>5\text{ mm}$ during harvest $\implies$ "Delay mechanical threshing; cover harvested earheads with tarpaulin (₹6,500/acre rot risk)."
     - *Safe Sowing Window:* Banavasi dry spell ($1.7\text{ mm}$, Wind $8\text{ km/h}$) $\implies$ "Favorable weather: proceed with sowing."
   - **Sugarcane (Grand Growth / Ripening):**
     - *Irrigation Withholding:* Rain $>15\text{ mm}$ $\implies$ "Withhold furrow irrigation; save electrical pumping cost."

---

## 4. Scientific Positioning: Augmenting Operational GP-Level Infrastructure

IMD currently operates the **Gram Panchayat Level Weather Forecast (GPLWF)** via `mausamgram.imd.gov.in`. Our system is designed to integrate into and augment this national infrastructure rather than replace it:

> **Core Positioning:** We augment existing operational infrastructure; we do not replace it.

| Evaluation Dimension | Existing Operational GP-Level Forecast Infrastructure (Mausamgram / MoPR Platforms) | Fast & Curious Refinement Layer (SIH 26074) |
| :--- | :--- | :--- |
| **Base NWP Resolution** | $12\text{ km}$ NCUM (National Centre for Medium Range Weather Forecasting) / $0.25^\circ$ operational grids | $0.25^\circ \approx 27\text{ km}$ operational coarse NWP grid |
| **Spatial Refinement** | Macro parent-cell forecast distribution | **Learned terrain-aware intra-grid refinement** conditioned on local topography |
| **Intra-Block Resolution** | Coarse block-mean or smooth gradient; washes out micro-convective extremes | Resolves sharp local extrema (e.g., **Nalligere $30.4\text{ mm}$** vs. **Banavasi $1.7\text{ mm}$** in the same district block) |
| **Parent-Cell Consistency** | Unconstrained across cell boundaries | **Strict $5\times 5$ Parent-Cell Precipitation Consistency** ($<0.001\%$ parent-cell volume error) |
| **Uncertainty Quantification** | Single deterministic value | **Calibrated uncertainty** via Conformal Quantile Regression (CQR ≈90% coverage) |
| **Delivery & Offline Capability** | Web portal requiring persistent connectivity | **Integration-ready local product**: Dual-mode PWA ($<400\text{ KB}$) with offline store-and-forward capability |
| **Local Agro-Context** | General forecast summaries | Stage-dependent Kannada audio weather summaries for rural field operators |
| **Interoperability** | Web tables / portal displays | **RESTful API + WMO/IMD standard AWS JSON feed** (`/api/v1/panchayat-feed/{lgd}`) |

---

## 5. Quickstart & Installation

### Prerequisites
- Python >= 3.11 (Recommended: Python 3.11 or 3.12)
- Node.js (Optional, for running automated frontend test scripts)

### Installation
```bash
# 1. Clone repository
git clone https://github.com/rohzhegde26/SIH-26074-Fast-and-Curious.git
cd SIH-FINALISTS-2026

# 2. Set up virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Data Setup
Download the LGD Panchayats geospatial boundaries before running the pipeline or GIS-dependent scripts:
```bash
python scripts/download_lgd_data.py
```
*(Fetches `LGD_panchayats.parquet` into `data/raw/geodata/`. Untracked by Git to maintain rapid repo clone times).*

### Running the Live Service
```bash
# Start the FastAPI server with live frontend mounting on port 8000
uvicorn src.api.main:app --reload --port 8000
```
Open your browser at:
- **Interactive Downscaled GIS & Mission Control PWA:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Model Health Check:** [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

### Running Automated Test Suite
```bash
# Run all 18+ automated test suites
pytest
```

---

## 6. Simulation Boundaries & Data Provenance

| Component | Status (TRAINED MODEL / SYNTHETIC HARNESS / STATIC DATA) | Evidence path |
| :--- | :--- | :--- |
| **UNet5x Deep Super-Resolution** | **TRAINED MODEL** | [`models/checkpoints/best_5x_model_v3_1.pt`](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/models/checkpoints/best_5x_model_v3_1.pt) |
| **Mass Conservation Kernel ($L_{\text{cons}}$)** | **TRAINED MODEL** | [`src/losses/conservation.py`](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/src/losses/conservation.py) |
| **Quantile Mapping Calibration** | **TRAINED MODEL** | [`src/eval/calibration.py`](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/src/eval/calibration.py), [`data/static/quantile_mapping_params.json`](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/data/static/quantile_mapping_params.json) |
| **IMD 0.25° Ingestion Engine** | **SYNTHETIC HARNESS** | [`scripts/run_pipeline.py`](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/scripts/run_pipeline.py) (`SyntheticIngestionHarness`), [`scripts/download_imd.py`](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/scripts/download_imd.py) |
| **KMF Dairy Validation Stream** | **SYNTHETIC HARNESS** | [`src/api/feedback_store.py`](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/src/api/feedback_store.py) (seed fixtures migrated to SQLite DB) |
| **Crop Economics & Cost-of-Error** | **STATIC DATA** | [`data/static/crop_economics.json`](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/data/static/crop_economics.json) (cited from UAS Bangalore 2022-2023 extension bulletins) |
| **Terrain Normalization & Centroids** | **STATIC DATA** | [`data/serving/mandya_centroids.json`](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/data/serving/mandya_centroids.json), [`data/static/norm_params.json`](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/data/static/norm_params.json) |

> **Plain Transparency Disclosure:** The UNet5x model, parent-cell precipitation consistency loss, and quantile mapper are trained and real. The inference engine is real-time capable; live operational feed integration is a deployment-stage step requiring authorized government access (the pilot ingests open operational NWP blends via Open-Meteo for multi-day forecasts). Real GLO-30 terrain will be ingested for operational deployment via authorized Copernicus Data Space access; the current demo runtime uses synthetic terrain. Initial KMF Nandini records are development fixtures migrated to SQLite. Crop financial risk numbers are cited estimates from published UAS Bangalore extension packages.

---

## 7. Repository Architecture

```
SIH-FINALISTS-2026/
├── README.md                      # Primary documentation & PS mapping
├── Dockerfile                     # Container deployment specification
├── requirements.txt               # Minimum-version production dependencies
├── conftest.py                    # Pytest harness configuration
├── data/
│   ├── raw/                       # Raw NetCDF/GRIB data + synthetic_terrain.nc (All-India DEM)
│   ├── processed/                 # GeoJSON boundaries (mandya_full, mandya_holdout_buffer)
│   ├── serving/                   # Live serving store (mandya_forecasts.json, mandya_centroids.json)
│   └── static/                    # Static terrain normalization parameters
├── docs/                          # Pitch deck, calibration artifacts, sprint contracts
├── frontend/                      # Dual-Mode PWA (HTML5, Vanilla CSS, JS, Service Worker)
│   ├── index.html                 # Unified single-page cockpit interface
│   ├── app.js                     # PWA client engine, IndexedDB manager, canvas plots
│   ├── styles.css                 # Responsive layout system & dark theme tokens
│   ├── service-worker.js          # Offline cache orchestrator (<400 KB budget)
│   └── audio/                     # Precached Kannada audio advisories
├── models/
│   └── checkpoints/               # Trained model weights (best_5x_model.pt)
├── src/
│   ├── advisory/                  # Agromet rules engine (Paddy, Ragi, Cane, multi-variable)
│   ├── api/                       # FastAPI routes, schemas, forecast repository, inference runner
│   ├── data/                      # NetCDF ingestion, patch extraction, spatial index
│   ├── eval/                      # Quantile mapping, CQR conformal intervals, metrics
│   ├── losses/                    # Parent-cell precipitation consistency & dual-domain loss functions
│   └── models/                    # UNet5x architecture, baselines (DeepSD, Bilinear), training loop
└── tests/                         # Automated unit & integration tests
    ├── api/                       # API endpoint, contract, and payload verification tests
    └── ...                        # Mathematical, GIS, and baseline tests
```

---

## 7. Rural Store-and-Forward SOP (Offline Hardening)

To ensure operational viability across rural Mandya where cellular connectivity is intermittent or absent at village level, the platform implements a **Rural Store-and-Forward Standard Operating Procedure (SOP)**:

| Parameter | Operational Specification |
| :--- | :--- |
| **Sync Node & Operator** | **Taluk RSK Officer / KMF Milk Collection Center Secretary** (sync occurs during daily 06:00–08:30 IST milk drop-off / RSK depot visit). |
| **Payload Size** | **~10 KB gzipped** (all 234 active Gram Panchayats in the current Mandya pilot dataset with 7 lead days, thermodynamic variables, and CQR bounds). |
| **Sync Cadence** | **Daily morning sync** via broadband/4G, stored to local IndexedDB/SQLite on the village device. |
| **Transport Medium** | Mobile PWA, Bluetooth store-and-forward, or USB shuttle between RSK and remote dairy centers. |

### 3-Tier Graceful Degradation Ladder
The frontend and API continuously monitor the forecast cycle age (`cycle_age_days` parsed from `fetched_at_utc`) and visually flag freshness:
1. **Tier 1 — Fresh / Operational ($\le 1$ day, Green Badge `Cycle: <date> (age 0d)`):**
   Full operational status. Day 1 verified downscaled analysis and Days 2–7 operational NWP downscaled guidance active.
2. **Tier 2 — Aging Cache ($2\text{--}3$ days, Amber Badge `Cycle: <date> (age 2d)`):**
   Synoptic multi-day trends remain directionally sound; lookahead chemical washoff hazard warnings flag declining NWP skill.
3. **Tier 3 — Stale Cycle ($> 3$ days, Red Badge `stale cycle: advisories from last sync`):**
   Advisories explicitly marked stale from last sync. Dynamic operational spraying alerts recommend field verification until next KMF/RSK store-and-forward sync.

---

## 8. License & Authors
Developed by **Fast and Curious** for **Smart India Hackathon (SIH) 2026** under the auspices of the **Ministry of Earth Sciences (MoES)** and **India Meteorological Department (IMD)**.
