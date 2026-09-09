# SIH 2026 — Problem Statement 26074: Downscaling Weather Forecast from Block to Panchayat Level

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
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
| **Output Grid** | Panchayat-level forecast ($0.05^\circ \approx 5.5\text{ km}$) | 5× super-resolution ($80\times 80$) mapped to 234 Mandya Gram Panchayats with official Census/LGD codes |
| **End Use** | Agro-meteorological advisory services | Multi-variable decision engine (Rainfall + Temp + RH + Wind) for Paddy, Ragi, and Sugarcane |

---

## 2. Mathematical Formulation

The operational model bridges the $27\text{ km} \to 5.5\text{ km}$ spatial scale through a **5× Super-Resolution U-Net (`UNet5x`)** with Group Normalization ($G=8$) and exact local mass conservation.

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
                                                |  234 Gram Panchayat Forecasts with LGD  |
                                                +-----------------------------------------+
```

### Local $5\times 5$ Block Mass Conservation
Unlike global scaling which moves precipitation across distant macro-boxes, our projection conserves atmospheric water volume **within each individual $27\text{ km}$ grid cell independently**:

$$\hat{y}_{\text{coarse}} = \frac{\text{avg\_pool2d}(y_{\text{HR}} \cdot \cos\phi, k=5, s=5)}{\text{avg\_pool2d}(\cos\phi, k=5, s=5)}$$

$$\text{scale} = \frac{x_{\text{coarse}}}{\text{clamp}(\hat{y}_{\text{coarse}}, \min=10^{-6})}$$

$$y_{\text{conserved}} = y_{\text{HR}} \times \text{repeat\_interleave}(\text{scale}, 5)$$

* **Mass Conservation Guarantee:** Integrated coarse-cell rainfall equals integrated fine-cell rainfall to $0.000\%$ mathematical precision.
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

## 4. Scientific Comparison: Our System vs. IMD Mausamgram (GPLWF)

IMD currently operates the **Gram Panchayat Level Weather Forecast (GPLWF)** via `mausamgram.imd.gov.in`. Below is the architectural comparison:

| Evaluation Dimension | IMD Mausamgram GPLWF (Operational) | Fast & Curious Solution (SIH 26074) |
| :--- | :--- | :--- |
| **Base NWP Resolution** | $12\text{ km}$ NCUM (National Centre for Medium Range Weather Forecasting) | $0.25^\circ \approx 27\text{ km}$ IMD GFS / NCUM coarse grid |
| **Downscaling Approach** | Bilinear interpolation to GP centroid point coordinates | **5× Deep Super-Resolution U-Net (`UNet5x`)** conditioned on local topography |
| **Intra-Block Resolution** | Smooth gradient; washes out micro-convective cloudbursts | Resolves sharp local extrema (e.g., **Nalligere $30.4\text{ mm}$** vs. **Banavasi $1.7\text{ mm}$** in the same district block) |
| **Physical Conservation** | Unconstrained interpolation; violates integrated atmospheric water mass | **Strict $5\times 5$ Block Mass Conservation** ($<0.001\%$ mass discrepancy) |
| **Delivery & Offline Capability** | Web portal requiring persistent 4G connectivity ($>3\text{ MB}$ payload) | **Dual-Mode PWA ($<400\text{ KB}$)** with 100% offline Service Worker & local IndexedDB storage |
| **Local Agro-Context** | Generic text tables | Stage-dependent Kannada audio advisories for rural intermediaries & farmers |
| **Interoperability** | HTML tables | **RESTful API + WMO/IMD standard AWS JSON feed** (`/api/v1/panchayat-feed/{lgd}`) |

---

## 5. Quickstart & Installation

### Prerequisites
- Python 3.10+ (Recommended: Python 3.11 or 3.12)
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

### Running the Live Service
```bash
# Start the FastAPI server with live frontend mounting on port 8000
uvicorn src.api.main:app --reload --port 8000
```
Open your browser at:
- **Village Cockpit & Mission Control PWA:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Model Health Check:** [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

### Running Automated Test Suite
```bash
# Run all 18+ automated test suites
pytest
```

---

## 6. Repository Architecture

```
SIH-FINALISTS-2026/
├── README.md                      # Primary documentation & PS mapping
├── Dockerfile                     # Container deployment specification
├── requirements.txt               # Pinned Python package dependencies
├── conftest.py                    # Pytest harness configuration
├── data/
│   ├── raw/                       # Raw NetCDF/GRIB data + glo30_terrain.nc (All-India DEM)
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
│   ├── losses/                    # Physical mass conservation & dual-domain loss functions
│   └── models/                    # UNet5x architecture, baselines (DeepSD, Bilinear), training loop
└── tests/                         # Automated unit & integration tests
    ├── api/                       # API endpoint, contract, and payload verification tests
    └── ...                        # Mathematical, GIS, and baseline tests
```

---

## 7. License & Authors
Developed by **Fast and Curious** for **Smart India Hackathon (SIH) 2026** under the auspices of the **Ministry of Earth Sciences (MoES)** and **India Meteorological Department (IMD)**.
