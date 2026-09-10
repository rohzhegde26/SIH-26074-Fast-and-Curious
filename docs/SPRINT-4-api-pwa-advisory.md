# Sprint 4 — Days 7–8: API Services, Offline PWA & Bilingual Agro-Advisory

**Reference Document:** [SIH26074-Sprint-Plan-FINAL-V2.md](file:///c:/Users/Rohith%20P%20Hegde/Desktop/SIH-FINALISTS-2026/sprint%20plan/SIH26074-Sprint-Plan-FINAL-V2.md)  
**Timeline:** Days 7–8  
**Status:** Commit-Ready (FINAL V2)

---

## 1. Goal
Deliver an integration-ready FastAPI backend serving calibrated forecasts and CQR uncertainty intervals, a lightweight mobile-first Progressive Web App (PWA) supporting offline viewing with an explicit airplane-mode banner, and an automated rule-based agro-advisory engine in Kannada and English tailored to Mandya's primary crops (Ragi and Paddy).

---

## 2. Team Responsibilities & Ownership

| Role | Sprint 4 Deliverables |
|---|---|
| **Backend Engineer** | Implement FastAPI application in `src/api/`, connect quantile mapping and CQR inference pipelines, implement `/api/forecast/{lgd_code}` and `/api/egramswaraj/mock` (strictly labeled as mock), export OpenAPI documentation. |
| **Frontend/PWA Engineer** | Build mobile-responsive frontend in `frontend/`, implement Service Worker caching (Network-First for forecast data, Cache-First for map tiles), configure IndexedDB persistence, build airplane-mode offline detection banner, integrate Mandya panchayat choropleth map. |
| **Domain/Product Lead** | Author bilingual agricultural decision rules in `src/advisory/` for Ragi and Paddy across phenological stages, verify Kannada translation naturalness and clarity for farmers. |
| **Data/GIS Lead** | Export optimized GeoJSON/TopoJSON for Mandya's panchayats (258 in source cadastral listing; 234 active in current Mandya pilot dataset) for client-side rendering in the PWA. |

---

## 3. Pinned Decisions & Operational Boundaries

* **Offline Claim Boundary:**
  - PWA provides **"Offline viewing of last-synced forecast"**.
  - **Never** claim or display "offline forecast generation" (a client browser cannot run downscaling models without coarse numerical input grids).
  - When disconnected from the internet, the application must display a prominent top banner:  
    `"Viewing cached forecast from {timestamp}. Offline mode active."`
* **Government Integration Boundary:**
  - State: *"Integration-ready API layer with mock interface"*.
  - **Never** claim write access or direct push to government portals (e.g., e-GramSwaraj).
  - The endpoint `/api/egramswaraj/mock` must be explicitly tagged as `[MOCK / INTEGRATION PROTOTYPE]` in source code comments, OpenAPI docstrings, and response headers.
* **Crop Scope:**
  - Restrict strictly to **2 crops: Ragi (Finger Millet) and Paddy (Rice)**, matching Mandya's dominant cropping pattern from the RDPR agricultural calendar.
* **Timebox Safety Rule:**
  - If service worker background synchronization exceeds a 1-day implementation budget, immediately pivot to a standard cache-fallback strategy using `localStorage` / `IndexedDB` to ensure delivery.

---

## 4. Sprint 4 Detailed Tasks

### A. Backend API Implementation (`src/api/main.py`)
- [ ] **FastAPI Application Setup:**
  - CORS middleware configured for PWA origin.
  - Pydantic response schemas for panchayat forecast and uncertainty bounds.
- [ ] **Core Endpoint `/api/forecast/{lgd_code}`:**
  - Accepts LGD panchayat code.
  - Queries latest downscaled, QM-calibrated precipitation for the polygon.
  - Retrieves pre-computed CQR 90% confidence bounds ($[q_{\text{min}}, q_{\text{max}}]$).
  - Triggers advisory engine to append stage-specific farming advice.
  - Response format:
    ```json
    {
      "lgd_code": 218542,
      "panchayat_name": "Keregodu",
      "district": "MANDYA",
      "forecast_date": "2026-09-04",
      "timestamp_utc": "2026-09-03T18:00:00Z",
      "rainfall_mm": {
        "expected": 14.2,
        "likely_min": 8.5,
        "likely_max": 22.1,
        "empirical_coverage": "90% calibrated (test 2023)"
      },
      "advisory": {
        "ragi": {
          "stage": "Vegetative",
          "action_en": "Postpone pesticide spraying; moderate rain expected.",
          "action_kn": "ಕೀಟನಾಶಕ ಸಿಂಪರಣೆಯನ್ನು ಮುಂದೂಡಿ; ಸಾಧಾರಣ ಮಳೆಯಾಗುವ ಸಾಧ್ಯತೆಯಿದೆ."
        },
        "paddy": {
          "stage": "Transplanting",
          "action_en": "Ensure drainage channels are clear to prevent waterlogging.",
          "action_kn": "ಹೆಚ್ಚುವರಿ ನೀರು ನಿಲ್ಲದಂತೆ ಕಾಲುವೆಗಳನ್ನು ಸ್ವಚ್ಛಗೊಳಿಸಿ."
        }
      },
      "is_cached": false
    }
    ```
- [ ] **Mock Government Interface `/api/egramswaraj/mock`:**
  - Demonstrates how panchayat secretary dashboards can ingest localized weather data.
  - Header explicitly set: `X-Integration-Status: Mock-Prototype`.
  - Swagger/OpenAPI documentation clearly describes it as a mock contract.

### B. Bilingual Agro-Advisory Engine (`src/advisory/engine.py`)
- [ ] Implement rule matrices based on:
  - 24-hour precipitation threshold ($< 2.5\text{ mm}$ dry, $2.5\text{--}15.5\text{ mm}$ light, $15.5\text{--}64.4\text{ mm}$ moderate, $> 64.5\text{ mm}$ heavy).
  - Extreme lower/upper CQR uncertainty bounds.
  - Crop stage (Sowing, Vegetative, Flowering, Harvest).
- [ ] Curate and verify English and Kannada advisory strings for:
  - Fertilizer application (skip if rain $> 10\text{ mm}$).
  - Pesticide spraying (requires dry window $> 24\text{ hours}$).
  - Irrigation scheduling (skip if likely rain $> 5\text{ mm}$).
  - Harvest and drying precautions.

### C. Frontend Progressive Web App (`frontend/`)
### C. Frontend Progressive Web App (`frontend/`)
- [ ] **Interactive Panchayat Map Interface:**
  - Render Mandya's panchayat polygons (258 in source cadastral listing; 234 active in current Mandya pilot dataset) using Leaflet / MapLibre, loading strictly from `data/processed/mandya_simplified.topojson` (<400 KB payload) for instant rural rendering and zero lag.
  - Color choropleth reflecting expected rainfall intensity.
  - Tap/click polygon to view localized forecast, CQR uncertainty card, and bilingual advisory toggle.
- [ ] **PWA Service Worker & Offline Storage:**
  - Register Service Worker with two caching strategies:
    - **Cache-First:** Static assets, map styling, base vector tiles.
    - **Network-First with IndexedDB Fallback:** `/api/forecast/*` data.
  - On first sync, cache all 234 active Gram Panchayats in the current Mandya pilot dataset (from 258 source cadastral listing) for offline use.
- [ ] **Offline Airplane-Mode Indicator:**
  - Listen to `window.addEventListener('online')` and `'offline'`.
  - When offline: display warning banner:  
    `"⚠️ Offline: Viewing cached forecast from [Timestamp]. Local generation not supported."`
  - Ensure full UI, map navigation, and cached advisories remain interactive in offline mode.

### D. Single-Command Demo Orchestrator (`scripts/run_pipeline.py`)
- [ ] **Implement Thin Orchestrator Script (Timeboxed to $\le 2$ hours):**
  - CLI invocation: `python scripts/run_pipeline.py --date 2023-07-15 --district MANDYA`
  - Acts strictly as a lightweight glue script calling established modules without duplicating logic:
    1. Calls `src/data/loaders.py` to retrieve the day's LR input grid.
    2. Calls `src/models/unet_5x.py` to produce raw 5× downscaled output.
    3. Calls `src/eval/calibration.py` to apply per-$0.25^\circ$ quantile mapping.
    4. Calls `src/eval/cqr.py` to evaluate prediction intervals.
    5. Calls `src/data/zonal_aggregation.py` using `data/processed/mandya_full.geojson` to produce the final GeoJSON payload.
  - Eliminates manual terminal friction and guarantees a flawless 1-line execution on stage during the jury demo.

---

## 5. Verification Gates & Definition of Done

- [ ] FastAPI backend starts with `uvicorn src.api.main:app` and passes OpenAPI validation.
- [ ] `/api/forecast/{lgd_code}` responds in $< 100\text{ ms}$ with valid JSON containing expected rainfall, calibrated CQR bounds, and Kannada/English advisories.
- [ ] `/api/egramswaraj/mock` is documented and labeled as a mock prototype in Swagger UI.
- [ ] `scripts/run_pipeline.py` runs end-to-end in $< 5\text{ seconds}$ on a single date, generating valid Mandya panchayat GeoJSON.
- [ ] PWA passes browser Lighthouse PWA audit (installable, service worker registered, offline-capable).
- [ ] PWA loads Mandya vector boundaries from `mandya_simplified.topojson` with payload size $< 400\text{ KB}$.
- [ ] Airplane-mode live test:
  1. Open PWA, load Mandya map.
  2. Disconnect Wi-Fi / enable airplane mode.
  3. Verify offline warning banner immediately appears with sync timestamp.
  4. Verify user can click and view cached forecasts and advisories across Mandya panchayats.
- [ ] Advisory content verified for both Ragi and Paddy in Kannada and English.
