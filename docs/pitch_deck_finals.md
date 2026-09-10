# SIH 2026 Finals — Pitch Deck & Stage Defense Master (PS 26074)
**Project:** Hyper-Local Agrometeorological Downscaling & Panchayat Spatial Intelligence<br/>
**Target District:** Mandya, Karnataka (234 active Gram Panchayats in the current Mandya pilot dataset)<br/>
**Team:** Fast and Curious (rohzhegde26/SIH-26074-Fast-and-Curious)

---

## 1. 12-Slide Final Presentation Deck

### Slide 1: Title & National Alignment
* **Title:** Hyper-Local Panchayat Weather Downscaling & High-Resolution Agromet Forecasts
* **Problem Statement:** SIH PS 26074 (Theme: Agriculture, FoodTech & Rural Development)
* **Tagline:** Turning 25 km coarse weather grids into 5.5 km panchayat-level spatial intelligence through existing rural operational infrastructure.

### Slide 2: The Core Problem: The 0.25° Resolution Blindspot
* IMD 0.25° grid (~752 km²) averages out all orographic and micro-climatic variation across 30–39 Gram Panchayats in Mandya.
* A smallholder in a rain-shadow valley gets the exact same forecast as an upland ridge location.
* Coarse forecasts lack phenological context: 15 mm of rain during vegetative growth is beneficial; 15 mm during harvest is ₹6,500/acre catastrophic crop rot.

### Slide 3: Meteorological Downscaling & Multi-Variable Agromet Advisory
* **The 0.25° Blindspot:** IMD 0.25° grid (~752 km²) averages out orographic and micro-climatic variation across 30–39 Gram Panchayats.
* A smallholder in a rain-shadow valley gets the exact same forecast as an upland ridge location.
* **Multi-Variable Agromet Context:** Coupled learned precipitation downscaling with physics-based thermodynamic refinement (Temp, RH, Wind Speed).
* **Crop Phenology Rules:**
  - 15 mm rain during vegetative growth = beneficial; 15 mm during harvest = ₹6,500/acre catastrophic rot.
  - Safe spray window requires: Rain < 2.5 mm AND Wind < 15 km/h AND RH < 80%.

### Slide 4: Data Pipeline & Spatial Integrity
* **Regional Training:** Peninsular India (Western Ghats & Southern Plateau) with spatial holdout validation on Mandya (2010–2023 daily monsoon data, land filter $\ge 70\%$).
* **Topography Conditioning:** Terrain-conditioned downscaling (elevation, slope, aspect, and curvature; pilot prototype utilizes synthetic terrain generator; operational deployment will ingest authorized spaceborne DEM).
* **Zero-Shot Spatial Holdout:** Mandya district $+ 0.5^\circ$ buffer (~50–100 km) strictly held out from training (`patch_index ∩ buffer == ∅`). Architecture designed for national deployment; Mandya is the controlled spatial-validation pilot.

### Slide 5: Physics & Local Parent-Cell Precipitation Consistency
* **Strict Local 5×5 Parent-Cell Precipitation Consistency ($L_{\text{cons}}$):** Area-weighted kernel=5 average-pooling preserves parent-cell precipitation volume cell-by-cell ($0.000\%$ local parent-grid precipitation-volume error vs $11.2\%$ hallucination in our unconstrained baseline super-resolution).
* **Grid Registration Realignment:** Fixed the critical 2.7 km half-pixel center-vs-corner coordinate offset.
* **On-Demand Live Inference:** `/api/v1/infer` runs 5× downscaling on arbitrary 16×16 coarse inputs in <50 ms on CPU.

### Slide 6: Refinement Layer vs. Existing Operational Forecasts
* **Core Positioning:** We augment existing operational infrastructure; we do not replace it.

| Dimension | Existing operational GP-level forecast infrastructure (Mausamgram / MoPR platforms) | Our refinement layer: learned terrain-aware intra-grid refinement + parent-cell precipitation consistency + calibrated uncertainty + integration-ready local product |
| :--- | :--- | :--- |
| **Spatial Granularity** | Coarse block-mean (0.25° / ~28 km) | **5× Deep Super-Resolution U-Net (`UNet5x`) to 0.05° (~5.5 km)** |
| **Intra-Block Resolution** | Single uniform value across block; blind to convective micro-bursts | Resolves **Nalligere 30.4 mm** convective peak vs **Banavasi 1.7 mm** in same block |
| **Precipitation Consistency** | Unconstrained across sub-grid cells | **Local parent-grid precipitation-volume consistency ($L_{\text{cons}}$)** |
| **Delivery Payload** | Web portal requiring persistent connectivity | **Dual-Mode PWA (<400 KB)** with offline Service Worker support |

### Slide 7: High-Resolution Panchayat Mapping & Cadastral Localization
* **Localized Spatial Intelligence:** Provides interactive 5.5 km choropleth mapping across all 234 active Gram Panchayats in the current Mandya pilot dataset.
* **Cadastral Multi-Polygon Handling:** Resolves disjoint cadastral parcels (89 multi-polygon GPs in Mandya) with area-weighted spatial consistency.
* **Reusable Refinement Layer:** Serves as an integration-ready, downstream-compatible data layer for agricultural extension planning and disaster management.
* **Future Roadmap (Planned Progression):** Role-adapted operational dashboards for village intermediaries and field workers are planned for deployment-stage integration.

### Slide 8: Ground Validation & Continuous Feedback Architecture
* **Addressing Gauge Sparsity:** Bridges the acute sparsity of physical rain gauges (Mandya has only 2 official IMD ARGs across 234 active Gram Panchayats in the current Mandya pilot dataset).
* **Crowdsourced Sensor Network Design:** Architecture designed to ingest binary ground confirmations from local touchpoints to flag divergent micro-clusters.
* **Closed-Loop API:** `/api/v1/validation/nandini` logs ground agreement for continuous quantile recalibration.

### Slide 9: 100% Offline Resilience (Airplane-Mode Ready)
* **Offline Audio Precache:** 7 canonical Mandya Kannada audio advisory files precached in Service Worker (`STATIC_ASSETS`, <500 KB total).
* **Offline Dispatch Queue:** When network drops, dispatch queues into IndexedDB, audio plays locally, and fallback template unfolds.
* **Truthful UI Diagnostics:** Banner states objective facts: *"⚠️ No network — operating on cached forecast. Offline map & dispatch queue active."*

### Slide 10: High-Resolution Panchayat Forecast Feed (Disaster-Management Schema)
* Downscaled 0.05° precipitation forecast feed for all 234 active Gram Panchayats in the current Mandya pilot dataset via `/api/v1/panchayat-feed/{lgd_code}`.
* Standard WMO/IMD-compatible GeoJSON/OpenAPI schema for state disaster-management integration and zero-code e-Governance.

### Slide 11: Scientific Honesty & The 11 Forbidden Claims
* Explicit transparency on operational boundaries:
  - Panchayat-scale resolution ceiling (1 pixel $\approx$ 1 GP / 19.2 km²).
  - Supervised downscaling model; the inference engine is real-time capable; live operational feed integration is a deployment-stage step requiring authorized government access (pilot currently ingests open operational NWP blends via Open-Meteo).
  - Architecture designed for national deployment; Mandya is the controlled spatial-validation pilot.
  - 100% adherence to all 11 scientific boundaries.

### Slide 12: Impact, Scalability & Roadmap
* **Financial Impact:** Prevents ₹1,800/acre fertilizer wash-off and ₹6,500/acre grain harvest rot across 234 active Gram Panchayats in the current Mandya pilot dataset.
* **Statewide Deployment:** Easily extensible across Karnataka's 6,000+ KMF milk cooperatives with zero additional hardware cost.

---

## 2. Simulation Boundaries & Data Provenance

| Component | Status (TRAINED MODEL / SYNTHETIC HARNESS / STATIC DATA) | Evidence path |
| :--- | :--- | :--- |
| **UNet5x Deep Super-Resolution** | **TRAINED MODEL** | Baseline U-Net benchmark (rain-only input); v3.1 adds terrain-conditioned refinement on top (`models/checkpoints/best_5x_model_v3_1.pt`) |
| **Parent-Cell Consistency Kernel ($L_{\text{cons}}$)** | **TRAINED MODEL** | `src/losses/conservation.py` |
| **Quantile Mapping Calibration** | **TRAINED MODEL** | ≈90% empirical coverage (held-out 2023 split; artifact: `docs/cqr_coverage.png`) (`src/eval/calibration.py`, `data/static/quantile_mapping_params.json`) |
| **Operational Data Ingestion Engine** | **OPEN NWP / EMULATION** | Real-time capable inference engine; open operational NWP blends (Open-Meteo) ingested; live IMD/NCUM integration is deployment-stage requiring authorized access (`scripts/run_pipeline.py`) |
| **KMF Dairy Validation Stream** | **SYNTHETIC HARNESS** | `src/api/feedback_store.py` (seed fixtures migrated to SQLite DB) |
| **Crop Economics & Cost-of-Error** | **STATIC DATA** | `data/static/crop_economics.json` (cited from UAS Bangalore 2022-2023 extension bulletins) |
| **Terrain Normalization & Centroids** | **STATIC DATA** | `data/serving/mandya_centroids.json`, `data/static/norm_params.json` |

> **Plain Transparency Disclosure:** The UNet5x model, parent-cell precipitation consistency loss, and quantile mapper are trained and real. The inference engine is real-time capable; live operational feed integration is a deployment-stage step requiring authorized government access (the pilot currently ingests open operational NWP blends via Open-Meteo for multi-day forecasts). Initial KMF Nandini records are development fixtures migrated to SQLite. Crop financial risk numbers are cited estimates from published UAS Bangalore extension packages.

---

## 3. 90-Second Stage Demonstration Script

* **[0:00 - 0:20] The Operational Gap & Framing:**
  "Judges, current operational GP forecast feeds provide a single uniform value across 25 km blocks. In Mandya, that leaves all 234 active Gram Panchayats with identical rainfall predictions, completely blind to convective micro-bursts and orographic rain-shadows. We do not replace existing operational forecast infrastructure; we augment it with a learned, mass-consistent spatial refinement layer."

* **[0:20 - 0:45] The Downscaling Proof & Side-by-Side Audit:**
  *(Presenter opens the Side-by-Side Spatial Resolution Audit)*
  "Look at this audit modal: on the left is the coarse parent block forecast—a flat 7.9 mm across the entire taluk. On the right is our 5× downscaled product at 5.5 km resolution. Notice how it captures the localized convective peak of 30.4 mm in Nalligere while simultaneously preserving local parent-grid precipitation-volume consistency down to 0.000% volume error."

* **[0:45 - 1:10] Multi-Day NWP Engine & Offline Resilience:**
  *(Presenter toggles day timeline and activates Airplane Mode)*
  "Our inference engine couples learned precipitation downscaling with physics-based thermodynamic refinement for temperature, humidity, and wind. When network connectivity drops, our banner transparently informs the user: *'⚠️ No network — operating on cached forecast.'* The cached choropleth, audio summaries, and offline dispatch queue continue operating seamlessly."

* **[1:10 - 1:30] Cadastral Precision & Virtual ARG Feed:**
  *(Presenter clicks a multi-polygon GP like Nalligere and inspects the Virtual ARG JSON)*
  "In Mandya, 89 of 234 Gram Panchayats are disjoint multi-polygon parcels. Our cadastral matrix downscales to each constituent parcel with strict area-weighted precipitation consistency. Finally, via `/api/v1/panchayat-feed/{lgd_code}`, we expose an IMD AWS-standard Virtual ARG data stream for all 234 active Gram Panchayats, ready for zero-code integration into state disaster management systems."
