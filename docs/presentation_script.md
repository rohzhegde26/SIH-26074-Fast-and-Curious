# SIH 2026 Finals — 6-Speaker Presentation Choreography Script (PS 26074)
**Project:** Hyper-Local Agrometeorological Downscaling & Panchayat Spatial Intelligence  
**Pilot District:** Mandya, Karnataka (234 active Gram Panchayats in the current Mandya pilot dataset)  
**Team:** Fast and Curious (rohzhegde26/SIH-26074-Fast-and-Curious)

---

### Speaker 1: Problem Framing & National Alignment (0:00 – 0:20)
"Respected judges, current operational agrometeorological feeds provide weather at the 25-kilometer macro-block scale. In a topographically complex district like Mandya, that leaves 30 to 39 Gram Panchayats sharing a single identical forecast number. A rain-shadow valley receives the exact same prediction as an upland ridge, obscuring localized convective bursts and causing catastrophic agricultural losses during sensitive crop phenology windows. Our mission with Problem Statement 26074 is to bridge this spatial gap by delivering 5.5-kilometer, panchayat-level spatial intelligence directly aligned with government disaster-management standards."

---

### Speaker 2: Operational Architecture & Augmentation Positioning (0:20 – 0:40)
"Existing operational GP-level forecast infrastructure, such as Mausamgram and MoPR platforms, establishes the foundational block-level baseline across India. We augment existing operational infrastructure; we do not replace it. Rather than attempting to re-invent numerical weather prediction, our system introduces a learned, physics-constrained spatial refinement layer. We ingest the coarse parent forecast at 0.25-degree resolution and resolve intra-block convective peaks that are otherwise invisible at the regional scale, creating an integration-ready product for state disaster management."

---

### Speaker 3: ML Downscaling Engine & Physical Consistency (0:40 – 1:00)
"Our core machine learning architecture couples learned precipitation downscaling with physics-based thermodynamic refinement for temperature, relative humidity, and surface wind. Crucially, unconstrained neural networks suffer from mass hallucination; to eliminate this, our custom loss function guarantees strict local parent-cell precipitation-volume consistency, ensuring the area-weighted sum of downscaled high-resolution predictions exactly matches the parent-grid coarse input down to 0.000% volume error. For topographic conditioning, our pilot prototype currently utilizes a synthetic demo terrain generator, while operational deployment will ingest authorized spaceborne digital elevation data."

---

### Speaker 4: Multi-Day NWP Engine & Live Ingestion Capabilities (1:00 – 1:20)
"Our inference engine is real-time capable; live operational feed integration is a deployment-stage step requiring authorized government access. Currently, the pilot ingests open operational NWP blends—specifically ECMWF and GFS via Open-Meteo—to produce multi-day forecasts with lead times up to 5 days. We apply psychrometric Magnus relationships and atmospheric lapse rates to refine temperature and humidity across elevations, blending spatial tile seams seamlessly with C¹ Hann windowing."

---

### Speaker 5: Cadastral Intelligence & Virtual ARG Feed (1:20 – 1:40)
"We deliver downscaled predictions for all 234 active Gram Panchayats in the current Mandya pilot dataset, filtered from the 258 source cadastral listing. Notably, 89 of Mandya's 234 Gram Panchayats are disjoint multi-polygons. Our system downscales to each individual cadastral parcel using area-weighted spatial consistency, ensuring no agricultural pocket is overlooked. Every panchayat's forecast is instantly accessible via our WMO/IMD-standard Virtual ARG API, providing a zero-code data stream for e-Governance platforms."

---

### Speaker 6: Rigorous Validation, Uncertainty & Scalability (1:40 – 2:00)
"In rigorous benchmarking, our baseline U-Net benchmark on rain-only input achieved 1.28 mm MAE; the v3.1 iteration adds terrain-conditioned refinement on top. To ensure reliable decision-making, Conformalized Quantile Regression provides calibrated prediction intervals with ≈90% empirical coverage on our held-out 2023 evaluation split. Our architecture is designed for national deployment; Mandya serves as our controlled spatial-validation pilot. We augment existing operational infrastructure; we do not replace it—providing India with a scalable, physics-consistent spatial refinement layer for rural resilience. Thank you."
