# SIH 2026 Finals — Pitch Deck & Stage Defense Master (PS 26074)
**Project:** Hyper-Local Agrometeorological Downscaling & Panchayat Intermediary Cockpit  
**Target District:** Mandya, Karnataka (234 Gram Panchayats)  
**Team:** Fast and Curious (rohzhegde26/SIH-26074-Fast-and-Curious)

---

## 1. 12-Slide Final Presentation Deck

### Slide 1: Title & National Alignment
* **Title:** Hyper-Local Panchayat Weather Downscaling & Village Intermediary Advisory
* **Problem Statement:** SIH PS 26074 (Theme: Agriculture, FoodTech & Rural Development)
* **Tagline:** Turning 25 km coarse weather grids into 5 km panchayat action through existing rural human infrastructure.

### Slide 2: The Core Problem: The 0.25° Resolution Blindspot
* IMD 0.25° grid (~752 km²) averages out all orographic and micro-climatic variation across 30–39 Gram Panchayats in Mandya.
* A smallholder in a rain-shadow valley gets the exact same forecast as an upland ridge farmer.
* Coarse forecasts lack phenological context: 15 mm of rain during vegetative growth is beneficial; 15 mm during harvest is ₹6,500/acre catastrophic crop rot.

### Slide 3: Meteorological Downscaling & Multi-Variable Agromet Advisory
* **The 0.25° Blindspot:** IMD 0.25° grid (~752 km²) averages out orographic and micro-climatic variation across 30–39 Gram Panchayats.
* A smallholder in a rain-shadow valley gets the exact same forecast as an upland ridge farmer.
* **Multi-Variable Agromet Context:** Coupled downscaled precipitation with Block NWP thermodynamics (Temp, RH, Wind Speed).
* **Crop Phenology Rules:**
  - 15 mm rain during vegetative growth = beneficial; 15 mm during harvest = ₹6,500/acre catastrophic rot.
  - Safe spray window requires: Rain < 2.5 mm AND Wind < 15 km/h AND RH < 80%.

### Slide 4: Data Pipeline & Spatial Integrity
* **All-India Training:** 2010–2023 daily monsoon data (1,708 days, 240k land patches $\ge 70\%$ land filter).
* **Topography Conditioning:** Copernicus GLO-30 DEM, slope, aspect, and curvature.
* **Zero-Shot Spatial Holdout:** Mandya district $+ 0.5^\circ$ buffer (~50–100 km) strictly held out from training (`patch_index ∩ buffer == ∅`). Proven generalization to unseen terrain.

### Slide 5: Physics & Local Mass Conservation
* **Strict Local 5×5 Mass Conservation ($L_{\text{cons}}$):** Area-weighted kernel=5 average-pooling preserves integrated precipitation volume cell-by-cell ($0.000\%$ local block mass error vs $11.2\%$ hallucination in vanilla super-resolution).
* **Grid Registration Realignment:** Fixed the critical 2.7 km half-pixel center-vs-corner coordinate offset.
* **On-Demand Live Inference:** `/api/v1/infer` runs 5× downscaling on arbitrary 16×16 coarse inputs in <50 ms on CPU.

### Slide 6: Scientific Comparison vs. IMD Mausamgram (GPLWF)
| Dimension | IMD Mausamgram GPLWF (Operational) | Fast & Curious Solution (SIH 26074) |
| :--- | :--- | :--- |
| **Downscaling Approach** | Bilinear interpolation from 12 km NCUM | **5× Deep Super-Resolution U-Net (`UNet5x`)** |
| **Intra-Block Resolution** | Smooth gradient; misses micro-convective events | Resolves **Nalligere 30.4 mm** vs **Banavasi 1.7 mm** in same block |
| **Mass Conservation** | None (integrated water mass violated) | **0.000% Local Block Mass Conservation** |
| **Delivery Payload** | Web portal requiring persistent 4G (>3 MB) | **Dual-Mode PWA (<400 KB)** with 100% offline Service Worker |

### Slide 7: The Intermediary Cockpit (Dynamic Role Reordering)
* **Democratized Cockpit:** Role toggle reorders cards dynamically so each operator sees their exact two-second routine first:
  - **RSK Officer:** Crop Growth Stage + Multi-Variable Spray Drift & Leaching Risk.
  - **GP Secretary:** A4 Print Notice / Chalkboard Slate + Panchayat Forecast Feed API.
  - **Lead Farmer:** High-contrast today/tomorrow field action notice (Kannada audio, zero jargon).
  - **Dairy Secretary:** 2-Tap ground-truth confirmation during morning milk intake.
* **Frictionless Routine:** Choice persisted in `localStorage` — boots instantly in field conditions.

### Slide 8: Ground Validation & Continuous Feedback Loop
* **Human Sensor Network:** Solves the acute sparsity of physical rain gauges (Mandya has only 2 official IMD ARGs across 234 GPs).
* **2-Second Ground Confirmation:** Secretary taps "🟢 ಹೌದು (Yes)" or "🔴 ಇಲ್ಲ (No)" while recording milk intake.
* **Closed-Loop API:** `/api/v1/validation/nandini` logs ground agreement and flags divergent micro-clusters for continuous quantile recalibration.

### Slide 9: 100% Offline Resilience (Airplane-Mode Ready)
* **Offline Audio Precache:** 7 canonical Mandya Kannada audio advisory files precached in Service Worker (`STATIC_ASSETS`, <500 KB total).
* **Offline Dispatch Queue:** When network drops, WhatsApp dispatch queues into IndexedDB, audio plays locally, and chalkboard template unfolds.
* **Truthful UI Diagnostics:** Banner states objective facts: *"⚠️ No network — operating on cached 06:00 IST advisory. Chalkboard & dispatch queue active."*

### Slide 10: High-Resolution Panchayat Forecast Feed (IMD AWS Schema)
* Downscaled 0.05° precipitation forecast feed for all 234 panchayats via `/api/v1/panchayat-feed/{lgd_code}`.
* Exact JSON compliance with IMD AWS standard schemas for zero-code e-Governance and State Disaster Management ingestion.

### Slide 11: Scientific Honesty & The 11 Forbidden Claims
* Explicit transparency on operational boundaries:
  - Panchayat-scale resolution ceiling (1 pixel $\approx$ 1 GP / 19.2 km²).
  - Perfect-model supervised downscaling (IMD/CHIRPS); GFS 0.25°/NCUM as format-compatible roadmap.
  - 100% adherence to all 11 scientific boundaries.

### Slide 12: Impact, Scalability & Roadmap
* **Financial Impact:** Prevents ₹1,800/acre fertilizer wash-off and ₹6,500/acre grain harvest rot across 234 Panchayats.
* **Statewide Deployment:** Easily extensible across Karnataka's 6,000+ KMF milk cooperatives with zero additional hardware cost.

---

## 2. 90-Second Stage Demonstration Script

* **[0:00 - 0:20] The Setup:**  
  "Judges, every agri-app tells the farmer: *'Download this, buy a phone, and log your rain.'* That model has failed. In Mandya's 234 Panchayats, we don't ask the farmer to be a data-entry clerk. We empower the village intermediary who is already standing at the milk scale at 06:00 AM."
* **[0:20 - 0:45] The Cockpit in Action:**  
  *(Presenter taps Dairy Secretary)*  
  "Notice how the cockpit morphs. The Dairy Secretary doesn't see raw neural networks or CQR equations. He sees his 2-second job: *'Did it rain in Halaguru in the last 12 hours?'* One tap on 'ಹೌದು' feeds our ground-truth validation loop. He taps Broadcast — the audio plays aloud in natural Mandya Kannada for the farmers waiting in line, and the chalkboard template appears."
* **[0:45 - 1:10] Airplane Mode Test:**  
  *(Presenter flips on Airplane Mode)*  
  "Now look: the tower is dead. Other apps crash. Our banner clearly states: *'No network — operating on cached 06:00 IST advisory.'* We tap Broadcast while completely offline — the audio plays immediately from the Service Worker cache, the chalkboard is rendered, and the WhatsApp dispatch is safely queued in IndexedDB to send the moment network returns."
* **[1:10 - 1:30] Intermediary Flexibility:**  
  *(Presenter switches to RSK Officer and GP Secretary)*  
  "For the RSK Officer, the cockpit instantly elevates the ₹ cost-of-error: ₹1,800 saved by withholding urea before tonight's rain. For the GP Secretary, it generates an official IMD-standard Virtual ARG data stream for all 232 unmonitored Panchayats."
