# Sprint 7 — Integration, Resilience, Pitch & Submission (Days 17–21)

Reference: `00-CANONICAL-SPEC.md`. Depends on all prior sprints being
functionally complete (mocks removed, real data flowing end-to-end).

## Goal
Everything works together live, survives a bad demo day, and is packaged the
way judges actually evaluate — completeness, honesty about limitations, and a
rehearsed narrative, not last-minute feature stuffing.

## Tasks

### Integration (all hands, Day 17–18)
- Run the full loop end-to-end: ingest → sea-ice forecast → iceberg drift →
  POLARIS risk → route optimization → both frontend views, for the actual
  demo scenario (A23A live tracking + Cape Town→Bharati route).
- Fix integration bugs as a team — this is not one person's job, everyone
  who touched a piece of the pipeline should be in the room.
- Re-run Sprint 1's validation notebook once more against the final pipeline
  state to confirm nothing regressed.

### Explainability polish (`frontend` + `models`)
- Wire SHAP feature attribution for the SIC forecast into the scientist
  view's explainability panel, if time allows — if it doesn't fit, cut it
  and say so in the roadmap slide rather than showing something broken live.

### Resilience (Data/DevOps Lead, Day 19)
- Pre-cache 3 days of demo data in the PWA so the captain view survives a
  live network drop.
- Confirm ONNX model artifacts are backed up somewhere retrievable
  independent of any one laptop (release artifact, HF Spaces, Vercel/Docker
  registry).
- Record a 2-minute backup demo video of the full working system, in case
  the live demo fails on stage.
- **Verify `docker compose up` on a completely clean clone**, ideally on a
  second machine, not just "it works on my laptop." This is the single most
  common way hackathon demos fail — an environment quirk nobody else's
  machine has.

### Pitch deck (Research & Pitch Lead, Day 18–20)
Official SIH format, roughly:
1. Problem + NCPOR operational context (charter cost, current manual
   workflow) — cite the real tender/bill references from `docs/references.md`
2. Novelty matrix vs. a generic "UNet + Leaflet" baseline team
3. Physics: Wagner gamma breakdown (small berg vs. tabular berg), POLARIS RIO
4. Architecture diagram (the actual one you built, not the aspirational one)
5. Results: RMSE/IoU/Integrated Ice Edge Error, FDE/ADE vs. physics baseline,
   route before/after — **all real measured numbers from Sprints 2–4**
6. India impact + future work — **every economic figure explicitly labeled
   "modeled estimate,"** per the canonical spec's honesty rule

### Demo script (Research & Pitch Lead + whoever demos)
- Write and rehearse a 2-minute script covering: problem framing → live
  Cesium globe with real SIC/iceberg data → click iceberg for force breakdown
  → RIO heatmap and route optimization before/after → captain PWA offline
  view → impact close.
- Rehearse at least 3 full run-throughs with a timer, on the actual machine
  and network you'll demo on.

### Judge Q&A prep (`docs/qa_prep.md`)
Write real, honest answers grounded in what was actually built:
- Latency — the measured number from Sprint 5, not an assumed target.
- "Why Antarctic not Arctic" — current-dominated drift, POLARIS RIV
  differences, the Wagner gamma breakdown.
- Data gaps — what actually happened when a provider pull failed during the
  hackathon, and what fallback you used (from Sprint 0's documented notes).
- Small iceberg / growler detection limits — the real minimum detectable
  size from whatever detector you actually used, not a literature number for
  a system you didn't build.
- Scalability to other regions/poles — honest answer about what's
  region-specific (RIV table, ice climatology) vs. what generalizes.
- Anything from the Finals-phase roadmap that got explicitly deferred
  (full YOLO fine-tuning, full NSGA-II, full Rotate Block + Gabor-Spectral
  IDRIFTNET, circumpolar coverage) — have the "here's what's next and why we
  scoped it out for the internal round" answer ready.

### Repo cleanup & submission
- Final `README.md`: architecture diagram, setup instructions, what's real
  vs. roadmap.
- Confirm no secrets/credentials are committed (re-check `.gitignore`
  actually caught everything).
- Add a LICENSE file.
- Submit: GitHub repo link, live URL (or Docker instructions if no live
  deployment), PPT, 2-minute demo video, technical doc with the novelty
  matrix.

## Definition of Done
- [ ] Full end-to-end demo scenario runs live without manual intervention
- [ ] Clean-clone `docker compose up` verified on a second machine
- [ ] Backup demo video recorded
- [ ] Pitch deck reviewed by the full team, every number checked against
      Sprint 2–5's actual measured results or labeled "modeled estimate"
- [ ] Demo script rehearsed at least 3 times with a timer
- [ ] Judge Q&A doc covers every claim made in the pitch honestly
- [ ] Repo submitted with README, no secrets, LICENSE, and working
      instructions for a judge to run it themselves if asked
