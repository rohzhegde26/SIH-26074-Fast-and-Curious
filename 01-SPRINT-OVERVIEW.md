# SIH26059 — Internal Round Sprint Plan Overview

Read `00-CANONICAL-SPEC.md` before starting any sprint. It resolves every
projection/grid/channel/date/labeling disagreement from earlier planning docs.

## Timeline (21-day internal round, adjust dates to your actual college deadline)

| Sprint | Days | Focus | Primary owner(s) |
|---|---|---|---|
| 0 | 0–1 | Foundation, environment, accounts, scaffolding | Everyone |
| 1 | 2–5 | Data ingestion, reprojection, storage | Data/DevOps Lead |
| 2 | 6–10 | Sea-ice forecasting model | AI Sea-Ice Lead |
| 3 | 6–10 (parallel w/ 2) | Iceberg detection + drift forecasting | AI Iceberg Lead |
| 4 | 11–13 | POLARIS risk + route optimization | AI Iceberg Lead + Backend |
| 5 | 11–15 (parallel w/ 4) | Backend API | Backend Lead |
| 6 | 11–16 (parallel w/ 4/5) | Frontend (scientist + captain views) | Frontend Lead |
| 7 | 17–21 | Integration, resilience, pitch, submission | Everyone |

Sprints 2/3 run in parallel (different people, independent workstreams).
Sprints 4/5/6 also overlap once Sprints 2/3 produce their first checkpoints —
the backend and frontend can build against mocked model outputs early and
swap in real outputs as soon as Sprint 2/3 finish, rather than waiting idle.

## Roles (team of 6, adjust to your actual headcount)
1. **AI Sea-Ice Lead** — Sprint 2, supports Sprint 7
2. **AI Iceberg Lead** — Sprint 3, Sprint 4, supports Sprint 7
3. **Backend Lead** — Sprint 5, supports Sprint 4 and 7
4. **Frontend Lead** — Sprint 6, supports Sprint 7
5. **Data & DevOps Lead** — Sprint 0, Sprint 1, supports everyone throughout
6. **Research & Pitch Lead** — runs continuously alongside every sprint
   (novelty matrix, references, slides, video script), heavy load in Sprint 7

## Definition of Done for the whole internal round
- `docker compose up` on a clean clone of the repo produces a working demo
  with no manual fixes.
- All four API endpoints return real model output, not mocked JSON.
- At least one full demo scenario (A23A tracking + Cape Town→Bharati route)
  runs end-to-end live.
- A 2-minute backup video exists in case the live demo fails.
- Every number in the pitch deck is either measured or explicitly labeled
  "modeled estimate."
- GitHub repo has no secrets committed, has a working README, and a fresh
  clone builds successfully — verify this on a second machine if you can.

## Sprint files
- `sprints/SPRINT-0-foundation.md`
- `sprints/SPRINT-1-data-pipeline.md`
- `sprints/SPRINT-2-sea-ice-forecasting.md`
- `sprints/SPRINT-3-iceberg-detection-drift.md`
- `sprints/SPRINT-4-polaris-routing.md`
- `sprints/SPRINT-5-backend-api.md`
- `sprints/SPRINT-6-frontend-cesium-pwa.md`
- `sprints/SPRINT-7-integration-demo.md`
