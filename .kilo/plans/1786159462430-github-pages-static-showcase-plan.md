# GitHub Pages Static Showcase — Deployment Plan

**Goal:** Publish `https://anandhu-96.github.io/Agentic_AI/` as a free, always-on static showcase. It auto-updates on every push to `main` (no manual steps after setup). Live edge-node hosting (FastAPI + Streamlit 24/7) is explicitly **out of scope for now**.

**Why GitHub Pages (context):** Pages serves static files only. It cannot run FastAPI/Streamlit/YOLO, so it hosts a showcase page, not the live detector. The real-time video feed is already only reachable from a locally-running backend (`127.0.0.1:8080`), so the showcase uses static preview images instead. Live hosting can be added later without rework.

**Current repo facts (verified):**
- GitHub repo exists: `Anandhu-96/Agentic_AI`, branch `main`, remote `origin` configured.
- Git identity set locally (`o0Joyal0o` / `joyalgeorge302005@gmail.com`).
- No `Dockerfile`, no `.github/`, no docker-compose, no Pages workflow yet.
- `dashboard/isip-dashboard.html` is a self-contained static HTML dashboard (CDN Tailwind/Chart.js) — it polls `127.0.0.1:8080` and refreshes the video `<img>` every 200ms, so it cannot be served as-is on Pages.
- Static assets: `dashboard/detection_output.jpg`, `dashboard/geofence_test.jpg`, `dashboard/isip-dashboard.html`.

---

## Design

Publish a **simple landing page** (`docs/index.html`) plus the two preview images and the full dashboard as a deep-linkable page. Everything is plain static files. GitHub Pages is enabled once (Settings → Pages → Deploy from branch `main` / folder `/docs`); the site rebuilds automatically on every push to `main`, which satisfies "updates as I update the code."

Site structure (`docs/` on `main` is the Pages root):
```
docs/
  index.html                <- landing/showcase page
  detection_output.jpg      <- copy of generated inference image
  geofence_test.jpg         <- copy of zone visuals
  isip-dashboard.html       <- copied demo dashboard
```
Deploying from `main:/docs` gives free, automatic, no-credit-card hosting with zero runtime cost and no sleep.

---

## Task list (implementation order)

1. **Add a `scripts/build_showcase.py`** (small helper; run locally to keep assets fresh):
   - Copies `dashboard/detection_output.jpg`, `dashboard/geofence_test.jpg`, and `dashboard/isip-dashboard.html` into `docs/`.
   - Writes `docs/index.html` as a self-contained landing page:
     - Page title: "ISIP — Industrial Safety Intelligence Platform".
     - Short description + bullet feature list (vision, geofencing, PPE, IIoT, RUL, dashboard).
     - Two `<img>`s of `detection-output.jpg` and `geofence_test.jpg` (responsive, `max-width:100%`).
     - Link "Open interactive dashboard" → `isip-dashboard.html`.
     - Note in the page that the live video/telemetry requires the locally-running edge node (`python scripts/run_edge.py`).
     - Minimal inline CSS (no frameworks) so it loads instantly on Pages.
   - Make script idempotent and safe to re-run (overwrites `docs/` contents, never touches other files).

2. **Enable GitHub Pages once (breathstep, cannot be scripted):** repo Settings → Pages → **Deploy from a branch** → `Branch: main`, `/docs` → save. First deploy uses new `docs/` content. Explain the check of public visibility (Pages is not free for private repos) and that the repo is currently public.

3. **Run `python scripts/build_showcase.py`** so `docs/` exists with real content, then `git add -A`, `git commit`, `git push origin main`.

4. **(Recommended) Add a lightweight CI sanity check** `.github/workflows/ci.yml` so pushing doesn't silently break the project:
   - Triggers on push/PR to `main`.
   - `actions/checkout@v4`, Python 3.11/3.12, `pip install -e ".[dev]"`.
   - Steps: `python -m pytest tests/ -q`, `python -m compileall -q src/`, run `python scripts/build_showcase.py`, then `git diff --exit-code` to ensure the showcase was regenerated (guards against stale landing images). (Optional if team prefers minimal infra.)

5. **Verify**:
   1. `python scripts/build_showcase.py` completes and `docs/index.html` renders in a browser (open file directly — no live-feed crash because it's a simple page).
   2. After pushing, visit `https://anandhu-96.github.io/Agentic_AI/` and confirm landing page + images load.
   3. Push another commit touching `dashboard/detection_output.jpg` → confirm the site updates within a few minutes (Pages auto-rebuild).

---

## Files that will be created/changed

| File | Purpose |
|---|---|
| `docs/index.html` | Landing showcase page (generated, committed so Pages can serve it). |
| `docs/detection-output.jpg` | copy of the inference preview image. |
| `docs/geofence_test.jpg` | copy of the geofence preview image. |
| `docs/isip-dashboard.html` | copy of interactive demo dashboard (optional but nice deep-link) |
| `scripts/build_showcase.py` | regenerates the files above so the site always reflects the latest dashboard images. |
| `.github/workflows/ci.yml` | optional push/PR validation + showcase freshness guard. |

No source modules (`src/`) or tests are modified — this is a static-site packaging task only.

---

## Scope / risks / out of scope

- **Out of scope:** 24/7 live FastAPI/Streamlit deployment (VPS, Render, etc.), Docker image, HTTPS custom domain.
- **Risk — stale images:** mitigated by shipping `build_showcase.py` and (optionally) the CI freshness check.
- **Risk — repo private by default breakdown:** Pages is free 24/7 **only** if deployment done from a branch in a **public** repo; check the repo visibility setting. If the team wants to keep it private, Pages needs a paid plan — that is a human/decision boundary, flagged in the plan.
- **Risk — Pages always up to date:** pages deploy-from-branch rebuilds on push automatically; just push a commit.

---

## Acceptance / exit criteria

- `https://anandhu-96.github.io/Agentic_AI/` loads and shows the landing page with both images.
- Pushing a repo update rebuilds the live page without any manual step (no server cost, no config apart from the one-time Settings toggle).
- `python scripts/build_showcase.py` regenerates `docs/` deterministically.
- (Optional) `pytest` + `compileall` pass in CI.