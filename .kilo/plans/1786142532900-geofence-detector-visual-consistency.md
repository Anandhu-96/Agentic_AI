# Fix geofence/detector visuals across all dashboards

## Goal

Make the restricted-zone polygons and their execution show correctly on both the
Streamlit dashboard and the standalone HTML dashboard, and make them consistent
with what the backend actually computes/publishes.

## Root causes found

1. **Multiple independent synthetic detector instances disagree.**
   `EdgeOrchestrator.detector` (orchestrator.py:59), `runtime.video_stream.detector`
   (runtime.py:55), and `runtime.detection_stream.detector` (runtime.py:220) are
   three separate `SyntheticDetector` objects, each with a private `_t` counter
   that increments at a different rate/time. Result: the video overlay shows a
   worker in one place while published `ZONE_INTRUSION`/`ZONE_CLEARED` events are
   computed from a different position. The polygons drawn on the live video feed
   therefore never match the zone status text / critical alerts on the dashboard.

2. **HTML dashboard script dies: missing `<canvas>`.**
   `dashboard/isip-dashboard.html:316` does `const cv = $('video-canvas')` /
   `cv.getContext('2d')`, but no `<canvas id="video-canvas">` exists in the HTML
   (only `<div id="video-wrap">` + `<img id="video-feed">`). `cv` is `null`, so the
   whole inline script throws and nothing live works (no polling, no chart, no
   buttons, no refresh). `drawDetections()` (line 530) also assumes the canvas.

3. **HTML dashboard hazards are hardcoded, not the real polygons.**
   `dz()` (line 324) draws a fixed rectangle at (0.52–0.74, 0.58–0.76) which does
   NOT match the actual `DANGER_ZONE_A` pentagon (0.40–0.72, 0.40–0.75) or
   `EXCLUSION_ZONE_B` (0.05–0.32, 0.10–0.35). The displayed "hazard zone" and the
   backend geofence execution are different geometry.

4. **Worker 3 collides with DANGER_ZONE_A.**
   Worker 3 (`(0.50 + 0.08cos, 0.50 + 0.08sin)` i.e. around (0.50, 0.50)) is inside
   `DANGER_ZONE_A`, while the detector comment says it is a "near center with vest
   only" example intended outside restricted zones. This makes a second worker
   always trigger the CRITICAL zone, cluttering the demo.

5. **Streamlit zone status renders only from the recent event window.**
   `render_zone_status()` derives "active zones" by folding `ZONE_INTRUSION` /
   `ZONE_CLEARED` over the returned `events` slice. With lingering/out-of-window
   events the status can disagree with the live video overlay. There is no
   authoritative "currently active zones" feed for the dashboard to consume.

## Design decision

Make the **synthetic detection trajectory deterministic by absolute time** instead
of per-instance monotonic counter. Every `detect()` call across all three
instances then produces identical worker positions for the same wall-clock time
(within sub-second tolerance), so the overlay, the published zone events, and the
dashboards all describe the SAME world. This is the minimal change that fixes the
visual‑vs‑execution mismatch without a pipeline rewrite.

Combine with:
- a shared zone-draw helper so all renderers scale polygons identically
  (normalized 0..1 × frame w/h);
- a live `/edge/zones` endpoint returning the currently-active zones so
  dashboards stop guess-rendering status from event windows;
- repairs to the HTML dashboard so its canvas exists and its hazard visuals are
  drawn from the real geofence geometry.

## Changes

### 1. `src/isip/vision/detector.py` — time-based synthetic detector

- Import `time` (already imported at top). In `SyntheticDetector.__init__`, record
  `self._epoch = time.time()` instead of `self._t = 0.0`.
- In `detect()`, compute shared phase from wall clock:
  `t = time.time() - self._epoch` (keep small to avoid drifting wildly; e.g.
  `t = (time.time() - self._epoch) * 0.5` for a pleasant demo speed).
- Replace all `self._t` usages inside `detect()` with the local `t`.
- Reposition **Worker 3** outside restricted zones (e.g. center-bottom area such
  as `(0.50 + 0.06*sin(t/31), 0.85 + 0.04*cos(t/43))`) so it does not sit inside
  `DANGER_ZONE_A`; comment stays "vest only, no helmet".
- Keep Worker 1 (drifts through `DANGER_ZONE_A`), Worker 2 (exits/enters
  `EXCLUSION_ZONE_B`), and Worker 4 (compliant, outside). Verify with the script
  in "Validation".
- Preserve `latency_ms` measurement and the `build_detector(...)` fallback logic
  unchanged.

### 2. `src/isip/vision/geofence.py` — expose polygon data

No algorithm changes (point-in-polygon is correct). Add convenience so dashboards
can render the same polygons:

- Add `GeofenceZone.as_pixels(frame_width, frame_height) -> List[Tuple[int,int]]`
  that scales `self.polygon` by `(w, h)` and `astype(int)`.
- Add `GeofenceEngine.overlay_polys(w, h)` returning
  `{name: {severity, action, points: [[x,y], ...], centroid}}` for API use.

### 3. `src/isip/runtime.py` — one detector engine for overlay + events

- Change `VideoStreamProducer.__init__` to accept optional
  `detector: ObjectDetector = None` and `geofences: GeofenceEngine = None`.
  Default to `build_detector(...)` / `from_yaml(...)` as today to keep
  standalone use working.
- In `EdgeRuntime.__init__`, build ONE detector and ONE geofence engine, pass the
  same instances to both `video_stream` and `detection_stream`.
- Replace the inline polygon drawing block (lines ~148-172) with a shared helper
  `draw_zones(frame, geofences, active_zones, w, h)` (module-level function),
  also used by `orchestrator.py` (see below) so overlay math stays in one place.
  Use `GeofenceHelper/zone.overlay_pixels` for the scaling.
- Track `runtime.active_zones()`: compute current active zones from the shared
  detector's latest worker positions each frame (existing code already computes
  `active_zones`; expose it via `self.latest_active_zones` on the producer,
  updated in `_run`).

### 4. `src/isip/orchestrator.py` — use the shared detector + publish live zones

- Reuse the runtime's detector instead of creating a separate one so events and
  overlay are consistent: `self.detector = self.runtime.detector`;
  `self.geofences = self.runtime.geofences`.
- In `EdgeRuntime`, the video stream threads are the overlay; the orchestrator's
  `_check_geofences` already runs from the same detector positions (same time
  based). Keep the event publish logic unchanged.
- Optionally set `self.runtime.latest_active_zones` inside `_check_geofences` so
  the `/zones` endpoint reflects the authoritative set (sync via a small lock).

### 4. `src/isip/api/server.py` — add live geofence/zones endpoint

- Add `GET {prefix}/zones` returning:
  ```json
  {
    "zones": {
      "DANGER_ZONE_A": {"severity": "CRITICAL", "active": bool, "polygon": [[x,y],...]},
      "EXCLUSION_ZONE_B": {"severity": "WARNING", "active": bool, "polygon": [[x,y],...]}
    }
  }
  ```
  using `runtime.geofences` and `runtime.latest_active_zones`.
- Add `GET {prefix}/geofences` returning the raw `GeofenceEngine` overlay payload
  (severity + normalized polygon) for the HTML dashboard to draw the true
  polygons.

### 5. `src/isip/dashboard/app.py` — Streamlit zone panel from live zones

- Add `api_get("/zones")` call in `main()`.
- Replace `render_zone_status(events)`'s active-zone derivation with the live
  `/zones` payload (fall back to event window when API offline).
- Keep the existing `Detection Feed` column; it already shows the snapshot URL
  (which will now be consistent once detectors are shared).
- No other layout changes required.

### 6. `dashboard/isip-dashboard.html` — repair broken script + real polygons

- Add a `<canvas id="video-canvas" class="absolute inset-0 w-full h-full">` inside
  `#video-wrap` (right after `<img id="video-feed">`). The existing `dz()`/
  `drawFrame`/`drawDetections` JS then has a real target.
- Change `dz()` and `drawFrame()` to plot the ACTUAL geofence polygons from
  `fetch('/edge/geofences')` (scaled to canvas w/h) instead of the hardcoded
  rectangle. Draw each with severity color; highlight when the zone's
  `active` flag is true (from `/edge/zones`).
- Wire `setInterval(fetchZones, 500)` + `fetchDetections` so hazard visuals track
  backend state (zone status pill + hazard-draw color) rather than a local
  `state.intrusion` boolean alone.
- Keep cosmetic canvas drawing (gradient background, worker boxes) but base the
  hazard rect/animation on the API polygons.
- Guard all canvas access (`if (cv)`), so a missing canvas never kills the app.

### 7. `config/geofences.yaml` — no change required

Geometry already matches; only make sure coordinates stay as source-of-truth.

## Validation

- `python -m pytest` — all 26 existing tests still pass.
- Add tests:
  - `test_detector_timebased.py`: patch/simulate wall-clock monotonic sequence;
    assert two `SyntheticDetector.detect` calls with the same epoch and tricked
    time produce identical worker coordinates; assert Worker 3 outside
    `DANGER_ZONE_A` polygon (copy zone list into test).
  - `test_geofence_overlay` (in `test_geofence.py`): `overlay_pixels` matches
    manual `pts*w, pts*h` computation.
- Run the shell computation script (Worker positions vs zone membership at
  several `t`) and confirm: Worker 1 sometimes in `DANGER_ZONE_A`, Worker 2
  oscillates in/out `EXCLUSION_ZONE_B`, Worker 3 NEVER in `DANGER_ZONE_A`,
  Worker 4 always outside.
- Smoke test: start edge (`python scripts/run_edge.py`), curl
  `127.0.0.1:8080/edge/zones` and `/edge/geofences`; confirm JSON valid.
- Verify HTML: open dashboard, ensure no console errors; hazard rect matches
  DANGER_ZONE_A positions; open Streamlit, confirm zone status text matches the
  live overlay's active zone highlight.

## Risks / open items

- Time-based trajectories drift between processes if `_epoch` differs; within one
  edge process this is fine since all instances share the same clock epoch.
  Document that the demo is single-process.
- The HTML dashboard canvas draws parallel to the already-overlaid snapshot
  (<img> with zones); to avoid double-drawing, either keep only canvas overlay for
  zones or hide the duplicate zone overlay from `runtime.video_stream` by using
  `detection_stream` (no zones) for the HTML `<img>`. **Decision:** use
  `detection_stream` (zones off) for `video-feed` img and draw zones themselves
  via canvas so it is unambiguous.
- `latest_active_zones` needs thread/async-safe update (simple
  `threading.Lock` in `RuntimeMetrics`, matching existing style).

## Out of scope

- Real PLC/GPIO wiring changes.
- Training the YOLO model or swapping `yolov8n.pt`.
- A visual "zone map" (SVG legend) beyond polygons on the video/ canvas.