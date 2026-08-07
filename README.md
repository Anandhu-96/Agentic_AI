# Agentic_AI — Industrial Safety Intelligence Platform (ISIP)

**Neurobots Championship 2026 · Round 1: Idea Validation & Team Understanding**

ISIP is an **edge-first, multi-modal safety AI system** for industrial
facilities. It fuses real-time YOLOv8 vision with IIoT telemetry (temperature,
vibration, toxic gas) to autonomously detect hazards, predict equipment
failure, and trip local PLC E-Stop hardware — completely offline, with a
sub-20ms safety loop.

Human CCTV operators miss up to 45% of safety incidents due to fatigue, and
cloud-based safety solutions can't meet sub-second response limits because of
network latency and bandwidth constraints. ISIP runs entirely on edge hardware
(NVIDIA Jetson Orin Nano / AGX Orin, Raspberry Pi 5 + Hailo-8, Intel NUC), so
it works in bandwidth-constrained, rural, and off-grid industrial facilities.

## Features

- **Vision pipeline** — YOLOv8 object detection, point-in-polygon geofencing of
  restricted zones, and PPE compliance rules (helmet/vest).
- **IIoT telemetry engine** — emulated temperature/vibration/gas sensors with
  threshold + statistical (z-score/EMA) anomaly detection and Arrhenius-based
  Remaining Useful Life (RUL) estimation.
- **Async event broker** — priority-aware pub/sub with per-subscriber queues so
  slow consumers never block the real-time safety path.
- **FastAPI control plane** — health, metrics, events, audit, and hardware
  E-Stop relay control over REST (`/docs` for interactive OpenAPI).
- **Two operator dashboards** — a live Streamlit dashboard with telemetry
  charts, RUL gauge, and E-Stop controls, plus a standalone static HTML
  terminal-style dashboard.
- **Audit ledger** — append-only JSONL log of every safety/control event.
- **Hardware abstraction** — `RPi.GPIO` on real Jetson/Pi boards, in-process
  mock relay elsewhere. Runs with zero hardware for demos.

## Repository layout

```
src/isip/
├── api/server.py        # FastAPI control plane
├── control/             # PLC E-Stop relay + audit ledger
├── dashboard/app.py     # Streamlit operator dashboard
├── events/              # pydantic schemas + async event broker
├── iiot/                # telemetry, anomaly detection, RUL estimation
├── vision/              # YOLOv8 detector, geofencing, PPE rules
├── orchestrator.py      # async edge orchestrator (entrypoint)
├── runtime.py           # shared edge runtime state
└── config.py            # YAML-driven typed configuration
config/                  # edge config + geofence definitions
dashboard/               # standalone static HTML dashboard
scripts/                 # one-command launchers
tests/                   # pytest unit tests
docs/architecture.md     # design document
```

## Quick start

Requires Python 3.11+.

```bash
# 1. Install
python -m pip install -e ".[dev]"

# 2. Run the edge node (inference + telemetry + control plane on :8080)
python scripts/run_edge.py

# 3. In a second terminal, open the operator dashboard on :8501
python -m streamlit run src/isip/dashboard/app.py --server.port 8501
```

Or launch everything with `scripts/run_all.sh` (Linux/macOS) or
`scripts/run_all.ps1` (Windows).

The edge node runs with **zero hardware**: the synthetic detector and emulated
sensors drive the full pipeline so geofencing, PPE violations, thermal
anomalies, RUL updates, and relay trips can all be demonstrated immediately.
Point a camera at `config/config.yaml → video.source` or set
`vision.inference_backend: yolov8` with `yolov8n.pt` to use real YOLOv8.

## REST API

| Method | Path                   | Description                          |
|--------|------------------------|--------------------------------------|
| GET    | `/edge/health`         | Node status snapshot                 |
| GET    | `/edge/metrics`        | latency, fps, alerts, E-Stop state   |
| GET    | `/edge/events`         | Recent broker events                 |
| GET    | `/edge/audit`          | Recent audit ledger rows             |
| POST   | `/edge/trigger-estop`  | Trip the PLC relay (`mode=HARD`)     |
| POST   | `/edge/release-estop`  | Release a locked relay               |
| GET    | `/edge/config`         | Loaded edge configuration            |

Interactive docs: `http://127.0.0.1:8080/docs`

## Tests

```bash
python -m pytest
```

## Team

| Member    | Contribution                                                        |
|-----------|---------------------------------------------------------------------|
| Joyal     | Computer Vision Pipeline & Edge Hardware Optimization (YOLOv8, geofencing, Jetson/Pi 5 benchmarking) |
| Anandhu   | Async Edge Orchestrator, Central Event Queue Broker, GitHub Repository Governance |
| Giri      | IIoT Telemetry Engine, Sensor Anomaly Detection, RUL Thermal Degradation Estimation |
| Boss      | FastAPI Control Plane Server, Mock PLC E-Stop Relay Integration, Audit Logging |
| Adithyan  | Streamlit Operator Dashboard, Live Video Overlay Engine, Telemetry Monitoring UI |

## License

MIT
