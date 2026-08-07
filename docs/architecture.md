# ISIP System Architecture

## Overview

The Industrial Safety Intelligence Platform (ISIP) is an edge-first, multi-modal
safety system for industrial facilities. A single edge node fuses:

1. **Vision** — real-time YOLOv8 object detection with PPE-compliance rules and
   polygon geofencing of restricted zones.
2. **IIoT telemetry** — temperature, vibration, and toxic-gas sensors with
   threshold + statistical anomaly detection and Remaining Useful Life (RUL)
   estimation.
3. **Control plane** — a FastAPI REST API that exposes health, metrics, events,
   the audit trail, and hardware E-Stop relay control.
4. **Operator dashboards** — a Streamlit live dashboard and a standalone HTML
   terminal-style dashboard.

All reasoning runs locally on edge hardware (NVIDIA Jetson Orin Nano / AGX Orin,
Raspberry Pi 5 + Hailo-8, Intel NUC) and targets a **sub-20ms** end-to-end safety
loop, entirely offline.

## Component diagram

```
┌───────────────────────────────────────────────────────────────┐
│                        Edge Node Process                      │
│                                                               │
│  ┌────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │ Cameras    │───▶│ Vision Pipeline  │───▶│               │  │
│  │ (CCTV/RGB) │    │ YOLOv8 · OpenCV  │    │               │  │
│  └────────────┘    │ Geofence · PPE   │    │               │  │
│  ┌────────────┐    └──────────────────┘    │  Async Event  │  │
│  │ IIoT       │───▶┌──────────────────┐    │    Broker     │──┼──▶ Audit JSONL
│  │ Sensors    │    │ Telemetry Engine │───▶│ (pub/sub)     │  │
│  │ (T/V/Gas)  │    │ Anomaly · RUL    │    │               │  │
│  └────────────┘    └──────────────────┘    │               │  │
│                                            └───────┬───────┘  │
│  ┌────────────┐    ┌──────────────────┐           │          │
│  │ PLC E-Stop │◀───│ FastAPI Control  │◀──────────┘          │
│  │ Relay      │    │ Plane (:8080)    │                       │
│  └────────────┘    └──────────────────┘                       │
│                                                               │
│  Streamlit dashboard (:8501)  ·  static HTML dashboard        │
└───────────────────────────────────────────────────────────────┘
```

## Data flow (safety loop)

1. Frame → YOLOv8 detections (persons, PPE items) with per-frame latency.
2. Worker centroids tested against polygon geofences
   (`src/isip/vision/geofence.py`).
3. PPE compliance evaluated per worker (`src/isip/vision/ppe.py`).
4. Telemetry samples (`src/isip/iiot/telemetry.py`) checked by
   `src/isip/iiot/anomaly.py`; temperature drives `src/isip/iiot/rul.py`.
5. Violations and trips are published on the async broker
   (`src/isip/events/broker.py`).
6. The control plane (`src/isip/api/server.py`) trips the PLC relay
   (`src/isip/control/plc.py`) and the audit logger
   (`src/isip/control/audit.py`) appends the JSONL ledger.

## Concurrency model

`src/isip/orchestrator.py` runs four independent `asyncio` tasks:

| Task              | Purpose                                        |
|-------------------|------------------------------------------------|
| `inference`       | Camera read → detect → geofence → PPE loop     |
| `telemetry`       | Sensor sampling → anomaly → RUL → publish      |
| `heartbeat`       | Periodic `HEARTBEAT` events                    |
| `api`             | Uvicorn serving the FastAPI control plane      |

Producers and consumers are decoupled through the broker; each subscriber owns
its own queue, so a slow consumer never blocks the sub-20ms safety path.

## Latency budget

Target: **< 20ms** frame-to-relay.

- Detector latency is measured per frame (`detector.latency_ms`).
- Relay trip latency is measured end-to-end (`plc.trip()`).
- Both surface in `/edge/metrics` and the dashboards.

## REST API

| Method | Path                 | Description                          |
|--------|----------------------|--------------------------------------|
| GET    | `/edge/health`       | Node status snapshot                 |
| GET    | `/edge/metrics`      | latency, fps, alerts, E-Stop state   |
| GET    | `/edge/events`       | Recent broker events                 |
| GET    | `/edge/audit`        | Recent audit ledger rows             |
| POST   | `/edge/trigger-estop`| Trip the PLC relay (`mode=HARD|SOFT`)|
| POST   | `/edge/release-estop`| Release a locked relay               |
| GET    | `/edge/config`       | Loaded edge configuration            |

Interactive docs at `http://127.0.0.1:8080/docs`.

## Running

See `README.md`. In short:

```bash
python -m pip install -e ".[dev]"
python scripts/run_edge.py        # edge node + control plane :8080
python -m streamlit run src/isip/dashboard/app.py --server.port 8501
```

Or use `scripts/run_all.sh` (Linux) / `scripts/run_all.ps1` (Windows).

## Project layout

```
src/isip/
├── api/server.py        # FastAPI control plane
├── control/             # PLC relay + audit ledger
├── dashboard/app.py     # Streamlit operator dashboard
├── events/              # schemas + async event broker
├── iiot/                # telemetry, anomaly, RUL
├── vision/              # detector, geofence, PPE rules
├── orchestrator.py      # async edge orchestrator
├── runtime.py           # composition root
└── config.py            # YAML-driven settings
config/                  # config.yaml + geofences.yaml
dashboard/               # standalone static HTML dashboard
scripts/                 # launchers
tests/                   # pytest suite
```
