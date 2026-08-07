# Neurobots Championship 2026 — Round 1: Idea Validation & Team Understanding

## Project: Industrial Safety Intelligence Platform (ISIP)

---

## Problem Statement Understanding

Industrial facilities contend with critical safety hazards, including unmonitored PPE violations and restricted zone breaches, while human CCTV operators miss up to 45% of safety incidents due to fatigue. Cloud-based safety solutions fail to meet sub-second response limits due to network latency and bandwidth constraints. The Industrial Safety Intelligence Platform (ISIP) solves this by deploying a sub-20ms, edge-first multi-modal AI system. It fuses real-time YOLOv8 vision with IIoT telemetry (temperature, vibration, toxic gas) to autonomously detect hazards, predict equipment failure, and trip local PLC E-Stop hardware completely offline.

---

## Proposed Technology Stack

- **Core & Async Runtime:** Python 3.11, asyncio
- **Vision & AI Pipeline:** YOLOv8 (Ultralytics), OpenCV, Pydantic
- **Operator Dashboard & Frontend:** Streamlit, Pandas
- **Backend & REST API:** FastAPI, Uvicorn
- **Telemetry & Protocols:** Modbus-TCP / MQTT Emulation, Async Event Queue
- **Target Edge Hardware:** NVIDIA Jetson Orin Nano / AGX Orin, Raspberry Pi 5 (Hailo-8 NPU), Intel NUC

---

## Team Member Contributions

- **Joyal:** Computer Vision Pipeline & Edge Hardware Optimization (YOLOv8, OpenCV point-in-polygon geofencing, Jetson/Pi 5 hardware benchmarking)
- **Anandhu:** Async Edge Orchestrator Architecture, Central Event Queue Broker, and GitHub Repository Governance
- **Giri:** IIoT Telemetry Engine, Sensor Anomaly Detection Rules, and Remaining Useful Life (RUL) Thermal Degradation Estimation
- **Boss:** FastAPI Control Plane Server, Mock PLC Emergency Stop (E-Stop) Relay Integration, & Audit Logging System
- **Adithyan:** Streamlit Operator Dashboard Development, Live Video Overlay Engine, & Telemetry Monitoring UI

---

## Business Impact

Drastically reduces industrial injury liabilities, compliance penalties, and insurance costs through automated hazard detection. The IIoT Remaining Useful Life (RUL) predictive tracking prevents unexpected equipment downtime and catastrophic machinery failure. Operating purely on edge hardware eliminates ongoing cloud bandwidth costs and guarantees sub-20ms response speeds regardless of internet connectivity.

---

## Social Impact

Protects factory and industrial workers by preventing workplace injuries and fatalities in high-risk operational environments. It democratizes enterprise-grade safety automation for bandwidth-constrained, rural, or off-grid industrial facilities, promoting higher occupational health standards globally.

---

## GitHub Repository

https://github.com/Anandhu-96/Agentic_AI

---

## Hardware Declaration

- [x] **Project involves hardware components** — NVIDIA Jetson Orin Nano / AGX Orin, Raspberry Pi 5 (Hailo-8 NPU), Intel NUC, Modbus-TCP / MQTT Emulation, PLC E-Stop Relay

---

## Submission Checklist

- [ ] Problem Statement Understanding filled
- [ ] Proposed Technology Stack filled
- [ ] Team Member Contributions filled
- [ ] Business Impact filled
- [ ] Social Impact filled
- [ ] GitHub Repository added
- [ ] Hardware project box checked
- [ ] Submit Round 1
