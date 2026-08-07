"""FastAPI control-plane server.

Exposes the edge node's health, metrics, audit trail, and the E-Stop control
surface so operators and the Streamlit dashboard can act on the system.
"""

from __future__ import annotations

import io
import logging
import time
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..config import VisionConfig
from ..events.schemas import (
    AuditEntry,
    BaseEvent,
    EStopEvent,
    EventType,
    Severity,
)
from ..runtime import EdgeRuntime
from ..vision.detector import build_detector, Detection
from ..vision.geofence import GeofenceEngine
from ..vision.ppe import evaluate_ppe

logger = logging.getLogger(__name__)


class EStopRequest(BaseModel):
    mode: str = Field(default="HARD", pattern="^(HARD|SOFT)$")


class EdgeStatus(BaseModel):
    node_id: str
    online: bool
    latency_ms: float
    fps: float
    alerts: int
    estop_locked: bool
    uptime_s: float


def create_app(runtime: EdgeRuntime) -> FastAPI:
    app = FastAPI(
        title="Industrial Safety Intelligence Platform",
        version="0.1.0",
        description="Edge-first multi-modal safety AI control plane.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = runtime.settings.api.edge_prefix

    @app.get(f"{prefix}/health", tags=["edge"])
    def health() -> EdgeStatus:
        return EdgeStatus(
            node_id=runtime.settings.edge.node_id,
            online=True,
            latency_ms=runtime.metrics.last_latency_ms,
            fps=runtime.metrics.last_fps,
            alerts=runtime.metrics.alert_count,
            estop_locked=runtime.plc.is_locked,
            uptime_s=runtime.metrics.snapshot()["uptime_s"],
        )

    @app.get(f"{prefix}/metrics", tags=["edge"])
    def metrics() -> dict:
        m = runtime.metrics.snapshot()
        m["node_id"] = runtime.settings.edge.node_id
        return m

    @app.get(f"{prefix}/events", tags=["edge"])
    def events(limit: int = Query(default=50, ge=1, le=500)) -> List[BaseEvent]:
        return runtime.broker.snapshot()[-limit:][::-1]

    @app.get(f"{prefix}/audit", tags=["edge"])
    def audit(limit: int = Query(default=100, ge=1, le=1000)) -> List[dict]:
        return runtime.audit.latest(limit)

    @app.post(f"{prefix}/trigger-estop", tags=["control"])
    async def trigger_estop(req: EStopRequest) -> dict:
        latency_ms = await runtime.plc.trip(req.mode)
        with runtime.metrics.lock:
            runtime.metrics.is_estop_locked = True
            runtime.metrics.alert_count += 1
        event = EStopEvent(
            event_type=EventType.E_STOP_TRIGGERED,
            severity=Severity.CRITICAL,
            latency_ms=latency_ms,
            relay=runtime.plc.relay_channel,
            mode=req.mode,
            line_id=runtime.plc.line_id,
        )
        await runtime.broker.publish(event)
        runtime.audit.record(
            AuditEntry(
                event_type=event.event_type,
                severity=event.severity,
                latency_ms=latency_ms,
                node_id=runtime.settings.edge.node_id,
            )
        )
        return {"status": "TRIPPED", "latency_ms": latency_ms, "relay": runtime.plc.relay_channel}

    @app.post(f"{prefix}/release-estop", tags=["control"])
    async def release_estop() -> dict:
        if not runtime.plc.is_locked:
            raise HTTPException(status_code=409, detail="E-Stop not locked")
        await runtime.plc.release()
        with runtime.metrics.lock:
            runtime.metrics.is_estop_locked = False
        event = EStopEvent(
            event_type=EventType.E_STOP_RELEASE,
            severity=Severity.INFO,
            relay=runtime.plc.relay_channel,
        )
        await runtime.broker.publish(event)
        runtime.audit.record(
            AuditEntry(
                event_type=event.event_type,
                severity=event.severity,
                latency_ms=0.0,
                node_id=runtime.settings.edge.node_id,
            )
        )
        return {"status": "RELEASED"}

    @app.get(f"{prefix}/config", tags=["edge"])
    def config() -> dict:
        return runtime.settings.model_dump()

    @app.get(f"{prefix}/video-feed", tags=["edge"])
    async def video_feed() -> Response:
        settings = runtime.settings
        vision_cfg = settings.vision
        video_cfg = settings.video

        detector = build_detector(vision_cfg)
        geofences = GeofenceEngine.from_yaml(vision_cfg.geofences_file)

        cap = None
        if not video_cfg.synthetic_camera:
            cap = cv2.VideoCapture(video_cfg.source)
            if not cap.isOpened():
                cap = None

        def draw_boxes(frame, detections, violations):
            h, w = frame.shape[:2]
            for det in detections:
                x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                color = (40, 180, 40)
                if det.class_name == "person":
                    worker_label = f"W-{int(det.bbox_center[0] * 1000)}"
                    has_violation = any(v[0] == worker_label for v in violations)
                    color = (40, 40, 180) if has_violation else (40, 180, 40)
                elif det.class_name == "helmet":
                    color = (180, 180, 40)
                elif det.class_name == "vest":
                    color = (180, 40, 180)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det.class_name} ({det.confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        def gen_frames():
            frame_idx = 0
            while True:
                try:
                    if cap is not None:
                        ret, frame = cap.read()
                        if not ret:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        if video_cfg.width and video_cfg.height:
                            frame = cv2.resize(frame, (video_cfg.width, video_cfg.height))
                    else:
                        frame = np.zeros((video_cfg.height, video_cfg.width, 3), dtype=np.uint8)

                    started = time.perf_counter()
                    detections = detector.detect(frame)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0

                    workers = [d for d in detections if d.class_name == "person"]
                    violations = evaluate_ppe(workers, detections, vision_cfg)

                    for zone in geofences.zones.values():
                        pts = np.array(zone.polygon, dtype=np.float32)
                        pts[:, 0] *= frame.shape[1]
                        pts[:, 1] *= frame.shape[0]
                        pts = pts.astype(int)
                        color = (0, 0, 180) if zone.severity == "CRITICAL" else (0, 120, 180)
                        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
                        cv2.putText(frame, zone.name, (pts[0][0], pts[0][1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    draw_boxes(frame, detections, violations)

                    cv2.putText(frame, f"FPS: {1000.0/max(elapsed_ms,1.0):.1f} | Latency: {elapsed_ms:.1f}ms",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ret:
                        continue
                    frame_bytes = buf.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                    ) + frame_bytes + b"\r\n"

                    frame_idx += 1
                except Exception as exc:
                    logger.exception("video feed error: %s", exc)
                    continue

        return StreamingResponse(
            gen_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    return app
