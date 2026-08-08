"""FastAPI control-plane server.

Exposes the edge node's health, metrics, audit trail, and the E-Stop control
surface so operators and the Streamlit dashboard can act on the system.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..events.schemas import (
    AuditEntry,
    BaseEvent,
    EStopEvent,
    EventType,
    Severity,
)
from ..runtime import EdgeRuntime

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
    async def health() -> EdgeStatus:
        snapshot = runtime.metrics.snapshot()
        return EdgeStatus(
            node_id=runtime.settings.edge.node_id,
            online=True,
            latency_ms=runtime.metrics.last_latency_ms,
            fps=runtime.metrics.last_fps,
            alerts=runtime.metrics.alert_count,
            estop_locked=runtime.plc.is_locked,
            uptime_s=snapshot["uptime_s"],
        )

    @app.get(f"{prefix}/metrics", tags=["edge"])
    async def metrics() -> dict:
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

    @app.get(f"{prefix}/geofences", tags=["edge"])
    def geofences() -> dict:
        """Normalized geofence geometry so dashboards can draw the real polygons."""
        return {"zones": runtime.geofences.overlay_polys(normalize=True)}

    @app.get(f"{prefix}/zones", tags=["edge"])
    def zones() -> dict:
        """Live zone state: static + dynamic machine zones with geometry."""
        active = runtime.detection_state.active_zones
        active |= set(runtime.video_stream.latest_active_zones)
        active |= set(runtime.detection_stream.latest_active_zones)
        overlay = runtime.geofences.overlay_polys(
            runtime.settings.video.width,
            runtime.settings.video.height,
        )
        for name, info in overlay.items():
            info["active"] = name in active
            info["dynamic"] = False
        # Merge the machine-derived danger zones (dynamic).
        for name, info in runtime.detection_state.dynamic_zones.items():
            z = info.to_dict()
            z["polygon"] = [
                [round(x * runtime.settings.video.width, 1),
                 round(y * runtime.settings.video.height, 1)]
                for x, y in z["polygon"]
            ]
            z["active"] = name in active
            z["dynamic"] = True
            overlay[name] = z
        return {"zones": overlay}

    @app.get(f"{prefix}/video-feed", tags=["edge"])
    async def video_feed() -> Response:
        producer = runtime.video_stream

        def gen_frames():
            while True:
                frame_bytes = producer.get_frame(timeout=1.0)
                if frame_bytes is None:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                ) + frame_bytes + b"\r\n"

        return StreamingResponse(
            gen_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get(f"{prefix}/video-snapshot", tags=["edge"])
    async def video_snapshot() -> Response:
        frame_bytes = runtime.video_stream.get_frame(timeout=1.0)
        if frame_bytes is None:
            frame_bytes = b""
        return Response(
            content=frame_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get(f"{prefix}/video-snapshot/detections", tags=["edge"])
    async def video_snapshot_detections() -> Response:
        """Snapshot from the detection-only stream (no zone overlay)."""
        frame_bytes = runtime.detection_stream.get_frame(timeout=1.0)
        if frame_bytes is None:
            frame_bytes = b""
        return Response(
            content=frame_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get(f"{prefix}/video-feed/detections", tags=["edge"])
    async def video_feed_detections() -> Response:
        producer = runtime.detection_stream

        def gen_frames():
            while True:
                frame_bytes = producer.get_frame(timeout=1.0)
                if frame_bytes is None:
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                ) + frame_bytes + b"\r\n"

        return StreamingResponse(
            gen_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get(f"{prefix}/stream", tags=["edge"])
    async def event_stream() -> Response:
        async def gen_events():
            async for event in runtime.broker.subscribe():
                payload = event.model_dump(mode="json")
                payload.pop("latency_ms", None)
                yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(
            gen_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app
