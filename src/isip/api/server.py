"""FastAPI control-plane server.

Exposes the edge node's health, metrics, audit trail, and the E-Stop control
surface so operators and the Streamlit dashboard can act on the system.

PRODUCTION NOTES
-----------------
- Control endpoints (trigger/release estop, config) require an API key via
  the `X-API-Key` header, read from `settings.api.api_key` or the
  `ISIP_API_KEY` env var. If neither is set, the server refuses to start in
  non-debug mode — silently running a safety control-plane with no auth is
  worse than failing loudly at boot.
- CORS is restricted to `settings.api.cors_origins` (a list). Defaults to
  no cross-origin access if unset, rather than "*".
- `/config` strips any field whose name looks secret-shaped
  (key/secret/token/password/credential) before returning it. This is a
  best-effort filter, not a substitute for keeping secrets out of the
  Settings object in the first place (prefer env vars / a secrets manager
  for those and exclude them from `model_dump()` at the schema level too).
- TLS termination, rate limiting, and network-level access control (e.g.
  keeping this port off the public internet) are assumed to be handled by
  the deployment layer (reverse proxy / firewall) — not implemented here.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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

_SECRET_NAME_HINTS = ("key", "secret", "token", "password", "credential")


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


def _resolve_api_key(settings) -> Optional[str]:
    return getattr(getattr(settings, "api", None), "api_key", None) or os.environ.get("ISIP_API_KEY")


def _resolve_cors_origins(settings) -> List[str]:
    origins = getattr(getattr(settings, "api", None), "cors_origins", None)
    if origins:
        return list(origins)
    return []


def _sanitize_config(raw: dict) -> dict:
    """Recursively drop keys that look like secrets before exposing config."""

    def scrub(value):
        if isinstance(value, dict):
            return {
                k: ("***REDACTED***" if any(h in k.lower() for h in _SECRET_NAME_HINTS) else scrub(v))
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value

    return scrub(raw)


def create_app(runtime: EdgeRuntime) -> FastAPI:
    debug_mode = bool(getattr(getattr(runtime.settings, "api", None), "debug", False))
    api_key = _resolve_api_key(runtime.settings)

    if not api_key and not debug_mode:
        raise RuntimeError(
            "No API key configured for the ISIP control plane (settings.api.api_key "
            "or ISIP_API_KEY env var). Refusing to start with unauthenticated "
            "E-Stop control endpoints. Set settings.api.debug=True to override for "
            "local development only."
        )
    if not api_key and debug_mode:
        logger.warning(
            "Starting with NO API key in debug mode — control endpoints are "
            "unauthenticated. Do not use this configuration in production."
        )

    app = FastAPI(
        title="Industrial Safety Intelligence Platform",
        version="0.1.0",
        description="Edge-first multi-modal safety AI control plane.",
    )

    cors_origins = _resolve_cors_origins(runtime.settings)
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["X-API-Key", "Content-Type"],
        )
    elif debug_mode:
        logger.warning("CORS: no origins configured; allowing all origins (debug mode only).")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    # else: no CORS middleware at all -> browsers block cross-origin by default.

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        runtime.shutdown()

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        detail = str(exc) if debug_mode else "Internal server error."
        return JSONResponse(status_code=500, content={"detail": detail})

    def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        if api_key is None:
            return  # debug mode, already warned at startup
        if x_api_key != api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    prefix = runtime.settings.api.edge_prefix

    @app.get(f"{prefix}/health", tags=["edge"])
    async def health() -> EdgeStatus:
        logger.debug("health check requested")
        snapshot = await runtime.metrics.snapshot()
        return EdgeStatus(
            node_id=runtime.settings.edge.node_id,
            online=True,
            latency_ms=snapshot["latency_ms"],
            fps=snapshot["fps"],
            alerts=snapshot["alerts"],
            estop_locked=runtime.plc.is_locked,
            uptime_s=snapshot["uptime_s"],
        )

    @app.get(f"{prefix}/metrics", tags=["edge"])
    async def metrics() -> dict:
        logger.debug("metrics requested")
        m = await runtime.metrics.snapshot()
        m["node_id"] = runtime.settings.edge.node_id
        return m

    @app.get(f"{prefix}/trackers", tags=["edge"])
    def trackers() -> dict:
        """Latest worker tracking-device positions (serial / emulated)."""
        logger.debug("trackers requested")
        return runtime.serial_tracker.snapshot()

    @app.get(f"{prefix}/events", tags=["edge"])
    def events(limit: int = Query(default=50, ge=1, le=500)) -> List[BaseEvent]:
        logger.debug("events requested limit=%d", limit)
        return runtime.broker.snapshot()[-limit:][::-1]

    @app.get(f"{prefix}/audit", tags=["edge"])
    def audit(limit: int = Query(default=100, ge=1, le=1000)) -> List[dict]:
        logger.debug("audit requested limit=%d", limit)
        return runtime.audit.latest(limit)

    @app.post(f"{prefix}/trigger-estop", tags=["control"], dependencies=[Depends(require_api_key)])
    async def trigger_estop(req: EStopRequest) -> dict:
        try:
            latency_ms = await runtime.plc.trip(req.mode)
        except Exception:
            logger.exception("PLC trip() failed for mode=%s", req.mode)
            raise HTTPException(status_code=503, detail="E-Stop hardware trip failed; see server logs")

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
        logger.warning("E-STOP TRIGGERED mode=%s latency_ms=%.2f", req.mode, latency_ms)
        return {"status": "TRIPPED", "latency_ms": latency_ms, "relay": runtime.plc.relay_channel}

    @app.post(f"{prefix}/release-estop", tags=["control"], dependencies=[Depends(require_api_key)])
    async def release_estop() -> dict:
        if not runtime.plc.is_locked:
            raise HTTPException(status_code=409, detail="E-Stop not locked")
        try:
            await runtime.plc.release()
        except Exception:
            logger.exception("PLC release() failed")
            raise HTTPException(status_code=503, detail="E-Stop hardware release failed; see server logs")

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
        logger.info("E-Stop released")
        return {"status": "RELEASED"}

    @app.get(f"{prefix}/config", tags=["edge"], dependencies=[Depends(require_api_key)])
    def config() -> dict:
        return _sanitize_config(runtime.settings.model_dump())

    @app.get(f"{prefix}/video-feed", tags=["edge"])
    async def video_feed() -> Response:
        producer = runtime.video_stream

        async def gen_frames():
            while True:
                frame_bytes = producer.get_frame(timeout=1.0)
                if frame_bytes is None:
                    if producer.error is not None:
                        logger.error("Video producer thread has an error: %s", producer.error)
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
    async def event_stream(request: Request) -> Response:
        async def gen_events():
            async for event in runtime.broker.subscribe():
                if await request.is_disconnected():
                    logger.debug("SSE client disconnected, stopping event stream")
                    break
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