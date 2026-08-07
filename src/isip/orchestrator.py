"""Async edge orchestrator.

Wires the vision pipeline, IIoT telemetry engine, event broker, control plane,
and REST API into a single edge node process. All loops are asyncio tasks so a
slow component never blocks the sub-20ms safety path.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import Settings
from .control.plc import PlcRelay
from .events.broker import EventBroker
from .events.schemas import (
    AuditEntry,
    BaseEvent,
    EventType,
    InferenceEvent,
    SafetyEvent,
    Severity,
)
from .iiot.anomaly import AnomalyEngine
from .iiot.rul import ThermalRulEstimator
from .iiot.telemetry import TelemetryEngine
from .runtime import EdgeRuntime
from .vision.detector import Detection, ObjectDetector, build_detector
from .vision.geofence import GeofenceEngine
from .vision.ppe import evaluate_ppe

logger = logging.getLogger(__name__)

ANOMALY_EVENT = {
    "temperature": EventType.THERMAL_ANOMALY,
    "vibration": EventType.VIBRATION_ANOMALY,
    "co_ppm": EventType.GAS_ANOMALY,
}

_STATUS_SEVERITY = {
    "NORMAL": Severity.INFO,
    "HEALTHY": Severity.INFO,
    "WARN": Severity.WARNING,
    "CRITICAL": Severity.CRITICAL,
}


class EdgeOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.runtime = EdgeRuntime(settings)
        self.detector: ObjectDetector = build_detector(settings.vision)
        self.geofences = GeofenceEngine.from_yaml(settings.vision.geofences_file)
        self.anomaly = AnomalyEngine(settings.iiot.thresholds)
        self.rul = ThermalRulEstimator(settings.rul)
        self.telemetry = TelemetryEngine(settings.iiot)

        self._active_zones: Set[str] = set()
        self._active_ppe: Set[Tuple[str, str]] = set()
        self._cap = None
        self._tasks: List[asyncio.Task] = []

    # ------------------------------------------------------------------ vision

    def _read_frame(self) -> Any:
        # Camera off by default: synthetic frames keep the demo hardware-free
        # and never enable a webcam LED.
        if self.settings.video.synthetic_camera:
            h, w = self.settings.video.height, self.settings.video.width
            return _synthetic_frame(w, h)
        try:
            import cv2
            import numpy as np

            if self._cap is None:
                self._cap = cv2.VideoCapture(self.settings.video.source)
                if not self._cap.isOpened():
                    self._cap = None
                    raise RuntimeError("camera unavailable")
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError("empty frame")
            if self.settings.video.width:
                frame = cv2.resize(
                    frame, (self.settings.video.width, self.settings.video.height)
                )
            return frame
        except Exception as exc:  # noqa: BLE001
            if getattr(self, "_cap", None) is not None:
                try:
                    self._cap.release()
                except Exception:  # noqa: BLE001
                    pass
                self._cap = None
            logger.debug("frame fallback: %s", exc)
            # Synthetic frame: content is ignored by SyntheticDetector.
            h, w = self.settings.video.height, self.settings.video.width
            return _synthetic_frame(w, h)

    async def _inference_loop(self) -> None:
        interval = 1.0 / max(1, self.settings.edge.fps_limit)
        frame_idx = 0
        while True:
            started = time.perf_counter()
            frame = self._read_frame()
            detections = self.detector.detect(frame)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            workers = [d for d in detections if d.class_name == "person"]
            await self._check_geofences(workers)
            await self._check_ppe(workers, detections)

            with self.runtime.metrics.lock:
                self.runtime.metrics.last_latency_ms = elapsed_ms
                self.runtime.metrics.last_fps = (
                    1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
                )
                self.runtime.metrics.inference_count += 1

            if frame_idx % 15 == 0:
                await self.runtime.broker.publish(
                    InferenceEvent(
                        event_type=EventType.INFERENCE_OK,
                        severity=Severity.INFO,
                        latency_ms=elapsed_ms,
                        detections=[d.__dict__ for d in detections[:8]],
                        fps=self.runtime.metrics.last_fps,
                    )
                )
            frame_idx += 1
            await asyncio.sleep(max(0.0, interval - elapsed_ms / 1000.0))

    async def _check_geofences(self, workers: List[Detection]) -> None:
        seen: Set[str] = set()
        for worker in workers:
            w, h = self.settings.video.width, self.settings.video.height
            cx, cy = worker.bbox_center
            zone = self.geofences.locate((cx / w, cy / h))
            if zone is None:
                continue
            seen.add(zone.name)
            if zone.name in self._active_zones:
                continue
            self._active_zones.add(zone.name)
            severity = Severity.CRITICAL if zone.severity == "CRITICAL" else Severity.WARNING
            event = SafetyEvent(
                event_type=EventType.ZONE_INTRUSION,
                severity=severity,
                zone_id=zone.name,
                confidence=worker.confidence,
                payload={"zone": zone.name, "description": zone.description},
            )
            await self.runtime.broker.publish(event)
            self._audit(event)
            if zone.action == "TRIP_RELAY":
                await self._trip_relay(reason=f"zone_intrusion:{zone.name}")

        for cleared in self._active_zones - seen:
            self._active_zones.discard(cleared)
            event = SafetyEvent(
                event_type=EventType.ZONE_CLEARED,
                severity=Severity.INFO,
                zone_id=cleared,
            )
            await self.runtime.broker.publish(event)
            self._audit(event)

    async def _check_ppe(self, workers: List[Detection], all_dets: List[Detection]) -> None:
        violations = evaluate_ppe(workers, all_dets, self.settings.vision)
        active = set()
        for label, gear, conf in violations:
            key = (label, gear)
            active.add(key)
            if key in self._active_ppe:
                continue
            self._active_ppe.add(key)
            rule = self.settings.vision.ppe_rules.get(f"no-{gear}")
            event = SafetyEvent(
                event_type=EventType.PPE_VIOLATION,
                severity=Severity(rule.get("severity", "WARNING")) if rule else Severity.WARNING,
                rule_id=f"no-{gear}",
                confidence=conf,
                payload={"worker": label, "missing_gear": gear},
            )
            await self.runtime.broker.publish(event)
            self._audit(event)
        self._active_ppe = active

    async def _trip_relay(self, reason: str) -> None:
        latency_ms = self.runtime.plc.trip("HARD")
        with self.runtime.metrics.lock:
            self.runtime.metrics.is_estop_locked = True
            self.runtime.metrics.alert_count += 1
        event = SafetyEvent(
            event_type=EventType.RELAY_TRIP,
            severity=Severity.CRITICAL,
            latency_ms=latency_ms,
            payload={"relay": self.runtime.plc.relay_channel, "reason": reason},
        )
        await self.runtime.broker.publish(event)
        self._audit(event)
        logger.critical("RELAY TRIP by %s in %.2fms", reason, latency_ms)

    # ----------------------------------------------------------------- iiot

    async def _telemetry_loop(self) -> None:
        async for sample in self.telemetry.stream():
            await self.runtime.broker.publish(
                BaseEvent(
                    event_type=EventType.TELEMETRY,
                    severity=_STATUS_SEVERITY.get(sample.status, Severity.INFO),
                    payload=sample.model_dump(),
                )
            )
            anomaly = self.anomaly.evaluate(sample.sensor_id, sample.value)
            if anomaly is not None:
                event_type = ANOMALY_EVENT.get(sample.metric, EventType.ANOMALY_CLEARED)
                severity = (
                    Severity.CRITICAL
                    if anomaly.kind == "THRESHOLD_CRITICAL"
                    else Severity.WARNING
                )
                event = BaseEvent(
                    event_type=event_type,
                    severity=severity,
                    payload={
                        "sensor_id": sample.sensor_id,
                        "kind": anomaly.kind,
                        "value": anomaly.value,
                        "threshold": anomaly.threshold,
                        "message": anomaly.message,
                    },
                )
                await self.runtime.broker.publish(event)
                self._audit(event)
                if sample.sensor_id == "TMP_BRG_02" and anomaly.kind == "THRESHOLD_CRITICAL":
                    await self._trip_relay(reason="thermal_critical")

            if sample.sensor_id == "TMP_BRG_02":
                rul_state = self.rul.observe(sample.value)
                await self.runtime.broker.publish(
                    BaseEvent(
                        event_type=EventType.RUL_UPDATE,
                        severity=_STATUS_SEVERITY.get(rul_state.status, Severity.INFO),
                        payload=rul_state.__dict__,
                    )
                )

    # ---------------------------------------------------------------- support

    def _audit(self, event: BaseEvent) -> None:
        self.runtime.audit.record(
            AuditEntry(
                event_type=event.event_type,
                severity=event.severity,
                latency_ms=event.latency_ms,
                node_id=self.settings.edge.node_id,
                payload=event.payload,
            )
        )

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(5.0)
            await self.runtime.broker.publish(
                BaseEvent(
                    event_type=EventType.HEARTBEAT,
                    severity=Severity.INFO,
                    payload={
                        "node_id": self.settings.edge.node_id,
                        "uptime_s": self.runtime.metrics.snapshot()["uptime_s"],
                        "edge_ok": not self.runtime.plc.is_locked,
                    },
                )
            )

    # ------------------------------------------------------------------- run

    async def run(self, with_api: bool = True) -> None:
        tasks = [
            asyncio.create_task(self._inference_loop(), name="inference"),
            asyncio.create_task(self._telemetry_loop(), name="telemetry"),
            asyncio.create_task(self._heartbeat_loop(), name="heartbeat"),
        ]
        if with_api:
            from .api.server import create_app
            import uvicorn

            app = create_app(self.runtime)
            cfg = self.settings.api
            tasks.append(
                asyncio.create_task(
                    uvicorn.Server(
                        uvicorn.Config(
                            app, host=cfg.host, port=cfg.port, log_level="warning"
                        )
                    ).serve(),
                    name="api",
                )
            )
        logger.info(
            "edge orchestrator online: backend=%s latency_budget=%dms",
            self.settings.edge.inference_backend,
            self.settings.edge.target_latency_ms,
        )
        await asyncio.gather(*tasks)


def _synthetic_frame(w: int, h: int) -> Any:
    try:
        import numpy as np

        return np.zeros((h, w, 3), dtype=np.uint8)
    except ImportError:  # pragma: no cover
        return None


def from_yaml(path: str | Path) -> EdgeOrchestrator:
    return EdgeOrchestrator(Settings.from_yaml(path))


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(from_yaml("config/config.yaml").run())
    except KeyboardInterrupt:
        print("\nedge orchestrator stopped")
