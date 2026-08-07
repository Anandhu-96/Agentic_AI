"""Shared edge runtime state wiring broker, control plane, and audit together."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .config import Settings
from .control.audit import AuditLogger
from .control.plc import PlcRelay
from .events.broker import EventBroker


@dataclass
class RuntimeMetrics:
    last_latency_ms: float = 0.0
    last_fps: float = 0.0
    alert_count: int = 0
    inference_count: int = 0
    started_at: float = field(default_factory=time.time)
    is_estop_locked: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "node_id": "",
                "uptime_s": round(time.time() - self.started_at, 1),
                "latency_ms": round(self.last_latency_ms, 2),
                "fps": round(self.last_fps, 2),
                "alerts": self.alert_count,
                "inferences": self.inference_count,
                "is_estop_locked": self.is_estop_locked,
            }


class EdgeRuntime:
    """Composition root for a single edge node."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.broker = EventBroker()
        self.metrics = RuntimeMetrics()
        self.audit = AuditLogger(
            audit_dir=settings.logging.audit_dir,
            node_id=settings.edge.node_id,
        )
        self.plc = PlcRelay(
            gpio=settings.control.estop_gpio,
            relay_channel=settings.control.relay_channel,
            line_id=settings.control.line_id,
            use_gpio=settings.control.use_gpio,
            trip_delay_ms=settings.control.trip_delay_ms,
        )
