"""Pydantic event schemas shared across the edge pipeline."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class EventType(str, Enum):
    INFERENCE_OK = "INFERENCE_OK"
    PPE_VIOLATION = "PPE_VIOLATION"
    ZONE_INTRUSION = "ZONE_INTRUSION"
    ZONE_CLEARED = "ZONE_CLEARED"
    DEVICE_PING = "DEVICE_PING"
    DEVICE_SIGNAL_LOST = "DEVICE_SIGNAL_LOST"
    DEVICE_SIGNAL_OK = "DEVICE_SIGNAL_OK"
    DEVICE_LOW_BATTERY = "DEVICE_LOW_BATTERY"
    THERMAL_ANOMALY = "THERMAL_ANOMALY"
    VIBRATION_ANOMALY = "VIBRATION_ANOMALY"
    GAS_ANOMALY = "GAS_ANOMALY"
    ANOMALY_CLEARED = "ANOMALY_CLEARED"
    RUL_UPDATE = "RUL_UPDATE"
    TELEMETRY = "TELEMETRY"
    RELAY_TRIP = "RELAY_TRIP"
    E_STOP_TRIGGERED = "E_STOP_TRIGGERED"
    E_STOP_RELEASE = "E_STOP_RELEASE"
    POWER_CUT = "POWER_CUT"
    SYSTEM_SAFE_STATE = "SYSTEM_SAFE_STATE"
    HEARTBEAT = "HEARTBEAT"


class BaseEvent(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    event_type: EventType
    severity: Severity = Severity.INFO
    latency_ms: float = 0.0
    payload: Dict[str, Any] = Field(default_factory=dict)


class InferenceEvent(BaseEvent):
    detections: List[Dict[str, Any]] = Field(default_factory=list)
    fps: float = 0.0


class SafetyEvent(BaseEvent):
    zone_id: Optional[str] = None
    rule_id: Optional[str] = None
    confidence: float = 0.0


class TelemetrySample(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    sensor_id: str
    metric: str
    value: float
    unit: str = ""
    status: Literal["NORMAL", "WARN", "CRITICAL"] = "NORMAL"


class EStopEvent(BaseEvent):
    relay: str = ""
    mode: str = "HARD"
    line_id: str = ""


class AuditEntry(BaseModel):
    """Structured row persisted to the append-only audit ledger."""

    timestamp: float = Field(default_factory=time.time)
    event_type: EventType
    severity: Severity
    latency_ms: float
    execution_target: str = "TensorRT_INT8_Mock"
    payload: Dict[str, Any] = Field(default_factory=dict)
    node_id: str = "edge-node-01"
