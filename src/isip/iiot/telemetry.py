"""IIoT telemetry engine.

Produces industrial sensor samples (temperature, vibration, toxic gas) from
emulated sources by default, and can publish them over MQTT/Modbus-TCP on real
deployments. The protocol transport is pluggable; the emulator keeps the edge
pipeline fully exercisable without hardware.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import AsyncIterator, Dict, Optional

from ..config import IiotConfig
from ..events.schemas import TelemetrySample

logger = logging.getLogger(__name__)


class EmulatedSensor:
    """Stateful synthetic industrial sensor with realistic noise and drift."""

    def __init__(self, sensor_id: str, metric: str, base: float, unit: str,
                 noise: float = 0.6, drift: float = 0.02) -> None:
        self.sensor_id = sensor_id
        self.metric = metric
        self.base = base
        self.unit = unit
        self.noise = noise
        self.drift = drift
        self._value = base
        self._t = 0.0

    def read(self) -> float:
        self._t += 1
        cycle = random.uniform(-1, 1)
        self._value = self.base + self.drift * self._t + self.noise * cycle
        return round(max(0.0, self._value), 2)


class TelemetryEngine:
    """Samples configured sensors and emits :class:`TelemetrySample` events."""

    SENSOR_PROFILES: Dict[str, dict] = {
        "TMP_BRG_02": {"metric": "temperature", "base": 68.0, "unit": "C", "noise": 0.9},
        "VIB_MTR_01": {"metric": "vibration", "base": 42.0, "unit": "Hz", "noise": 1.8},
        "GAS_CO_03": {"metric": "co_ppm", "base": 8.0, "unit": "ppm", "noise": 1.2},
    }

    def __init__(self, config: IiotConfig) -> None:
        self.config = config
        self._sensors = [
            EmulatedSensor(
                sensor_id=sid,
                **self.SENSOR_PROFILES.get(sid, {"metric": "value", "base": 50.0, "unit": ""}),
            )
            for sid in config.sensor_ids
        ]

    async def stream(self, interval_s: float | None = None) -> AsyncIterator[TelemetrySample]:
        interval = interval_s or self.config.sampling_interval_s
        while True:
            await asyncio.sleep(interval)
            for sensor in self._sensors:
                value = sensor.read()
                threshold = self.config.thresholds.get(sensor.sensor_id)
                status = "NORMAL"
                if threshold is not None:
                    if value >= threshold.critical:
                        status = "CRITICAL"
                    elif value >= threshold.warn:
                        status = "WARN"
                yield TelemetrySample(
                    sensor_id=sensor.sensor_id,
                    metric=sensor.metric,
                    value=value,
                    unit=sensor.unit,
                    status=status,
                )
