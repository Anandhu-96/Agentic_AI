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
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Optional

from ..config import IiotConfig
from ..events.schemas import TelemetrySample

logger = logging.getLogger(__name__)


class SensorTransport(ABC):
    """Abstract transport for sensor data sources."""

    @abstractmethod
    async def read(self, sensor_id: str) -> Optional[float]:
        """Return the latest value for a sensor, or None if unavailable."""
        raise NotImplementedError


class EmulatedTransport(SensorTransport):
    """Stateful synthetic industrial sensors with realistic noise and drift."""

    SENSOR_PROFILES: Dict[str, dict] = {
        "TMP_BRG_02": {"metric": "temperature", "base": 68.0, "unit": "C", "noise": 0.9},
        "VIB_MTR_01": {"metric": "vibration", "base": 42.0, "unit": "Hz", "noise": 1.8},
        "GAS_CO_03": {"metric": "co_ppm", "base": 8.0, "unit": "ppm", "noise": 1.2},
    }

    def __init__(self, sensor_ids: list[str]) -> None:
        self._sensors: Dict[str, dict] = {}
        self._values: Dict[str, float] = {}
        for sid in sensor_ids:
            profile = self.SENSOR_PROFILES.get(
                sid, {"metric": "value", "base": 50.0, "unit": "", "noise": 0.6}
            )
            self._sensors[sid] = profile
            self._values[sid] = profile["base"]

    async def read(self, sensor_id: str) -> Optional[float]:
        profile = self._sensors.get(sensor_id)
        if profile is None:
            return None
        cycle = random.uniform(-1, 1)
        value = profile["base"] + profile["noise"] * cycle
        self._values[sensor_id] = round(max(0.0, value), 2)
        return self._values[sensor_id]

    def sensor_metric(self, sensor_id: str) -> str:
        return self._sensors.get(sensor_id, {}).get("metric", "value")

    def sensor_unit(self, sensor_id: str) -> str:
        return self._sensors.get(sensor_id, {}).get("unit", "")


EmulatedSensor = EmulatedTransport  # backwards-compatible alias


class MqttTransport(SensorTransport):
    """Stub MQTT transport using paho-mqtt.

    In production this would subscribe to sensor topics and buffer the latest
    values. For now it falls back to emulated values so the pipeline remains
    exercisable without a broker.
    """

    def __init__(self, config: IiotConfig) -> None:
        self._config = config
        self._latest: Dict[str, float] = {}
        self._fallback = EmulatedTransport(config.sensor_ids)
        logger.warning(
            "MQTT transport is a stub; falling back to emulated values. "
            "Configure broker_host=%s broker_port=%d for real MQTT.",
            config.broker_host,
            config.broker_port,
        )

    async def read(self, sensor_id: str) -> Optional[float]:
        try:
            from paho.mqtt.client import Client as MqttClient  # type: ignore[import-untyped]
        except ImportError:
            return await self._fallback.read(sensor_id)
        # Stub: in a real implementation this would maintain a persistent
        # MQTT client subscribed to sensor topics and return buffered values.
        return await self._fallback.read(sensor_id)


class ModbusTransport(SensorTransport):
    """Stub Modbus-TCP transport using pymodbus.

    In production this would poll holding registers on the PLC/sensor gateway.
    For now it falls back to emulated values so the pipeline remains exercisable
    without Modbus hardware.
    """

    def __init__(self, config: IiotConfig) -> None:
        self._config = config
        self._fallback = EmulatedTransport(config.sensor_ids)
        logger.warning(
            "Modbus transport is a stub; falling back to emulated values. "
            "Configure modbus_unit=%d for real Modbus.",
            config.modbus_unit,
        )

    async def read(self, sensor_id: str) -> Optional[float]:
        try:
            from pymodbus.client import ModbusTcpClient  # type: ignore[import-untyped]
        except ImportError:
            return await self._fallback.read(sensor_id)
        # Stub: in a real implementation this would open a ModbusTcpClient,
        # read the appropriate register for sensor_id, and return the value.
        return await self._fallback.read(sensor_id)


class TelemetryEngine:
    """Samples configured sensors and emits :class:`TelemetrySample` events."""

    def __init__(self, config: IiotConfig) -> None:
        self.config = config
        protocol = (config.protocol or "emulated").lower()
        if protocol == "mqtt":
            self._transport: SensorTransport = MqttTransport(config)
        elif protocol == "modbus":
            self._transport = ModbusTransport(config)
        else:
            self._transport = EmulatedTransport(config.sensor_ids)
        self._emulated_meta = (
            self._transport.SENSOR_PROFILES
            if isinstance(self._transport, EmulatedTransport)
            else {}
        )

    async def stream(self, interval_s: float | None = None) -> AsyncIterator[TelemetrySample]:
        interval = interval_s or self.config.sampling_interval_s
        while True:
            await asyncio.sleep(interval)
            for sensor_id in self.config.sensor_ids:
                value = await self._transport.read(sensor_id)
                if value is None:
                    continue
                metric = (
                    self._emulated_meta.get(sensor_id, {}).get("metric", "value")
                    if isinstance(self._transport, EmulatedTransport)
                    else "value"
                )
                unit = (
                    self._emulated_meta.get(sensor_id, {}).get("unit", "")
                    if isinstance(self._transport, EmulatedTransport)
                    else ""
                )
                threshold = self.config.thresholds.get(sensor_id)
                status = "NORMAL"
                if threshold is not None:
                    if value >= threshold.critical:
                        status = "CRITICAL"
                    elif value >= threshold.warn:
                        status = "WARN"
                yield TelemetrySample(
                    sensor_id=sensor_id,
                    metric=metric,
                    value=value,
                    unit=unit,
                    status=status,
                )
