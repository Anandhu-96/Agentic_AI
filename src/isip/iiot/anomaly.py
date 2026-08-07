"""Sensor anomaly detection: threshold rules + rolling statistics.

Combines configured warn/critical thresholds with an exponential moving
average (EMA) and z-score test to catch slow drift and sudden spikes that a
fixed threshold would miss.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from ..config import ThresholdConfig


@dataclass
class SensorAnomaly:
    sensor_id: str
    kind: str                      # THRESHOLD_WARN | THRESHOLD_CRITICAL | SPIKE | DRIFT
    value: float
    threshold: Optional[float]
    message: str


@dataclass
class _RollingWindow:
    window: Deque[float] = field(default_factory=lambda: deque(maxlen=120))
    ema: Optional[float] = None

    def push(self, value: float, alpha: float = 0.3) -> None:
        self.window.append(value)
        self.ema = value if self.ema is None else alpha * value + (1 - alpha) * self.ema

    def zscore(self, value: float) -> float:
        if len(self.window) < 15:
            return 0.0
        mean = statistics.fmean(self.window)
        stdev = statistics.pstdev(self.window)
        if stdev == 0:
            return 0.0
        return (value - mean) / stdev


class AnomalyEngine:
    def __init__(self, thresholds: Dict[str, ThresholdConfig],
                 spike_z: float = 4.0, drift_z: float = 3.0) -> None:
        self._thresholds = thresholds
        self._spike_z = spike_z
        self._drift_z = drift_z
        self._roll: Dict[str, _RollingWindow] = {}

    def evaluate(self, sensor_id: str, value: float) -> Optional[SensorAnomaly]:
        roll = self._roll.setdefault(sensor_id, _RollingWindow())
        z = roll.zscore(value)

        threshold = self._thresholds.get(sensor_id)
        if threshold is not None:
            if value >= threshold.critical:
                roll.push(value)
                return SensorAnomaly(
                    sensor_id, "THRESHOLD_CRITICAL", value, threshold.critical,
                    f"{sensor_id} critical: {value:.1f} >= {threshold.critical}",
                )
            if value >= threshold.warn:
                roll.push(value)
                return SensorAnomaly(
                    sensor_id, "THRESHOLD_WARN", value, threshold.warn,
                    f"{sensor_id} warn: {value:.1f} >= {threshold.warn}",
                )

        if z > self._spike_z:
            roll.push(value)
            return SensorAnomaly(
                sensor_id, "SPIKE", value, None,
                f"{sensor_id} sudden spike (z={z:.1f}): {value:.1f}",
            )
        if z < -self._drift_z:
            roll.push(value)
            return SensorAnomaly(
                sensor_id, "DRIFT", value, None,
                f"{sensor_id} drift detected (z={z:.1f}): {value:.1f}",
            )
        roll.push(value)
        return None
