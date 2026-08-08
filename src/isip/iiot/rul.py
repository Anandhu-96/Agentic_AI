"""Remaining Useful Life (RUL) estimation for thermal degradation.

Uses the Arrhenius-based 10-degree rule: each 10C above the nominal operating
temperature doubles the rate of thermal aging. Cumulative life consumption
tracks toward a rated lifetime so maintenance can be scheduled before failure.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ..config import RulConfig

logger = logging.getLogger(__name__)


@dataclass
class RulState:
    sensor_id: str
    remaining_hours: float
    health_pct: float
    consumed_hours: float
    accumulated_damage: float
    status: str


class ThermalRulEstimator:
    def __init__(self, config: RulConfig) -> None:
        self.config = config
        self._damage = 0.0
        self._last_sample_ts: float | None = None
        self._last_temp = config.nominal_temp_c

    @staticmethod
    def _acceleration_factor(temp_c: float, nominal_c: float) -> float:
        if temp_c <= nominal_c:
            return 1.0
        # Doubling rate per 10C (Q10 rule).
        return 2.0 ** ((temp_c - nominal_c) / 10.0)

    def consume(self, temp_c: float, elapsed_h: float) -> float:
        """Add accrued thermal damage for an elapsed interval."""
        af = self._acceleration_factor(temp_c, self.config.nominal_temp_c)
        damage = elapsed_h * af
        self._damage += damage
        self._last_temp = temp_c
        return damage

    def observe(self, temp_c: float) -> "RulState":
        now = time.time()
        if self._last_sample_ts is not None:
            elapsed_h = (now - self._last_sample_ts) / 3600.0
            self.consume(temp_c, elapsed_h)
        self._last_sample_ts = now

        rated = self.config.max_rated_life_h
        remaining = max(0.0, rated - self._damage)
        health = max(0.0, min(100.0, (remaining / rated) * 100.0))
        status = (
            "CRITICAL" if health < 20.0
            else "WARN" if health < 50.0
            else "HEALTHY"
        )
        if status == "CRITICAL":
            logger.warning("RUL CRITICAL health=%.1f%% remaining=%.1fh damage=%.3f",
                           health, remaining, self._damage)
        elif status == "WARN":
            logger.info("RUL WARN health=%.1f%% remaining=%.1fh damage=%.3f",
                        health, remaining, self._damage)
        return RulState(
            sensor_id="TMP_BRG_02",
            remaining_hours=remaining,
            health_pct=health,
            consumed_hours=self._damage,
            accumulated_damage=self._damage,
            status=status,
        )
