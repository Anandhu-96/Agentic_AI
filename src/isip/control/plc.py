"""Mock PLC Emergency Stop (E-Stop) relay.

Abstraction over the local hardware relay that cuts power to machinery. Uses
RPi.GPIO on real edge devices (Jetson/Pi) and falls back to an in-process mock
that records trips and measures latency, so the full safety loop can be
demonstrated on any machine.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class PlcRelay:
    def __init__(self, gpio: int, relay_channel: str, line_id: str,
                 use_gpio: bool = False, trip_delay_ms: int = 0) -> None:
        self._gpio = gpio
        self.relay_channel = relay_channel
        self.line_id = line_id
        self._trip_delay_s = trip_delay_ms / 1000.0
        self._locked = False
        self._powered = True
        self._last_latency_ms: Optional[float] = None
        self._gpio_iface = None
        if use_gpio:
            self._init_gpio()

    def _init_gpio(self) -> None:
        try:
            import RPi.GPIO as GPIO

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._gpio, GPIO.OUT, initial=GPIO.HIGH)
            self._gpio_iface = GPIO
            logger.info("PLC relay bound to GPIO-%d", self._gpio)
        except (ImportError, RuntimeError) as exc:
            logger.warning("RPi.GPIO unavailable (%s) - mock relay", exc)

    @property
    def is_locked(self) -> bool:
        return self._locked

    @property
    def is_powered(self) -> bool:
        return self._powered

    @property
    def last_latency_ms(self) -> Optional[float]:
        return self._last_latency_ms

    def trip(self, mode: str = "HARD") -> float:
        """Cut power to the line; returns trip latency in milliseconds."""
        started = time.perf_counter()
        if self._trip_delay_s > 0:
            time.sleep(self._trip_delay_s)
        self._powered = False
        self._locked = True
        if self._gpio_iface is not None:
            self._gpio_iface.output(self._gpio, self._gpio_iface.LOW)
        self._last_latency_ms = (time.perf_counter() - started) * 1000.0
        logger.warning(
            "E-STOP TRIP mode=%s relay=%s line=%s latency=%.2fms",
            mode, self.relay_channel, self.line_id, self._last_latency_ms,
        )
        return self._last_latency_ms

    def release(self) -> None:
        self._powered = True
        self._locked = False
        if self._gpio_iface is not None:
            self._gpio_iface.output(self._gpio, self._gpio_iface.HIGH)
        logger.info("E-STOP released for line %s", self.line_id)

    def status(self) -> dict:
        return {
            "relay_channel": self.relay_channel,
            "line_id": self.line_id,
            "powered": self._powered,
            "locked": self._locked,
            "last_latency_ms": self._last_latency_ms,
        }
