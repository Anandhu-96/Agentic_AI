"""Serial-port worker tracking device reader.

Workers carry tracking badges/devices that stream one position ping per line
over a serial port (USB/UART):

    DEV-001,W-1,0.52,0.64,96
    <device_id>,<worker_id>,<x_norm>,<y_norm>,<battery_pct>

Coordinates are normalized 0..1 so they line up with the dashboard's zone map
and the geofence engine. When no serial port is configured (or pyserial is
unavailable) the tracker falls back to an emulated stream so the full
real-world pipeline stays testable without hardware.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Dict, List, Optional

from ..config import SerialTrackerConfig

logger = logging.getLogger(__name__)

DEFAULT_DEVICES = [
    ("DEV-001", "W-1"),
    ("DEV-002", "W-2"),
    ("DEV-003", "W-3"),
]


def _worker_for(device_id: str) -> str:
    for dev, worker in DEFAULT_DEVICES:
        if dev == device_id:
            return worker
    return device_id


def parse_ping(line: str) -> Optional[dict]:
    """Parse one serial line into ``{id, worker, x, y, battery, signal}``."""
    line = line.strip()
    if not line or line.startswith(("#", "//")):
        return None
    parts = [p.strip() for p in line.split(",")]
    try:
        if len(parts) >= 5:
            dev_id, worker = parts[0], parts[1]
            x, y, batt = float(parts[2]), float(parts[3]), float(parts[4])
        elif len(parts) >= 4:
            dev_id = parts[0]
            x, y, batt = float(parts[1]), float(parts[2]), float(parts[3])
            worker = _worker_for(dev_id)
        else:
            return None
        if not dev_id:
            return None
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None
        return {
            "id": dev_id,
            "worker": worker or dev_id,
            "x": x,
            "y": y,
            "battery": max(0.0, min(100.0, batt)),
            "signal": True,
        }
    except (ValueError, IndexError):
        return None


class SerialDeviceTracker:
    """Reads worker-device pings from a serial port (or an emulated stream)."""

    def __init__(self, config: SerialTrackerConfig) -> None:
        self.config = config
        self._devices: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._source = "none"
        self._last_ping = 0.0

    @property
    def source(self) -> str:
        return self._source

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("serial device tracker started port=%r emulate=%s", self.config.port, self.config.emulate)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        ser = self._open_serial()
        if ser is None:
            if self.config.emulate:
                self._source = "emulated"
                logger.info("serial tracker: no serial port, emulating device stream")
                self._emulate()
            else:
                self._source = "none"
                logger.warning("serial tracker: no serial port and emulation disabled")
            return

        self._source = "serial"
        logger.info("serial tracker reading %s @ %s baud", self.config.port, self.config.baudrate)
        try:
            while self._running:
                try:
                    line = ser.readline()
                    if not line:
                        continue
                    self._handle(line.decode(errors="ignore"))
                except Exception as exc:  # pragma: no cover - hw error path
                    logger.warning("serial tracker read error: %s", exc)
                    time.sleep(0.5)
        finally:
            try:
                ser.close()
            except Exception:  # pragma: no cover
                pass

    def _open_serial(self):
        port = self.config.port.strip()
        if not port:
            return None
        try:
            import serial  # type: ignore

            return serial.Serial(port, self.config.baudrate, timeout=self.config.timeout_s)
        except Exception as exc:
            logger.warning("serial tracker: cannot open %r (%s)", port, exc)
            return None

    def _emulate(self) -> None:
        t = 0.0
        while self._running:
            t += 1
            for idx, (dev_id, worker) in enumerate(DEFAULT_DEVICES):
                x = 0.18 + 0.18 * idx + 0.12 * math.sin(t / 22.0 + idx)
                y = 0.55 + 0.22 * math.sin(t / 26.0 + idx * 1.7)
                batt = 100.0 - ((t * 0.03) % 45.0)
                self._handle(f"{dev_id},{worker},{x:.3f},{y:.3f},{batt:.1f}")
            time.sleep(self.config.ping_interval_s)

    def _handle(self, line: str) -> None:
        ping = parse_ping(line)
        if ping is None:
            return
        with self._lock:
            self._devices[ping["id"]] = {**ping, "last_ping": time.time()}
        self._last_ping = time.time()

    def snapshot(self) -> dict:
        with self._lock:
            devices = [dict(d) for d in self._devices.values()]
        devices.sort(key=lambda d: d["id"])
        return {
            "source": self._source,
            "devices": devices,
            "last_ping": self._last_ping,
        }
