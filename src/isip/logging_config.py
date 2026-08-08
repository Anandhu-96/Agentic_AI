"""Centralized logging configuration for ISIP.

Configures console + rotating file handlers, log level from Settings, and
injects a ``component`` extra into every record so operators can filter by
module (vision, iiot, control, api, dashboard, etc.).
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .config import LoggingConfig


_COMPONENT_PREFIXES = {
    "isip.orchestrator": "orchestrator",
    "isip.runtime": "runtime",
    "isip.api.server": "api",
    "isip.control.plc": "control",
    "isip.control.audit": "control",
    "isip.events.broker": "events",
    "isip.iiot.telemetry": "iiot",
    "isip.iiot.anomaly": "iiot",
    "isip.iiot.rul": "iiot",
    "isip.vision.detector": "vision",
    "isip.vision.geofence": "vision",
    "isip.vision.ppe": "vision",
    "isip.dashboard.app": "dashboard",
    "isip.training.train": "training",
}


class _ComponentFilter(logging.Filter):
    """Attach a human-readable component name to each log record based on logger name."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.component = _COMPONENT_PREFIXES.get(record.name, record.name.split(".")[0])
        return True


def setup_logging(config: Optional[LoggingConfig] = None, node_id: str = "edge-node-01") -> None:
    """Configure root logger from Settings.

    Call this once at process startup before importing other ISIP modules so
    every subsequent ``logging.getLogger(__name__)`` inherits the handlers.
    """
    if config is None:
        config = LoggingConfig()

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    if root.handlers:
        return

    fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)-7s [%(component)-10s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    console.setFormatter(fmt)
    root.addHandler(console)

    log_path = Path(config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.component = _COMPONENT_PREFIXES.get(record.name, record.name.split(".")[0])
        return record
    logging.setLogRecordFactory(record_factory)

    logging.getLogger("isip").info(
        "logging initialized node=%s level=%s file=%s", node_id, config.level, log_path
    )
