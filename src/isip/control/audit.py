"""Append-only audit ledger.

Persists every safety/telemetry/control event as a JSONL row for compliance
review, with a small in-memory ring for the REST/dashboard surface.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

from ..events.schemas import AuditEntry

logger = logging.getLogger(__name__)


class AuditLogger:
    def __init__(self, audit_dir: str = "logs", node_id: str = "edge-node-01",
                 ring_size: int = 500) -> None:
        self.node_id = node_id
        self._dir = Path(audit_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "eval_audit.jsonl"
        self._ring: Deque[Dict] = deque(maxlen=ring_size)
        self._lock = threading.Lock()

    def record(self, entry: AuditEntry) -> None:
        row = entry.model_dump()
        with self._lock:
            self._ring.append(row)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")

    def latest(self, limit: int = 100) -> List[Dict]:
        with self._lock:
            return list(self._ring)[-limit:][::-1]

    def clear(self) -> None:
        with self._lock:
            self._ring.clear()
            self._path.unlink(missing_ok=True)
        logger.info("audit ledger cleared")
