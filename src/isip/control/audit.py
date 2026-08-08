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
                 ring_size: int = 500, max_bytes: int = 10 * 1024 * 1024,
                 backup_count: int = 3, supabase_client=None) -> None:
        self.node_id = node_id
        self._dir = Path(audit_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "eval_audit.jsonl"
        self._ring: Deque[Dict] = deque(maxlen=ring_size)
        self._lock = threading.Lock()
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._supabase = supabase_client

    def _rotate(self) -> None:
        for i in range(self._backup_count - 1, 0, -1):
            src = self._dir / f"eval_audit.jsonl.{i}"
            dst = self._dir / f"eval_audit.jsonl.{i + 1}"
            if src.exists():
                src.replace(dst)
        rotated = self._dir / "eval_audit.jsonl.1"
        if rotated.exists():
            rotated.unlink()
        self._path.replace(rotated)

    def record(self, entry: AuditEntry) -> None:
        row = entry.model_dump()
        with self._lock:
            self._ring.append(row)
            try:
                if self._path.exists() and self._path.stat().st_size >= self._max_bytes:
                    self._rotate()
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, default=str) + "\n")
            except OSError as exc:
                logger.error("failed to write audit entry to %s: %s", self._path, exc)
        if self._supabase is not None and getattr(self._supabase, "enabled", False):
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                sb_row = {
                    "node_id": entry.node_id,
                    "event_type": entry.event_type.value if hasattr(entry.event_type, "value") else str(entry.event_type),
                    "severity": entry.severity.value if hasattr(entry.severity, "value") else str(entry.severity),
                    "latency_ms": entry.latency_ms,
                    "payload": entry.payload,
                    "created_at": time.time(),
                }
                asyncio.run_coroutine_threadsafe(
                    self._supabase.insert("audit_entries", [sb_row]),
                    loop,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to enqueue audit entry to supabase: %s", exc)

    def latest(self, limit: int = 100) -> List[Dict]:
        with self._lock:
            return list(self._ring)[-limit:][::-1]

    def clear(self) -> None:
        with self._lock:
            self._ring.clear()
            self._path.unlink(missing_ok=True)
        logger.info("audit ledger cleared")
