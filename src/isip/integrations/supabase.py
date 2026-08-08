"""Supabase integration for ISIP event persistence."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

from ..config import SupabaseConfig
from ..events.schemas import BaseEvent

logger = logging.getLogger(__name__)

try:
    from supabase import create_client as _supabase_create_client, Client as _SupabaseClient
    _SUPABASE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _SUPABASE_AVAILABLE = False
    _SupabaseClient = Any  # type: ignore[misc,assignment]


class SupabaseClient:
    """Async-friendly wrapper around the Supabase Python client.

    Provides batched insert helpers for the ISIP event stream. If Supabase
    is unavailable or disabled, all methods are no-ops so the rest of the
    system can keep running without it.
    """

    def __init__(self, config: SupabaseConfig) -> None:
        self._config = config
        self._client: Optional[Any] = None
        self._enabled = False
        if _SUPABASE_AVAILABLE and config.enabled and config.url and config.key:
            try:
                self._client = _supabase_create_client(config.url, config.key)
                self._enabled = True
                logger.info("Supabase client initialized url=%s", config.url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Supabase client init failed: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def insert(self, table: str, rows: List[Dict[str, Any]]) -> None:
        if not self._enabled or not rows:
            return
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._insert_sync, table, rows)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Supabase insert failed table=%s rows=%d: %s", table, len(rows), exc)

    def _insert_sync(self, table: str, rows: List[Dict[str, Any]]) -> None:
        try:
            self._client.table(table).insert(rows).execute()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Supabase sync insert failed table=%s: %s", table, exc)


class SupabaseEventStore:
    """Subscribe to broker events and persist them to Supabase.

    Batches inserts by table type to stay within the configured
    ``batch_size`` and ``flush_interval_s``.
    """

    def __init__(self, client: SupabaseClient, node_id: str = "") -> None:
        self._client = client
        self._node_id = node_id
        self._batch_size = client._config.batch_size
        self._flush_interval = client._config.flush_interval_s
        self._queues: Dict[str, deque[Dict[str, Any]]] = {
            "safety_events": deque(),
            "audit_entries": deque(),
            "telemetry_samples": deque(),
            "rul_updates": deque(),
            "inference_events": deque(),
        }
        self._last_flush = time.perf_counter()
        self._flush_lock = asyncio.Lock()

    async def enqueue(self, event: BaseEvent, node_id: str = "") -> None:
        table = _event_table(event.event_type)
        if table is None:
            return
        row = _event_to_row(event, node_id=node_id or self._node_id)
        self._queues[table].append(row)
        if len(self._queues[table]) >= self._batch_size:
            await self._flush_table(table)

    async def maybe_flush(self) -> None:
        now = time.perf_counter()
        if now - self._last_flush < self._flush_interval:
            return
        self._last_flush = now
        for table in list(self._queues):
            if self._queues[table]:
                await self._flush_table(table)

    async def _flush_table(self, table: str) -> None:
        async with self._flush_lock:
            q = self._queues[table]
            if not q:
                return
            rows: List[Dict[str, Any]] = []
            while q and len(rows) < self._batch_size:
                rows.append(q.popleft())
            if rows:
                await self._client.insert(table, rows)

    async def shutdown(self) -> None:
        for table in list(self._queues):
            await self._flush_table(table)


def _event_table(event_type: Any) -> Optional[str]:
    from ..events.schemas import EventType
    mapping = {
        EventType.ZONE_INTRUSION: "safety_events",
        EventType.ZONE_CLEARED: "safety_events",
        EventType.PPE_VIOLATION: "safety_events",
        EventType.RELAY_TRIP: "safety_events",
        EventType.E_STOP_TRIGGERED: "safety_events",
        EventType.E_STOP_RELEASE: "safety_events",
        EventType.TELEMETRY: "telemetry_samples",
        EventType.RUL_UPDATE: "rul_updates",
        EventType.INFERENCE_OK: "inference_events",
    }
    return mapping.get(event_type)


def _event_to_row(event: BaseEvent, node_id: str = "") -> Dict[str, Any]:
    row = {
        "node_id": getattr(event, "node_id", node_id),
        "event_type": event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
        "severity": event.severity.value if hasattr(event.severity, "value") else str(event.severity),
        "latency_ms": getattr(event, "latency_ms", 0.0),
        "payload": event.payload if hasattr(event, "payload") else {},
        "created_at": time.time(),
    }
    if hasattr(event, "zone_id") and event.zone_id:
        row["zone_id"] = event.zone_id
    if hasattr(event, "rule_id") and event.rule_id:
        row["rule_id"] = event.rule_id
    if hasattr(event, "confidence") and event.confidence is not None:
        row["confidence"] = event.confidence
    if hasattr(event, "sensor_id") and event.sensor_id:
        row["sensor_id"] = event.sensor_id
    return row
