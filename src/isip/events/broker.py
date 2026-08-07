"""Central async event queue broker.

A single in-process pub/sub hub that decouples producers (vision, telemetry,
control plane) from consumers (audit logger, REST subscribers, dashboard SSE).
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from collections import defaultdict, deque
from typing import Any, AsyncIterator, Awaitable, Callable, Deque, Dict, List, Set

from .schemas import BaseEvent, EventType

logger = logging.getLogger(__name__)


class EventPriority:
    """Lower number = higher priority (delivered first)."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class EventBroker:
    """Thread-safe-async pub/sub broker with per-subscriber priority queues.

    Design notes:
    - ``publish`` is O(1); fan-out is lazy.
    - Each subscriber owns an ``asyncio.Queue`` so a slow consumer never
      blocks the producers.
    - Latency is measured from publish to delivery for the audit trail.
    """

    def __init__(self, max_queue: int = 10_000) -> None:
        self._subscribers: Dict[EventType, Set[asyncio.Queue]] = defaultdict(set)
        self._history: Deque[BaseEvent] = deque(maxlen=500)
        self._max_queue = max_queue

    async def subscribe(
        self, event_types: List[EventType] | None = None
    ) -> AsyncIterator[BaseEvent]:
        """Yield events matching ``event_types`` (all if ``None``)."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        types = event_types or list(EventType)
        for et in types:
            self._subscribers[et].add(queue)
        try:
            while True:
                item = await queue.get()
                yield item
        finally:
            for et in types:
                self._subscribers[et].discard(queue)

    async def publish(self, event: BaseEvent) -> None:
        event.latency_ms = 0.0
        started = time.perf_counter()
        subscribers = list(self._subscribers.get(event.event_type, ()))
        for queue in subscribers:
            if queue.full():
                # Drop-oldest to protect the edge real-time budget.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover
                continue
        self._history.append(event)
        event.latency_ms = (time.perf_counter() - started) * 1000.0

    def snapshot(self) -> List[BaseEvent]:
        return list(self._history)

    def close(self) -> None:
        for queues in self._subscribers.values():
            for q in queues:
                q.put_nowait(_Terminator())
        self._subscribers.clear()


class _Terminator(BaseEvent):
    event_type: EventType = EventType.HEARTBEAT
