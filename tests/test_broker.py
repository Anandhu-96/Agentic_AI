import asyncio

from isip.events.broker import EventBroker
from isip.events.schemas import BaseEvent, EventType, Severity


def test_publish_subscribe_roundtrip():
    broker = EventBroker()

    async def scenario():
        received = []
        consumer = asyncio.create_task(_consume(broker, received))
        await asyncio.sleep(0.05)
        await broker.publish(
            BaseEvent(event_type=EventType.ZONE_INTRUSION, severity=Severity.CRITICAL)
        )
        await asyncio.sleep(0.05)
        consumer.cancel()
        return received

    received = asyncio.run(scenario())
    assert len(received) == 1
    assert received[0].event_type == EventType.ZONE_INTRUSION


def test_filtered_subscription():
    broker = EventBroker()

    async def scenario():
        received = []
        consumer = asyncio.create_task(
            _consume(broker, received, [EventType.HEARTBEAT])
        )
        await asyncio.sleep(0.05)
        await broker.publish(
            BaseEvent(event_type=EventType.ZONE_INTRUSION, severity=Severity.CRITICAL)
        )
        await broker.publish(
            BaseEvent(event_type=EventType.HEARTBEAT, severity=Severity.INFO)
        )
        await asyncio.sleep(0.05)
        consumer.cancel()
        return received

    received = asyncio.run(scenario())
    assert len(received) == 1
    assert received[0].event_type == EventType.HEARTBEAT


def test_snapshot_holds_history():
    broker = EventBroker()

    async def scenario():
        for _ in range(3):
            await broker.publish(
                BaseEvent(event_type=EventType.INFERENCE_OK, severity=Severity.INFO)
            )

    asyncio.run(scenario())
    assert len(broker.snapshot()) == 3


async def _consume(broker, out, types=None):
    async for event in broker.subscribe(types):
        out.append(event)
