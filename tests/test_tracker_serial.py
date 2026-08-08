"""Tests for the serial-port worker tracking device parser."""

from unittest.mock import patch

from isip.config import SerialTrackerConfig
from isip.vision.tracker_serial import SerialDeviceTracker, parse_ping


def test_parse_full_line():
    assert parse_ping("DEV-001,W-1,0.52,0.64,96") == {
        "id": "DEV-001",
        "worker": "W-1",
        "x": 0.52,
        "y": 0.64,
        "battery": 96.0,
        "signal": True,
    }


def test_parse_short_line_infers_worker():
    ping = parse_ping("DEV-002,0.30,0.70,88")
    assert ping["id"] == "DEV-002"
    assert ping["worker"] == "W-2"
    assert ping["battery"] == 88.0


def test_parse_rejects_garbage():
    assert parse_ping("") is None
    assert parse_ping("# comment") is None
    assert parse_ping("not-a-ping") is None
    assert parse_ping("DEV-001,W-1,5.0,0.5,99") is None  # x out of range


def test_snapshot_after_handle():
    tracker = SerialDeviceTracker(SerialTrackerConfig())
    tracker._handle("DEV-001,W-1,0.4,0.6,50")
    tracker._handle("DEV-002,W-2,0.2,0.8,40")
    snap = tracker.snapshot()
    assert snap["source"] == "none"  # no port opened yet
    assert [d["id"] for d in snap["devices"]] == ["DEV-001", "DEV-002"]


def test_emulated_stream_populates_devices():
    tracker = SerialDeviceTracker(SerialTrackerConfig(ping_interval_s=0.01))
    tracker.start()
    for _ in range(20):
        tracker.snapshot()
    import time
    time.sleep(0.05)
    tracker.stop()
    snap = tracker.snapshot()
    assert snap["source"] == "emulated"
    assert len(snap["devices"]) == 3
    assert all(0.0 <= d["x"] <= 1.0 for d in snap["devices"])
