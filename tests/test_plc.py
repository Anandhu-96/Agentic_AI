from isip.control.plc import PlcRelay


def test_trip_cuts_power_and_locks():
    relay = PlcRelay(gpio=17, relay_channel="GPIO-3", line_id="LINE_01")
    latency = relay.trip("HARD")
    assert latency >= 0.0
    assert not relay.is_powered
    assert relay.is_locked


def test_release_restores_power():
    relay = PlcRelay(gpio=17, relay_channel="GPIO-3", line_id="LINE_01")
    relay.trip("HARD")
    relay.release()
    assert relay.is_powered
    assert not relay.is_locked


def test_trip_latency_is_measured():
    relay = PlcRelay(gpio=17, relay_channel="GPIO-3", line_id="LINE_01", trip_delay_ms=5)
    latency = relay.trip("HARD")
    assert latency >= 5.0
    assert relay.last_latency_ms == latency


def test_status_payload():
    relay = PlcRelay(gpio=17, relay_channel="GPIO-3", line_id="LINE_01")
    relay.trip("HARD")
    status = relay.status()
    assert status["locked"] is True
    assert status["line_id"] == "LINE_01"
