from isip.config import ThresholdConfig
from isip.iiot.anomaly import AnomalyEngine

THRESHOLDS = {
    "TMP": ThresholdConfig(warn=85.0, critical=95.0),
}


def test_normal_value():
    engine = AnomalyEngine(THRESHOLDS)
    assert engine.evaluate("TMP", 60.0) is None


def test_warn_threshold():
    engine = AnomalyEngine(THRESHOLDS)
    anomaly = engine.evaluate("TMP", 90.0)
    assert anomaly is not None
    assert anomaly.kind == "THRESHOLD_WARN"


def test_critical_threshold():
    engine = AnomalyEngine(THRESHOLDS)
    anomaly = engine.evaluate("TMP", 96.0)
    assert anomaly.kind == "THRESHOLD_CRITICAL"


def test_spike_detection():
    import random

    engine = AnomalyEngine(THRESHOLDS, spike_z=3.0)
    for _ in range(60):
        engine.evaluate("TMP", 60.0 + random.uniform(-1.0, 1.0))
    anomaly = engine.evaluate("TMP", 75.0)  # below warn threshold (85)
    assert anomaly is not None
    assert anomaly.kind == "SPIKE"
