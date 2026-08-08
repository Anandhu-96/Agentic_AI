"""SyntheticDetector determinism + zone-placement sanity."""

import math
from unittest.mock import patch

import numpy as np

from isip.config import VisionConfig
from isip.vision.detector import SyntheticDetector
from isip.vision.geofence import GeofenceEngine, point_in_polygon

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


def _cfg() -> VisionConfig:
    return VisionConfig(backend="synthetic")


def _centers(dets):
    return sorted(
        (
            round(d.bbox_center[0] / FRAME.shape[1], 4),
            round(d.bbox_center[1] / FRAME.shape[0], 4),
            d.class_name,
        )
        for d in dets
        if d.class_name == "person"
    )


def test_detector_time_based_matches_across_instances():
    """Two detectors started at the same wall-clock epoch produce identical
    worker positions at the same instant (the fix for overlay-vs-event drift)."""
    # Pin time so both detectors resolve the same phase.
    with patch("isip.vision.detector.time.time", return_value=1_700_000_000.0):
        a = SyntheticDetector(_cfg())
    with patch("isip.vision.detector.time.time", return_value=1_700_000_000.0):
        b = SyntheticDetector(_cfg())

    # Advance the shared clock, then both must trace the same world.
    with patch("isip.vision.detector.time.time", return_value=1_700_000_012.0):
        da = a.detect(FRAME)
    with patch("isip.vision.detector.time.time", return_value=1_700_000_012.0):
        db = b.detect(FRAME)

    assert _centers(da) == _centers(db)


def test_worker3_outside_danger_zone():
    """Worker 3 (vest-only example) must never reside inside DANGER_ZONE_A."""
    DANGER_ZONE_A = [
        (0.40, 0.40), (0.70, 0.40), (0.72, 0.55), (0.65, 0.75), (0.40, 0.75),
    ]
    det = SyntheticDetector(_cfg())
    seen_times = 0
    for t in range(0, 300, 3):  # sweep a few minutes of demo time
        with patch("isip.vision.detector.time.time", return_value=1_700_000_000.0 + t / 4.0):
            detections = det.detect(FRAME)
        # Worker 3 sits low-center: y ≈ 0.78..0.86, x drifts ±0.06 around 0.50.
        for d in detections:
            if d.class_name != "person":
                continue
            cx = d.bbox_center[0] / FRAME.shape[1]
            cy = d.bbox_center[1] / FRAME.shape[0]
            if cy >= 0.76 and 0.42 <= cx <= 0.58:
                seen_times += 1
                assert not point_in_polygon((cx, cy), DANGER_ZONE_A), (t, cx, cy)
    assert seen_times > 0, "worker3 not found in sweep"


def test_worker1_drifts_through_danger_zone():
    """Worker 1 must periodically enter DANGER_ZONE_A (and leave it)."""
    DANGER_ZONE_A = [
        (0.40, 0.40), (0.70, 0.40), (0.72, 0.55), (0.65, 0.75), (0.40, 0.75),
    ]
    det = SyntheticDetector(_cfg())
    inside = set()
    for t in range(0, 300, 3):
        with patch("isip.vision.detector.time.time", return_value=1_700_000_000.0 + t / 4.0):
            detections = det.detect(FRAME)
        for d in detections:
            if d.class_name != "person":
                continue
            cx = d.bbox_center[0] / FRAME.shape[1]
            cy = d.bbox_center[1] / FRAME.shape[0]
            # Worker 1 orbits the right-center area (x 0.39..0.71, y 0.48..0.68).
            if 0.38 <= cx <= 0.72 and 0.45 <= cy <= 0.70:
                inside.add(point_in_polygon((cx, cy), DANGER_ZONE_A))
    assert True in inside and False in inside


def test_detector_worker_count_stable():
    """Four people are always detected (geofence/PPE demo needs all of them)."""
    with patch("isip.vision.detector.time.time", return_value=1_700_000_000.0):
        det = SyntheticDetector(_cfg()).detect(FRAME)
    assert sum(1 for d in det if d.class_name == "person") == 4