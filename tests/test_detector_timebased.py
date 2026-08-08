"""SyntheticDetector determinism + zone-placement sanity."""

from unittest.mock import patch

import numpy as np

from isip.config import VisionConfig
from isip.vision.detector import SyntheticDetector
from isip.vision.geofence import point_in_polygon

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)

DANGER_ZONE_A = [
    (0.40, 0.40), (0.70, 0.40), (0.72, 0.55), (0.65, 0.75), (0.40, 0.75),
]
EXCLUSION_ZONE_B = [
    (0.05, 0.10), (0.30, 0.10), (0.32, 0.20), (0.25, 0.35), (0.05, 0.35),
]


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
    with patch("isip.vision.detector.time.time", return_value=1_700_000_000.0):
        a = SyntheticDetector(_cfg())
    with patch("isip.vision.detector.time.time", return_value=1_700_000_000.0):
        b = SyntheticDetector(_cfg())

    with patch("isip.vision.detector.time.time", return_value=1_700_000_012.0):
        da = a.detect(FRAME)
    with patch("isip.vision.detector.time.time", return_value=1_700_000_012.0):
        db = b.detect(FRAME)

    assert _centers(da) == _centers(db)


def test_worker1_drifts_through_danger_zone():
    """Worker 1 (x 0.39..0.71, y 0.48..0.68) must periodically enter DANGER_ZONE_A."""
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
            if 0.38 <= cx <= 0.72 and 0.45 <= cy <= 0.70:
                inside.add(point_in_polygon((cx, cy), DANGER_ZONE_A))
    assert True in inside and False in inside


def test_worker2_oscillates_in_exclusion_zone():
    """Worker 2 (x=0.17, y 0.05..0.30) must enter and leave EXCLUSION_ZONE_B."""
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
            if 0.10 <= cx <= 0.25:  # Worker 2 stays at x ~0.17
                inside.add(point_in_polygon((cx, cy), EXCLUSION_ZONE_B))
    assert True in inside and False in inside


def test_detector_worker_count_stable():
    """Two people are always detected (geofence/PPE demo needs both)."""
    with patch("isip.vision.detector.time.time", return_value=1_700_000_000.0):
        det = SyntheticDetector(_cfg()).detect(FRAME)
    assert sum(1 for d in det if d.class_name == "person") == 2