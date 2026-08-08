"""Tests for detections with segmentation polygons and person-foot geometry."""

from unittest.mock import patch

import numpy as np

from isip.config import VisionConfig
from isip.vision.detector import Detection, get_feet_position, normalize_class_name
from isip.vision.detector import SyntheticDetector


def _person_with_polygon(center_x=0.5):
    w, h = 1280, 720
    cx, cy = center_x * w, 0.55 * h
    bw, bh = 0.1 * w, 0.25 * h
    return Detection("person", 0.9, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2,
                     w, h, polygon=np.array([
                         [cx, cy - bh], [cx + bw * 0.4, cy - bh * 0.5],
                         [cx, cy],  # single bottom point -> feet = (center_x, 0.55)
                     ], dtype=np.float32))


def test_detection_defaults_backward_compatible():
    d = Detection("person", 0.9, 10, 20, 50, 120)
    assert d.bbox_center == (30.0, 70.0)
    assert d.polygon is None
    assert d.mask is None
    assert d.norm_feet_position[0] == 30.0 / 1280
    assert d.norm_feet_position[1] == 120.0 / 720


def test_get_feet_position_uses_polygon_bottom():
    d = _person_with_polygon()
    from isip.vision.detector import get_feet_position
    feet = get_feet_position(d)
    assert abs(feet[1] - 0.55) < 1e-4  # polygon bottom sits at y=0.55
    assert abs(feet[0] - 0.5) < 1e-4


def test_get_feet_position_fallback_bbox():
    d = Detection("person", 0.9, 100, 100, 300, 300, 1280, 720)
    fx, fy = get_feet_position(d)
    assert fx == 200.0 / 1280
    assert fy == 300.0 / 720


def test_synthetic_workers_have_polygons():
    with patch("isip.vision.detector.time.time", return_value=1_700_000_000.0):
        det = SyntheticDetector(VisionConfig(backend="synthetic")).detect(
            np.zeros((720, 1280, 3), dtype=np.uint8))
    persons = [d for d in det if d.is_person]
    assert len(persons) == 2
    for p in persons:
        assert p.polygon is not None and len(p.polygon) >= 3


def test_synthetic_machine_present():
    with patch("isip.vision.detector.time.time", return_value=1_700_000_000.0):
        det = SyntheticDetector(VisionConfig(backend="synthetic")).detect(
            np.zeros((720, 1280, 3), dtype=np.uint8))
    machines = [d for d in det if d.is_machine]
    assert len(machines) == 1
    assert machines[0].bbox_center[0] > 0


def test_normalize_class_name_synonyms():
    assert normalize_class_name("helmet") == "helmet"
    assert normalize_class_name("safety vest") == "vest"
    assert normalize_class_name("hard-hat") == "helmet"
    assert normalize_class_name("heavy machinery") == "machine"
    assert normalize_class_name("engine") == "machine"
    assert normalize_class_name("motor") == "machine"