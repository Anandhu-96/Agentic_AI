"""Configuration validation and renderer smoke tests."""

import numpy as np
import pytest

from isip.config import Settings, VisionConfig
from isip.vision.detector import (
    Detection,
    SyntheticDetector,
    YoloDetector,
    build_detector,
    validate_configuration,
)
from isip.vision.geofence import GeofenceEngine
from isip.vision.renderer import draw_detections, draw_dynamic_zones, draw_static_zones
from isip.vision.zones import DynamicZone


def test_world_backend_requires_world_model():
    with pytest.raises(ValueError, match="yolov8_world"):
        validate_configuration(VisionConfig(backend="yolov8_world", model="yolov8n.pt"))


def test_seg_backend_requires_seg_model():
    with pytest.raises(ValueError, match="yolov8_seg"):
        validate_configuration(VisionConfig(backend="yolov8_seg", model="yolov8n.pt"))


def test_unknown_backend_raises():
    with pytest.raises(RuntimeError, match="unknown vision.backend"):
        build_detector(VisionConfig(backend="magic"))


def test_synthetic_backend_builds():
    det = build_detector(VisionConfig(backend="synthetic"))
    assert isinstance(det, SyntheticDetector)


def test_settings_loads_new_sections():
    s = Settings()
    assert s.tracking.max_age == 10
    assert s.dynamic_zones.machine_danger.buffer == 0.05
    assert s.visualization.show_person_polygon is True
    assert s.vision.machine_classes


def test_renderer_person_polygon_vs_bbox():
    from isip.config import VisualizationConfig

    W, H = 640, 360
    poly = np.array([
        [200, 60], [260, 80], [300, 120], [310, 180], [300, 240],
        [260, 300], [200, 330], [140, 300], [100, 240], [90, 180],
        [100, 120], [140, 80],
    ], dtype=np.float32)
    viz = VisualizationConfig(show_person_bbox=False, show_person_polygon=True,
                              fill_person_polygon=True)
    person = Detection("person", 0.95, 90, 60, 310, 330, W, H, polygon=poly)
    f_poly = np.zeros((H, W, 3), dtype=np.uint8)
    draw_detections(f_poly, [person], [], viz=viz)
    f_bbox = np.zeros((H, W, 3), dtype=np.uint8)
    draw_detections(f_bbox, [Detection("person", 0.95, 90, 60, 310, 330, W, H)], [],
                    viz=viz)
    colored = lambda f: int(np.sum(np.any(f > 0, axis=2)))
    assert colored(f_poly) > colored(f_bbox)


def test_renderer_static_and_dynamic_zones_no_crash():
    W, H = 640, 360
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    geofences = GeofenceEngine({
        "ZONE_A": {"severity": "CRITICAL",
                   "polygon": [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]]}
    })
    draw_static_zones(frame, geofences, {"ZONE_A"})
    zone = DynamicZone("MACHINE_DANGER_M-1", "M-1",
                       [(0.5, 0.5), (0.7, 0.5), (0.7, 0.7), (0.5, 0.7)])
    draw_dynamic_zones(frame, [zone])
    assert frame.any()


def test_renderer_no_mask_bbox_fallback():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_detections(frame, [Detection("person", 0.9, 10, 10, 60, 80, 100, 100)], [])
    assert frame.any()