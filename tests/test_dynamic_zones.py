"""Tests for machine tracking and dynamic machine danger zones."""

from unittest.mock import patch

import numpy as np

from isip.config import MachineDangerConfig, VisionConfig
from isip.vision.detector import Detection, SyntheticDetector
from isip.vision.geofence import point_in_polygon
from isip.vision.tracking import MachineTracker
from isip.vision.zones import DynamicZoneEngine, machine_danger_zones


def _machine(x1, y1, x2, y2, conf=0.9):
    return Detection("machine", conf, x1, y1, x2, y2, 1280, 720)


def _track(box):
    """Feed a normalized box to the engine through a fake track."""
    class _T:
        id = "M-1"
        bbox = box
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        age = 0
    return _T()


def test_machine_tracker_stable_id():
    tracker = MachineTracker(iou_threshold=0.3, max_age=10)
    b1 = (0.20, 0.20, 0.45, 0.55)
    b2 = (0.22, 0.21, 0.47, 0.56)
    t1 = tracker.update([b1])
    t2 = tracker.update([b2])
    assert [t.id for t in t1 if t] == [t.id for t in t2 if t]


def test_machine_tracker_new_object_new_id():
    tracker = MachineTracker(iou_threshold=0.3, max_age=10)
    t1 = tracker.update([(0.1, 0.1, 0.3, 0.4)])
    t2 = tracker.update([(0.7, 0.7, 0.9, 0.95)])
    assert t1[0].id != t2[0].id


def test_machine_tracker_ages_out():
    tracker = MachineTracker(iou_threshold=0.3, max_age=2)
    tracker.update([(0.1, 0.1, 0.3, 0.4)])
    tracker.update([])
    tracker.update([])
    assert tracker.active_tracks == []


def test_dynamic_zone_produced_from_machine():
    cfg = MachineDangerConfig(enabled=True, buffer=0.05, smoothing=0.0)
    track = _track((0.20, 0.20, 0.45, 0.55))
    zones = machine_danger_zones([track], cfg)
    assert len(zones) == 1
    zone = zones[0]
    assert zone.name == "MACHINE_DANGER_M-1"
    assert zone.severity == "CRITICAL"
    # buffer expands the machine bbox
    xs = [p[0] for p in zone.polygon]
    ys = [p[1] for p in zone.polygon]
    assert min(xs) < 0.20 and max(xs) > 0.45
    assert min(ys) < 0.20 and max(ys) > 0.55


def test_dynamic_zone_moves_with_machine():
    cfg = MachineDangerConfig(enabled=True, buffer=0.05, smoothing=0.0)
    engine = DynamicZoneEngine(cfg)
    t1 = _track((0.20, 0.20, 0.45, 0.55))
    z1 = engine.update([t1])
    t2 = _track((0.60, 0.20, 0.85, 0.55))
    z2 = engine.update([t2])
    assert [p[0] for p in z1[0].polygon] != [p[0] for p in z2[0].polygon]
    assert z2[0].name == z1[0].name  # same machine, moved zone


def test_dynamic_zone_disappears_on_timeout():
    cfg = MachineDangerConfig(enabled=True, buffer=0.05, smoothing=0.0)
    engine = DynamicZoneEngine(cfg)
    engine.update([_track((0.2, 0.2, 0.45, 0.55))])
    assert engine.active != {}
    # No machines left -> the zone is cleared (its track is gone).
    engine.update([])
    assert engine.active == {}


def test_dynamic_zone_disabled():
    cfg = MachineDangerConfig(enabled=False, buffer=0.05, smoothing=0.0)
    zones = machine_danger_zones([_track((0.2, 0.2, 0.45, 0.55))], cfg)
    assert zones == []


def test_worker_inside_machine_zone_is_in_danger():
    cfg = MachineDangerConfig(enabled=True, buffer=0.02, smoothing=0.0)
    engine = DynamicZoneEngine(cfg)
    engine.update([_track((0.30, 0.30, 0.60, 0.70))])
    # Person inside the machine danger zone
    person_inside = Detection("person", 0.9, 0.4 * 1280, 0.4 * 720,
                              0.5 * 1280, 0.6 * 720, 1280, 720)
    assert engine.locate(person_inside.norm_center) is not None
    # Person far away is safe
    person_far = Detection("person", 0.9, 0.05 * 1280, 0.05 * 720,
                           0.12 * 1280, 0.2 * 720, 1280, 720)
    assert engine.locate(person_far.norm_center) is None


def test_locate_uses_feet_position():
    cfg = MachineDangerConfig(enabled=True, buffer=0.02, smoothing=0.0)
    engine = DynamicZoneEngine(cfg)
    engine.update([_track((0.3, 0.3, 0.6, 0.7))])
    from isip.vision.detector import get_feet_position
    # Person polygon whose feet land inside the zone gets flagged even when the
    # bbox centre is not strictly inside.
    poly = np.array([
        [0.30 * 1280, 0.30 * 720], [0.60 * 1280, 0.30 * 720],
        [0.60 * 1280, 0.70 * 720], [0.30 * 1280, 0.70 * 720],
    ], dtype=np.float32)
    det = Detection("person", 0.9, 0.3 * 1280, 0.3 * 720,
                    0.6 * 1280, 0.7 * 720, 1280, 720, polygon=poly)
    assert engine.locate(get_feet_position(det)) is not None