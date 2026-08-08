"""End-to-end dynamic machine danger zone test.

Frame 1: machine at x=0.4, person far away (x=0.9) -> zone exists, safe.
Frame 2: machine moves to x=0.7; person unchanged -> still safe.
Frame 3: person steps into the (moved) machine zone -> violation.
Frame 4: person leaves -> safe again.
"""

from isip.config import MachineDangerConfig
from isip.vision.detector import Detection, get_feet_position
from isip.vision.tracking import MachineTracker
from isip.vision.zones import DynamicZoneEngine

CFG = MachineDangerConfig(enabled=True, buffer=0.02, smoothing=0.0)
W, H = 1280, 720


def _machine(x, y, half_w=0.08, half_h=0.14):
    return Detection(
        "machine", 0.94,
        (x - half_w) * W, (y - half_h) * H,
        (x + half_w) * W, (y + half_h) * H, W, H,
    )


def _person(feet_x, feet_y):
    half_w, half_h = 0.06, 0.12
    return Detection(
        "person", 0.9,
        (feet_x - half_w) * W, (feet_y - 2 * half_h) * H,
        (feet_x + half_w) * W, feet_y * H, W, H,
    )


def _run(machines, persons):
    tracker = MachineTracker(iou_threshold=0.3, max_age=10)
    engine = DynamicZoneEngine(CFG)
    tracks = [
        t for t in tracker.update([m.norm_bbox for m in machines])
        if t is not None
    ]
    zones = engine.update(tracks)
    in_danger = [any(z.contains(get_feet_position(p)) for z in zones) for p in persons]
    return zones, in_danger


def test_machine_zone_moves_and_safety_changes():
    # Frame 1: machine at x=0.4, person far away (x=0.9) -> zone exists, safe.
    zones1, safe1 = _run([_machine(0.4, 0.5)], [_person(0.9, 0.5)])
    assert zones1
    assert not any(safe1)

    # Frame 2: machine moves to x=0.7; person unchanged -> still safe.
    _, safe2 = _run([_machine(0.7, 0.5)], [_person(0.9, 0.5)])
    assert not any(safe2)

    # Frame 3: person enters the moved machine zone -> violation.
    _, safe3 = _run([_machine(0.7, 0.5)], [_person(0.7, 0.5)])
    assert any(safe3)

    # Frame 4: person leaves -> safe again.
    _, safe4 = _run([_machine(0.7, 0.5)], [_person(0.9, 0.5)])
    assert not any(safe4)


def test_zone_machine_id_stable():
    tracker = MachineTracker(iou_threshold=0.3, max_age=10)
    engine = DynamicZoneEngine(CFG)
    machines = [_machine(0.4, 0.5)]
    tracks = [t for t in tracker.update([m.norm_bbox for m in machines]) if t]
    zones = engine.update(tracks)
    assert zones[0].machine_id == tracks[0].id
    assert zones[0].name == f"MACHINE_DANGER_{tracks[0].id}"