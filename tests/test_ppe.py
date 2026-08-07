from isip.config import VisionConfig
from isip.vision.detector import Detection
from isip.vision.ppe import evaluate_ppe


def _person(x, y, w, h, conf=0.9):
    return Detection("person", conf, x, y, x + w, y + h)


def test_compliant_worker_no_violation():
    worker = _person(10, 10, 50, 120)
    helmet = Detection("helmet", 0.9, 12, 12, 58, 22)
    vest = Detection("vest", 0.8, 12, 20, 58, 40)
    config = VisionConfig(ppe_required=["helmet", "vest"])
    assert evaluate_ppe([worker], [helmet, vest], config) == []


def test_missing_gear_detected():
    worker = _person(10, 10, 50, 120)
    vest = Detection("vest", 0.8, 12, 20, 58, 40)
    config = VisionConfig(ppe_required=["helmet", "vest"])
    violations = evaluate_ppe([worker], [vest], config)
    assert len(violations) == 1
    assert violations[0][1] == "helmet"


def test_distant_gear_not_counted():
    worker = _person(10, 10, 50, 120)
    far_helmet = Detection("helmet", 0.9, 300, 300, 350, 320)
    config = VisionConfig(ppe_required=["helmet"])
    violations = evaluate_ppe([worker], [far_helmet], config)
    assert len(violations) == 1
