"""PPE compliance rules: match required safety gear to worker detections."""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..config import VisionConfig
from .detector import Detection


def _iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _matches(worker: Detection, gear: Detection, iou_threshold: float) -> bool:
    """A gear item belongs to a worker when its center falls inside the
    worker's box (PPE items are much smaller than the person, so pure IoU is
    unreliable), or when it overlaps substantially.
    """
    gx, gy = gear.bbox_center
    if worker.x1 <= gx <= worker.x2 and worker.y1 <= gy <= worker.y2:
        return True
    return _iou(worker, gear) >= iou_threshold


def evaluate_ppe(
    workers: List[Detection],
    all_detections: List[Detection],
    config: VisionConfig,
    iou_threshold: float = 0.15,
    worker_ids: List[str] | None = None,
) -> List[Tuple[str, str, float]]:
    """Return ``(worker_label, missing_gear, confidence)`` violations.

    A worker is considered compliant for a gear class if any detection of that
    class overlaps its bounding box beyond ``iou_threshold``. When
    ``worker_ids`` is provided it overrides the position-derived label so events
    keep a stable identity across frames.
    """
    violations: List[Tuple[str, str, float]] = []
    required = set(config.ppe_required)
    for idx, worker in enumerate(workers):
        label = (
            worker_ids[idx]
            if worker_ids is not None and idx < len(worker_ids)
            else f"W-{int(worker.bbox_center[0] * 1000)}"
        )
        present: set[str] = set()
        for det in all_detections:
            if det.class_name in required and _matches(worker, det, iou_threshold):
                present.add(det.class_name)
        for gear in sorted(required - present):
            violations.append((label, gear, worker.confidence))
    return violations