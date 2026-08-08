"""Generate annotated detection image for the dashboard."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from isip.config import Settings
from isip.vision.detector import build_detector
from isip.vision.geofence import GeofenceEngine
from isip.vision.ppe import evaluate_ppe

logger = logging.getLogger(__name__)

OUTPUT_PATH = Path("dashboard/detection_output.jpg")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        settings = Settings.from_yaml("config/config.yaml")
    except Exception:
        logger.exception("Failed to load config/config.yaml")
        return 1

    try:
        detector = build_detector(settings.vision)
        geofences = GeofenceEngine.from_yaml(settings.vision.geofences_file)
    except Exception:
        logger.exception("Failed to initialize detector or geofence engine")
        return 1

    # Synthetic frame (no live camera needed for this static-preview script).
    frame = np.zeros((settings.video.height, settings.video.width, 3), dtype=np.uint8)
    frame[:] = (30, 40, 30)  # dark green-gray background

    try:
        detections = detector.detect(frame)
    except Exception:
        logger.exception("Detection failed on synthetic frame")
        return 1

    workers = [d for d in detections if d.class_name == "person"]

    # Draw geofence zones
    for zone in geofences.zones.values():
        pts = np.array(zone.polygon, dtype=np.float32)
        centroid = pts.mean(axis=0)
        pts = centroid + (pts - centroid) * 0.5
        pts[:, 0] *= settings.video.width
        pts[:, 1] *= settings.video.height
        pts = pts.astype(int)
        color = (0, 0, 180) if zone.severity == "CRITICAL" else (0, 120, 180)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(frame, zone.name, (pts[0][0], pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw detections
    violations = evaluate_ppe(workers, detections, settings.vision)

    for det in detections:
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        is_person = det.class_name == "person"

        if is_person:
            worker_label = f"W-{int(det.bbox_center[0] * 1000)}"
            has_violation = any(v[0] == worker_label for v in violations)
            color = (40, 40, 180) if has_violation else (40, 180, 40)  # Blue=violation, Green=compliant

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Worker {det.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if has_violation:
                missing = [v[1] for v in violations if v[0] == worker_label]
                cv2.putText(frame, f"Missing: {', '.join(missing)}",
                            (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 180), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 40), 1)
            cv2.putText(frame, f"{det.class_name} {det.confidence:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 40), 1)

    cv2.putText(frame, f"ISIP Edge Node | Detections: {len(detections)} | Workers: {len(workers)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Violations: {len(violations)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 180), 2)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(OUTPUT_PATH), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        logger.error("cv2.imwrite reported failure writing to %s", OUTPUT_PATH)
        return 1

    logger.info("Saved annotated frame to %s", OUTPUT_PATH)
    logger.info("Shape: %s | Detections: %d | Violations: %d", frame.shape, len(detections), len(violations))
    for label, gear, conf in violations:
        logger.info("  - %s missing %s (confidence %.2f)", label, gear, conf)

    return 0


if __name__ == "__main__":
    sys.exit(main())