"""Generate annotated detection image for the dashboard."""
import cv2
import numpy as np
import random

from isip.config import Settings
from isip.vision.detector import build_detector, SyntheticDetector
from isip.vision.geofence import GeofenceEngine
from isip.vision.ppe import evaluate_ppe

settings = Settings.from_yaml("config/config.yaml")
detector = build_detector(settings.vision)

# Generate synthetic frame
frame = np.zeros((settings.video.height, settings.video.width, 3), dtype=np.uint8)
frame[:] = (30, 40, 30)  # dark green-gray background

# Get detections
detections = detector.detect(frame)
workers = [d for d in detections if d.class_name == "person"]

# Load geofences for zone visualization
geofences = GeofenceEngine.from_yaml(settings.vision.geofences_file)

# Draw geofence zones
for zone in geofences.zones.values():
    pts = np.array(zone.polygon, dtype=np.float32)
    pts[:, 0] *= settings.video.width
    pts[:, 1] *= settings.video.height
    pts = pts.astype(int)
    color = (0, 0, 180) if zone.severity == "CRITICAL" else (0, 120, 180)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    cv2.putText(frame, zone.name, (pts[0][0], pts[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Draw detections
violations = evaluate_ppe(workers, detections, settings.vision)
violation_map = {(label, gear) for label, gear, conf in violations}

for det in detections:
    x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
    is_person = det.class_name == "person"
    
    if is_person:
        # Check if worker has PPE violations
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
        # PPE items
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 40), 1)
        cv2.putText(frame, f"{det.class_name} {det.confidence:.2f}", 
                   (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 40), 1)

# Add info overlay
cv2.putText(frame, f"ISIP Edge Node | Detections: {len(detections)} | Workers: {len(workers)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(frame, f"Violations: {len(violations)}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 180), 2)

# Save image
output_path = "dashboard/detection_output.jpg"
cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(f"Saved annotated frame to {output_path}")
print(f"Shape: {frame.shape}")
print(f"Detections: {len(detections)}")
print(f"PPE Violations: {len(violations)}")
for label, gear, conf in violations:
    print(f"  - {label} missing {gear}")
