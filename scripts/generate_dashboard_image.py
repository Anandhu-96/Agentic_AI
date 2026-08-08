"""Generate annotated detection image for the dashboard.

Uses the same renderer as the live video streams so the still image shows the
same person polygons, machine danger zones and static geofences as the feed.
"""
import cv2
import numpy as np

from isip.config import Settings
from isip.vision.detector import build_detector
from isip.vision.geofence import GeofenceEngine
from isip.vision.ppe import evaluate_ppe
from isip.vision.renderer import (
    draw_detections,
    draw_dynamic_zones,
    draw_static_zones,
    draw_status_bar,
)
from isip.vision.tracking import MachineTracker, WorkerTracker
from isip.vision.zones import DynamicZoneEngine

settings = Settings.from_yaml("config/config.yaml")
detector = build_detector(settings.vision)

# Generate synthetic frame
frame = np.zeros((settings.video.height, settings.video.width, 3), dtype=np.uint8)
frame[:] = (30, 40, 30)  # dark green-gray background

# Run the core pipeline
detections = detector.detect(frame)
workers = [d for d in detections if d.class_name == "person"]
geofences = GeofenceEngine.from_yaml(settings.vision.geofences_file)

# Worker + machine trackers for stable ids / dynamic zones
workers_ids = WorkerTracker().update([w.norm_center for w in workers])
violations = evaluate_ppe(workers, detections, settings.vision, worker_ids=workers_ids)

machines = [d for d in detections if d.class_name == "machine"]
machine_tracker = MachineTracker(iou_threshold=0.3, max_age=10)
tracks = [
    t for t in machine_tracker.update([(m.x1, m.y1, m.x2, m.y2) for m in machines])
    if t is not None
]
dynamic = DynamicZoneEngine().update(tracks)

active = set()
for worker in workers:
    zone = geofences.locate_detection(worker)
    if zone is not None:
        active.add(zone.name)

# Render with the shared renderer
draw_static_zones(frame, geofences, active, viz=settings.visualization)
draw_dynamic_zones(frame, dynamic, viz=settings.visualization)
draw_detections(frame, detections, violations, viz=settings.visualization)
draw_status_bar(frame, max(detector.latency_ms and 1000.0 / max(detector.latency_ms, 1.0), 0),
                detector.latency_ms, len(detections), len(machines))

# Save image
output_path = "dashboard/detection_output.jpg"
cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(f"Saved annotated frame to {output_path}")
print(f"Shape: {frame.shape}")
print(f"Detections: {len(detections)}")
print(f"Machines: {len(machines)}")
print(f"Dynamic zones: {[z.name for z in dynamic]}")
print(f"PPE Violations: {len(violations)}")
for label, gear, conf in violations:
    print(f"  - {label} missing {gear}")