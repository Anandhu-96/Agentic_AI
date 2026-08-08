"""Integrated AI Safety pipeline (legacy entry point).

This module is a thin wrapper over the authoritative detection / geofence /
safety modules in :mod:`isip.vision`. It exists so the historical standalone
entry point still runs, but it no longer carries a second copy of the vision
logic — all detection, tracking, dynamic-zone generation and rendering are
shared with the edge orchestrator.

    python -m isip.safety_pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .config import VisionConfig, Settings
from .vision.detector import build_detector, Detection
from .vision.geofence import GeofenceEngine
from .vision.ppe import evaluate_ppe
from .vision.renderer import (
    draw_detections,
    draw_dynamic_zones,
    draw_static_zones,
    draw_status_bar,
)
from .vision.tracking import MachineTracker, WorkerTracker
from .vision.zones import DynamicZoneEngine

logger = logging.getLogger(__name__)


class SafetyPipeline:
    """Video runner over the shared core vision components."""

    def __init__(self, config: VisionConfig, zones: Dict[str, dict]) -> None:
        self.config = config
        self.detector = build_detector(config)
        self.geofence = GeofenceEngine(zones)
        self.tracker = WorkerTracker()
        self.machine_tracker = MachineTracker(iou_threshold=0.3, max_age=10)
        self.dynamic = DynamicZoneEngine()

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video source: %s", video_path)
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        logger.info("Processing stream (%dx%d @ %.1f FPS)... Press 'q' to exit.", w, h, fps)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detector.detect(frame)
            workers = [d for d in detections if d.is_person]
            worker_ids = self.tracker.update([x.norm_center for x in workers])
            violations = evaluate_ppe(workers, detections, self.config, worker_ids=worker_ids)

            machines = [d for d in detections if d.is_machine]
            tracks = []
            if machines:
                tracks = [t for t in self.machine_tracker.update(
                    [m.norm_bbox for m in machines]) if t is not None]
            dynamic = self.dynamic.update(tracks)

            active_zones = set()
            for worker in workers:
                zone = self.geofence.locate_detection(worker)
                if zone is not None:
                    active_zones.add(zone.name)

            draw_detections(frame, detections, violations)
            draw_static_zones(frame, self.geofence, active_zones)
            draw_dynamic_zones(frame, dynamic)
            draw_status_bar(frame, fps, self.detector.latency_ms, len(detections), len(machines))

            if writer:
                writer.write(frame)

            cv2.imshow("AI Vision Safety Pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = Settings.from_yaml("config/config.yaml")
    zones = GeofenceEngine.from_yaml(settings.vision.geofences_file)
    pipeline = SafetyPipeline(settings.vision, {
        name: {
            "severity": z.severity,
            "action": z.action,
            "description": z.description,
            "polygon": z.polygon,
        }
        for name, z in zones.zones.items()
    })
    pipeline.process_video(settings.video.source)


if __name__ == "__main__":
    main()