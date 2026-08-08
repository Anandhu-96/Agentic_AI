"""Generate a gallery of annotated detection frames from the demo video.

Used by the GitHub Pages showcase so the live landing page shows REAL
detection output extracted from ``We_want_the_vedio_of_the_engin.mp4`` (not just
one synthetic still). Frames are sampled evenly across the clip, run through
the same renderer as the edge node, and written under ``dashboard/gallery/``.

Usage:
    python scripts/generate_detection_gallery.py                # real model
    python scripts/generate_detection_gallery.py --synthetic     # demo world
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "dashboard" / "gallery"
N_SAMPLES = 6


def generate_video_samples(settings: Settings, synthetic: bool) -> list[Path]:
    if synthetic:
        settings.vision.backend = "synthetic"
    detector = build_detector(settings.vision)
    geofences = GeofenceEngine.from_yaml(settings.vision.geofences_file)
    worker_tracker = WorkerTracker()
    machine_tracker = MachineTracker(iou_threshold=0.3, max_age=10)
    dynamic_engine = DynamicZoneEngine(settings.dynamic_zones.machine_danger)

    video_path = ROOT / settings.video.source
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 240
    out = []
    OUT_DIR.mkdir(exist_ok=True)

    for i in range(N_SAMPLES * 2):
        if len(out) >= N_SAMPLES:
            break
        pos = int((i + 0.5) * total / (N_SAMPLES * 2))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            continue

        detections = detector.detect(frame)
        workers = [d for d in detections if d.is_person]
        worker_ids = worker_tracker.update([w.norm_center for w in workers])
        violations = evaluate_ppe(workers, detections, settings.vision, worker_ids=worker_ids)

        machines = [d for d in detections if d.is_machine]
        tracks = [t for t in machine_tracker.update([m.norm_bbox for m in machines]) if t]
        dynamic = dynamic_engine.update(tracks)

        active = set()
        for worker in workers:
            zone = geofences.locate_detection(worker)
            if zone is not None:
                active.add(zone.name)

        draw_detections(frame, detections, violations, worker_ids=worker_ids,
                        viz=settings.visualization)
        draw_static_zones(frame, geofences, active, viz=settings.visualization)
        draw_dynamic_zones(frame, dynamic, viz=settings.visualization)
        draw_status_bar(frame, detector.latency_ms and 30.0, detector.latency_ms,
                        len(detections), len(machines))

        out_path = OUT_DIR / f"detection_{len(out):02d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        out.append(out_path)
        print(f"wrote {out_path.name} det={len(detections)} people={len(workers)} machines={len(machines)} zones={list(active)}")

    cap.release()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use synthetic demo world")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    settings = Settings.from_yaml(args.config)
    files = generate_video_samples(settings, args.synthetic)
    print(f"generated {len(files)} detection frames in dashboard/gallery")


if __name__ == "__main__":
    main()