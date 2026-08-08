"""Object detection pipeline: YOLOv8 (Ultralytics) with a deterministic
synthetic fallback so the full edge pipeline runs on any machine without a
camera or GPU. Latency is measured per frame to enforce the sub-20ms budget.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..config import VisionConfig

logger = logging.getLogger(__name__)

PPE_MAP = {
    "person": "person",
    "helmet": "helmet",
    "hard-hat": "helmet",
    "vest": "vest",
    "safety-vest": "vest",
}


@dataclass
class Detection:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    bbox_center: tuple = field(init=False)

    def __post_init__(self) -> None:
        self.bbox_center = ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class ObjectDetector:
    """Base detector interface."""

    def detect(self, frame: Any) -> List[Detection]:
        raise NotImplementedError

    @property
    def latency_ms(self) -> float:
        return 0.0


class YoloDetector(ObjectDetector):
    """Ultralytics YOLOv8 wrapper.

    Optional dependency: if ``ultralytics``/torch is unavailable the factory
    automatically falls back to the synthetic detector.
    """

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        from ultralytics import YOLO

        self._model = YOLO(config.model)
        self._class_filter = set(config.classes)
        self._last_latency = 0.0

    def detect(self, frame: Any) -> List[Detection]:
        started = time.perf_counter()
        results = self._model.predict(
            source=frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
            device=self.config.device,
        )
        self._last_latency = (time.perf_counter() - started) * 1000.0

        detections: List[Detection] = []
        names = self._model.names
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                name = names.get(class_id, "unknown")
                if self._class_filter and name not in self._class_filter:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append(
                    Detection(name, conf, x1, y1, x2, y2)
                )
        return detections

    @property
    def latency_ms(self) -> float:
        return self._last_latency


class SyntheticDetector(ObjectDetector):
    """Deterministic detection source used in demo/no-camera mode.

    Produces plausible worker/PPE detections that drift around the frame so
    the geofencing, PPE rule engine, and dashboard remain fully exercisable
    without any hardware.
    """

    # Slow wall-clock multiplier for the synthetic trajectories. Keeps the
    # demo lively (a full drift cycle takes ~40s) while guaranteeing that every
    # detector instance in the process computes identical positions at the same
    # instant, so video overlay and published zone events always agree.
    _TIME_SCALE = 4.0

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._epoch = time.time()
        self._last_latency = 0.0

    def detect(self, frame: Any) -> List[Detection]:
        import math

        started = time.perf_counter()
        # Absolute wall-clock phase: identical for every SyntheticDetector
        # instance in the process, so the dashboard overlay, the orchestrator's
        # event loop, and both video streams describe the same world.
        t = (time.time() - self._epoch) * self._TIME_SCALE
        h, w = frame.shape[:2] if hasattr(frame, "shape") else (720, 1280)

        def norm(x: float, y: float) -> tuple:
            return x * w, y * h

        def worker_box(cx, cy, scale=1.0):
            bw = w * 0.045 * scale
            bh = h * 0.14 * scale
            return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2

        detections: List[Detection] = []

        # Worker 1: drifts through DANGER_ZONE_A with helmet
        wx = 0.55 + 0.16 * math.sin(t / 28.0)
        wy = 0.58 + 0.10 * math.cos(t / 37.0)
        x, y = norm(wx, wy)
        x1, y1, x2, y2 = worker_box(x, y)
        detections.append(Detection("person", 0.94, x1, y1, x2, y2))
        detections.append(Detection("helmet", 0.91, x1 + w * 0.002, y1 - h * 0.012, x1 + w * 0.042, y1 + h * 0.012))

        # Worker 2: moves in and out of EXCLUSION_ZONE_B without helmet
        ex, ey = norm(0.17, 0.05 + 0.25 * (0.5 + 0.5 * math.sin(t / 23.0)))
        ex1, ey1, ex2, ey2 = worker_box(ex, ey)
        detections.append(Detection("person", 0.88, ex1, ey1, ex2, ey2))

        # Worker 3: lower-center with vest only, no helmet - kept OUTSIDE the
        # restricted zones so it exercises the PPE rule alone, never the zone.
        vx, vy = norm(0.50 + 0.06 * math.sin(t / 31.0), 0.82 + 0.04 * math.cos(t / 43.0))
        vx1, vy1, vx2, vy2 = worker_box(vx, vy)
        detections.append(Detection("person", 0.90, vx1, vy1, vx2, vy2))
        detections.append(Detection("vest", 0.87, vx1 + w * 0.005, vy1 + h * 0.02, vx2 - w * 0.005, vy1 + h * 0.10))

        # Worker 4: compliant worker with helmet + vest, outside all zones
        cx, cy = norm(0.78 + 0.05 * math.sin(t / 33.0), 0.65 + 0.05 * math.cos(t / 29.0))
        cx1, cy1, cx2, cy2 = worker_box(cx, cy)
        detections.append(Detection("person", 0.92, cx1, cy1, cx2, cy2))
        detections.append(Detection("helmet", 0.93, cx1 + w * 0.002, cy1 - h * 0.012, cx1 + w * 0.042, cy1 + h * 0.012))
        detections.append(Detection("vest", 0.89, cx1 + w * 0.003, cy1 + h * 0.02, cx2 - w * 0.003, cy1 + h * 0.10))

        self._last_latency = (time.perf_counter() - started) * 1000.0
        return detections

    @property
    def latency_ms(self) -> float:
        return self._last_latency


def build_detector(config: VisionConfig) -> ObjectDetector:
    """Instantiate the configured detector with graceful fallbacks.

    ``config.backend`` = "synthetic" (deterministic demo, no hardware) or
    "yolov8" (real Ultralytics inference).
    """
    backend = (config.backend or "yolov8").lower()
    if config.model and backend == "yolov8":
        try:
            return YoloDetector(config)
        except ImportError as exc:
            logger.warning("Ultralytics unavailable (%s) - using synthetic detector", exc)
        except Exception as exc:  # model download / torch init failure
            logger.warning("YOLOv8 init failed (%s) - using synthetic detector", exc)
    return SyntheticDetector(config)
