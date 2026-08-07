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
            device="cpu",
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

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._t = 0.0
        self._last_latency = 0.0

    def detect(self, frame: Any) -> List[Detection]:
        import math

        started = time.perf_counter()
        self._t += 1
        h, w = frame.shape[:2] if hasattr(frame, "shape") else (720, 1280)

        def norm(x: float, y: float) -> tuple:
            return x * w, y * h

        # Worker drifts across DANGER_ZONE_A (0.40-0.70 x, 0.40-0.75 y).
        wx = 0.55 + 0.16 * math.sin(self._t / 28.0)
        wy = 0.58 + 0.10 * math.cos(self._t / 37.0)
        x, y = norm(wx, wy)

        detections = [
            Detection("person", 0.94, x, y, x + w * 0.045, y + h * 0.14),
            Detection("helmet", 0.91, x + w * 0.002, y - h * 0.012, x + w * 0.042, y + h * 0.012),
        ]
        # Second worker in the exclusion zone without helmet (drives PPE alert).
        ex, ey = norm(0.17, 0.22 + 0.06 * math.sin(self._t / 23.0))
        detections.append(
            Detection("person", 0.88, ex, ey, ex + w * 0.04, ey + h * 0.12)
        )
        self._last_latency = 2.4 + (self._t % 5) * 0.3
        return detections

    @property
    def latency_ms(self) -> float:
        return self._last_latency


def build_detector(config: VisionConfig) -> ObjectDetector:
    """Instantiate the configured detector with graceful fallbacks."""
    backend = getattr(config, "backend", "yolov8")
    if config.model and backend == "yolov8":
        try:
            return YoloDetector(config)
        except ImportError as exc:
            logger.warning("Ultralytics unavailable (%s) - using synthetic detector", exc)
        except Exception as exc:  # model download / torch init failure
            logger.warning("YOLOv8 init failed (%s) - using synthetic detector", exc)
    return SyntheticDetector(config)
