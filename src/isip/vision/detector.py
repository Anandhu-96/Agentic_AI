"""Object detection pipeline.

Backends
--------
- ``synthetic``     deterministic demo world with workers, PPE and a moving
                    machine. Never used outside test/demo mode.
- ``yolov8``        standard Ultralytics detect model (COCO).
- ``yolov8_world``  open-vocabulary YOLO-World model; machine classes are
                    resolved from ``VisionConfig.machine_classes``.
- ``yolov8_seg``    segmentation model (``-seg``); persons get a real mask /
                    polygon used for person-shaped rendering and foot position.

Persons carry an optional segmentation ``polygon`` so renderers can draw
person-shaped outlines instead of plain rectangles. Everything degrades to the
bounding box when a model does not produce masks.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from ..config import VisionConfig

logger = logging.getLogger(__name__)

# Class-name normalization for PPE + machines. YOLO models report their own
# vocabulary; this collapses synonyms to the categories the safety rules and
# trackers understand.
PPE_MAP = {
    "person": "person",
    "helmet": "helmet",
    "hard-hat": "helmet",
    "safety helmet": "helmet",
    "safety vest": "vest",
    "vest": "vest",
    "safety-vest": "vest",
    "glove": "gloves",
    "gloves": "gloves",
    "safety glove": "gloves",
    "safety gloves": "gloves",
    "hand gloves": "gloves",
}

MACHINE_MAP = {
    "machine": "machine",
    "heavy machinery": "machine",
    "heavy machine": "machine",
    "heavy equipment": "machine",
    "industrial machinery": "machine",
    "industrial machine": "machine",
    "construction equipment": "machine",
    "factory machinery": "machine",
    "gear": "machine",
    "gears": "machine",
    "metal gears": "machine",
    "engine": "machine",
    "engines": "machine",
    "motor": "machine",
    "turbine": "machine",
    "industrial": "machine",
    "industrial robots": "machine",
    "robot": "machine",
    "machinery": "machine",
}


def normalize_class_name(raw: str, machine_classes: Optional[List[str]] = None) -> str:
    """Map a raw YOLO class name to a canonical category.

    PPE synonyms collapse to person/helmet/vest/gloves; anything matching the
    ``machine_classes`` vocabulary (or the built-in machine synonyms) becomes
    ``machine`` so the tracker and dynamic-zone layer get one category.
    """
    if not raw:
        return "unknown"
    if raw in PPE_MAP:
        return PPE_MAP[raw]
    if raw in MACHINE_CLASS | (set(machine_classes or ()).union(MACHINE_ALIASES)):
        return "machine"
    return raw


MACHINE_CLASS = {"machine", "heavy machinery", "heavy machine", "heavy equipment",
    "industrial machine", "industrial machinery", "construction equipment",
    "engine", "engines", "motor", "turbine", "gear", "gears", "metal gears",
    "machinery", "robot", "industrial robots", "factory machinery"}
MACHINE_ALIASES = {
    "heavy-metal-machine": "machine",
    "construction-machine": "machine",
    "industrial-robot": "machine",
    "factory-equipment": "machine",
}


@dataclass
class Detection:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    frame_width: int = 1280
    frame_height: int = 720
    mask: Optional[np.ndarray] = None
    polygon: Optional[np.ndarray] = None
    bbox_center: Tuple[float, float] = field(init=False)
    norm_center: Tuple[float, float] = field(init=False)
    norm_feet_position: Tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        cx = (self.x1 + self.x2) / 2.0
        cy = (self.y1 + self.y2) / 2.0
        self.bbox_center = (cx, cy)
        w = max(self.frame_width, 1)
        h = max(self.frame_height, 1)
        self.norm_center = (cx / w, cy / h)
        self.norm_feet_position = (cx / w, self.y2 / h)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def norm_bbox(self) -> Tuple[float, float, float, float]:
        """Normalized (0..1) bounding box (x1, y1, x2, y2)."""
        w = max(self.frame_width, 1)
        h = max(self.frame_height, 1)
        return (self.x1 / w, self.y1 / h, self.x2 / w, self.y2 / h)

    @property
    def is_machine(self) -> bool:
        return self.class_name == "machine"

    @property
    def is_person(self) -> bool:
        return self.class_name == "person"

    @property
    def latency_ms(self) -> float:
        return 0.0

    def norm_polygon(self) -> Optional[np.ndarray]:
        """Return the segmentation polygon in normalized (0..1) pixels, or None."""
        if self.polygon is None or len(self.polygon) < 3:
            return None
        w = max(self.frame_width, 1)
        h = max(self.frame_height, 1)
        return np.asarray(self.polygon, dtype=np.float32) / np.asarray([w, h])


def get_feet_position(detection: Detection) -> Tuple[float, float]:
    """Return the worker's contact point on the floor, normalized (0..1).

    When a segmentation polygon exists the bottom-most polygon point is used
    (feet accurately track the silhouette); otherwise the bbox bottom-centre.
    """
    if detection.polygon is not None and len(detection.polygon) >= 3:
        poly = np.asarray(detection.polygon, dtype=np.float32)
        bottom = poly[np.argmax(poly[:, 1])]
        w = max(detection.frame_width, 1)
        h = max(detection.frame_height, 1)
        return float(bottom[0] / w), float(bottom[1] / h)
    return detection.norm_feet_position


class ObjectDetector:
    """Base detector interface."""

    def detect(self, frame: Any) -> List[Detection]:
        raise NotImplementedError

    @property
    def latency_ms(self) -> float:
        return 0.0

    @property
    def names(self) -> List[str]:
        return []


class YoloDetector(ObjectDetector):
    """Ultralytics YOLOv8 / YOLO-World / YOLO-Seg wrapper.

    - ``model`` containing ``world`` or selected via ``yolov8_world`` backend
      gets an open-vocabulary ``set_classes`` call with the configured
      ``classes + machine_classes``.
    - segmentation models (``...-seg`` or ``yolov8_seg`` backend) populate
      ``Detection.mask`` and ``Detection.polygon``.
    """

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        from ultralytics import YOLO

        self._model = YOLO(config.model)
        self._last_latency = 0.0
        self._world = self._is_world_model() or (
            (config.backend or "").lower() == "yolov8_world"
        )
        self._is_seg = "seg" in (config.model or "").lower() or (
            (config.backend or "").lower() == "yolov8_seg"
        )

        if self._world:
            # Instruct YOLO-World which objects to look for.
            vocab = list(config.classes) + list(config.machine_classes or [])
            self._model.set_classes(vocab)

    def _is_world_model(self) -> bool:
        name = (self.config.model or "").lower()
        return "world" in name

    def detect(self, frame: Any) -> List[Detection]:
        started = time.perf_counter()
        h, w = frame.shape[:2] if hasattr(frame, "shape") else (720, 1280)

        results = self._model.predict(
            source=frame,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            verbose=False,
            device=self.config.device,
        )
        self._last_latency = (time.perf_counter() - started) * 1000.0

        detections: List[Detection] = []
        names = self._model.names

        for r in results:
            if r.boxes is None:
                continue
            masks = getattr(r, "masks", None)
            for i, box in enumerate(r.boxes):
                class_id = int(box.cls[0])
                raw_name = names.get(class_id, "unknown")
                mapped_name = normalize_class_name(raw_name, self.config.machine_classes)

                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])

                mask = None
                polygon = None
                if masks is not None and masks.xy is not None and i < len(masks.xy):
                    contour = np.asarray(masks.xy[i], dtype=np.float32)
                    if contour.shape[0] >= 3:
                        polygon = contour.copy()
                    if masks.data is not None and i < masks.data.shape[0]:
                        mask = np.asarray(masks.data[i], dtype=np.uint8)

                detections.append(
                    Detection(
                        class_name=mapped_name,
                        confidence=conf,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        frame_width=w,
                        frame_height=h,
                        mask=mask,
                        polygon=polygon,
                    )
                )
        return detections

    @property
    def latency_ms(self) -> float:
        return self._last_latency


class SyntheticDetector(ObjectDetector):
    """Deterministic demo detector.

    All positions are functions of a shared wall-clock phase (``_TIME_SCALE``),
    so every detector instance in the process returns the SAME world at the
    same instant. Used only when ``vision.backend: synthetic``.
    """

    _TIME_SCALE = 4.0

    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._epoch = time.time()
        self._last_latency = 0.0

    def detect(self, frame: Any) -> List[Detection]:
        started = time.perf_counter()
        t = (time.time() - self._epoch) * self._TIME_SCALE
        h, w = frame.shape[:2] if hasattr(frame, "shape") else (720, 1280)

        def worker_polygon(cx: float, cy: float, bw: float, bh: float) -> np.ndarray:
            """Person-ish silhouette polygon in pixel coordinates (head + body)."""
            return np.asarray([
                [cx, cy - bh], [cx + bw * 0.34, cy - bh * 0.72],
                [cx + bw * 0.42, cy - bh * 0.40], [cx + bw * 0.48, cy],
                [cx + bw * 0.42, cy + bh * 0.1], [cx + bw * 0.3, cy + bh * 0.32],
                [cx + bw * 0.18, cy + bh * 0.62], [cx + bw * 0.08, cy + bh * 0.9],
                [cx - bw * 0.08, cy + bh * 0.9], [cx - bw * 0.18, cy + bh * 0.62],
                [cx - bw * 0.3, cy + bh * 0.32], [cx - bw * 0.42, cy + bh * 0.30],
                [cx - bw * 0.48, cy - bh * 0.40], [cx - bw * 0.34, cy - bh * 0.72],
            ], dtype=np.float32)

        detections: List[Detection] = []

        # Worker 1: drifts through DANGER_ZONE_A wearing a helmet + vest.
        wx = 0.55 + 0.16 * math.sin(t / 28.0)
        wy = 0.58 + 0.10 * math.cos(t / 37.0)
        bw, bh = w * 0.085, h * 0.16
        px1, py1 = wx * w, wy * h
        x1, y1 = px1 - bw / 2, py1 - bh / 2
        detections.append(Detection(
            "person", 0.94, x1, y1, x1 + bw, y1 + bh, w, h,
            polygon=worker_polygon(px1, y1 + bh, bw, bh),
        ))
        detections.append(Detection(
            "helmet", 0.91, x1 + w * 0.002, y1 - h * 0.012,
            x1 + w * 0.042, y1 + h * 0.012, w, h,
        ))
        detections.append(Detection(
            "vest", 0.90, x1 + w * 0.002, y1 + h * 0.03,
            x1 + w * 0.042, y1 + h * 0.10, w, h,
        ))

        # Worker 2: moves in and out of EXCLUSION_ZONE_B, no helmet.
        ey = 0.05 + 0.25 * (0.5 + 0.5 * math.sin(t / 23.0))
        ex = 0.17
        px1, py1 = ex * w, ey * h
        x1, y1 = px1 - bw / 2, py1 - bh / 2
        detections.append(Detection(
            "person", 0.88, x1, y1, x1 + bw, y1 + bh, w, h,
            polygon=worker_polygon(px1, y1 + bh, bw, bh),
        ))

        # A real demo machine: moves so the dynamic danger zone moves with it.
        mx = 0.72 + 0.12 * math.sin(t / 25.0)
        my = 0.30 + 0.08 * math.sin(t / 31.0)
        mbw, mbh = w * 0.22, h * 0.3
        px1, py1 = mx * w, my * h
        x1, y1 = px1 - mbw / 2, py1 - mbh / 2
        detections.append(Detection(
            "machine", 0.94, x1, y1, x1 + mbw, y1 + mbh, w, h,
        ))

        self._last_latency = (time.perf_counter() - started) * 1000.0
        return detections

    @property
    def latency_ms(self) -> float:
        return self._last_latency


def validate_configuration(config: VisionConfig, logger_obj=None) -> None:
    """Fail loudly on impossible backend/model combinations.

    Never silently downgrade a requested real backend to the synthetic one.
    """
    log = logger_obj or logger
    backend = (config.backend or "yolov8").lower()
    model = (config.model or "").lower()

    if backend in ("yolov8_world",):
        if "world" not in model:
            raise ValueError(
                "Configuration error: backend='yolov8_world' requires a"
                " YOLO-World-compatible model (e.g. yolov8s-world.pt)."
            )
    if backend == "yolov8_seg":
        if "-seg" not in model and "seg" not in model:
            raise ValueError(
                "Configuration error: backend='yolov8_seg' requires a"
                " segmentation model (e.g. yolov8n-seg.pt, yolov8m-seg.pt)."
            )
    if config.machine_classes and backend == "yolov8":
        # Plain COCO det models have no machine classes; the tracker simply
        # finds no machines. Warn so the operator isn't surprised by the
        # empty dynamic machine zones.
        log.warning(
            "VisionConfig: machine_classes=[%s] requested with a COCO detect "
            "backend ('yolov8'). No machines will be recognized unless the "
            "model is open-vocabulary (use 'yolov8_world').",
            ", ".join(config.machine_classes[:3]),
        )


def build_detector(config: VisionConfig) -> ObjectDetector:
    """Instantiate the configured detector with explicit fallbacks.

    Never silently switches a real backend to synthetic: configuration/model
    errors are raised so the operator can fix the deployment.
    """
    backend = (config.backend or "synthetic").lower()

    if backend == "synthetic":
        return SyntheticDetector(config)

    if not config.model:
        raise RuntimeError("Configuration error: vision.model is empty; ")

    if backend in ("yolov8", "yolov8_world", "yolov8_seg"):
        validate_configuration(config)
        return YoloDetector(config)

    raise RuntimeError(
        f"Configuration error: unknown vision.backend '{backend}'. "
        "Use synthetic, yolov8, yolov8_world or yolov8_seg."
    )