"""Shared edge runtime state wiring broker, control plane, and audit together."""

from __future__ import annotations

import asyncio
import io
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import cv2
import numpy as np

from .config import Settings
from .control.audit import AuditLogger
from .control.plc import PlcRelay
from .events.broker import EventBroker
from .vision.detector import ObjectDetector, build_detector
from .vision.geofence import GeofenceEngine
from .vision.ppe import evaluate_ppe
from .vision.renderer import (
    draw_detections,
    draw_dynamic_zones,
    draw_static_zones,
    draw_status_bar,
)


@dataclass
class RuntimeMetrics:
    last_latency_ms: float = 0.0
    last_fps: float = 0.0
    alert_count: int = 0
    inference_count: int = 0
    started_at: float = field(default_factory=time.time)
    is_estop_locked: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "node_id": "",
                "uptime_s": round(time.time() - self.started_at, 1),
                "latency_ms": round(self.last_latency_ms, 2),
                "fps": round(self.last_fps, 2),
                "alerts": self.alert_count,
                "inferences": self.inference_count,
                "is_estop_locked": self.is_estop_locked,
            }


class DetectionState:
    """Thread-safe snapshot of the latest inference results.

    The orchestrator's inference loop writes here once per frame; the two
    video producers read it afterwards and only render boxes/zones. This keeps
    YOLO running exactly ONE stream on the GPU instead of three.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._detections = []
        self._active_zones: Set[str] = set()
        self._fps = 0.0
        self._latency_ms = 0.0
        self._dynamic_zones = {}  # name -> dict (normalized polygon, machine_id)
        self._machine_ids: Dict[int, Optional[str]] = {}

    def update(self, detections, active_zones, fps: float, latency_ms: float,
               dynamic_zones=None, machine_ids=None) -> None:
        with self.lock:
            self._detections = list(detections)
            self._active_zones = set(active_zones)
            self._fps = fps
            self._latency_ms = latency_ms
            self._dynamic_zones = dict(dynamic_zones or {})
            self._machine_ids = dict(machine_ids or {})

    def snapshot(self):
        with self.lock:
            return (
                list(self._detections),
                set(self._active_zones),
                self._fps,
                self._latency_ms,
                dict(self._dynamic_zones),
                dict(self._machine_ids),
            )

    @property
    def active_zones(self) -> Set[str]:
        with self.lock:
            return set(self._active_zones)

    @property
    def dynamic_zones(self) -> dict:
        with self.lock:
            return dict(self._dynamic_zones)

    @property
    def machine_ids(self) -> dict:
        with self.lock:
            return dict(self._machine_ids)


class VideoStreamProducer:
    """Background thread that continuously produces MJPEG frames."""

    def __init__(
        self,
        settings: Settings,
        *,
        draw_zones: bool = True,
        detector: ObjectDetector | None = None,
        geofences: GeofenceEngine | None = None,
        shared_state: DetectionState | None = None,
    ) -> None:
        self.settings = settings
        self.vision_cfg = settings.vision
        self.video_cfg = settings.video
        self.vision_overlay_cfg = settings.visualization
        self.detector = detector or build_detector(self.vision_cfg)
        self.geofences = geofences or GeofenceEngine.from_yaml(self.vision_cfg.geofences_file)
        self._draw_zones = draw_zones
        self._shared_state = shared_state
        self._inference_lock = threading.Lock()
        self._queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=20)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._error: Optional[Exception] = None
        self.latest_active_zones: Set[str] = set()

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_frame(self, timeout: float = 1.0) -> Optional[bytes]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self) -> None:
        try:
            cap = None
            if not self.video_cfg.synthetic_camera:
                cap = cv2.VideoCapture(self.video_cfg.source)
                if not cap.isOpened():
                    cap = None

            # Display-only streams don't need to match the inference FPS;
            # throttle them so CPU video decode + JPEG encoding doesn't starve
            # the YOLO pipeline (which runs its own loop).
            target_fps = 12 if self._shared_state is not None else max(
                1, int(getattr(self.settings.edge, "fps_limit", 4))
            )
            interval = 1.0 / target_fps

            while self._running:
                try:
                    frame_start = time.perf_counter()
                    if cap is not None:
                        ret, frame = cap.read()
                        if not ret:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        if self.video_cfg.width and self.video_cfg.height:
                            frame = cv2.resize(frame, (self.video_cfg.width, self.video_cfg.height))
                    else:
                        frame = np.zeros((self.video_cfg.height, self.video_cfg.width, 3), dtype=np.uint8)

                    h, w = frame.shape[:2]

                    # One YOLO pipeline owns inference and shares the result.
                    # When connected to a shared state we only render, so the
                    # GPU runs one stream instead of three.
                    if self._shared_state is not None:
                        (detections, active_zones, _fps, lat_ui,
                         dynamic_zones, machine_ids) = self._shared_state.snapshot()
                        dynamic = list(dynamic_zones.values())
                        violations = evaluate_ppe(
                            [d for d in detections if d.class_name == "person"],
                            detections,
                            self.vision_cfg,
                        )
                    else:
                        with self._inference_lock:
                            started = time.perf_counter()
                            detections = self.detector.detect(frame)
                            lat_ui = (time.perf_counter() - started) * 1000.0
                        violations = evaluate_ppe(
                            [d for d in detections if d.class_name == "person"],
                            detections,
                            self.vision_cfg,
                        )
                        active_zones = set()
                        dynamic = []
                        machine_ids = {}
                        for worker in (d for d in detections if d.class_name == "person"):
                            zone = self.geofences.locate_detection(worker)
                            if zone is not None:
                                active_zones.add(zone.name)
                        self.latest_active_zones = active_zones

                    # Render detections (person polygons when available),
                    # machines, static geofences and machine danger zones.
                    draw_detections(frame, detections, violations,
                                    machine_ids=machine_ids,
                                    viz=self.vision_overlay_cfg)
                    if self._draw_zones:
                        draw_static_zones(frame, self.geofences, active_zones,
                                          viz=self.vision_overlay_cfg)
                        draw_dynamic_zones(frame, dynamic, viz=self.vision_overlay_cfg)
                    n_machines = sum(1 for d in detections if d.is_machine)
                    draw_status_bar(frame, 1000.0 / max(lat_ui, 1.0), lat_ui,
                                    len(detections), n_machines)

                    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        frame_bytes = buf.tobytes()
                        try:
                            self._queue.put_nowait(frame_bytes)
                        except queue.Full:
                            try:
                                self._queue.get_nowait()
                                self._queue.put_nowait(frame_bytes)
                            except queue.Empty:
                                pass

                    elapsed = time.perf_counter() - frame_start
                    time.sleep(max(0.0, interval - elapsed))
                except Exception as exc:
                    time.sleep(0.1)
        except Exception as exc:
            self._error = exc


class EdgeRuntime:
    """Composition root for a single edge node."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.broker = EventBroker()
        self.metrics = RuntimeMetrics()
        self.audit = AuditLogger(
            audit_dir=settings.logging.audit_dir,
            node_id=settings.edge.node_id,
        )
        self.plc = PlcRelay(
            gpio=settings.control.estop_gpio,
            relay_channel=settings.control.relay_channel,
            line_id=settings.control.line_id,
            use_gpio=settings.control.use_gpio,
            trip_delay_ms=settings.control.trip_delay_ms,
        )
        # Single shared detection + geofence engine so the overlay, the event
        # loop, and both video streams all reason about the SAME world.
        self.detector = build_detector(settings.vision)
        self.geofences = GeofenceEngine.from_yaml(settings.vision.geofences_file)
        # One inference owner writes here; both video producers render it.
        self.detection_state = DetectionState()
        self.video_stream = VideoStreamProducer(
            settings,
            draw_zones=True,
            detector=self.detector,
            geofences=self.geofences,
            shared_state=self.detection_state,
        )
        self.detection_stream = VideoStreamProducer(
            settings,
            draw_zones=False,
            detector=self.detector,
            geofences=self.geofences,
            shared_state=self.detection_state,
        )
