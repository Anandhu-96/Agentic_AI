"""Shared edge runtime state wiring broker, control plane, and audit together."""

from __future__ import annotations

import asyncio
import io
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import Settings
from .control.audit import AuditLogger
from .control.plc import PlcRelay
from .events.broker import EventBroker
from .vision.detector import build_detector
from .vision.geofence import GeofenceEngine
from .vision.ppe import evaluate_ppe


@dataclass
class RuntimeMetrics:
    last_latency_ms: float = 0.0
    last_fps: float = 0.0
    alert_count: int = 0
    inference_count: int = 0
    started_at: float = field(default_factory=time.time)
    is_estop_locked: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def snapshot(self) -> dict:
        async with self.lock:
            return {
                "node_id": "",
                "uptime_s": round(time.time() - self.started_at, 1),
                "latency_ms": round(self.last_latency_ms, 2),
                "fps": round(self.last_fps, 2),
                "alerts": self.alert_count,
                "inferences": self.inference_count,
                "is_estop_locked": self.is_estop_locked,
            }


class VideoStreamProducer:
    """Background thread that continuously produces MJPEG frames."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.vision_cfg = settings.vision
        self.video_cfg = settings.video
        self.detector = build_detector(self.vision_cfg)
        self.geofences = GeofenceEngine.from_yaml(self.vision_cfg.geofences_file)
        self._queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=20)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._error: Optional[Exception] = None

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

            target_fps = max(1, int(getattr(self.settings.edge, "fps_limit", 8)))
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

                    started = time.perf_counter()
                    detections = self.detector.detect(frame)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0

                    workers = [d for d in detections if d.class_name == "person"]
                    violations = evaluate_ppe(workers, detections, self.vision_cfg)

                    h, w = frame.shape[:2]
                    for zone in self.geofences.zones.values():
                        pts = np.array(zone.polygon, dtype=np.float32)
                        pts[:, 0] *= w
                        pts[:, 1] *= h
                        pts = pts.astype(int)
                        color = (0, 0, 180) if zone.severity == "CRITICAL" else (0, 120, 180)
                        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
                        cv2.putText(frame, zone.name, (pts[0][0], pts[0][1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    for det in detections:
                        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                        color = (40, 180, 40)
                        if det.class_name == "person":
                            worker_label = f"W-{int(det.bbox_center[0] * 1000)}"
                            has_violation = any(v[0] == worker_label for v in violations)
                            color = (40, 40, 180) if has_violation else (40, 180, 40)
                        elif det.class_name == "helmet":
                            color = (180, 180, 40)
                        elif det.class_name == "vest":
                            color = (180, 40, 180)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{det.class_name} ({det.confidence:.2f})"
                        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    cv2.putText(frame, f"FPS: {1000.0/max(elapsed_ms,1.0):.1f} | Latency: {elapsed_ms:.1f}ms",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret:
                        frame_bytes = buf.tobytes()
                        while not self._queue.full():
                            try:
                                self._queue.put_nowait(frame_bytes)
                                break
                            except queue.Full:
                                try:
                                    self._queue.get_nowait()
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
        self.video_stream = VideoStreamProducer(settings)
