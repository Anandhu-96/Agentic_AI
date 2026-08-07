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


class VideoStreamProducer:
    """Background thread that continuously produces MJPEG frames."""

    def __init__(self, settings: Settings, *, draw_zones: bool = True) -> None:
        self.settings = settings
        self.vision_cfg = settings.vision
        self.video_cfg = settings.video
        self.detector = build_detector(self.vision_cfg)
        self.geofences = GeofenceEngine.from_yaml(self.vision_cfg.geofences_file)
        self._draw_zones = draw_zones
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

            target_fps = max(1, int(getattr(self.settings.edge, "fps_limit", 4)))
            interval = 1.0 / target_fps
            last_inference = 0.0
            inference_interval = 1.0 / max(1, target_fps - 1)

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

                    now = time.perf_counter()
                    if now - last_inference >= inference_interval:
                        last_inference = now
                        started = time.perf_counter()
                        detections = self.detector.detect(frame)
                        elapsed_ms = (time.perf_counter() - started) * 1000.0

                        workers = [d for d in detections if d.class_name == "person"]
                        violations = evaluate_ppe(workers, detections, self.vision_cfg)

                        h, w = frame.shape[:2]
                        active_zones = set()
                        for worker in workers:
                            wx, wy = worker.bbox_center
                            zone = self.geofences.locate((wx / w, wy / h))
                            if zone is not None:
                                active_zones.add(zone.name)

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
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            label = f"{det.class_name} ({det.confidence:.2f})"
                            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                        if self._draw_zones:
                            for zone in self.geofences.zones.values():
                                pts = np.array(zone.polygon, dtype=np.float32)
                                pts[:, 0] *= w
                                pts[:, 1] *= h
                                pts = pts.astype(int)
                                if len(pts) < 3:
                                    continue
                                is_active = zone.name in active_zones
                                base_color = (0, 0, 180) if zone.severity == "CRITICAL" else (0, 120, 180)
                                line_thickness = 2 if is_active else 1
                                cv2.polylines(frame, [pts], isClosed=True, color=base_color,
                                              thickness=line_thickness, lineType=cv2.LINE_AA)
                                for i, (px, py) in enumerate(pts):
                                    radius = 4 if is_active else 3
                                    cv2.circle(frame, (int(px), int(py)), radius, base_color, -1, lineType=cv2.LINE_AA)
                                    cv2.circle(frame, (int(px), int(py)), radius + 2, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                                label_y = max(16, int(pts[:, 1].min()) - 8)
                                label_text = f"{zone.name} [ACTIVE]" if is_active else zone.name
                                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                                cv2.rectangle(frame, (int(pts[:, 0].min()), label_y - th - 4),
                                              (int(pts[:, 0].min()) + tw + 6, label_y + 4), (0, 0, 0), -1)
                                cv2.putText(frame, label_text, (int(pts[:, 0].min()) + 3, label_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, base_color if is_active else (200, 200, 200), 1)

                            if active_zones:
                                banner = f"ZONE BREACH: {', '.join(sorted(active_zones))}"
                                cv2.rectangle(frame, (0, 0), (w, 32), (0, 0, 180), -1)
                                cv2.putText(frame, banner, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        cv2.putText(frame, f"FPS: {1000.0/max(elapsed_ms,1.0):.1f} | Latency: {elapsed_ms:.1f}ms",
                                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
        self.video_stream = VideoStreamProducer(settings, draw_zones=True)
        self.detection_stream = VideoStreamProducer(settings, draw_zones=False)
