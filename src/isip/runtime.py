"""Shared edge runtime state wiring broker, control plane, and audit together.

PRODUCTION NOTES
----------------
- RuntimeMetrics is written from two contexts: FastAPI's asyncio event loop
  (e.g. trigger_estop) and the VideoStreamProducer background thread. A plain
  ``asyncio.Lock`` only protects against concurrent *coroutines* on the same
  event loop -- it does nothing for a real OS thread, and using it from a
  thread with no running loop will raise. This version uses a
  ``threading.Lock`` for the actual critical section (cheap, non-blocking for
  the small dict updates here) and exposes both a sync ``snapshot_sync()`` and
  an async ``snapshot()`` wrapper so both call sites stay correct.
- VideoStreamProducer now takes the shared RuntimeMetrics instance and
  actually updates latency/fps/inference_count after each inference --
  previously these fields were computed locally and only used for the
  on-frame text overlay, so `/metrics` and the dashboard KPIs never reflected
  real numbers.
- The camera capture is released on stop/exit, loop errors are logged instead
  of swallowed, and a stalled/errored thread is surfaced via `.error` so the
  API layer (and a future liveness probe) can detect it instead of quietly
  serving a frozen feed.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .config import Settings
from .control.audit import AuditLogger
from .control.plc import PlcRelay
from .events.broker import EventBroker
from .integrations.supabase import SupabaseClient
from .vision.detector import build_detector
from .vision.geofence import GeofenceEngine
from .vision.ppe import evaluate_ppe
from .vision.tracker_serial import SerialDeviceTracker

logger = logging.getLogger(__name__)


@dataclass
class RuntimeMetrics:
    last_latency_ms: float = 0.0
    last_fps: float = 0.0
    alert_count: int = 0
    inference_count: int = 0
    started_at: float = field(default_factory=time.time)
    is_estop_locked: bool = False
    # threading.Lock, not asyncio.Lock: this object is written from a
    # background OS thread (VideoStreamProducer) as well as from asyncio
    # coroutines, and asyncio.Lock is not valid across that boundary.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def lock(self) -> threading.Lock:
        """Kept as `.lock` for call-site compatibility; use `with metrics.lock:`."""
        return self._lock

    def record_inference(self, latency_ms: float, fps: float) -> None:
        with self._lock:
            self.last_latency_ms = latency_ms
            self.last_fps = fps
            self.inference_count += 1

    def snapshot_sync(self) -> dict:
        with self._lock:
            return {
                "node_id": "",
                "uptime_s": round(time.time() - self.started_at, 1),
                "latency_ms": round(self.last_latency_ms, 2),
                "fps": round(self.last_fps, 2),
                "alerts": self.alert_count,
                "inferences": self.inference_count,
                "is_estop_locked": self.is_estop_locked,
            }

    async def snapshot(self) -> dict:
        # threading.Lock critical section here is tiny (dict construction),
        # so taking it synchronously inside a coroutine is fine and avoids
        # the asyncio/threading lock mismatch entirely.
        return self.snapshot_sync()


class VideoStreamProducer:
    """Background thread that continuously produces MJPEG frames."""

    def __init__(self, settings: Settings, metrics: Optional[RuntimeMetrics] = None) -> None:
        self.settings = settings
        self.vision_cfg = settings.vision
        self.video_cfg = settings.video
        self.metrics = metrics
        self.detector = build_detector(self.vision_cfg)
        self.geofences = GeofenceEngine.from_yaml(self.vision_cfg.geofences_file)
        self._queue: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=20)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._error: Optional[Exception] = None
        self._source_error: Optional[Exception] = None
        self._frame_times: "deque[float]" = deque(maxlen=30)

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        """Return safe diagnostics for the API and operator dashboard."""
        return {
            "ready": self.is_alive and self._error is None and self._source_error is None,
            "alive": self.is_alive,
            "source": str(self.video_cfg.source),
            "synthetic_camera": bool(self.video_cfg.synthetic_camera),
            "error": str(self._error or self._source_error) if (self._error or self._source_error) is not None else None,
        }

    def start(self) -> None:
        if self._running:
            logger.debug("video stream already running")
            return
        self._running = True
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("video stream producer started target_fps=%d", self.settings.edge.fps_limit)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning("Video producer thread did not stop within timeout")

    def get_frame(self, timeout: float = 1.0) -> Optional[bytes]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self) -> None:
        cap = None
        try:
            if not self.video_cfg.synthetic_camera:
                cap = cv2.VideoCapture(self.video_cfg.source)
                if not cap.isOpened():
                    error = RuntimeError(f"could not open video source: {self.video_cfg.source!r}")
                    self._source_error = error
                    logger.error(
                        "%s; using generated frames until the source is fixed",
                        error,
                    )
                    cap = None

            target_fps = max(1, int(getattr(self.settings.edge, "fps_limit", 4)))
            interval = 1.0 / target_fps
            last_inference = 0.0
            inference_interval = 1.0 / max(1, target_fps - 1)
            elapsed_ms = 0.0
            consecutive_errors = 0

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

                        self._frame_times.append(time.perf_counter())
                        if len(self._frame_times) >= 2:
                            dt = self._frame_times[-1] - self._frame_times[0]
                            actual_fps = (len(self._frame_times) - 1) / dt if dt > 0 else 0.0
                        else:
                            actual_fps = 0.0
                        current_fps = min(actual_fps, float(self.settings.edge.fps_limit) * 1.5)
                        if self.metrics is not None:
                            self.metrics.record_inference(elapsed_ms, current_fps)

                        workers = [d for d in detections if d.class_name == "person"]
                        violations = evaluate_ppe(workers, detections, self.vision_cfg)

                        h, w = frame.shape[:2]
                        active_zones = set()
                        for worker in workers:
                            wx, wy = worker.bbox_center
                            zone = self.geofences.locate((wx / w, wy / h))
                            if zone is not None:
                                active_zones.add(zone.name)
                        for zone in self.geofences.zones.values():
                            pts = np.array(zone.polygon, dtype=np.float32)
                            centroid = pts.mean(axis=0)
                            pts = centroid + (pts - centroid) * 0.5
                            pts[:, 0] *= w
                            pts[:, 1] *= h
                            pts = pts.astype(int)
                            if len(pts) >= 3:
                                is_active = zone.name in active_zones
                                base_color = (0, 0, 180) if zone.severity == "CRITICAL" else (0, 120, 180)
                                overlay = frame.copy()
                                fill_alpha = 0.35 if is_active else 0.15
                                cv2.fillPoly(overlay, [pts], color=base_color)
                                cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)
                                line_thickness = 4 if is_active else 2
                                cv2.polylines(frame, [pts], isClosed=True, color=base_color, thickness=line_thickness)
                                for i, (px, py) in enumerate(pts):
                                    radius = 6 if is_active else 4
                                    cv2.circle(frame, (int(px), int(py)), radius, base_color, -1)
                                    cv2.circle(frame, (int(px), int(py)), radius + 3, (255, 255, 255), 2)
                                label_y = max(14, int(pts[:, 1].min()) - 10)
                                if is_active:
                                    cv2.putText(frame, f"{zone.name} [ACTIVE]", (int(pts[:, 0].min()), label_y),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
                                else:
                                    cv2.putText(frame, zone.name, (int(pts[:, 0].min()), label_y),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, base_color, 2)

                        for det in detections:
                            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                            color = (40, 180, 40)
                            if det.class_name == "person":
                                wx, wy = det.bbox_center
                                worker_label = f"W-{int(wx * 50)}-{int(wy * 50)}"
                                has_violation = any(v[0] == worker_label for v in violations)
                                color = (40, 40, 180) if has_violation else (40, 180, 40)
                            elif det.class_name == "helmet":
                                color = (180, 180, 40)
                            elif det.class_name == "vest":
                                color = (180, 40, 180)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{det.class_name} ({det.confidence:.2f})"
                            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    cv2.putText(frame, f"FPS: {target_fps} | Latency: {elapsed_ms:.1f}ms",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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

                    consecutive_errors = 0
                    elapsed = time.perf_counter() - frame_start
                    time.sleep(max(0.0, interval - elapsed))
                except Exception:
                    consecutive_errors += 1
                    logger.exception(
                        "Error in video producer frame loop (consecutive=%d)", consecutive_errors
                    )
                    if consecutive_errors >= 20:
                        logger.critical(
                            "Video producer failing repeatedly (%d consecutive errors); "
                            "stopping thread rather than spinning silently.",
                            consecutive_errors,
                        )
                        raise
                    time.sleep(0.1)
        except Exception as exc:
            self._error = exc
            logger.exception("Video producer thread exiting due to unrecoverable error")
        finally:
            if cap is not None:
                cap.release()
                logger.info("Released video capture device")


class EdgeRuntime:
    """Composition root for a single edge node."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.broker = EventBroker()
        self.metrics = RuntimeMetrics()
        self._supabase_client = SupabaseClient(settings.supabase)
        self.audit = AuditLogger(
            audit_dir=settings.logging.audit_dir,
            node_id=settings.edge.node_id,
            max_bytes=settings.logging.audit_max_bytes,
            backup_count=settings.logging.audit_backup_count,
            supabase_client=self._supabase_client,
        )
        self.plc = PlcRelay(
            gpio=settings.control.estop_gpio,
            relay_channel=settings.control.relay_channel,
            line_id=settings.control.line_id,
            use_gpio=settings.control.use_gpio,
            trip_delay_ms=settings.control.trip_delay_ms,
        )
        self.video_stream = VideoStreamProducer(settings, metrics=self.metrics)
        self.serial_tracker = SerialDeviceTracker(settings.tracking.serial)

    def shutdown(self) -> None:
        """Best-effort graceful shutdown; call from a FastAPI shutdown event."""
        logger.info("EdgeRuntime shutting down")
        self.serial_tracker.stop()
        self.video_stream.stop()