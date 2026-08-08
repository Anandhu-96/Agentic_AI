"""Rendering helpers shared by the video streams and dashboard snapshots.

Single source of truth for how detections and zones are drawn, so the edge
node, dashboard images and legacy pipeline all render the same world:

- persons are drawn as their segmentation polygon (person-shaped) whenever one
  is available, otherwise the bounding box;
- machines get a highlighted bbox plus their dynamic danger zone;
- static geofences and dynamic machine zones are visually distinct.

Coordinate convention: all geometry handed to the renderer is either already
pixel-sized (``Detection`` boxes/polygons) or normalized-and-then-scaled by the
helper itself (zones). Nothing here invents positions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from ..config import VisualizationConfig
from .detector import Detection, get_feet_position
from .geofence import GeofenceEngine
from .zones import DynamicZone

_SCHEME = {
    "person_safe": (40, 180, 40),
    "person_violation": (40, 40, 180),
    "helmet": (180, 180, 40),
    "vest": (180, 40, 180),
    "gloves": (40, 120, 180),
    "machine": (200, 120, 20),
    "machine_zone": (40, 60, 220),  # bright blue danger zone
    "static_zone": (0, 120, 180),
    "static_zone_active": (0, 0, 180),
}


def _draw_person_shape(frame: np.ndarray, det: Detection, color, viz: VisualizationConfig) -> None:
    """Person-shaped rendering: polygon when segmentation is available."""
    x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
    if viz.show_person_bbox:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    if viz.show_person_polygon and det.polygon is not None and len(det.polygon) >= 3:
        pts = np.asarray(det.polygon, dtype=np.float32).reshape(-1, 1, 2)
        pts = pts.astype(np.int32)
        if viz.fill_person_polygon and len(pts) >= 3:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    fx, fy = get_feet_position(det)
    w, h = frame.shape[1], frame.shape[0]
    cv2.circle(frame, (int(fx * w), int(fy * h)), 4, (255, 0, 255), -1)


def draw_detections(frame: np.ndarray, detections: List[Detection],
                    violations: List[Tuple[str, str, float]],
                    machine_ids: Optional[Dict[int, Optional[str]]] = None,
                    worker_ids: Optional[List[str]] = None,
                    viz: Optional[VisualizationConfig] = None) -> None:
    """Draw all detections; violations tie workers to their missing gear.

    Persons are matched to ``violations`` by the same stable ``W-N`` labels the
    tracker emits. ``worker_ids`` must line up 1:1 with the person detections;
    when omitted a position-derived label is used as a fallback.
    """
    viz = viz or VisualizationConfig()
    person_idx = 0  # index into worker_ids for person detections only

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        if det.is_person:
            if worker_ids is not None and person_idx < len(worker_ids):
                label = worker_ids[person_idx]
            else:
                label = f"W-{int(det.bbox_center[0] * 1000)}"
            person_idx += 1
            if any(v[0] == label for v in violations):
                color = _SCHEME["person_violation"]
            else:
                color = _SCHEME["person_safe"]
            _draw_person_shape(frame, det, color, viz)
            missing = [v[1] for v in violations if v[0] == label]
            tag = f"{label} MISSING {','.join(missing)}" if missing else "SAFE"
            _draw_box_label(frame, x1, y1, tag, color)
        elif det.class_name == "helmet":
            cv2.rectangle(frame, (x1, y1), (x2, y2), _SCHEME["helmet"], 1)
        elif det.class_name == "vest":
            cv2.rectangle(frame, (x1, y1), (x2, y2), _SCHEME["vest"], 1)
        elif det.is_machine:
            color = _SCHEME["machine"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            mid = None
            if machine_ids:
                mid = machine_ids.get(idx)
            tag = f"MACHINE {mid}" if mid else "MACHINE"
            _draw_box_label(frame, (x1 + x2) // 2, y1, tag, color)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)


def _draw_box_label(frame, cx: int, y: int, text: str, color) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x = max(0, cx - tw // 2)
    y = max(th + 4, y)
    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)


def draw_static_zones(frame: np.ndarray, geofences: GeofenceEngine,
                      active_zones: Optional[Set[str]] = None,
                      viz: Optional[VisualizationConfig] = None) -> None:
    """Draw the fixed geofences from config/geofences.yaml (static)."""
    viz = viz or VisualizationConfig()
    if not viz.show_static_zones:
        return
    h, w = frame.shape[:2]
    active = active_zones or set()
    for zone in geofences.zones.values():
        pts = np.array([(x * w, y * h) for x, y in zone.polygon], dtype=np.float32)
        if len(pts) < 3:
            continue
        pts = pts.astype(np.int32).reshape(-1, 1, 2)
        is_active = zone.name in active
        color = _SCHEME["static_zone_active"] if is_active else _SCHEME["static_zone"]
        if is_active:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        label = f"STATIC: {zone.name}" + (" [ACTIVE]" if is_active else "")
        cv2.putText(frame, label, (pts[0][0][0], max(16, pts[0][0][1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_dynamic_zones(frame: np.ndarray, zones: List[DynamicZone] | Dict[str, DynamicZone],
                       viz: Optional[VisualizationConfig] = None) -> None:
    """Draw machine-derived danger zones (dynamic), visually distinct."""
    viz = viz or VisualizationConfig()
    if not viz.show_machine_zone:
        return
    if isinstance(zones, dict):
        zones = list(zones.values())
    h, w = frame.shape[:2]
    for zone in zones:
        pts = zone.as_pixels(w, h)
        if len(pts) < 3:
            continue
        pts = np.asarray(pts, np.int32).reshape(-1, 1, 2)
        color = _SCHEME["machine_zone"]
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        label = f"DANGER {zone.name}"
        cv2.putText(frame, label, (pts[0][0][0], max(16, pts[0][0][1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_status_bar(frame: np.ndarray, fps: float, latency_ms: float,
                    detections: int, machines: int) -> None:
    h = frame.shape[0]
    text = f"FPS:{fps:.1f} Lat:{latency_ms:.1f}ms det:{detections} mach:{machines}"
    cv2.putText(frame, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)