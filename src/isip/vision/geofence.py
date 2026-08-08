"""Restricted-zone geofencing using ray-casting point-in-polygon tests."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)

Point = Tuple[float, float]


def point_in_polygon(point: Point, polygon: List[Point]) -> bool:
    """Ray-casting PIP test. Vertices are normalized (0..1) coordinates.

    Edge cases handled: vertex crossing, horizontal ray, points on the
    polygon boundary are considered inside.
    """
    x, y = point
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Boundary point -> inside.
        if _on_segment(x, y, xi, yi, xj, yj):
            return True
        intersect = (yi > y) != (yj > y)
        if intersect:
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _on_segment(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    cross = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
    if abs(cross) > 1e-9:
        return False
    return min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and (
        min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9
    )


class GeofenceZone(BaseModel):
    name: str
    description: str = ""
    severity: str = "WARNING"
    polygon: List[Point]
    action: str = "ALERT_ONLY"

    @property
    def centroid(self) -> Point:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def as_pixels(self, frame_width: int, frame_height: int) -> List[Tuple[int, int]]:
        """Scale the normalized polygon to pixel coordinates for drawing."""
        return [
            (int(round(x * frame_width)), int(round(y * frame_height)))
            for x, y in self.polygon
        ]


class GeofenceEngine:
    """Loads geofence definitions and tests detections against them."""

    def __init__(self, zones: Dict[str, dict] | None = None) -> None:
        self._zones: Dict[str, GeofenceZone] = {}
        for name, data in (zones or {}).items():
            zone = GeofenceZone(name=name, **data)
            self._zones[name] = zone

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GeofenceEngine":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        zones = data.get("zones", {})
        logger.info("geofence engine loaded zones=%s from %s", list(zones.keys()), path)
        return cls(zones)

    @property
    def zones(self) -> Dict[str, GeofenceZone]:
        return self._zones

    def locate(self, point: Point) -> Optional[GeofenceZone]:
        """Return the first zone containing ``point``, or ``None``."""
        for zone in self._zones.values():
            if point_in_polygon(point, zone.polygon):
                logger.debug("point %s inside zone=%s", point, zone.name)
                return zone
        return None

    def locate_point(self, point: Point) -> Optional[GeofenceZone]:
        """Alias of :meth:`locate` for callers that prefer the explicit name."""
        return self.locate(point)

    def locate_detection(
        self, detection, use_feet: bool = True
    ) -> Optional[GeofenceZone]:
        """Locate a :class:`Detection` using its foot (or centre) position.

        Feet prefer the segmentation polygon's bottom-most point (via
        :func:`get_feet_position`) and fall back to the bbox bottom-centre, so
        geofence tests follow the person's actual contact point.
        """
        if use_feet:
            from .detector import get_feet_position

            point = get_feet_position(detection)
        else:
            point = detection.norm_center
        return self.locate(point)

    def overlay_polys(self, w: int = 1280, h: int = 720, normalize: bool = False) -> Dict[str, dict]:
        """Serializable zone geometry used by dashboards/overlays.

        Every renderer should scale the polygon by the target frame width/height
        exactly once so all visuals agree with the geofence engine's own
        point-in-polygon tests. ``normalize=True`` returns the raw 0..1 polygon
        for clients that scale to their own canvas size.
        """
        return {
            name: {
                "severity": zone.severity,
                "action": zone.action,
                "description": zone.description,
                "polygon": (
                    [[float(x), float(y)] for x, y in zone.polygon]
                    if normalize
                    else [[round(x * w, 1), round(y * h, 1)] for x, y in zone.polygon]
                ),
                "centroid": zone.centroid,
            }
            for name, zone in self._zones.items()
        }