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
