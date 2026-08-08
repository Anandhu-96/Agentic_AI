"""Dynamic machine danger zones.

Static geofences (``config/geofences.yaml``) describe fixed physical areas.
Dynamic machine zones are generated *per frame* from tracked machine
detections: the machine's bounding box is expanded by a configurable safety
buffer, smoothed over time, and exposed to the safety evaluation + renderers
under a stable ``MACHINE_DANGER_<id>`` name.

Using normalized (0..1) coordinates keeps the zone geometry consistent with the
static geofence engine and the point-in-polygon tests.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import MachineDangerConfig
from .geofence import point_in_polygon
from .tracking import MachineTrack

Point = Tuple[float, float]


class DynamicZone:
    """A machine-derived danger zone with a stable identity."""

    def __init__(self, name: str, machine_id: str, polygon: List[Point],
                 severity: str = "CRITICAL") -> None:
        self.name = name
        self.machine_id = machine_id
        self.polygon = polygon  # normalized 0..1 points
        self.severity = severity

    def as_pixels(self, frame_width: int, frame_height: int) -> List[Tuple[int, int]]:
        return [
            (int(round(x * frame_width)), int(round(y * frame_height)))
            for x, y in self.polygon
        ]

    def contains(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "machine_id": self.machine_id,
            "severity": self.severity,
            "polygon": [[float(x), float(y)] for x, y in self.polygon],
        }


def _expanded_box(bbox: Tuple[float, float, float, float], buffer: float,
                  smoothing: float, prev: Optional[List[Point]]) -> List[Point]:
    """Expand a normalized machine bbox by ``buffer`` into a danger polygon.

    ``smoothing`` blends against the previous polygon (EMA) so the zone doesn't
    flicker/jump frame to frame. 0 disables smoothing.
    """
    x1, y1, x2, y2 = bbox
    new_poly = [
        (max(0.0, x1 - buffer), max(0.0, y1 - buffer)),
        (min(1.0, x2 + buffer), max(0.0, y1 - buffer)),
        (min(1.0, x2 + buffer), min(1.0, y2 + buffer)),
        (max(0.0, x1 - buffer), min(1.0, y2 + buffer)),
    ]
    if prev is not None and len(prev) == 4 and 0 < smoothing <= 1.0:
        return [
            (round(smoothing * n[0] + (1 - smoothing) * o[0], 6),
             round(smoothing * n[1] + (1 - smoothing) * o[1], 6))
            for n, o in zip(new_poly, prev)
        ]
    return new_poly


class DynamicZoneEngine:
    """Maintains one smoothed danger zone per tracked machine."""

    def __init__(self, config: Optional[MachineDangerConfig] = None) -> None:
        self._config = config or MachineDangerConfig()
        self._zones: Dict[str, DynamicZone] = {}
        self._prev: Dict[str, List[Point]] = {}

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def update(self, tracks: List[MachineTrack]) -> List[DynamicZone]:
        """Recompute danger zones from the current machine tracks.

        Zones for machines that disappeared this frame are kept (smoothed and
        still rendered) until their track ages out, so a momentary miss doesn't
        make the zone blink off.
        """
        if not self.enabled:
            self._zones.clear()
            self._prev.clear()
            return []

        present = {t.id for t in tracks if t is not None}
        zones: List[DynamicZone] = []
        for track in tracks:
            if track is None:
                continue
            name = f"MACHINE_DANGER_{track.id}"
            poly = _expanded_box(
                track.bbox, self._config.buffer, self._config.smoothing,
                self._prev.get(name),
            )
            zone = DynamicZone(name, track.id, poly)
            self._zones[name] = zone
            self._prev[name] = poly
            zones.append(zone)

        # Drop stale zones whose track is long gone.
        for name in list(self._zones):
            if name not in {f"MACHINE_DANGER_{t.id}" for t in tracks if t is not None}:
                del self._zones[name]
                self._prev.pop(name, None)
        return zones

    @property
    def active(self) -> Dict[str, DynamicZone]:
        return dict(self._zones)

    def locate(self, point: Point) -> Optional[DynamicZone]:
        """First machine danger zone containing ``point``, or None."""
        for zone in self._zones.values():
            if zone.contains(point):
                return zone
        return None

    def overlay_polys(self, w: int = 1280, h: int = 720, normalize: bool = False) -> Dict[str, dict]:
        """Serializable geometry for dashboards (mirrors GeofenceEngine)."""
        out: Dict[str, dict] = {}
        for name, zone in self._zones.items():
            out[name] = {
                "machine_id": zone.machine_id,
                "severity": zone.severity,
                "dynamic": True,
                "polygon": (
                    [[float(x), float(y)] for x, y in zone.polygon]
                    if normalize
                    else [[round(x * w, 1), round(y * h, 1)] for x, y in zone.polygon]
                ),
            }
        return out


def machine_danger_zones(tracks: List[MachineTrack],
                         config: MachineDangerConfig) -> List[DynamicZone]:
    """Convenience factory used by tests/scripts that don't need the engine."""
    return DynamicZoneEngine(config).update(tracks)