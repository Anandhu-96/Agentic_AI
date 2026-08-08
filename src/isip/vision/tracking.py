"""Lightweight centroid-based worker tracker and IoU-based machine tracker.

Both provide stable identities across frames so events (PPE, zone intrusions,
dynamic machine zones) don't re-fire on every inference tick.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float]


class WorkerTracker:
    """Matches normalized (0..1) worker centroids into stable IDs."""

    def __init__(self, match_distance: float = 0.12, max_workers: int = 32) -> None:
        self._match_distance = match_distance
        self._max_workers = max_workers
        self._next_id = 1
        self._frame = 0
        # id -> (last_center, last_frame_seen)
        self._tracks: Dict[str, Tuple[Point, int]] = {}

    def update(self, centers: List[Point]) -> List[str]:
        """Return a stable ID for each center, in the same order."""
        self._frame += 1
        used: set[str] = set()
        ids: List[str] = []

        for cx, cy in centers:
            best_id: str | None = None
            best_dist = self._match_distance
            for wid, (center, _last) in self._tracks.items():
                if wid in used:
                    continue
                d = math.hypot(cx - center[0], cy - center[1])
                if d < best_dist:
                    best_dist = d
                    best_id = wid
            if best_id is None:
                best_id = f"W-{self._next_id}"
                self._next_id += 1
            self._tracks[best_id] = ((cx, cy), self._frame)
            used.add(best_id)
            ids.append(best_id)

        # Drop stale tracks once we exceed the cap so IDs keep recycling
        # instead of growing without bound.
        if len(self._tracks) > self._max_workers:
            stale = sorted(self._tracks, key=lambda wid: self._tracks[wid][1])
            for wid in stale[: len(self._tracks) - self._max_workers]:
                del self._tracks[wid]

        return ids

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame = 0


def _iou_norm(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Intersection-over-union of two normalized (x1, y1, x2, y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


class MachineTrack:
    """A tracked machine with a stable ID and recent box history."""

    def __init__(self, track_id: str, bbox: Tuple[float, float, float, float],
                 center: Point, frame: int) -> None:
        self.id = track_id
        self.bbox = bbox  # normalized (x1, y1, x2, y2)
        self.center = center
        self.last_frame = frame
        self.age = 0  # frames since last observation; increments when unseen
        self.first_frame = frame

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"MachineTrack(id={self.id}, age={self.age}, bbox={tuple(round(v, 3) for v in self.bbox)})"


class MachineTracker:
    """Greedy IoU/centroid association for machine detections.

    Reuses an existing ID when a new machine box sufficiently overlaps a
    previously seen machine (IoU >= ``iou_threshold``); otherwise creates a new
    ``M-N`` ID. Tracks age out after ``max_age`` frames without a match.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10) -> None:
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._next_id = 1
        self._frame = 0
        self._tracks: Dict[str, MachineTrack] = {}

    def update(self, boxes: List[Tuple[float, float, float, float]]) -> List[Optional[MachineTrack]]:
        """Associate normalized boxes to stable track IDs.

        Returns one entry per input box: the matched :class:`MachineTrack` or
        ``None`` when the box was too small / invalid.
        """
        self._frame += 1
        used: set[str] = set()
        result: List[Optional[MachineTrack]] = []

        for box in boxes:
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            if box[2] - box[0] < 1e-6 or box[3] - box[1] < 1e-6:
                result.append(None)
                continue
            best_id: Optional[str] = None
            best_score = self._iou_threshold
            for tid, trk in self._tracks.items():
                if tid in used:
                    continue
                score = _iou_norm(box, trk.bbox)
                if score >= best_score:
                    best_score = score
                    best_id = tid
            if best_id is None:
                best_id = f"M-{self._next_id}"
                self._next_id += 1
                self._tracks[best_id] = MachineTrack(best_id, box, (cx, cy), self._frame)
            else:
                trk = self._tracks[best_id]
                trk.bbox = box
                trk.center = (cx, cy)
                trk.last_frame = self._frame
                trk.age = 0
            used.add(best_id)
            result.append(self._tracks[best_id])

        # Age out tracks that were not matched this frame.
        for tid in list(self._tracks):
            if tid in used:
                continue
            self._tracks[tid].age += 1
            if self._tracks[tid].age >= self._max_age:
                del self._tracks[tid]

        return result

    @property
    def active_tracks(self) -> List[MachineTrack]:
        return [t for t in self._tracks.values() if t.age < self._max_age]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame = 0