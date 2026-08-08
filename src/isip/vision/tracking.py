"""Lightweight centroid-based worker tracker.

Stable labels for the demo pipeline: without tracking, worker IDs derived from
raw pixel coordinates change on every frame (the video moves by a pixel or two),
so the orchestrator would re-fire PPE/zone events for a different "worker" each
frame. This tracker reuses an ID when a new detection is close to a previously
seen worker, and assigns a fresh `W-N` only when a genuinely new worker appears.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

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