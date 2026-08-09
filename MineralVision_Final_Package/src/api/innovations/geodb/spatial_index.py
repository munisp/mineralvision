"""
Pure-Python spatial index for the geodb bridge (sqlite fallback path).

Implements a uniform sorted-grid index (rtree-style semantics) over point
entities. Uses the ``rtree`` package when it is importable, otherwise falls
back to the built-in grid implementation. All operations are real - entries
come from database rows and queries return actual entity IDs/distances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import rtree as _rtree  # type: ignore
    RTREE_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    _rtree = None
    RTREE_AVAILABLE = False


@dataclass
class IndexedEntity:
    """A point entity stored in the spatial index."""

    entity_id: str
    entity_type: str  # "drillhole" | "sample" | "project"
    x: float
    y: float
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.entity_id,
            "entity_type": self.entity_type,
            "x": self.x,
            "y": self.y,
            "properties": self.properties,
        }


class SpatialIndex:
    """Uniform-grid spatial index (rtree-style) over point entities."""

    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or ("rtree" if RTREE_AVAILABLE else "grid")
        self._entities: Dict[str, IndexedEntity] = {}
        self._cell_size: float = 1.0
        self._grid: Dict[Tuple[int, int], List[str]] = {}
        self._rtree_idx = None
        self._rtree_counter = 0
        self._rtree_ids: Dict[int, str] = {}
        if self.backend == "rtree":
            self._rtree_idx = _rtree.index.Index()

    # ------------------------------------------------------------------ build
    def clear(self) -> None:
        self._entities.clear()
        self._grid.clear()
        self._rtree_ids.clear()
        self._rtree_counter = 0
        if self.backend == "rtree":
            self._rtree_idx = _rtree.index.Index()

    def _recompute_cell_size(self) -> None:
        if len(self._entities) < 2:
            self._cell_size = 1.0
            return
        xs = [e.x for e in self._entities.values()]
        ys = [e.y for e in self._entities.values()]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        self._cell_size = max(span / 50.0, 1e-9)

    def _cells_for_point(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor(x / self._cell_size)), int(math.floor(y / self._cell_size)))

    def insert(self, entity: IndexedEntity) -> None:
        self._entities[entity.entity_id] = entity
        if self.backend == "rtree":
            self._rtree_counter += 1
            self._rtree_ids[self._rtree_counter] = entity.entity_id
            self._rtree_idx.insert(
                self._rtree_counter, (entity.x, entity.y, entity.x, entity.y)
            )
        # grid is always maintained so bbox reporting is consistent
        self._grid.setdefault(self._cells_for_point(entity.x, entity.y), []).append(
            entity.entity_id
        )

    def rebuild(self, entities: List[IndexedEntity]) -> None:
        """Bulk-load entities, recomputing grid resolution."""
        self.clear()
        for e in entities:
            self._entities[e.entity_id] = e
        self._recompute_cell_size()
        self._grid.clear()
        for e in entities:
            self._grid.setdefault(self._cells_for_point(e.x, e.y), []).append(e.entity_id)
            if self.backend == "rtree":
                self._rtree_counter += 1
                self._rtree_ids[self._rtree_counter] = e.entity_id
                self._rtree_idx.insert(self._rtree_counter, (e.x, e.y, e.x, e.y))

    # ---------------------------------------------------------------- queries
    def query_bbox(
        self, min_x: float, min_y: float, max_x: float, max_y: float
    ) -> List[IndexedEntity]:
        """Return all entities whose point falls inside the bbox."""
        if self.backend == "rtree":
            hits = self._rtree_idx.intersection((min_x, min_y, max_x, max_y))
            return [self._entities[self._rtree_ids[i]] for i in hits]
        results: List[IndexedEntity] = []
        cx0, cy0 = self._cells_for_point(min_x, min_y)
        cx1, cy1 = self._cells_for_point(max_x, max_y)
        seen = set()
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                for eid in self._grid.get((cx, cy), ()):  # candidate cell
                    if eid in seen:
                        continue
                    seen.add(eid)
                    e = self._entities[eid]
                    if min_x <= e.x <= max_x and min_y <= e.y <= max_y:
                        results.append(e)
        return results

    def query_near(
        self,
        x: float,
        y: float,
        k: int = 10,
        max_distance: Optional[float] = None,
        entity_types: Optional[List[str]] = None,
    ) -> List[Tuple[IndexedEntity, float]]:
        """Return the k nearest entities with real Euclidean distances.

        Type filtering happens *before* truncation so k honoured per type.
        """
        if self.backend == "rtree" and max_distance is None and entity_types is None:
            hits = list(self._rtree_idx.nearest((x, y, x, y), k))
            distances = []
            for i in hits:
                e = self._entities[self._rtree_ids[i]]
                distances.append((math.hypot(e.x - x, e.y - y), e))
        else:
            distances = []
            for e in self._entities.values():
                if entity_types is not None and e.entity_type not in entity_types:
                    continue
                d = math.hypot(e.x - x, e.y - y)
                if max_distance is None or d <= max_distance:
                    distances.append((d, e))
            distances.sort(key=lambda t: t[0])
            distances = distances[:k]
        return [(e, d) for d, e in distances]

    # ------------------------------------------------------------------ stats
    def bounds(self) -> Optional[Dict[str, float]]:
        if not self._entities:
            return None
        xs = [e.x for e in self._entities.values()]
        ys = [e.y for e in self._entities.values()]
        return {"min_x": min(xs), "min_y": min(ys), "max_x": max(xs), "max_y": max(ys)}

    @property
    def count(self) -> int:
        return len(self._entities)
