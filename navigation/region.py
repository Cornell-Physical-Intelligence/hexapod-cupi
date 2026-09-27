"""Survey boundary, exclusions and the clearance margin a route must respect."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


__all__ = ["PlanMargin", "Point", "Polygon", "SurveyRegion"]

#: An ``(east, north)`` pair in metres, in the open map's local east-north frame.
Point = tuple[float, float]


@dataclass(frozen=True)
class PlanMargin:
    """Clearance a route keeps from every boundary and exclusion edge.

    R-05 requires the full moving footprint to stay inside the approved region
    including localization and stopping uncertainty. The three terms stay separate
    because they come from different owners and move independently:
    ``footprint_radius_m`` from the robot, ``stop_tolerance_m`` from the follower,
    ``position_error_bound_m`` from R-03's GPS bound.

    ``position_error_bound_m`` is an **input, never a constant**. A rehearsal with
    ideal poses passes ``0.0``; the plain-GPS bound of several metres is what makes
    a small drawn area infeasible, and the caller -- not this module -- owns that
    number.
    """

    footprint_radius_m: float
    stop_tolerance_m: float
    position_error_bound_m: float

    def __post_init__(self) -> None:
        for name in (
            "footprint_radius_m",
            "stop_tolerance_m",
            "position_error_bound_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative, got {value!r}"
                )
        if float(self.footprint_radius_m) <= 0.0:
            raise ValueError(
                f"footprint_radius_m must be positive, got {self.footprint_radius_m!r}"
            )

    @property
    def total_m(self) -> float:
        """Clearance every passable cell must have from the nearest edge."""

        return (
            float(self.footprint_radius_m)
            + float(self.stop_tolerance_m)
            + float(self.position_error_bound_m)
        )


@dataclass(frozen=True)
class Polygon:
    """A closed polygon in the local east-north frame.

    ``vertices`` are ``(east, north)`` metre pairs in order. The closing edge from
    the last vertex back to the first is implicit and must not be repeated.

    Simplicity is **not** checked on construction, deliberately: an operator's
    drawing has to be *reported* as invalid rather than raise while a request is
    being parsed. Callers check :meth:`is_simple` before approving a request.
    """

    vertices: tuple[Point, ...]

    def __post_init__(self) -> None:
        raw = self.vertices
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise TypeError(
                f"vertices must be a sequence of (east, north) pairs, "
                f"got {type(raw).__name__}"
            )
        points: list[Point] = []
        for index, vertex in enumerate(raw):
            if isinstance(vertex, (str, bytes)) or not isinstance(vertex, Sequence):
                raise TypeError(
                    f"vertices[{index}] must be an (east, north) pair, got {vertex!r}"
                )
            pair = tuple(float(value) for value in vertex)
            if len(pair) != 2 or not all(math.isfinite(value) for value in pair):
                raise ValueError(
                    f"vertices[{index}] must be a finite (east, north) pair, "
                    f"got {vertex!r}"
                )
            points.append((pair[0], pair[1]))
        if len(points) < 3:
            raise ValueError(
                f"vertices must hold at least three points, got {len(points)}"
            )
        # Normalise to an immutable tuple of float pairs. A frozen dataclass that
        # stored the caller's list would be mutable through the back door and
        # unhashable, so the value is replaced rather than merely validated.
        object.__setattr__(self, "vertices", tuple(points))

    @property
    def edges(self) -> tuple[tuple[Point, Point], ...]:
        """Every edge as a ``(start, end)`` pair, including the closing edge."""

        points = self.vertices
        return tuple((points[i], points[(i + 1) % len(points)]) for i in range(len(points)))

    @property
    def bounds_m(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds as ``(min_east, min_north, max_east, max_north)``."""

        easts = [point[0] for point in self.vertices]
        norths = [point[1] for point in self.vertices]
        return (min(easts), min(norths), max(easts), max(norths))

    @property
    def signed_area_m2(self) -> float:
        """Shoelace area; positive when the vertices wind counterclockwise."""

        total = 0.0
        for (x0, y0), (x1, y1) in self.edges:
            total += x0 * y1 - x1 * y0
        return 0.5 * total

    def is_simple(self) -> bool:
        """True when no two non-adjacent edges intersect and no edge is degenerate.

        # CHALLENGE 5 -- pinned by ``test_self_crossing_polygon_is_rejected``.
        #
        # Write a segment-intersection predicate and apply it to every pair of
        # edges. Two points to get right, because they are what the naive version
        # misses:
        #   * Adjacent edges legitimately touch at their shared vertex, so they
        #     must be exempt -- but only at that vertex. An adjacent pair that
        #     overlaps along a length is still self-crossing.
        #   * Collinear overlap is an intersection even though the usual
        #     orientation test reports zero for every combination.
        # Prefer exact orientation signs (cross products) over computing an
        # intersection point; dividing invites both precision loss and a
        # zero-division branch.
        """

        raise NotImplementedError("CHALLENGE 5: implement the simplicity predicate")

    def contains(self, point: Point) -> bool:
        """True when ``point`` lies inside this polygon.

        # CHALLENGE -- exercised throughout ``test_grid.py`` and ``test_region.py``.
        #
        # Ray casting is the usual approach: count boundary crossings of a ray from
        # the point and return True on an odd count. The cases that decide whether
        # this is correct are the degenerate ones -- a ray that passes exactly
        # through a vertex must not be counted twice, and a horizontal edge at the
        # ray's height must not be counted at all. Using a half-open rule on each
        # edge's north span handles both without special-casing.
        #
        # Points exactly on an edge may return either answer; no test depends on
        # it, and the clearance margin makes the ambiguity unreachable in practice.
        """

        raise NotImplementedError("CHALLENGE: implement point-in-polygon")

    def distance_to_boundary_m(self, point: Point) -> float:
        """Shortest distance from ``point`` to any edge of this polygon.

        # CHALLENGE -- pinned by ``test_inflation_matches_a_brute_force_distance_check``.
        #
        # The minimum over edges of the point-to-segment distance. The part people
        # get wrong is the segment, not the line: project onto the segment and
        # clamp the parameter to ``[0, 1]`` before measuring, or points beyond an
        # endpoint report the distance to the infinite line instead. Handle a
        # zero-length edge without dividing by zero.
        """

        raise NotImplementedError("CHALLENGE: implement point-to-segment distance")


@dataclass(frozen=True)
class SurveyRegion:
    """An approved survey area: one boundary, optional exclusions and a corridor.

    Exclusions are the operator's drawn no-go areas of R-01. This type models the
    *request*, so it holds no sensed obstacle: under the GPS-only architecture
    (`ARCHITECTURE.md` R-10, and #20's out-of-scope list) obstacles are drawn
    before motion, not discovered during it.
    """

    boundary: Polygon
    exclusions: tuple[Polygon, ...] = ()
    corridor: Polygon | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, Polygon):
            raise TypeError(
                f"boundary must be a Polygon, got {type(self.boundary).__name__}"
            )
        raw = self.exclusions
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise TypeError(
                f"exclusions must be a sequence of Polygon, got {type(raw).__name__}"
            )
        for index, exclusion in enumerate(raw):
            if not isinstance(exclusion, Polygon):
                raise TypeError(
                    f"exclusions[{index}] must be a Polygon, got {exclusion!r}"
                )
        if self.corridor is not None and not isinstance(self.corridor, Polygon):
            raise TypeError(
                f"corridor must be a Polygon or None, got {type(self.corridor).__name__}"
            )
        object.__setattr__(self, "exclusions", tuple(raw))

    @property
    def bounds_m(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds of the boundary, plus the corridor when present."""

        min_east, min_north, max_east, max_north = self.boundary.bounds_m
        if self.corridor is not None:
            other = self.corridor.bounds_m
            min_east, min_north = min(min_east, other[0]), min(min_north, other[1])
            max_east, max_north = max(max_east, other[2]), max(max_north, other[3])
        return (min_east, min_north, max_east, max_north)

    def is_simple(self) -> bool:
        """True when the boundary, every exclusion and the corridor are simple."""

        parts = [self.boundary, *self.exclusions]
        if self.corridor is not None:
            parts.append(self.corridor)
        return all(part.is_simple() for part in parts)

    def contains(self, point: Point) -> bool:
        """True when ``point`` is inside the boundary or corridor, outside exclusions."""

        inside = self.boundary.contains(point)
        if not inside and self.corridor is not None:
            inside = self.corridor.contains(point)
        if not inside:
            return False
        return not any(exclusion.contains(point) for exclusion in self.exclusions)

    def clearance_m(self, point: Point) -> float:
        """Distance from ``point`` to the nearest edge it must stay clear of.

        Returns ``0.0`` for a point that is not inside the region at all, so that
        "passable" is uniformly ``clearance_m(point) >= margin.total_m`` and no
        caller has to special-case being outside.
        """

        if not self.contains(point):
            return 0.0
        distances = [self.boundary.distance_to_boundary_m(point)]
        if self.corridor is not None:
            # Inside the corridor the boundary edge it shares is not a wall, so the
            # corridor's own edges are the binding constraint there.
            distances.append(self.corridor.distance_to_boundary_m(point))
            if self.corridor.contains(point):
                distances = [self.corridor.distance_to_boundary_m(point)]
        distances.extend(
            exclusion.distance_to_boundary_m(point) for exclusion in self.exclusions
        )
        return min(distances)
