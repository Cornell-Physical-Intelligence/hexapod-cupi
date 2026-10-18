"""Survey boundary, exclusions and the clearance margin a route must respect."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence, Literal

import shapely
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

__all__ = ["PlanMargin", "Point", "RegionRejected", "SurveyRegion"]

# TODO: Who decides the origin of the local frame? How is the frame represented?
#   Both teams must agree (ARCHITECTURE.md section 3; Pose and Survey request
#   contracts in section 4).
#: An ``(east, north)`` pair in meters, in the open map's local east-north frame.
Point = tuple[float, float]


class RegionRejected(ValueError):
    """The operator's drawing cannot be accepted: ask them to redraw ``part``.

    Raised only for problems that can be fixed by redrawing:
        - A part is empty: nothing was drawn.
        - A part is not valid, for example self-intersecting or with a non-finite
          vertex.
        - The corridor does not share an edge with the boundary: it is detached,
          or touches it only at a corner.
        - The exclusions together cover the whole boundary, so nothing is left to
          survey.

    Mistakes in the calling code, such as passing something that is not a
    polygon, raise ``TypeError`` instead.

    Attributes:
        part: What to redraw: one of ``"boundary"``, ``"corridor"``, ``"exclusions"``.
        reason: Why it was rejected, phrased to follow ``part``.
        index: The index ``i`` indicating that ``exclusions[i]`` is at fault.
            ``None`` when ``part`` is not ``"exclusions"``, or when the exclusions
            are at fault only together.
    """

    def __init__(
        self,
        part: Literal["boundary", "corridor", "exclusions"],
        reason: str,
        index: int | None = None,
    ) -> None:
        assert index is None or part == "exclusions"
        super().__init__(part, reason, index)
        self.part = part
        self.reason = reason
        self.index = index

    def __str__(self) -> str:
        part = self.part if self.index is None else f"{self.part}[{self.index}]"
        return f"{part} {self.reason}"


@dataclass(frozen=True)
class PlanMargin:
    """Clearance a route keeps from every boundary and exclusion edge.

    It is required that the full moving footprint stay inside the approved region,
    including localization and stopping uncertainty.

    Args:
        ``footprint_radius_m``: Length of largest line fitting across the hexapod.
        ``stop_tolerance_m``: Perscribed by the specifications of the controller + robot build.
        ``position_error_bound_m``: the specificed GPS bound.

    Raises:
        ValueError: A term is negative or not finite, or ``footprint_radius_m`` is
            zero. The message names the term and the value given.
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
class SurveyRegion:
    """An approved survey area: Region Boundary, optional exclusions and an entry corridor.

    Every part is a :class:`shapely.Polygon` in the local east-north frame (see
    `Point`), and must be valid (in particular, not self-intersecting); an invalid
    drawing is rejected here, before approval, rather than after motion.

    Raises:
        TypeError: ``exclusions`` is not a sequence (a generator, say), or a part is
            not a ``shapely.Polygon``.
        RegionRejected: The operator must redraw ``part``. See
            :class:`RegionRejected` for when this is raised.
    """

    boundary: shapely.Polygon
    exclusions: tuple[shapely.Polygon, ...] = ()
    corridor: shapely.Polygon | None = None
    #: Boundary plus corridor, minus every exclusion: everywhere the robot may be.
    free_space: BaseGeometry = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        raw = self.exclusions
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise TypeError(
                f"exclusions must be a sequence of shapely.Polygon, got {type(raw).__name__}"
            )
        exclusions = tuple(raw)
        parts = [
            ("boundary", None, self.boundary),
            ("corridor", None, self.corridor),
        ]
        parts.extend(
            ("exclusions", index, exclusion)
            for index, exclusion in enumerate(exclusions)
        )
        for part, index, polygon in parts:
            if polygon is None and part == "corridor":
                continue
            if not isinstance(polygon, shapely.Polygon):
                name = part if index is None else f"{part}[{index}]"
                raise TypeError(
                    f"{name} must be a shapely.Polygon, got {type(polygon).__name__}"
                )
            if polygon.is_empty:
                raise RegionRejected(part, "is empty: nothing was drawn", index)
            if not polygon.is_valid:
                raise RegionRejected(
                    part, f"is not valid: {explain_validity(polygon)}", index
                )
        object.__setattr__(self, "exclusions", exclusions)

        area = self.boundary
        if self.corridor is not None:
            area = area.union(self.corridor)
            # Disjoint, or touching at a single point, gives a MultiPolygon.
            if not isinstance(area, shapely.Polygon):
                raise RegionRejected(
                    "corridor", "does not share an edge with the boundary"
                )
        blocked = shapely.union_all(exclusions)
        # The corridor is only a way in, so the boundary is what must survive.
        if self.boundary.difference(blocked).is_empty:
            raise RegionRejected(
                "exclusions",
                "cover the whole boundary: nothing is left to survey",
            )
        object.__setattr__(self, "free_space", area.difference(blocked))

    @property
    def bounds_m(self) -> tuple[float, float, float, float]:
        """``(min_east, min_north, max_east, max_north)`` of the boundary and corridor."""

        bounds = shapely.total_bounds([self.boundary, self.corridor])
        return tuple(float(value) for value in bounds)

    def feasible(self, margin: PlanMargin) -> BaseGeometry:
        """Where the robot's center may go: free space shrunk by ``margin.total_m``.

        Shrinking the free space and inflating every edge by the margin are the same
        operation. A margin too large for the area gives an empty geometry -- reported
        rather than raised.
        """

        # NOTE: Not exact. Buffer rounds corners with straight segments that cut
        # slightly inside the true arc, up to about 1% of the margin.
        return self.free_space.buffer(-margin.total_m)
