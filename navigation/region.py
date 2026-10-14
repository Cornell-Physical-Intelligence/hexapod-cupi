"""Survey boundary, exclusions and the clearance margin a route must respect."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import shapely
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

__all__ = ["PlanMargin", "Point", "SurveyRegion"]

#: An ``(east, north)`` pair in metres, in the open map's local east-north frame.
Point = tuple[float, float]


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

    Every part is a :class:`shapely.Polygon` in the local east-north frame, and must be
    valid (in particular, not self-intersecting); an invalid drawing is rejected here,
    before approval, rather than after motion.

    Raises:
        TypeError: ``exclusions`` is not a sequence (a generator, say), or a part is
            not a ``shapely.Polygon``.
        ValueError: A part is not valid, for example self-intersecting. The message
            names the part (``boundary``, ``corridor`` or ``exclusions[i]``) and
            gives shapely's reason.
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
        parts = {"boundary": self.boundary, "corridor": self.corridor}
        parts.update(
            (f"exclusions[{index}]", exclusion)
            for index, exclusion in enumerate(exclusions)
        )
        for name, polygon in parts.items():
            if polygon is None and name == "corridor":
                continue
            if not isinstance(polygon, shapely.Polygon):
                raise TypeError(
                    f"{name} must be a shapely.Polygon, got {type(polygon).__name__}"
                )
            if not polygon.is_valid:
                raise ValueError(f"{name} is not valid: {explain_validity(polygon)}")
        object.__setattr__(self, "exclusions", exclusions)

        area = self.boundary
        if self.corridor is not None:
            area = area.union(self.corridor)
        object.__setattr__(
            self, "free_space", area.difference(shapely.union_all(exclusions))
        )

    @property
    def bounds_m(self) -> tuple[float, float, float, float]:
        """``(min_east, min_north, max_east, max_north)`` of the boundary and corridor."""

        bounds = shapely.total_bounds([self.boundary, self.corridor])
        return tuple(float(value) for value in bounds)

    def feasible(self, margin: PlanMargin) -> BaseGeometry:
        """Where the robot's centre may go: free space shrunk by ``margin.total_m``.

        Shrinking the free space and inflating every edge by the margin are the same
        operation. A margin too large for the area gives an empty geometry -- reported
        rather than raised.
        """

        return self.free_space.buffer(-margin.total_m)
