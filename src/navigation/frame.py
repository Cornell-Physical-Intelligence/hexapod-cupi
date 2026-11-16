"""A fixed-origin, two-dimensional approximation of a WGS84 map frame.

The local scale formulas remain a draft awaiting review. Forward and inverse
conversion use the same fixed scales; polygon helpers preserve vertex order.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from shapely import Polygon, get_coordinate_dimension, transform

from .region import Point

__all__ = ["LocalFrame"]

# https://proj.org/en/stable/usage/ellipsoids.html
_WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
_WGS84_FLATTENING = 1 / 298.257223563
_WGS84_ECCENTRICITY_SQUARED = _WGS84_FLATTENING * (2 - _WGS84_FLATTENING)


def _validate_lon_lat(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    """
    Raises if latitude is outside (-90, 90) or longitude is outside [-180, 180].
    """

    lon_deg, lat_deg = float(lon_deg), float(lat_deg)
    if not math.isfinite(lat_deg) or not -90 <= lat_deg <= 90:
        raise ValueError(f"latitude must be finite and in [-90, 90], got {lat_deg!r}")
    if abs(lat_deg) == 90:
        raise ValueError("latitude must be strictly between -90 and 90 degrees")
    # TODO: Should the above two really be separated? I could see argument for and against...

    if not math.isfinite(lon_deg) or not -180 <= lon_deg <= 180:
        raise ValueError(f"longitude must be finite and in [-180, 180], got {lon_deg!r}")
    return lon_deg, lat_deg


def _transform_polygon(polygon: Polygon, convert: Callable[[float, float], Point]) -> Polygon:
    if not isinstance(polygon, Polygon):
        raise TypeError(f"polygon must be a Polygon, got {type(polygon).__name__}")
    if get_coordinate_dimension(polygon) != 2:
        raise ValueError("polygon coordinates must be two-dimensional")

    def convert_coordinates(coordinates: np.ndarray) -> np.ndarray:
        converted = np.empty_like(coordinates)
        for index, (x, y) in enumerate(coordinates):
            converted[index] = convert(x, y)
        return converted

    # Compatible with Shapely 2.0: the callback takes an (N, 2) coordinate array.
    # https://shapely.readthedocs.io/en/2.0.7/reference/shapely.transform.html
    return transform(polygon, convert_coordinates)


class LocalFrame:
    """Convert (longitude, latitude) degrees to local (east, north) metres and back.

    The caller supplies an origin that remains fixed for this object's lifetime.
    The origin maps to (0, 0); +x is east and +y is north at the origin.
    Heights are unsupported.

    Notes:
        Intended for local survey areas on the order of 100 x 100 m. This describes
        intended use, not a guaranteed accuracy range. A zero-height WGS84 ECEF/ENU
        comparison at latitude 42.4447 degrees, with a southwest origin, observed
        maximum horizontal forward errors of 1.60 mm at 100 x 100 m, 6.42 mm at
        200 x 200 m and 160.43 mm at 1000 x 1000 m. Each area used a 41 x 41 grid
        and three seeded 2048-point samples (7825 points). These are sampled
        observations, not bounds for other sites or all points between samples.

        The local ellipsoid scales are held fixed at the origin. Heights and
        terrain distances are not modelled.
    """

    def __init__(self, *, origin_lon_deg: float, origin_lat_deg: float) -> None:
        """Set the read-only origin and calculate its local metres-per-degree scales.

        Args:
            origin_lon_deg: Origin longitude in WGS84 decimal degrees.
            origin_lat_deg: Origin latitude in WGS84 decimal degrees.

        Raises:
            ValueError: If either coordinate is non-finite, latitude is outside
                (-90, 90), or longitude is outside [-180, 180]. The poles are
                excluded because the longitude scale vanishes there.
        """
        origin_lon_deg, origin_lat_deg = _validate_lon_lat(origin_lon_deg, origin_lat_deg)

        self._origin_lon_deg = origin_lon_deg
        self._origin_lat_deg = origin_lat_deg

        # Curvature radii: https://www.oc.nps.edu/oc2902w/geodesy/radiigeo.pdf, p. 4.
        # TODO: I need to review this logic
        origin_lat_rad = math.radians(origin_lat_deg)
        curvature_factor = 1 - _WGS84_ECCENTRICITY_SQUARED * math.sin(origin_lat_rad) ** 2
        prime_vertical_radius_m = _WGS84_SEMI_MAJOR_AXIS_M / math.sqrt(curvature_factor)
        meridional_radius_m = (
            _WGS84_SEMI_MAJOR_AXIS_M * (1 - _WGS84_ECCENTRICITY_SQUARED) / curvature_factor**1.5
        )
        self._north_m_per_deg = meridional_radius_m * math.pi / 180
        self._east_m_per_deg = prime_vertical_radius_m * math.cos(origin_lat_rad) * math.pi / 180

    @property
    def origin_lon_deg(self) -> float:
        """Read-only origin longitude in WGS84 decimal degrees."""
        return self._origin_lon_deg

    @property
    def origin_lat_deg(self) -> float:
        """Read-only origin latitude in WGS84 decimal degrees."""
        return self._origin_lat_deg

    def to_local(self, lon_deg: float, lat_deg: float) -> Point:
        """Convert WGS84 (longitude, latitude) degrees to (east, north) metres.

        Args:
            lon_deg: Longitude in WGS84 decimal degrees.
            lat_deg: Latitude in WGS84 decimal degrees.

        Returns:
            An (east, north) pair in metres relative to the fixed origin.
            Longitude differences take the shorter direction across +/-180 degrees.

        Raises:
            ValueError: If either coordinate is non-finite, latitude is outside
                (-90, 90), or longitude is outside [-180, 180].
        """
        # TODO: I need to review this logic
        lon_deg, lat_deg = _validate_lon_lat(lon_deg, lat_deg)
        lon_delta_deg = math.remainder(lon_deg - self._origin_lon_deg, 360.0)
        east_m = lon_delta_deg * self._east_m_per_deg
        north_m = (lat_deg - self._origin_lat_deg) * self._north_m_per_deg
        return east_m, north_m

    def to_lon_lat(self, east_m: float, north_m: float) -> tuple[float, float]:
        """Convert local (east, north) metres to (longitude, latitude) degrees.

        Divide offsets by the same fixed scales used by :meth:`to_local`.

        Args:
            east_m: Eastward offset from the fixed origin in metres.
            north_m: Northward offset from the fixed origin in metres.

        Returns:
            A (longitude, latitude) pair in WGS84 decimal degrees, with longitude
            in [-180, 180].

        Raises:
            ValueError: If either coordinate is non-finite or the resulting
                latitude is outside (-90, 90), or conversion overflows.
        """
        east_m, north_m = float(east_m), float(north_m)
        if not math.isfinite(east_m) or not math.isfinite(north_m):
            raise ValueError("east_m and north_m must be finite")

        lat_deg = self._origin_lat_deg + north_m / self._north_m_per_deg
        lon_delta_deg = east_m / self._east_m_per_deg
        if not math.isfinite(lon_delta_deg):
            raise ValueError("east_m is too large to convert to longitude")
        lon_deg = math.remainder(self._origin_lon_deg + math.remainder(lon_delta_deg, 360.0), 360.0)
        return _validate_lon_lat(lon_deg, lat_deg)

    def polygon_to_local(self, polygon: Polygon) -> Polygon:
        """Convert polygon vertices from (longitude, latitude) to (east, north).

        Each vertex uses the conversion and checks of :meth:`to_local`.

        Args:
            polygon: A two-dimensional polygon with (longitude, latitude) vertices
                in WGS84 decimal degrees.

        Returns:
            A new polygon with (east, north) vertices in metres. The outer boundary
            and any holes retain their vertex order and count. Empty input stays empty.

        Raises:
            TypeError: If the input is not a Polygon.
            ValueError: If the polygon has non-2D coordinates or any vertex fails
                the coordinate checks of :meth:`to_local`.
        """
        return _transform_polygon(polygon, self.to_local)

    def polygon_to_lon_lat(self, polygon: Polygon) -> Polygon:
        """Convert polygon vertices from (east, north) to (longitude, latitude).

        Each vertex uses the conversion and checks of :meth:`to_lon_lat`.

        Args:
            polygon: A two-dimensional polygon with (east, north) vertices in metres.

        Returns:
            A new polygon with (longitude, latitude) vertices in WGS84 decimal
            degrees. The outer boundary and any holes retain their vertex order
            and count. Empty input stays empty.

        Raises:
            TypeError: If the input is not a Polygon.
            ValueError: If the polygon has non-2D coordinates or any vertex fails
                the coordinate checks of :meth:`to_lon_lat`.
        """
        return _transform_polygon(polygon, self.to_lon_lat)
