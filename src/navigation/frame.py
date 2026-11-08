"""A fixed-origin, two-dimensional approximation of a WGS84 map frame.

Origin construction and forward conversion are drafted. Inverse conversion and
polygon helpers remain unimplemented.
"""

from __future__ import annotations

import math

from shapely import Polygon

from .region import Point

__all__ = ["LocalFrame"]

# https://proj.org/en/stable/usage/ellipsoids.html
_WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
_WGS84_FLATTENING = 1 / 298.257223563
_WGS84_ECCENTRICITY_SQUARED = _WGS84_FLATTENING * (2 - _WGS84_FLATTENING)


def _validate_lat_lon(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """
    Raises if latitude is outside (-90, 90) or longitude is outside [-180, 180].
    """

    lat_deg, lon_deg = float(lat_deg), float(lon_deg)
    if not math.isfinite(lat_deg) or not -90 <= lat_deg <= 90:
        raise ValueError(f"latitude must be finite and in [-90, 90], got {lat_deg!r}")
    if abs(lat_deg) == 90:
        raise ValueError("origin latitude must be strictly between -90 and 90 degrees")
    # TODO: Should the above two really be separated? I could see argument for and against...

    if not math.isfinite(lon_deg) or not -180 <= lon_deg <= 180:
        raise ValueError(f"longitude must be finite and in [-180, 180], got {lon_deg!r}")
    return lat_deg, lon_deg


class LocalFrame:
    """Convert geographic degrees to local east/north metres and back.

    The caller supplies an origin that remains fixed for this object's lifetime.
    The origin maps to (0, 0); +x is east and +y is north at the origin.
    Heights are unsupported.

    Notes:
        Intended for local survey areas on the order of 100 x 100 m. This describes
        intended use, not a guaranteed accuracy range. Accuracy against full WGS84
        tangent-plane coordinates over this survey scale has not yet been characterized.

        The local ellipsoid scales are held fixed at the origin. Heights and
        terrain distances are not modelled.
    """

    def __init__(self, *, origin_lat_deg: float, origin_lon_deg: float) -> None:
        """Set the read-only origin and calculate its local metres-per-degree scales.

        Args:
            origin_lat_deg: Origin latitude in WGS84 decimal degrees.
            origin_lon_deg: Origin longitude in WGS84 decimal degrees.

        Raises:
            ValueError: If either coordinate is non-finite, latitude is outside
                (-90, 90), or longitude is outside [-180, 180]. The poles are
                excluded because the longitude scale vanishes there.
        """
        origin_lat_deg, origin_lon_deg = _validate_lat_lon(origin_lat_deg, origin_lon_deg)

        self._origin_lat_deg = origin_lat_deg
        self._origin_lon_deg = origin_lon_deg

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
    def origin_lat_deg(self) -> float:
        """Read-only origin latitude in WGS84 decimal degrees."""
        return self._origin_lat_deg

    @property
    def origin_lon_deg(self) -> float:
        """Read-only origin longitude in WGS84 decimal degrees."""
        return self._origin_lon_deg

    def to_local(self, *, lat_deg: float, lon_deg: float) -> Point:
        """Convert WGS84 degrees to an (east, north) pair in metres.

        Args:
            lat_deg: Latitude in WGS84 decimal degrees.
            lon_deg: Longitude in WGS84 decimal degrees.

        Returns:
            An (east, north) pair in metres relative to the fixed origin.
            Longitude differences take the shorter direction across +/-180 degrees.

        Raises:
            ValueError: If either coordinate is non-finite, latitude is outside
                [-90, 90], or longitude is outside [-180, 180].
        """
        # TODO: I need to review this logic
        lat_deg, lon_deg = _validate_lat_lon(lat_deg, lon_deg)
        lon_delta_deg = math.remainder(lon_deg - self._origin_lon_deg, 360.0)
        east_m = lon_delta_deg * self._east_m_per_deg
        north_m = (lat_deg - self._origin_lat_deg) * self._north_m_per_deg
        return east_m, north_m

    def to_lat_lon(self, *, east_m: float, north_m: float) -> tuple[float, float]:
        """Convert local metres to a (latitude, longitude) pair in degrees.

        Args:
            east_m: Eastward offset from the fixed origin in metres.
            north_m: Northward offset from the fixed origin in metres.

        Returns:
            A (latitude, longitude) pair in WGS84 decimal degrees, with longitude
            in [-180, 180].

        Raises:
            ValueError: If either coordinate is non-finite or the resulting
                latitude is outside [-90, 90].
        """
        raise NotImplementedError("LocalFrame inverse conversion is not implemented.")

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
        raise NotImplementedError("LocalFrame polygon conversion is not implemented.")

    def polygon_to_lat_lon(self, polygon: Polygon) -> Polygon:
        """Convert polygon vertices from (east, north) to (longitude, latitude).

        Each vertex uses the conversion and checks of :meth:`to_lat_lon`.
        Polygon vertices store longitude first (x), then latitude (y);
        :meth:`to_lat_lon` returns latitude first, then longitude.

        Args:
            polygon: A two-dimensional polygon with (east, north) vertices in metres.

        Returns:
            A new polygon with (longitude, latitude) vertices in WGS84 decimal
            degrees. The outer boundary and any holes retain their vertex order
            and count. Empty input stays empty.

        Raises:
            TypeError: If the input is not a Polygon.
            ValueError: If the polygon has non-2D coordinates or any vertex fails
                the coordinate checks of :meth:`to_lat_lon`.
        """
        raise NotImplementedError("LocalFrame polygon conversion is not implemented.")
