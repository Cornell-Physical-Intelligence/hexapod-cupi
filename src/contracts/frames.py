"""Anatomical axes: forward -body Y, left +body X, up +body Z."""
from __future__ import annotations


__all__ = [
    "BODY_TO_NAVIGATION_MATRIX",
    "FORWARD_AXIS_INDEX",
    "LATERAL_AXIS_INDEX",
    "NAVIGATION_FORWARD_IN_BODY",
    "NAVIGATION_LATERAL_IN_BODY",
    "NAVIGATION_UP_IN_BODY",
    "SCHEMA_VERSION",
    "UP_AXIS_INDEX",
    "body_to_navigation",
    "navigation_to_body",
]

SCHEMA_VERSION = 1

#: Unit vectors of the navigation axes, written in imported-body coordinates.
NAVIGATION_FORWARD_IN_BODY: tuple[float, float, float] = (0.0, -1.0, 0.0)
NAVIGATION_LATERAL_IN_BODY: tuple[float, float, float] = (1.0, 0.0, 0.0)
NAVIGATION_UP_IN_BODY: tuple[float, float, float] = (0.0, 0.0, 1.0)

#: Row-major rotation taking a body vector to ``[forward, lateral, up]``. Each
#: row is the corresponding navigation axis expressed in body coordinates.
BODY_TO_NAVIGATION_MATRIX: tuple[tuple[float, float, float], ...] = (
    NAVIGATION_FORWARD_IN_BODY,
    NAVIGATION_LATERAL_IN_BODY,
    NAVIGATION_UP_IN_BODY,
)

#: Component indices inside a navigation-frame 3-vector, matching the command
#: order ``[forward, lateral, yaw]`` used by :mod:`contracts.command`.
FORWARD_AXIS_INDEX = 0
LATERAL_AXIS_INDEX = 1
UP_AXIS_INDEX = 2


def body_to_navigation(
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate an imported-body vector into ``[forward, lateral, up]``."""

    body_x, body_y, body_z = (float(component) for component in vector)
    return (-body_y, body_x, body_z)


def navigation_to_body(
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a ``[forward, lateral, up]`` vector back into imported-body axes."""

    forward, lateral, up = (float(component) for component in vector)
    return (lateral, -forward, up)
