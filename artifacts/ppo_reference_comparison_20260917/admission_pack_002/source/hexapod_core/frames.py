"""Frozen v1 coordinate contract: anatomical navigation axes.

```text
anatomical/navigation forward = imported body -Y
navigation lateral            = imported body +X
navigation up                 = imported body +Z
```

This is the contract that fixed the earlier policy which walked visually
sideways while reporting "forward" progress. It is implemented by
``body_to_navigation_frame`` in
``packages/hexapod_env/hexapod_env/command_sampling.py``:

    return torch.stack((-vectors[..., 1], vectors[..., 0], vectors[..., 2]), dim=-1)

i.e. ``[forward, lateral, up] = [-body_y, body_x, body_z]``. ``env.py`` applies
it through ``_vector_in_command_frame`` to the root linear velocity, the root
angular velocity and the projected gravity whenever
``command_frame == "navigation"``, which is every stage from Phase1 v5 onward,
including the deployed Stage2C. The rotation is a plain +90 degree turn about
+Z and is its own inverse only after four applications, so the transpose below
is provided explicitly.

The mapping applies to ordinary vectors and to axial vectors (angular velocity)
alike, because it is a proper rotation: ``det = +1``.

Do not revert this contract, and do not add a second convention beside it. A
different mapping is a v2 module, not an edit here.
"""

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
#: order ``[forward, lateral, yaw]`` used by :mod:`hexapod_core.command`.
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
