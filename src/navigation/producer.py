"""A non-blocking pose-to-command protocol."""
from __future__ import annotations

import math
from typing import Protocol, runtime_checkable
from contracts.pose import Pose2D

from contracts.command import VelocityCommand


__all__ = ["CommandProducer", "Pose2D", "wrap_to_pi"]


@runtime_checkable
class CommandProducer(Protocol):
    """Anything that can produce a velocity command for the policy."""

    def update(self, pose: Pose2D, t: float) -> VelocityCommand:
        """Return the command for the tick at time ``t`` seconds.

        ``t`` is monotonic seconds since the control session started, not wall
        clock. Implementations should be usable at the 50 Hz policy rate and
        must not block.
        """
        ...


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap an angle to ``(-pi, pi]``.

    Shared because every producer that compares headings needs it, and getting
    it wrong shows up as a robot that turns the long way around.
    """

    value = float(angle_rad)
    if not math.isfinite(value):
        raise ValueError(f"angle_rad must be finite, got {angle_rad!r}")
    wrapped = math.remainder(value, 2.0 * math.pi)
    if wrapped <= -math.pi:
        wrapped += 2.0 * math.pi
    return wrapped
