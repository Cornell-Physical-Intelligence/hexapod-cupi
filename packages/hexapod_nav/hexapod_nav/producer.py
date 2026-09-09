"""The command-producer seam: what path planning has to look like from below.

A command producer answers exactly one question, once per control tick: given
where the robot is and what time it is, what velocity command should the policy
receive? The answer is a :class:`hexapod_core.command.VelocityCommand` -- three
scalars -- and that is the entire interface. A producer may be a waypoint
follower, a joystick bridge, a replayed trace, a full planner with a costmap,
or a stub that always stands still; the locomotion side cannot tell the
difference, and that is the design.

Two rules keep this seam honest:

* A producer never reaches downward. It does not know about joints, gaits,
  torques, or the simulator. Everything it emits goes through the command
  contract.
* A producer's output must be admissible. Emitting a command outside the
  envelope the policy was trained on is not "asking for more" -- it is asking
  for undefined behavior from a network that has never seen it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hexapod_core.command import VelocityCommand


__all__ = ["CommandProducer", "Pose2D", "wrap_to_pi"]


@dataclass(frozen=True)
class Pose2D:
    """Planar pose in the world/navigation frame.

    ``heading_rad`` is the robot's forward direction measured counterclockwise
    from world +x, so a robot at ``heading_rad = 0`` moving at ``vx_mps > 0``
    travels toward increasing ``x_m``. Forward is the anatomical forward of
    :mod:`hexapod_core.frames`, not the imported body +X.
    """

    x_m: float
    y_m: float
    heading_rad: float

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "heading_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")


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
