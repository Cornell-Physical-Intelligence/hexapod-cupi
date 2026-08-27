"""A minimal waypoint follower: an example of the seam, not a tuned planner.

What this is: about forty lines of trigonometry that turn a list of goal points
into velocity commands. Heading error drives a clamped yaw rate; distance to
the current waypoint drives a clamped forward speed; inside the stop radius the
waypoint is retired, and with no waypoints left the command is a stand.

What this is not: it has no obstacle awareness, no path smoothing, no velocity
feed-forward, no acceleration limits, no notion of the robot's actual dynamics,
and no state estimation of any kind -- it trusts the pose it is handed. It is
memoryless apart from which waypoint is current, so its ``t`` argument is
accepted for the :class:`~hexapod_nav.producer.CommandProducer` protocol and
otherwise unused. A real planner replaces this class wholesale; what it must
not change is the interface, which is the reason this file exists.

Two deliberate choices worth knowing before copying it:

* Lateral velocity is always exactly zero. The deployed Stage2C policy was
  trained with the lateral command pinned to zero, so a follower that emitted
  side velocity would be asking for behavior no accepted checkpoint has.
* Forward speed is gated on heading: past ``heading_gate_rad`` of heading
  error the robot turns in place rather than driving an arc. That is the
  conservative choice for a hexapod with modest yaw authority, and it makes the
  behavior easy to reason about; it is not the fastest path.

Everything emitted is clamped into a :class:`~hexapod_core.command
.CommandEnvelope` and validated against it, so a follower can never hand the
policy a command outside what it was trained on.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from hexapod_core.command import (
    PHASE2_TARGET_ENVELOPE,
    CommandEnvelope,
    Frame,
    VelocityCommand,
)

from .producer import Pose2D, wrap_to_pi


__all__ = ["WaypointFollower"]

STAND = VelocityCommand(vx_mps=0.0, vy_mps=0.0, wz_rad_s=0.0, frame=Frame.NAVIGATION)


class WaypointFollower:
    """Drive a sequence of ``(x, y)`` goals with forward speed and yaw rate."""

    def __init__(
        self,
        waypoints: Iterable[Sequence[float]],
        *,
        cruise_speed_mps: float = 0.24,
        approach_gain_per_s: float = 0.60,
        max_yaw_rate_rad_s: float = 0.40,
        heading_gain_per_s: float = 1.20,
        heading_gate_rad: float = math.pi / 3.0,
        stop_radius_m: float = 0.12,
        envelope: CommandEnvelope = PHASE2_TARGET_ENVELOPE,
    ) -> None:
        self.waypoints: list[tuple[float, float]] = []
        for index, waypoint in enumerate(waypoints):
            point = tuple(float(value) for value in waypoint)
            if len(point) != 2 or not all(math.isfinite(value) for value in point):
                raise ValueError(
                    f"waypoints[{index}] must be a finite (x, y) pair, got {waypoint!r}"
                )
            self.waypoints.append((point[0], point[1]))
        for name, value in (
            ("cruise_speed_mps", cruise_speed_mps),
            ("approach_gain_per_s", approach_gain_per_s),
            ("max_yaw_rate_rad_s", max_yaw_rate_rad_s),
            ("heading_gain_per_s", heading_gain_per_s),
            ("heading_gate_rad", heading_gate_rad),
            ("stop_radius_m", stop_radius_m),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        self.cruise_speed_mps = float(cruise_speed_mps)
        self.approach_gain_per_s = float(approach_gain_per_s)
        self.max_yaw_rate_rad_s = float(max_yaw_rate_rad_s)
        self.heading_gain_per_s = float(heading_gain_per_s)
        self.heading_gate_rad = float(heading_gate_rad)
        self.stop_radius_m = float(stop_radius_m)
        self.envelope = envelope
        self._index = 0

    @property
    def is_finished(self) -> bool:
        """True once every waypoint has been retired."""

        return self._index >= len(self.waypoints)

    @property
    def current_waypoint(self) -> tuple[float, float] | None:
        """The goal being driven toward, or ``None`` when finished."""

        if self.is_finished:
            return None
        return self.waypoints[self._index]

    def update(self, pose: Pose2D, t: float) -> VelocityCommand:
        """Return the command for this tick. ``t`` is unused; see the module docstring."""

        del t  # memoryless: the follower's only state is the waypoint index
        if not isinstance(pose, Pose2D):
            raise TypeError(f"pose must be a Pose2D, got {type(pose).__name__}")

        # Retire every waypoint already satisfied, so a pose that lands inside
        # several small goals at once does not need several ticks to catch up.
        while not self.is_finished:
            goal_x, goal_y = self.waypoints[self._index]
            if math.hypot(goal_x - pose.x_m, goal_y - pose.y_m) <= self.stop_radius_m:
                self._index += 1
                continue
            break
        if self.is_finished:
            return STAND

        goal_x, goal_y = self.waypoints[self._index]
        delta_x = goal_x - pose.x_m
        delta_y = goal_y - pose.y_m
        distance = math.hypot(delta_x, delta_y)
        heading_error = wrap_to_pi(math.atan2(delta_y, delta_x) - pose.heading_rad)

        yaw_rate = _clamp(
            self.heading_gain_per_s * heading_error,
            -self.max_yaw_rate_rad_s,
            self.max_yaw_rate_rad_s,
        )
        if abs(heading_error) >= self.heading_gate_rad:
            forward = 0.0
        else:
            forward = min(self.approach_gain_per_s * distance, self.cruise_speed_mps)

        command = VelocityCommand(
            vx_mps=forward,
            vy_mps=0.0,
            wz_rad_s=yaw_rate,
            frame=Frame.NAVIGATION,
        )
        if command.is_stand():
            return STAND
        command = self.envelope.clamp(command)
        # Cheap and worth it: a producer that emits an inadmissible command
        # should fail here, next to the bug, not inside the policy.
        command.validate(self.envelope)
        return command


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))
