"""Waypoint follower whose every command is a training-bank entry or stand.

The omniwalk policy was trained on a finite command bank: eight translation
bearings at 45 degree spacing at two speeds, pure yaw at one rate, and two
arcs. A command between bank entries asks the network for behaviour it has not
seen, so this follower emits bank entries only. It holds the body heading at a
multiple of 45 degrees in the map frame and drives holonomically, so an
eight-connected grid route maps onto bank bearings without turning.

Training also held each command for 2 to 5 seconds before changing it, and the
policy's destination restarts on every change. A translation command is
therefore held for ``hold_min_s`` before the bearing or speed is chosen again;
without the hold, a goal near a 22.5 degree boundary makes the follower
alternate between two neighbouring entries every tick. Reaching a goal or
tripping the heading gate ends a hold at once.

The heading gate has hysteresis: a turn starts once the heading error reaches
``heading_gate_rad`` and ends once it falls inside ``heading_release_rad``.
Without it, a policy whose heading creeps while walking trips the gate every
few ticks and the follower alternates one-tick turns with walks. A goal is
retired at ``stop_radius_m``, which sits inside the mission's stop tolerance by
the body's stopping distance, so the stand command arrives before the goal
and the body rolls to rest inside the tolerance.

Navigation and training build their command banks from the shared command set
in ``contracts.command`` and the owner in ``configs/walking_requirements.json``.
A narrower envelope filters the bank; unavailable motion raises ``ValueError``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from contracts.command import (
    CommandEnvelope,
    CommandSet,
    Frame,
    VelocityCommand,
    load_requirement_set,
)

from .producer import Pose2D, wrap_to_pi

__all__ = [
    "BANK_ARC_SPEED_MPS",
    "TRAINING_HOLD_MIN_S",
    "BANK_ARC_YAW_RATE_RAD_S",
    "BANK_ENVELOPE",
    "BANK_SPEED_LEVELS_MPS",
    "BANK_YAW_RATE_RAD_S",
    "STAND",
    "WaypointFollower",
    "command_bank",
]

# Resolve owner data at import, before a follower starts its control loop.
_DEFAULT_RULES, _ = load_requirement_set()
COMMAND_SET = CommandSet.from_mapping(_DEFAULT_RULES["commands"])
BANK_SPEED_LEVELS_MPS = COMMAND_SET.translate_speeds_mps
BANK_YAW_RATE_RAD_S = COMMAND_SET.turn_rate_rad_s
BANK_ARC_SPEED_MPS = COMMAND_SET.arc_speed_mps
BANK_ARC_YAW_RATE_RAD_S = COMMAND_SET.arc_turn_rate_rad_s
BANK_BEARINGS = COMMAND_SET.translate_bearings
BEARING_STEP_RAD = COMMAND_SET.bearing_step_rad
TRAINING_HOLD_MIN_S = _DEFAULT_RULES["hold_min_s"]
# Retire a goal well inside the 0.12 m stop tolerance of configs/walking_requirements.json:
# the body rolls up to the 0.05 m stopping distance after the stand command, and the
# surrogate closed loop of 7 November 2026 ended 117 to 122 mm from goals retired at 0.12 m.
RETIRE_RADIUS_M = 0.06


def command_bank() -> list[list[float]]:
    """The nonzero navigation-frame bank in its recorded training order."""
    return [list(command.as_tuple()) for command in COMMAND_SET.nonzero()]


BANK_ENVELOPE = COMMAND_SET.envelope()

STAND = VelocityCommand(vx_mps=0.0, vy_mps=0.0, wz_rad_s=0.0, frame=Frame.NAVIGATION)


class WaypointFollower:
    """Drive a sequence of ``(x, y)`` goals with bank translation and bank yaw."""

    def __init__(
        self,
        waypoints: Iterable[Sequence[float]],
        *,
        approach_gain_per_s: float = 0.60,
        heading_gate_rad: float = 0.05,
        heading_release_rad: float = 0.01,
        stop_radius_m: float = RETIRE_RADIUS_M,
        hold_min_s: float = TRAINING_HOLD_MIN_S,
        envelope: CommandEnvelope | None = None,
        command_set: CommandSet = COMMAND_SET,
    ) -> None:
        if not command_set.stand:
            raise ValueError("WaypointFollower requires a permitted stand command")
        self.command_set = command_set
        self.envelope = command_set.envelope() if envelope is None else envelope
        STAND.validate(self.envelope)
        self._translations = [
            [
                (speed, command)
                for speed in command_set.translate_speeds_mps
                if (command := command_set.translation(speed, bearing)).is_within(self.envelope)
            ]
            for bearing in range(command_set.translate_bearings)
        ]
        self.waypoints: list[tuple[float, float]] = []
        for index, waypoint in enumerate(waypoints):
            point = tuple(float(value) for value in waypoint)
            if len(point) != 2 or not all(math.isfinite(value) for value in point):
                raise ValueError(
                    f"waypoints[{index}] must be a finite (x, y) pair, got {waypoint!r}"
                )
            self.waypoints.append((point[0], point[1]))
        for name, value in (
            ("approach_gain_per_s", approach_gain_per_s),
            ("heading_gate_rad", heading_gate_rad),
            ("stop_radius_m", stop_radius_m),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if not math.isfinite(float(hold_min_s)) or float(hold_min_s) < 0.0:
            raise ValueError(f"hold_min_s must be finite and non-negative, got {hold_min_s!r}")
        release = float(heading_release_rad)
        if not math.isfinite(release) or not 0.0 <= release < float(heading_gate_rad):
            raise ValueError(
                "heading_release_rad must lie in [0, heading_gate_rad), "
                f"got {heading_release_rad!r}"
            )
        # The held heading is the nearest bank bearing, so the error is at most
        # half one bearing step; a gate at or past that never trips.
        if float(heading_gate_rad) >= command_set.bearing_step_rad / 2.0:
            raise ValueError(
                f"heading_gate_rad must be below {command_set.bearing_step_rad / 2.0:.4f}, "
                f"got {heading_gate_rad!r}"
            )
        self.approach_gain_per_s = float(approach_gain_per_s)
        self.heading_gate_rad = float(heading_gate_rad)
        self.heading_release_rad = release
        self._turning = False
        self.stop_radius_m = float(stop_radius_m)
        self.hold_min_s = float(hold_min_s)
        self._index = 0
        self._held: VelocityCommand | None = None
        self._hold_until_s = -math.inf

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

    @staticmethod
    def held_heading(heading_rad: float, bearing_step_rad: float = BEARING_STEP_RAD) -> float:
        """The bank bearing nearest ``heading_rad``, wrapped to ``(-pi, pi]``."""

        return wrap_to_pi(round(float(heading_rad) / bearing_step_rad) * bearing_step_rad)

    def update(self, pose: Pose2D, t: float) -> VelocityCommand:
        """Return the bank command for the tick at ``t`` seconds; see the module docstring."""

        if not isinstance(pose, Pose2D):
            raise TypeError(f"pose must be a Pose2D, got {type(pose).__name__}")
        t = float(t)
        if not math.isfinite(t):
            raise ValueError(f"t must be finite, got {t!r}")

        # Retire every waypoint already satisfied, so a pose that lands inside
        # several small goals at once does not need several ticks to catch up.
        while not self.is_finished:
            goal_x, goal_y = self.waypoints[self._index]
            if math.hypot(goal_x - pose.x_m, goal_y - pose.y_m) <= self.stop_radius_m:
                self._index += 1
                self._held = None
                self._turning = False
                continue
            break
        if self.is_finished:
            return STAND

        heading_error = wrap_to_pi(
            self.held_heading(pose.heading_rad, self.command_set.bearing_step_rad)
            - pose.heading_rad
        )
        # A turn starts past the gate and ends inside the release, so a heading that
        # creeps across the gate does not alternate one-tick turns with walks.
        self._turning = abs(heading_error) >= self.heading_gate_rad or (
            self._turning and abs(heading_error) > self.heading_release_rad
        )
        if self._turning:
            # The gate ends a hold: the correction is a few ticks long and
            # holding it for hold_min_s would overshoot the held heading.
            self._held = None
            yaw_rate = self.command_set.turn_rate_rad_s * (1 if heading_error > 0.0 else -1)
            command = VelocityCommand(0.0, 0.0, yaw_rate, frame=Frame.NAVIGATION)
            if not command.is_within(self.envelope):
                raise ValueError(f"Required yaw {yaw_rate} rad/s is unavailable in the envelope")
        elif self._held is not None and t < self._hold_until_s:
            command = self._held
        else:
            goal_x, goal_y = self.waypoints[self._index]
            delta_x = goal_x - pose.x_m
            delta_y = goal_y - pose.y_m
            distance = math.hypot(delta_x, delta_y)
            body_bearing = wrap_to_pi(math.atan2(delta_y, delta_x) - pose.heading_rad)
            bearing = (
                round(body_bearing / self.command_set.bearing_step_rad)
                % self.command_set.translate_bearings
            )
            choices = self._translations[bearing]
            if not choices:
                raise ValueError(
                    f"Required translation bearing {bearing} is unavailable in the envelope"
                )
            speed_goal = self.approach_gain_per_s * distance
            _, command = min(choices, key=lambda entry: (abs(entry[0] - speed_goal), entry[0]))
            self._held = command
            self._hold_until_s = t + self.hold_min_s

        # Cheap and worth it: a producer that emits an inadmissible command
        # should fail here, next to the bug, not inside the policy.
        command.validate(self.envelope)
        return command
