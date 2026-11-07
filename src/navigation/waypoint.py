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

Navigation must not import locomotion (ARCHITECTURE.md), so the bank values are
restated here; ``tests/test_producer.py`` asserts they equal
``locomotion.omniwalk.mdp.commands.command_bank(TaskConfig())``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from contracts.command import (
    CommandEnvelope,
    Frame,
    VelocityCommand,
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

# Restated from locomotion.omniwalk.mdp.commands.TaskConfig; the navigation
# tests assert equality with command_bank(TaskConfig()).
BANK_SPEED_LEVELS_MPS: tuple[float, ...] = (0.025, 0.05)
BANK_YAW_RATE_RAD_S = 0.2
BANK_ARC_SPEED_MPS = 0.04
BANK_ARC_YAW_RATE_RAD_S = 0.15
BANK_BEARINGS = 8
BEARING_STEP_RAD = math.pi / 4
# Restated from TaskConfig.hold_min_seconds: the shortest command hold the
# policy saw in training.
TRAINING_HOLD_MIN_S = 2.0
# Retire a goal well inside the 0.12 m stop tolerance of configs/walking_requirements.json:
# the body rolls up to the 0.05 m stopping distance after the stand command, and the
# surrogate closed loop of 7 November 2026 ended 117 to 122 mm from goals retired at 0.12 m.
RETIRE_RADIUS_M = 0.06


def command_bank() -> list[list[float]]:
    """The nonzero navigation-frame bank, in the training module's order and arithmetic."""

    result: list[list[float]] = []
    for speed in BANK_SPEED_LEVELS_MPS:
        for bearing in range(BANK_BEARINGS):
            angle = bearing * math.pi / 4
            result.append([speed * math.cos(angle), speed * math.sin(angle), 0.0])
    result.extend([[0.0, 0.0, -BANK_YAW_RATE_RAD_S], [0.0, 0.0, BANK_YAW_RATE_RAD_S]])
    result.extend(
        [
            [BANK_ARC_SPEED_MPS, 0.0, -BANK_ARC_YAW_RATE_RAD_S],
            [BANK_ARC_SPEED_MPS, 0.0, BANK_ARC_YAW_RATE_RAD_S],
        ]
    )
    return result


def _bank_envelope() -> CommandEnvelope:
    bank = command_bank()
    limits = []
    for axis in range(3):
        values = [entry[axis] for entry in bank]
        limits.append((min(values), max(values)))
    return CommandEnvelope(limits[0], limits[1], limits[2], name="bank")


# Per-axis extremes of the bank. The envelope admits more than the bank (any
# point inside the box); the follower itself only emits bank entries.
BANK_ENVELOPE = _bank_envelope()

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
        envelope: CommandEnvelope = BANK_ENVELOPE,
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
        # The held heading is the nearest multiple of 45 degrees, so the error
        # never exceeds 22.5 degrees; a gate at or past that never trips.
        if float(heading_gate_rad) >= BEARING_STEP_RAD / 2.0:
            raise ValueError(
                f"heading_gate_rad must be below {BEARING_STEP_RAD / 2.0:.4f}, "
                f"got {heading_gate_rad!r}"
            )
        self.approach_gain_per_s = float(approach_gain_per_s)
        self.heading_gate_rad = float(heading_gate_rad)
        self.heading_release_rad = release
        self._turning = False
        self.stop_radius_m = float(stop_radius_m)
        self.hold_min_s = float(hold_min_s)
        self.envelope = envelope
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
    def held_heading(heading_rad: float) -> float:
        """The multiple of 45 degrees nearest ``heading_rad``, wrapped to ``(-pi, pi]``."""

        return wrap_to_pi(round(float(heading_rad) / BEARING_STEP_RAD) * BEARING_STEP_RAD)

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

        heading_error = wrap_to_pi(self.held_heading(pose.heading_rad) - pose.heading_rad)
        # A turn starts past the gate and ends inside the release, so a heading that
        # creeps across the gate does not alternate one-tick turns with walks.
        self._turning = abs(heading_error) >= self.heading_gate_rad or (
            self._turning and abs(heading_error) > self.heading_release_rad
        )
        if self._turning:
            # The gate ends a hold: the correction is a few ticks long and
            # holding it for hold_min_s would overshoot the held heading.
            self._held = None
            yaw_rate = BANK_YAW_RATE_RAD_S if heading_error > 0.0 else -BANK_YAW_RATE_RAD_S
            command = VelocityCommand(0.0, 0.0, yaw_rate, frame=Frame.NAVIGATION)
        elif self._held is not None and t < self._hold_until_s:
            command = self._held
        else:
            goal_x, goal_y = self.waypoints[self._index]
            delta_x = goal_x - pose.x_m
            delta_y = goal_y - pose.y_m
            distance = math.hypot(delta_x, delta_y)
            body_bearing = wrap_to_pi(math.atan2(delta_y, delta_x) - pose.heading_rad)
            bearing = round(body_bearing / BEARING_STEP_RAD) % BANK_BEARINGS
            angle = bearing * math.pi / 4
            speed = _nearest_level(self.approach_gain_per_s * distance)
            command = VelocityCommand(
                vx_mps=speed * math.cos(angle),
                vy_mps=speed * math.sin(angle),
                wz_rad_s=0.0,
                frame=Frame.NAVIGATION,
            )
            self._held = command
            self._hold_until_s = t + self.hold_min_s

        command = self.envelope.clamp(command)
        # Cheap and worth it: a producer that emits an inadmissible command
        # should fail here, next to the bug, not inside the policy.
        command.validate(self.envelope)
        return command


def _nearest_level(speed_mps: float) -> float:
    """The bank speed level nearest ``speed_mps``; a tie takes the lower level."""

    return min(BANK_SPEED_LEVELS_MPS, key=lambda level: (abs(level - speed_mps), level))
