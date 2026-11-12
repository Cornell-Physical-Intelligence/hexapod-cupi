"""Velocity commands with an explicit caller-supplied envelope."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = [
    "CommandEnvelope",
    "CommandSet",
    "Frame",
    "VelocityCommand",
    "SCHEMA_VERSION",
    "default_requirements_path",
    "load_requirement_set",
]

SCHEMA_VERSION = 1


class Frame(Enum):
    """Frame a velocity command is expressed in."""

    #: Anatomical navigation axes -- the deployed convention.
    NAVIGATION = "navigation"
    #: Legacy imported-body axes, Phase 1 only. Retained so a historical
    #: artifact can be labelled honestly; never produce new commands in it.
    BODY = "body"


@dataclass(frozen=True)
class CommandEnvelope:
    """Inclusive per-axis limits a velocity command must satisfy."""

    forward_mps: tuple[float, float]
    lateral_mps: tuple[float, float]
    yaw_rate_rad_s: tuple[float, float]
    name: str = "custom"
    #: The all-zero stand command is always legal, even when zero lies outside
    #: ``forward_mps``: the training mixtures emit it as an explicit category.
    allow_stand: bool = True

    def __post_init__(self) -> None:
        for axis in ("forward_mps", "lateral_mps", "yaw_rate_rad_s"):
            bounds = getattr(self, axis)
            if len(bounds) != 2:
                raise ValueError(f"{axis} must hold two values, got {bounds!r}")
            low, high = float(bounds[0]), float(bounds[1])
            if not (math.isfinite(low) and math.isfinite(high)):
                raise ValueError(f"{axis} must be finite, got {bounds!r}")
            if low > high:
                raise ValueError(f"{axis} must be ordered low-to-high, got {bounds!r}")


@dataclass(frozen=True)
class VelocityCommand:
    """One velocity command: ``[forward, lateral, yaw rate]``.

    Units are m/s, m/s and rad/s. Consumers own observation placement.
    """

    vx_mps: float
    vy_mps: float
    wz_rad_s: float
    frame: Frame = Frame.NAVIGATION

    def as_tuple(self) -> tuple[float, float, float]:
        """Return the 3-tuple the policy observation consumes."""

        return (float(self.vx_mps), float(self.vy_mps), float(self.wz_rad_s))

    def is_stand(self) -> bool:
        """True when every axis is exactly zero."""

        return self.as_tuple() == (0.0, 0.0, 0.0)

    def is_within(self, envelope: CommandEnvelope) -> bool:
        """True when this command is admissible inside ``envelope``."""

        values = self.as_tuple()
        if not all(math.isfinite(value) for value in values):
            return False
        if envelope.allow_stand and self.is_stand():
            return True
        bounds = (envelope.forward_mps, envelope.lateral_mps, envelope.yaw_rate_rad_s)
        return all(float(low) <= value <= float(high) for value, (low, high) in zip(values, bounds))

    def validate(self, envelope: CommandEnvelope) -> None:
        """Raise ``ValueError`` unless this command is finite and in ``envelope``."""

        if not isinstance(self.frame, Frame):
            raise ValueError(f"frame must be a Frame, got {self.frame!r}")
        axes = (
            ("vx_mps", self.vx_mps, envelope.forward_mps),
            ("vy_mps", self.vy_mps, envelope.lateral_mps),
            ("wz_rad_s", self.wz_rad_s, envelope.yaw_rate_rad_s),
        )
        for name, value, _ in axes:
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if envelope.allow_stand and self.is_stand():
            return
        for name, value, (low, high) in axes:
            if not float(low) <= float(value) <= float(high):
                raise ValueError(
                    f"{name}={value!r} is outside the {envelope.name} envelope [{low}, {high}]"
                )


def default_requirements_path() -> Path:
    """Locate the owner file in a frozen source pack or the repository."""
    source = Path(__file__).resolve().parents[1]
    for root in (source, source.parent):
        path = root / "configs/walking_requirements.json"
        if path.is_file():
            return path
    raise FileNotFoundError("No configs/walking_requirements.json beside the runtime packages")


def load_requirement_set(path=None, name="stage2_flat") -> tuple[dict, str]:
    """Read one requirement set and hash the same bytes, before the control loop."""
    raw = (default_requirements_path() if path is None else Path(path)).read_bytes()
    return json.loads(raw)["sets"][name], hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class CommandSet:
    """The finite navigation-frame commands permitted by one requirement set."""

    translate_speeds_mps: tuple[float, ...]
    translate_bearings: int
    turn_rate_rad_s: float
    arc_speed_mps: float
    arc_turn_rate_rad_s: float
    stand: bool

    @classmethod
    def from_mapping(cls, commands: dict) -> "CommandSet":
        return cls(
            tuple(commands["translate_speeds_mps"]),
            commands["translate_bearings"],
            commands["turn_rate_rad_s"],
            commands["arc"]["speed_mps"],
            commands["arc"]["turn_rate_rad_s"],
            commands["stand"],
        )

    def __post_init__(self) -> None:
        speeds = tuple(float(value) for value in self.translate_speeds_mps)
        if not speeds or len(set(speeds)) != len(speeds):
            raise ValueError("Translation speeds must be nonempty and distinct")
        for value in (*speeds, self.turn_rate_rad_s, self.arc_speed_mps, self.arc_turn_rate_rad_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Command speeds and rates must be finite and positive")
        if type(self.translate_bearings) is not int or self.translate_bearings < 1:
            raise ValueError("Translation bearings must be a positive integer")
        if type(self.stand) is not bool:
            raise ValueError("The stand command declaration must be boolean")
        object.__setattr__(self, "translate_speeds_mps", speeds)

    @property
    def bearing_step_rad(self) -> float:
        return 2 * math.pi / self.translate_bearings

    def translation(self, speed: float, bearing: int) -> VelocityCommand:
        """One declared speed and bearing, with the recorded bank arithmetic."""
        if speed not in self.translate_speeds_mps or not 0 <= bearing < self.translate_bearings:
            raise ValueError("Translation speed or bearing is outside the command set")
        angle = bearing * self.bearing_step_rad
        return VelocityCommand(speed * math.cos(angle), speed * math.sin(angle), 0.0)

    def nonzero(self, maximum_speed_mps=None) -> tuple[VelocityCommand, ...]:
        """Translations, turns, then arcs in their recorded training order."""
        maximum = math.inf if maximum_speed_mps is None else maximum_speed_mps
        result = [
            self.translation(speed, bearing)
            for speed in self.translate_speeds_mps
            if speed <= maximum
            for bearing in range(self.translate_bearings)
        ]
        result.extend(VelocityCommand(0.0, 0.0, sign * self.turn_rate_rad_s) for sign in (-1, 1))
        if self.arc_speed_mps <= maximum:
            result.extend(
                VelocityCommand(self.arc_speed_mps, 0.0, sign * self.arc_turn_rate_rad_s)
                for sign in (-1, 1)
            )
        return tuple(result)

    def envelope(self) -> CommandEnvelope:
        values = [command.as_tuple() for command in self.nonzero()]
        bounds = [
            (min(row[axis] for row in values), max(row[axis] for row in values))
            for axis in range(3)
        ]
        return CommandEnvelope(*bounds, name="bank", allow_stand=self.stand)
