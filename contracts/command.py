"""Velocity commands with an explicit caller-supplied envelope."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


__all__ = ["CommandEnvelope", "Frame", "VelocityCommand", "SCHEMA_VERSION"]

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

    def clamp(self, command: "VelocityCommand") -> "VelocityCommand":
        """Return ``command`` with every axis clamped into this envelope."""

        return VelocityCommand(
            vx_mps=_clamp(command.vx_mps, self.forward_mps),
            vy_mps=_clamp(command.vy_mps, self.lateral_mps),
            wz_rad_s=_clamp(command.wz_rad_s, self.yaw_rate_rad_s),
            frame=command.frame,
        )


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
        return all(
            float(low) <= value <= float(high)
            for value, (low, high) in zip(values, bounds)
        )

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
                    f"{name}={value!r} is outside the {envelope.name} envelope "
                    f"[{low}, {high}]"
                )


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    low, high = float(bounds[0]), float(bounds[1])
    return min(max(float(value), low), high)
