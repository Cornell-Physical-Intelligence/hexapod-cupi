"""Frozen v1 velocity-command contract: the planner/policy boundary.

The policy consumes exactly three command scalars -- forward, lateral, yaw rate
-- occupying observation slots 9..12 (see :mod:`hexapod_core.observation`).
That 3-tuple is the whole interface between path planning and locomotion.
Nothing on the policy side knows about waypoints, maps, or goals; nothing on
the planner side knows about joints, gaits, or torques. Command *producers*
live in ``hexapod_nav``; this module only defines the type and the envelopes it
is legal inside.

Frames: ``env.py`` validates ``command_frame`` against ``{"body",
"navigation"}``. Every deployed stage from Phase1 v5 onward -- Stage2C
included -- sets ``command_frame = "navigation"``, where forward is the
robot's anatomical forward (imported body -Y); see :mod:`hexapod_core.frames`.
``Frame.BODY`` exists only to name the legacy Phase 1 convention.

Envelopes are read from ``packages/hexapod_env/hexapod_env/phase2_cfg.py``:

* :data:`STAGE2C_ENVELOPE` -- what the deployed policy was actually trained on
  (``HexapodStage2CStabilizedForwardCommandCfg``): forward only.
* :data:`PHASE2_TARGET_ENVELOPE` -- the wider Phase 2 goal
  (``HexapodPhase2FinalEnvCfg.velocity_command``). No accepted checkpoint
  covers it yet; it is the target the curriculum is walking toward.

The all-zero stand command is admissible in every envelope: the Stage2C mixture
emits it 20% of the time (``standing_probability = 0.20``) even though 0.0 m/s
is outside its 0.16..0.32 forward range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


__all__ = [
    "CommandEnvelope",
    "Frame",
    "PHASE2_TARGET_ENVELOPE",
    "SCHEMA_VERSION",
    "STAGE2C_ENVELOPE",
    "STAGE2C_FORWARD_RANGE_MPS",
    "STAGE2C_STANDING_PROBABILITY",
    "STAGE2C_LATERAL_RANGE_MPS",
    "STAGE2C_YAW_RATE_RANGE_RAD_S",
    "PHASE2_FORWARD_RANGE_MPS",
    "PHASE2_LATERAL_RANGE_MPS",
    "PHASE2_YAW_RATE_RANGE_RAD_S",
    "VelocityCommand",
]

SCHEMA_VERSION = 1


class Frame(Enum):
    """Frame a velocity command is expressed in."""

    #: Anatomical navigation axes -- the deployed convention.
    NAVIGATION = "navigation"
    #: Legacy imported-body axes, Phase 1 only. Retained so a historical
    #: artifact can be labelled honestly; never produce new commands in it.
    BODY = "body"


# phase2_cfg.py :: HexapodStage2CStabilizedForwardCommandCfg -- the command
# distribution the deployed Stage2C policy was trained against. Forward only:
# lateral and yaw are pinned to exactly zero.
STAGE2C_FORWARD_RANGE_MPS = (0.16, 0.32)
STAGE2C_LATERAL_RANGE_MPS = (0.0, 0.0)
STAGE2C_YAW_RATE_RANGE_RAD_S = (0.0, 0.0)
#: phase2_cfg.py :: HexapodStage2CStabilizedForwardCommandCfg.standing_probability
STAGE2C_STANDING_PROBABILITY = 0.20

# phase2_cfg.py :: HexapodPhase2FinalEnvCfg.velocity_command -- the full Phase 2
# omnidirectional envelope. A target, not a demonstrated capability.
PHASE2_FORWARD_RANGE_MPS = (-0.40, 0.60)
PHASE2_LATERAL_RANGE_MPS = (-0.35, 0.35)
PHASE2_YAW_RATE_RANGE_RAD_S = (-0.75, 0.75)


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


STAGE2C_ENVELOPE = CommandEnvelope(
    forward_mps=STAGE2C_FORWARD_RANGE_MPS,
    lateral_mps=STAGE2C_LATERAL_RANGE_MPS,
    yaw_rate_rad_s=STAGE2C_YAW_RATE_RANGE_RAD_S,
    name="stage2c",
)

PHASE2_TARGET_ENVELOPE = CommandEnvelope(
    forward_mps=PHASE2_FORWARD_RANGE_MPS,
    lateral_mps=PHASE2_LATERAL_RANGE_MPS,
    yaw_rate_rad_s=PHASE2_YAW_RATE_RANGE_RAD_S,
    name="phase2_target",
)


@dataclass(frozen=True)
class VelocityCommand:
    """One velocity command: ``[forward, lateral, yaw rate]``.

    Units are m/s, m/s, rad/s. The scalars are written into observation slots
    9, 10 and 11 in exactly this order.
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

    def is_within(self, envelope: CommandEnvelope = PHASE2_TARGET_ENVELOPE) -> bool:
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

    def validate(self, envelope: CommandEnvelope = PHASE2_TARGET_ENVELOPE) -> None:
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
