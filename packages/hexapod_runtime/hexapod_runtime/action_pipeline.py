"""The deployed action path, in pure Python, shared by sim and real.

This is a line-by-line re-implementation of ``HexapodEnv._pre_physics_step``
(``packages/hexapod_env/hexapod_env/env.py``) and the two helpers it calls in
``packages/hexapod_env/hexapod_env/rewards/actions.py``. Lists of floats in,
lists of floats out, standard library only -- no torch, no numpy, no Isaac --
so the same code runs on the robot's companion computer and inside a test.

Stage order, matching the environment exactly:

1. clip to ``[-1, 1]``
2. command-conditioned stand scale (Stage2C: 0.0 while every command axis is
   inactive, i.e. an inactive command asks for the default pose exactly)
3. ``action_scale * action + default_joint_position``
4. clamp to the articulation's soft joint position limits, when the caller can
   supply them
5. slew-limit the *post-clamp* target at the radians-per-20 ms budget, rescaled
   by the actual step dt

Step 4 is the one place the environment has information a bare runtime does
not: the soft limits come from the URDF limits shrunk by
``soft_joint_pos_limit_factor`` inside the articulation. ``soft_limits=None``
skips the clamp and is honest about it; on hardware, pass measured per-joint
limits. Skipping the clamp changes what step 5 sees, so parity with training
holds only when the clamp is either irrelevant (targets inside the limits) or
supplied.

Divergence from the training functions is a bug, not a tuning choice; it is
enforced by ``isaaclab/tests/test_runtime_parity.py``, which runs these
functions against the real torch implementations on identical inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from hexapod_core import action as action_contract
from hexapod_core import joints as joint_contract


__all__ = [
    "ActionPipeline",
    "SlewResult",
    "apply_stand_action_scale",
    "clamp_to_soft_limits",
    "clip_action",
    "limit_joint_target_slew",
    "scale_and_offset",
]


def _floats(values: Sequence[float], *, name: str, length: int | None = None) -> list[float]:
    result = [float(value) for value in values]
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must hold {length} values, got {len(result)}")
    if any(math.isnan(value) for value in result):
        raise ValueError(f"{name} must not contain NaN")
    return result


def clip_action(action: Sequence[float]) -> list[float]:
    """Clip a normalized action to ``[-1, 1]``.

    Mirrors ``actions.clone().clamp(-1.0, 1.0)``.
    """

    values = _floats(action, name="action", length=action_contract.ACTION_DIM)
    return [
        min(max(value, action_contract.ACTION_CLIP_MIN), action_contract.ACTION_CLIP_MAX)
        for value in values
    ]


def apply_stand_action_scale(
    action: Sequence[float],
    command: Sequence[float],
    *,
    stand_action_scale: float = action_contract.STAND_ACTION_SCALE,
    command_active_threshold: float = action_contract.COMMAND_ACTIVE_THRESHOLD,
) -> list[float]:
    """Scale the action toward zero while every command axis is inactive.

    Mirrors ``apply_command_conditioned_stand_action_scale``. A scale of one is
    an exact pass-through that deliberately does not inspect the command, which
    is why the command may be any 3-sequence in that case.
    """

    if not math.isfinite(float(stand_action_scale)) or not 0.0 <= float(stand_action_scale) <= 1.0:
        raise ValueError(
            f"stand_action_scale must be finite and in [0, 1], got {stand_action_scale!r}"
        )
    values = _floats(action, name="action")
    if not values:
        raise ValueError("action must have a non-empty action dimension")
    if float(stand_action_scale) == 1.0:
        return values
    commands = _floats(command, name="command", length=3)
    if (
        not math.isfinite(float(command_active_threshold))
        or float(command_active_threshold) < 0.0
    ):
        raise ValueError(
            "command_active_threshold must be finite and non-negative, "
            f"got {command_active_threshold!r}"
        )
    command_active = any(abs(value) > float(command_active_threshold) for value in commands)
    if command_active:
        return values
    return [value * float(stand_action_scale) for value in values]


def scale_and_offset(
    action: Sequence[float],
    default_joint_positions: Sequence[float],
    *,
    action_scale: float = action_contract.ACTION_SCALE_RAD,
) -> list[float]:
    """Turn a normalized action into an absolute joint target, in radians.

    Mirrors ``cfg.action_scale * actions + default_joint_pos``.
    """

    if not math.isfinite(float(action_scale)):
        raise ValueError(f"action_scale must be finite, got {action_scale!r}")
    values = _floats(action, name="action", length=action_contract.ACTION_DIM)
    defaults = _floats(
        default_joint_positions,
        name="default_joint_positions",
        length=action_contract.ACTION_DIM,
    )
    return [float(action_scale) * value + default for value, default in zip(values, defaults)]


def clamp_to_soft_limits(
    target: Sequence[float],
    soft_limits: Sequence[Sequence[float]] | None,
) -> list[float]:
    """Clamp a joint target to per-joint ``(lower, upper)`` soft limits.

    ``None`` skips the clamp. Mirrors the ``torch.clamp(targets,
    soft_limits[:, :, 0], soft_limits[:, :, 1])`` step.
    """

    values = _floats(target, name="target", length=action_contract.ACTION_DIM)
    if soft_limits is None:
        return values
    bounds = list(soft_limits)
    if len(bounds) != action_contract.ACTION_DIM:
        raise ValueError(
            f"soft_limits must hold {action_contract.ACTION_DIM} pairs, got {len(bounds)}"
        )
    clamped: list[float] = []
    for index, (value, pair) in enumerate(zip(values, bounds)):
        if len(pair) != 2:
            raise ValueError(f"soft_limits[{index}] must be a (lower, upper) pair")
        lower, upper = float(pair[0]), float(pair[1])
        if not (math.isfinite(lower) and math.isfinite(upper)) or lower > upper:
            raise ValueError(
                f"soft_limits[{index}] must be finite and ordered low-to-high, got {pair!r}"
            )
        clamped.append(min(max(value, lower), upper))
    return clamped


@dataclass(frozen=True)
class SlewResult:
    """A slew-limited joint target and how much of it the limiter touched."""

    target: list[float]
    #: Fraction of joints whose target the limiter actually moved, matching the
    #: ``joint_target_slew_limited_fraction`` telemetry the environment logs.
    limited_fraction: float


def limit_joint_target_slew(
    target: Sequence[float],
    previous_target: Sequence[float],
    has_previous_target: bool,
    *,
    max_delta_rad_per_20ms: float | None,
    step_dt: float,
) -> SlewResult:
    """Limit the per-step change of a processed joint target.

    Mirrors ``limit_processed_joint_target_slew``, including its edge cases:

    * ``max_delta_rad_per_20ms=None`` is an exact pass-through that
      deliberately does not inspect the history at all.
    * Without history the target passes through unchanged and nothing counts as
      limited -- the environment reaches this only before its first reset,
      because ``_reset_idx`` anchors the history to the episode's randomized
      reset pose whenever the limiter is enabled.
    * The budget is quoted per 20 ms and rescaled by the real ``step_dt``:
      ``allowed = max_delta_rad_per_20ms * step_dt / 0.020``.
    * The clamp is per joint and symmetric, so a saturated sign flip is spread
      over as many steps as it needs.
    * A joint counts as limited when it moved by more than 1e-12, so a step
      that lands exactly on the budget is not counted.
    """

    values = _floats(target, name="target")
    if max_delta_rad_per_20ms is None:
        return SlewResult(target=values, limited_fraction=0.0)
    previous = _floats(previous_target, name="previous_target")
    if len(previous) != len(values):
        raise ValueError("target and previous_target must have matching shapes")
    if not isinstance(has_previous_target, bool):
        raise ValueError(
            f"has_previous_target must be a bool, got {type(has_previous_target).__name__}"
        )
    if (
        not math.isfinite(float(max_delta_rad_per_20ms))
        or float(max_delta_rad_per_20ms) <= 0.0
    ):
        raise ValueError(
            "max_delta_rad_per_20ms must be finite and positive when enabled, "
            f"got {max_delta_rad_per_20ms!r}"
        )
    if not math.isfinite(float(step_dt)) or float(step_dt) <= 0.0:
        raise ValueError(f"step_dt must be finite and positive, got {step_dt!r}")

    if not has_previous_target:
        return SlewResult(target=values, limited_fraction=0.0)

    allowed_delta = (
        float(max_delta_rad_per_20ms)
        * float(step_dt)
        / action_contract.SLEW_REFERENCE_STEP_S
    )
    limited: list[float] = []
    limited_count = 0
    for value, prior in zip(values, previous):
        bounded_delta = min(max(value - prior, -allowed_delta), allowed_delta)
        candidate = prior + bounded_delta
        limited.append(candidate)
        if abs(value - candidate) > 1.0e-12:
            limited_count += 1
    return SlewResult(
        target=limited,
        limited_fraction=limited_count / len(limited) if limited else 0.0,
    )


class ActionPipeline:
    """Stateful, single-robot action path with the deployed defaults.

    The only state is the previous processed joint target, which the slew
    limiter needs. Construct one per robot and call :meth:`reset` at every
    episode or control-session boundary; carrying a target across a reset is
    exactly the bug the environment's ``has_previous`` flag exists to prevent.
    """

    def __init__(
        self,
        *,
        default_joint_positions: Sequence[float] = (
            joint_contract.STAGE2C_DEFAULT_JOINT_POSITIONS_RAD
        ),
        action_scale: float = action_contract.ACTION_SCALE_RAD,
        stand_action_scale: float = action_contract.STAND_ACTION_SCALE,
        command_active_threshold: float = action_contract.COMMAND_ACTIVE_THRESHOLD,
        slew_limit_rad_per_20ms: float | None = (
            action_contract.PROCESSED_JOINT_TARGET_SLEW_LIMIT_RAD_PER_20MS
        ),
        step_dt: float = action_contract.POLICY_STEP_DT_S,
        soft_limits: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.default_joint_positions = _floats(
            default_joint_positions,
            name="default_joint_positions",
            length=action_contract.ACTION_DIM,
        )
        self.action_scale = float(action_scale)
        self.stand_action_scale = float(stand_action_scale)
        self.command_active_threshold = float(command_active_threshold)
        self.slew_limit_rad_per_20ms = (
            None if slew_limit_rad_per_20ms is None else float(slew_limit_rad_per_20ms)
        )
        self.step_dt = float(step_dt)
        self.soft_limits = soft_limits
        self._previous_target: list[float] = list(self.default_joint_positions)
        self._has_previous_target = False
        self.last_limited_fraction = 0.0

    def reset(self, measured_joint_positions: Sequence[float] | None = None) -> None:
        """Clear the slew history at an episode or session boundary.

        Passing the robot's measured joint positions anchors the first limited
        step to the real pose, which is what ``_reset_idx`` does with the
        episode's randomized reset pose. Passing nothing leaves the first step
        unlimited.
        """

        self.last_limited_fraction = 0.0
        if measured_joint_positions is None:
            self._previous_target = list(self.default_joint_positions)
            self._has_previous_target = False
            return
        self._previous_target = _floats(
            measured_joint_positions,
            name="measured_joint_positions",
            length=action_contract.ACTION_DIM,
        )
        self._has_previous_target = self.slew_limit_rad_per_20ms is not None

    def step(
        self,
        action: Sequence[float],
        command: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> list[float]:
        """Run one policy step and return the absolute joint target, in radians."""

        clipped = clip_action(action)
        scaled = apply_stand_action_scale(
            clipped,
            command,
            stand_action_scale=self.stand_action_scale,
            command_active_threshold=self.command_active_threshold,
        )
        target = scale_and_offset(
            scaled, self.default_joint_positions, action_scale=self.action_scale
        )
        target = clamp_to_soft_limits(target, self.soft_limits)
        result = limit_joint_target_slew(
            target,
            self._previous_target,
            self._has_previous_target,
            max_delta_rad_per_20ms=self.slew_limit_rad_per_20ms,
            step_dt=self.step_dt,
        )
        self.last_limited_fraction = result.limited_fraction
        if self.slew_limit_rad_per_20ms is not None:
            # The environment updates the history only while the limiter is
            # enabled; a disabled limiter leaves ``has_previous`` False forever.
            self._previous_target = list(result.target)
            self._has_previous_target = True
        return result.target
