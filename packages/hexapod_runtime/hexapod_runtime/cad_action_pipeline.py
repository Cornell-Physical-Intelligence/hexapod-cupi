"""Name-bound CAD action pipeline, stdlib only; this does not drive motors.

Policy vectors use the manifest order. Feedback and returned targets use the
explicitly bound articulation order. Every positional input carries its names;
a changed order or nonfinite value is rejected before slew history changes.
Unlike the legacy mock pipeline, a measured reset pose is mandatory before
the first target, and the soft-limit clamp cannot be omitted.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

from hexapod_core import cad_manifest_v2 as contract
from .action_pipeline import ActionPipeline


def _finite(values: Sequence[float], *, field: str, length: int) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{field} must contain {length} finite values")
    return result


class CadActionPipeline:
    """Validated simulation adapter with explicit policy/articulation orders.

    Construct through :func:`create_cad_action_pipeline`. A shuffled reported
    articulation order is supported by a named permutation, never by guessing
    motor indices. This is not a CAN mapping or a hardware deployment gate.
    """

    def __init__(self, *, manifest: Mapping, urdf_path: str | Path,
                 asset_name: str, articulation_joint_names: Sequence[str]) -> None:
        contract.validate_simulation_manifest(manifest)
        if asset_name != contract.ASSET_NAME:
            raise ValueError(f"Expected asset {contract.ASSET_NAME!r}, got {asset_name!r}")
        contract.verify_urdf_identity(urdf_path)
        self.articulation_joint_names = contract.validate_joint_names(
            articulation_joint_names, field="articulation_joint_names"
        )
        self.policy_joint_names = tuple(manifest["policy_joint_names"])
        # Build the adapter from names even when both tuples currently match.
        self._policy_to_articulation = tuple(
            self.policy_joint_names.index(name) for name in self.articulation_joint_names
        )
        self._articulation_to_policy = tuple(
            self.articulation_joint_names.index(name) for name in self.policy_joint_names
        )
        self.default_joint_positions = tuple(
            contract.DEFAULT_JOINT_POSITIONS_BY_NAME[name] for name in self.policy_joint_names
        )
        self.soft_limits = tuple(contract.SOFT_LIMITS_BY_NAME[name] for name in self.policy_joint_names)
        self._pipeline = ActionPipeline(
            default_joint_positions=self.default_joint_positions,
            action_scale=contract.ACTION_SCALE_RAD,
            stand_action_scale=contract.STAND_ACTION_SCALE,
            command_active_threshold=contract.COMMAND_ACTIVE_THRESHOLD,
            slew_limit_rad_per_20ms=contract.SLEW_LIMIT_RAD_PER_20MS,
            step_dt=contract.POLICY_STEP_DT_S,
            soft_limits=self.soft_limits,
        )
        self._ready = False

    @property
    def last_limited_fraction(self) -> float:
        return self._pipeline.last_limited_fraction

    def reset(self, measured_joint_positions: Sequence[float], *,
              joint_names: Sequence[str]) -> None:
        """Bind slew history to a finite in-limit pose in the reported order.

        Out-of-limit starts need a separate recovery decision: clamping a bad
        feedback pose here could command an unbounded first correction.
        """
        self._ready = False
        if tuple(joint_names) != self.articulation_joint_names:
            raise ValueError("Feedback joint order changed from the bound articulation order")
        measured = _finite(measured_joint_positions, field="measured_joint_positions", length=18)
        ordered = [measured[index] for index in self._articulation_to_policy]
        for name, position, (lower, upper) in zip(self.policy_joint_names, ordered, self.soft_limits):
            if not lower <= position <= upper:
                raise ValueError(f"Reset joint {name} lies outside the CAD v2 soft limits")
        self._pipeline.reset(ordered)
        self._ready = True

    def step(self, action: Sequence[float], command: Sequence[float], *,
             action_joint_names: Sequence[str], command_frame: str) -> list[float]:
        """Return absolute-radian targets in ``articulation_joint_names`` order."""
        if not self._ready:
            raise RuntimeError("A valid measured reset pose is required before CAD targets")
        if tuple(action_joint_names) != self.policy_joint_names:
            raise ValueError("Action joint order does not match the CAD v2 policy manifest")
        if command_frame != contract.COMMAND_FRAME:
            raise ValueError("Commands must use anatomical navigation axes (-Y forward, +X left)")
        checked_action = _finite(action, field="action", length=18)
        checked_command = _finite(command, field="command", length=3)
        target = self._pipeline.step(checked_action, checked_command)
        # The inputs and immutable construction parameters are finite; this is
        # also an explicit output boundary if the underlying path ever changes.
        checked_target = _finite(target, field="target", length=18)
        return [checked_target[index] for index in self._policy_to_articulation]


def create_cad_action_pipeline(*, manifest: Mapping, urdf_path: str | Path,
                               asset_name: str,
                               articulation_joint_names: Sequence[str]) -> CadActionPipeline:
    """No default mock pose, implicit asset, omitted hash check or order guess."""
    return CadActionPipeline(manifest=manifest, urdf_path=urdf_path, asset_name=asset_name,
                             articulation_joint_names=articulation_joint_names)
