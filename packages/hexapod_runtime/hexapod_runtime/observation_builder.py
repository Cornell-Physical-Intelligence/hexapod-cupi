"""Assemble the 66-vector the policy consumes, from named runtime inputs.

The layout, the widths and the order all come from
:mod:`hexapod_core.observation`; nothing here restates them. This module's only
jobs are to make the caller name every block, to refuse a block of the wrong
length, and to refuse a non-finite value before it reaches the policy.

That last one is deliberate. A silently NaN observation does not crash a
policy: it produces a NaN action, which becomes a NaN joint target, which on
hardware is a dropped or undefined command mid-stride. Failing loudly at the
boundary is the whole point of building the vector here rather than
concatenating lists at the call site.

This is a skeleton, honestly. It does not read an IMU, filter anything, or
estimate root velocity -- those belong to the sensor layer that will call it.
What it does guarantee is that whatever that layer produces lands in the exact
slots the training environment used.
"""

from __future__ import annotations

import math
from typing import Sequence

from hexapod_core import observation as observation_contract


__all__ = [
    "build_observation",
    "split_observation",
    "validate_observation",
]


def _block(values: Sequence[float], *, name: str) -> list[float]:
    width = observation_contract.width_of(name)
    block = [float(value) for value in values]
    if len(block) != width:
        raise ValueError(
            f"observation field {name!r} must hold {width} values, got {len(block)}"
        )
    for index, value in enumerate(block):
        if math.isnan(value):
            raise ValueError(f"observation field {name!r} is NaN at index {index}")
        if math.isinf(value):
            raise ValueError(f"observation field {name!r} is infinite at index {index}")
    return block


def build_observation(
    *,
    root_linear_velocity: Sequence[float],
    root_angular_velocity: Sequence[float],
    projected_gravity: Sequence[float],
    command: Sequence[float],
    joint_position_error: Sequence[float],
    joint_velocity: Sequence[float],
    action: Sequence[float],
) -> list[float]:
    """Return the 66-dimensional policy observation.

    Every argument is keyword-only so a caller cannot silently transpose two
    blocks of equal width -- ``joint_position_error``, ``joint_velocity`` and
    ``action`` are all 18 wide and swapping them is undetectable at runtime.

    Expectations the caller is responsible for, all inherited from the training
    environment and not checkable here:

    * the first three blocks are in the *navigation* frame
      (:mod:`hexapod_core.frames`), not the imported-body frame;
    * ``joint_position_error`` is measured minus default joint position, in
      runtime joint order (:mod:`hexapod_core.joints`);
    * ``action`` is the previous step's action *after* clipping and the
      command-conditioned stand scale, which under the Stage2C
      ``stand_action_scale = 0.0`` is exactly zero while the command is
      inactive.
    """

    blocks = {
        "root_linear_velocity": root_linear_velocity,
        "root_angular_velocity": root_angular_velocity,
        "projected_gravity": projected_gravity,
        "command": command,
        "joint_position_error": joint_position_error,
        "joint_velocity": joint_velocity,
        "action": action,
    }
    vector: list[float] = []
    for name in observation_contract.FIELD_NAMES:
        vector.extend(_block(blocks[name], name=name))
    if len(vector) != observation_contract.OBSERVATION_DIM:
        raise AssertionError(
            "observation layout is inconsistent with hexapod_core: built "
            f"{len(vector)} values, contract says "
            f"{observation_contract.OBSERVATION_DIM}"
        )
    return vector


def validate_observation(vector: Sequence[float]) -> list[float]:
    """Check an assembled observation's width and finiteness; return it as floats."""

    values = [float(value) for value in vector]
    if len(values) != observation_contract.OBSERVATION_DIM:
        raise ValueError(
            f"observation must hold {observation_contract.OBSERVATION_DIM} values, "
            f"got {len(values)}"
        )
    for name, (start, stop) in observation_contract.field_layout().items():
        for offset, value in enumerate(values[start:stop]):
            if math.isnan(value):
                raise ValueError(f"observation field {name!r} is NaN at index {offset}")
            if math.isinf(value):
                raise ValueError(
                    f"observation field {name!r} is infinite at index {offset}"
                )
    return values


def split_observation(vector: Sequence[float]) -> dict[str, list[float]]:
    """Split an assembled observation back into its named blocks."""

    values = validate_observation(vector)
    return {
        name: values[start:stop]
        for name, (start, stop) in observation_contract.field_layout().items()
    }
