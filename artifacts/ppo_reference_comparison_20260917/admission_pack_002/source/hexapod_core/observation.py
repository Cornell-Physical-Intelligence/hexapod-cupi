"""Frozen v1 policy-observation layout: 66 dimensions, no privileged state.

The order below was read off ``HexapodEnv._get_observations`` in
``packages/hexapod_env/hexapod_env/env.py``, which builds

    observation_terms = [
        root_lin_vel,            # command-frame root linear velocity   (3)
        root_ang_vel,            # command-frame root angular velocity  (3)
        projected_gravity,       # command-frame projected gravity      (3)
        self._commands,          # [forward, lateral, yaw] command      (3)
        joint_pos - default_joint_pos,   # joint position error        (18)
        joint_vel,               # joint velocity                      (18)
        self._actions,           # current clipped action              (18)
    ]
    observations = torch.cat(observation_terms, dim=-1)

and matches the documented order in ``docs/TRAINING.md`` §2 exactly.

Two facts the dimension table does not show, both verified in ``env.py``:

* The first three blocks are expressed in the *command* frame. Under
  ``command_frame = "navigation"`` (Phase1 v5 and every later stage, including
  the deployed Stage2C) ``_vector_in_command_frame`` applies
  ``body_to_navigation_frame``; see :mod:`hexapod_core.frames`.
* The trailing action block is ``self._actions``, which is the clipped action
  *after* ``apply_command_conditioned_stand_action_scale``. With the Stage2C
  ``stand_action_scale = 0.0``, this block is exactly zero on every step where
  no command axis exceeds the active threshold. It is not the raw pre-clip
  policy output.

The 68-dimensional Stage2G scratch variant (``include_gait_phase_observation``
appends ``[sin, cos]`` of the gait clock) is deliberately *not* schema v1: a
changed layout is a new schema version alongside this one, never an edit here.
"""

from __future__ import annotations

from types import MappingProxyType


__all__ = [
    "ACTION",
    "COMMAND",
    "FIELDS",
    "FIELD_NAMES",
    "FIELD_SLICES",
    "JOINT_POSITION_ERROR",
    "JOINT_VELOCITY",
    "OBSERVATION_DIM",
    "PROJECTED_GRAVITY",
    "ROOT_ANGULAR_VELOCITY",
    "ROOT_LINEAR_VELOCITY",
    "SCHEMA_VERSION",
    "field_layout",
    "slice_of",
    "width_of",
]

SCHEMA_VERSION = 1

# env_cfg.py :: HexapodFlatEnvCfg.observation_space (base default; no deployed
# stage overrides it -- the Stage2G scratch arm widens it to 68 and is a
# separate schema, not this one).
OBSERVATION_DIM = 66

ROOT_LINEAR_VELOCITY = slice(0, 3)
ROOT_ANGULAR_VELOCITY = slice(3, 6)
PROJECTED_GRAVITY = slice(6, 9)
COMMAND = slice(9, 12)
JOINT_POSITION_ERROR = slice(12, 30)
JOINT_VELOCITY = slice(30, 48)
ACTION = slice(48, 66)

# Declaration order is the concatenation order. Nothing may be reordered or
# resized in v1.
FIELDS: tuple[tuple[str, slice], ...] = (
    ("root_linear_velocity", ROOT_LINEAR_VELOCITY),
    ("root_angular_velocity", ROOT_ANGULAR_VELOCITY),
    ("projected_gravity", PROJECTED_GRAVITY),
    ("command", COMMAND),
    ("joint_position_error", JOINT_POSITION_ERROR),
    ("joint_velocity", JOINT_VELOCITY),
    ("action", ACTION),
)

FIELD_NAMES: tuple[str, ...] = tuple(name for name, _ in FIELDS)

FIELD_SLICES = MappingProxyType({name: field for name, field in FIELDS})


def field_layout() -> dict[str, tuple[int, int]]:
    """Return ``name -> (start, stop)`` in concatenation order."""

    return {name: (field.start, field.stop) for name, field in FIELDS}


def slice_of(name: str) -> slice:
    """Return the slice for one named observation field."""

    try:
        return FIELD_SLICES[name]
    except KeyError:
        raise KeyError(f"Unknown observation field {name!r}") from None


def width_of(name: str) -> int:
    """Return the number of scalars occupied by one named field."""

    field = slice_of(name)
    return field.stop - field.start
