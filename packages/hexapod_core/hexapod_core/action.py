"""Frozen v1 action interface and policy/physics timing.

The deployed action path, in order, is ``HexapodEnv._pre_physics_step`` in
``packages/hexapod_env/hexapod_env/env.py``:

1. ``actions.clone().clamp(-1.0, 1.0)``
2. ``apply_command_conditioned_stand_action_scale(...)`` -- rows whose every
   command axis is inactive are multiplied by ``stand_action_scale``
3. ``cfg.action_scale * actions + default_joint_pos``
4. ``torch.clamp(targets, soft_limits[..., 0], soft_limits[..., 1])`` -- the
   articulation's soft joint position limits, i.e. the URDF limits shrunk by
   ``soft_joint_pos_limit_factor``; see :mod:`hexapod_core.actuator`
5. ``limit_processed_joint_target_slew(...)`` -- per-joint clamp of the change
   in the *post-soft-limit* target, budgeted in radians per 20 ms and scaled by
   the actual ``step_dt``
6. ``_apply_action`` writes the result as the joint position target

Every value below is a literal read out of the training sources. Each carries
the file and class it came from and whether it is the base default from
``env_cfg.HexapodFlatEnvCfg`` or the value the deployed Stage2C task actually
runs at (task ID
``Isaac-Velocity-Omni-Recovery-Stage2C-Stabilized-Forward-Hexapod-RobStride-Direct-v0``,
config ``HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg``).

Note on ``ACTION_SCALE_RAD``: the Stage2C class does **not** declare
``action_scale``. It inherits 0.20 through
Stage2C -> Stage2 -> Stage1 -> Phase1V5 -> Phase1V4, where
``HexapodPhase1V4EnvCfg.action_scale = 0.20`` is the nearest declaration
(``HexapodPhase1V2EnvCfg`` declares the same 0.20 one level further up). The
0.30 in ``env_cfg`` is the untouched base default and is *not* the deployed
value. ``test_core_contracts.py`` resolves that inheritance chain rather than
trusting either the comment in ``phase2_cfg.py`` or the docs.
"""

from __future__ import annotations

from types import MappingProxyType


__all__ = [
    "ACTION_CLIP_MAX",
    "ACTION_CLIP_MIN",
    "ACTION_DIM",
    "ACTION_SCALE_RAD",
    "BASE_ACTION_SCALE_RAD",
    "BASE_PROCESSED_JOINT_TARGET_SLEW_LIMIT_RAD_PER_20MS",
    "BASE_STAND_ACTION_SCALE",
    "COMMAND_ACTIVE_THRESHOLD",
    "DECIMATION",
    "EPISODE_LENGTH_S",
    "PHYSICS_DT_S",
    "PHYSICS_RATE_HZ",
    "POLICY_RATE_HZ",
    "POLICY_STEP_DT_S",
    "PROCESSED_JOINT_TARGET_SLEW_LIMIT_RAD_PER_20MS",
    "PROVENANCE",
    "SCHEMA_VERSION",
    "SLEW_REFERENCE_STEP_S",
    "STAND_ACTION_SCALE",
]

SCHEMA_VERSION = 1

# env_cfg.py :: HexapodFlatEnvCfg.action_space -- base default, never overridden.
ACTION_DIM = 18

# env.py :: _pre_physics_step -> actions.clone().clamp(-1.0, 1.0).
ACTION_CLIP_MIN = -1.0
ACTION_CLIP_MAX = 1.0

# Radians of joint offset per unit of normalized action.
# Deployed (Stage2C, inherited): phase1_v4_cfg.py :: HexapodPhase1V4EnvCfg.
ACTION_SCALE_RAD = 0.20
# Base default: env_cfg.py :: HexapodFlatEnvCfg.action_scale -- overridden.
BASE_ACTION_SCALE_RAD = 0.30

# Post-soft-limit joint-target slew budget.
# Stage2C override: phase2_cfg.py ::
# HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg
# .processed_joint_target_slew_limit_rad_per_20ms = 0.04.
PROCESSED_JOINT_TARGET_SLEW_LIMIT_RAD_PER_20MS = 0.040
# Base default: env_cfg.py :: HexapodFlatEnvCfg -- ``None`` means the limiter is
# a pass-through and the historical processed-action semantics are unchanged.
BASE_PROCESSED_JOINT_TARGET_SLEW_LIMIT_RAD_PER_20MS: float | None = None
# rewards/actions.py :: limit_processed_joint_target_slew --
# ``allowed_delta = max_delta_rad_per_20ms * step_dt / 0.020``. The budget is
# quoted per 20 ms and rescaled to the real policy step, so changing the policy
# rate does not change the physical rate limit.
SLEW_REFERENCE_STEP_S = 0.020

# Multiplier applied to the clipped action while every command axis is inactive.
# Stage2C override: phase2_cfg.py ::
# HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg.stand_action_scale = 0.0,
# i.e. a fully inactive command requests the default joint pose exactly.
STAND_ACTION_SCALE = 0.0
# Base default: env_cfg.py :: HexapodFlatEnvCfg.stand_action_scale = 1.0, an
# exact legacy pass-through -- overridden by Stage2C.
BASE_STAND_ACTION_SCALE = 1.0

# A command axis counts as active at strictly greater than this magnitude.
# env_cfg.py :: HexapodFlatEnvCfg.axis_command_active_threshold (base default)
# and phase2_cfg.py :: HexapodPhase2RecoveryStage2EnvCfg restates the same
# 0.01, so the deployed value equals the base default.
COMMAND_ACTIVE_THRESHOLD = 0.01

# env_cfg.py :: HexapodFlatEnvCfg -- base defaults, unchanged by every deployed
# stage. ``sim.dt = 1.0 / 200.0`` with ``decimation = 4`` gives the 50 Hz policy
# step that the 20 ms slew budget is quoted against.
DECIMATION = 4
PHYSICS_RATE_HZ = 200.0
PHYSICS_DT_S = 1.0 / 200.0
POLICY_STEP_DT_S = PHYSICS_DT_S * DECIMATION
POLICY_RATE_HZ = 1.0 / POLICY_STEP_DT_S
EPISODE_LENGTH_S = 20.0

#: Where each value above was read from, and whether the deployed Stage2C task
#: overrides the base default. Consumed by the contract test and by anyone
#: auditing the runtime against training.
PROVENANCE = MappingProxyType(
    {
        "ACTION_DIM": "env_cfg.py HexapodFlatEnvCfg.action_space (base default)",
        "ACTION_CLIP_MIN": "env.py _pre_physics_step clamp (base default)",
        "ACTION_CLIP_MAX": "env.py _pre_physics_step clamp (base default)",
        "ACTION_SCALE_RAD": (
            "phase1_v4_cfg.py HexapodPhase1V4EnvCfg.action_scale "
            "(Stage2C override of the env_cfg base default, inherited)"
        ),
        "BASE_ACTION_SCALE_RAD": "env_cfg.py HexapodFlatEnvCfg.action_scale (base default)",
        "PROCESSED_JOINT_TARGET_SLEW_LIMIT_RAD_PER_20MS": (
            "phase2_cfg.py HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg"
            ".processed_joint_target_slew_limit_rad_per_20ms (Stage2C override)"
        ),
        "BASE_PROCESSED_JOINT_TARGET_SLEW_LIMIT_RAD_PER_20MS": (
            "env_cfg.py HexapodFlatEnvCfg"
            ".processed_joint_target_slew_limit_rad_per_20ms (base default, None)"
        ),
        "SLEW_REFERENCE_STEP_S": (
            "rewards/actions.py limit_processed_joint_target_slew divisor (base default)"
        ),
        "STAND_ACTION_SCALE": (
            "phase2_cfg.py HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg"
            ".stand_action_scale (Stage2C override)"
        ),
        "BASE_STAND_ACTION_SCALE": (
            "env_cfg.py HexapodFlatEnvCfg.stand_action_scale (base default)"
        ),
        "COMMAND_ACTIVE_THRESHOLD": (
            "env_cfg.py HexapodFlatEnvCfg.axis_command_active_threshold (base default; "
            "phase2_cfg.py HexapodPhase2RecoveryStage2EnvCfg restates the same value)"
        ),
        "DECIMATION": "env_cfg.py HexapodFlatEnvCfg.decimation (base default)",
        "PHYSICS_DT_S": "env_cfg.py HexapodFlatEnvCfg.sim SimulationCfg(dt=...) (base default)",
        "EPISODE_LENGTH_S": "env_cfg.py HexapodFlatEnvCfg.episode_length_s (base default)",
    }
)
