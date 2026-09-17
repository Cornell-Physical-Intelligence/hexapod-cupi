"""Frozen v1 RobStride RS05 actuator contract, applied to all 18 joints.

Every value is a literal from ``packages/hexapod_env/hexapod_env/asset_cfg.py``
(``ROBSTRIDE_RS05_CFG`` / ``HEXAPOD_CFG``) or, for the termination thresholds,
from the deployed Stage2C config in ``phase2_cfg.py``. The two velocity limits
are kept in their source form -- an RPM literal converted with ``2*pi/60`` --
rather than as the rounded 50.27 / 55.29 rad/s the docs quote, so the contract
and the simulator agree bit for bit.

The torque story has three distinct numbers and they are not interchangeable:

* :data:`CONTINUOUS_TORQUE_NM` (1.6) is the *applied* ceiling. ``effort_limit``
  clips what the actuator model actually delivers, so measured applied torque
  saturates here.
* :data:`PEAK_TORQUE_NM` (5.5) is the published short-duration peak. It is both
  ``saturation_effort`` and ``effort_limit_sim``, so raw PD demand inside the
  simulator may reach it.
* :data:`TERMINATION_RAW_DEMAND_NM` (5.5) is the safety gate: raw demand held
  above it for :data:`TERMINATION_RAW_DEMAND_DURATION_S` after
  :data:`TERMINATION_RAW_DEMAND_GRACE_S` of episode grace terminates the
  episode.

``rated_torque_nm = 1.6`` in ``env_cfg.py`` is the same continuous rating seen
from the reward side; it is what the over-rating penalties measure against.
"""

from __future__ import annotations

import math


__all__ = [
    "ARMATURE_KG_M2",
    "CONTINUOUS_TORQUE_NM",
    "DAMPING_NM_S_PER_RAD",
    "DYNAMIC_FRICTION",
    "NOMINAL_VELOCITY_RAD_S",
    "NOMINAL_VELOCITY_RPM",
    "PEAK_TORQUE_NM",
    "RATED_TORQUE_NM",
    "SCHEMA_VERSION",
    "SELF_COLLISIONS_ENABLED",
    "SIM_VELOCITY_RAD_S",
    "SIM_VELOCITY_RPM",
    "SOFT_JOINT_LIMIT_FACTOR",
    "STATIC_FRICTION",
    "STIFFNESS_NM_PER_RAD",
    "TERMINATION_RAW_DEMAND_DURATION_S",
    "TERMINATION_RAW_DEMAND_GRACE_S",
    "TERMINATION_RAW_DEMAND_NM",
    "VISCOUS_FRICTION",
]

SCHEMA_VERSION = 1

# asset_cfg.py :: ROBSTRIDE_RS05_CFG.effort_limit -- published continuous
# output-side rating at 48 V; the applied-torque ceiling.
CONTINUOUS_TORQUE_NM = 1.6
# env_cfg.py :: HexapodFlatEnvCfg.rated_torque_nm -- the same rating as used by
# the over-rating reward terms. Must equal CONTINUOUS_TORQUE_NM.
RATED_TORQUE_NM = 1.6
# asset_cfg.py :: ROBSTRIDE_RS05_CFG.saturation_effort and .effort_limit_sim --
# short-duration peak (about one second at stall).
PEAK_TORQUE_NM = 5.5

# phase2_cfg.py :: HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg
# (Stage2C override; first introduced by the Stage2B lateral config, and there
# is no base default -- the base env_cfg has no torque-demand termination).
TERMINATION_RAW_DEMAND_NM = 5.5
TERMINATION_RAW_DEMAND_DURATION_S = 0.10
TERMINATION_RAW_DEMAND_GRACE_S = 0.50

# asset_cfg.py :: ROBSTRIDE_RS05_CFG.velocity_limit / .velocity_limit_sim.
# 480 rpm -> 50.2655 rad/s (docs round to 50.27); 528 rpm -> 55.2920 rad/s
# (docs round to 55.29).
NOMINAL_VELOCITY_RPM = 480.0
SIM_VELOCITY_RPM = 528.0
NOMINAL_VELOCITY_RAD_S = NOMINAL_VELOCITY_RPM * 2.0 * math.pi / 60.0
SIM_VELOCITY_RAD_S = SIM_VELOCITY_RPM * 2.0 * math.pi / 60.0

# asset_cfg.py :: ROBSTRIDE_RS05_CFG -- implicit PD gains and mechanical model.
STIFFNESS_NM_PER_RAD = 30.0
DAMPING_NM_S_PER_RAD = 0.6
ARMATURE_KG_M2 = 7.0e-4
# ``friction`` (static) and ``dynamic_friction`` are both 0.01 in asset_cfg;
# the docs quote the single value "friction: 0.01".
STATIC_FRICTION = 0.01
DYNAMIC_FRICTION = 0.01
VISCOUS_FRICTION = 0.002

# asset_cfg.py :: HEXAPOD_CFG.soft_joint_pos_limit_factor -- the URDF joint
# limits are shrunk by this factor, and the resulting soft limits are what the
# action pipeline clamps the offset joint target against.
SOFT_JOINT_LIMIT_FACTOR = 0.95

# asset_cfg.py :: HEXAPOD_CFG.spawn.articulation_props.enabled_self_collisions.
# Recorded because it is a known model gap, not a design choice to rely on.
SELF_COLLISIONS_ENABLED = False
