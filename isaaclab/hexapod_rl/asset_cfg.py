"""Physics and actuator configuration for the RobStride RS05 hexapod."""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg


USD_PATH = os.environ.get(
    "HEXAPOD_USD_PATH",
    "/workspace/hexapod/robot/hexapod_mkii_mock_assy/usd/"
    "hexapod_mkii_robstride/hexapod_mkii_robstride.usda",
)

COXA_JOINTS = (
    "revolute_1_1",
    "revolute_1_7",
    "revolute_2_5",
    "revolute_3",
    "revolute_4",
    "revolute_5",
)
FEMUR_JOINTS = (
    "revolute_1",
    "revolute_1_2",
    "revolute_1_3",
    "revolute_1_4",
    "revolute_1_5",
    "revolute_1_6",
)
TIBIA_JOINTS = (
    "revolute_2",
    "revolute_2_1",
    "revolute_2_2",
    "revolute_2_3",
    "revolute_2_4",
    "revolute_2_6",
)


ROBSTRIDE_RS05_CFG = DCMotorCfg(
    joint_names_expr=[".*"],
    # Published output-side limits at 48 V. Peak torque is available only
    # briefly; the environment separately penalizes torque above 1.6 N*m.
    saturation_effort=5.5,
    # The 5.5 N*m value is a short-duration peak (about one second at stall).
    # Until a measured thermal/I-squared-t model is available, train against
    # the published continuous 1.6 N*m rating so the policy cannot exploit it.
    effort_limit=1.6,
    effort_limit_sim=5.5,
    velocity_limit=480.0 * 2.0 * math.pi / 60.0,
    velocity_limit_sim=528.0 * 2.0 * math.pi / 60.0,
    stiffness=30.0,
    damping=0.6,
    armature=7.0e-4,
    friction=0.01,
    dynamic_friction=0.01,
    viscous_friction=0.002,
)


HEXAPOD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=20.0,
            max_angular_velocity=50.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Conservative six-foot reset stance for the 6.3 kg target model.
        # The standing validator, not this comment, is the torque acceptance gate.
        pos=(0.0, 0.0, 0.210),
        joint_pos={
            **{name: 0.0 for name in COXA_JOINTS},
            **{name: 0.40 for name in FEMUR_JOINTS},
            **{name: 2.10 for name in TIBIA_JOINTS},
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={"legs": ROBSTRIDE_RS05_CFG},
)
