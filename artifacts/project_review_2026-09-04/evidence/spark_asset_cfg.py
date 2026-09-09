"""Physics and actuator configuration for the RobStride RS05 hexapod.

The articulation is the CAD assembly package ``robot/hexapod_mkii_assy``
(``hexapod_mkii_serial.urdf`` imported as a floating-base USD, see
``tools/import_urdf_to_usd.py``): 19 links, 18 revolute joints, 8.26 kg with
every RS05 at its published 191 g.
"""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg


USD_PATH = os.environ.get(
    "HEXAPOD_USD_PATH",
    "/workspace/hexapod/robot/hexapod_mkii_assy/usd/"
    "hexapod_mkii_serial/hexapod_mkii_serial.usda",
)

# Body frame: URDF export frame, z up, forward = -y, left = +x.
ROOT_LINK_NAME = "body"
# left/right x front/middle/rear.
LEG_NAMES = ("lf", "lm", "lr", "rf", "rm", "rr")
COXA_JOINTS = tuple(f"{leg}_coxa_yaw" for leg in LEG_NAMES)
FEMUR_JOINTS = tuple(f"{leg}_femur_pitch" for leg in LEG_NAMES)
TIBIA_JOINTS = tuple(f"{leg}_tibia_pitch" for leg in LEG_NAMES)
LEG_LINK_NAMES = tuple(
    (f"{leg}_coxa", f"{leg}_femur", f"{leg}_tibia") for leg in LEG_NAMES
)

# URDF joint limits (robot/hexapod_mkii_assy/joint_limits.json).  Zero is the
# CAD pose: coxa_yaw zero points the leg straight out of the body (positive is
# counter-clockwise about body +z); femur_pitch positive raises the femur,
# which already sits 46 deg above horizontal at zero; tibia_pitch positive
# opens the knee, whose interior angle is 74 deg at zero (closed at -1.29,
# straight at +1.85).  The femur/tibia ranges are derived from the CAD
# kinematics and must be replaced by measured hardware stops.
COXA_LIMITS = (-0.872665, 0.872665)
FEMUR_LIMITS = (-1.745329, 0.55)
TIBIA_LIMITS = (-0.95, 1.75)

# Reset/validation stance (robot/hexapod_mkii_assy/stance.json), derived from
# the URDF kinematics: feet 0.11 m outboard of the yaw axes, knee interior
# angle 42 deg, femur 32 deg above horizontal, bottom plate 0.124 m above the
# ground, static hip and knee torque both 0.9 N*m at 8.26 kg.
STANCE_ROOT_HEIGHT_M = 0.078
STANCE_RESET_ROOT_HEIGHT_M = 0.084
STANCE_FEMUR_RAD = 0.5
STANCE_TIBIA_RAD = -0.8


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
        # Six-foot reset stance for the 8.26 kg CAD assembly, released 6 mm
        # above its settled height.  The standing validator, not this comment,
        # is the torque acceptance gate.
        pos=(0.0, 0.0, STANCE_RESET_ROOT_HEIGHT_M),
        joint_pos={
            **{name: 0.0 for name in COXA_JOINTS},
            **{name: STANCE_FEMUR_RAD for name in FEMUR_JOINTS},
            **{name: STANCE_TIBIA_RAD for name in TIBIA_JOINTS},
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={"legs": ROBSTRIDE_RS05_CFG},
)
