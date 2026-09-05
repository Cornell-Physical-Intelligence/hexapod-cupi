"""Explicit physical-linkage task configuration; no legacy actuator inheritance."""
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from hexapod_core import fourbar_v1 as contract
from ...actuators.rs05_v2 import make_rs05_v2_cfg
from ...ppo_cfg import HexapodPPORunnerCfg

ROOT = Path(__file__).resolve().parents[5]
KINEMATICS = contract.load_kinematics(ROOT / contract.KINEMATICS_PATH)
ROBOT_PRIM = "/World/envs/env_.*/Robot"
GROUND = "/World/ground/terrain/GroundPlane/CollisionPlane"
USD_PATH = os.environ.get("HEXAPOD_MKII_FOURBAR_USD_PATH", str(ROOT / KINEMATICS["usd_path_relative"]))


def _sensor(name, path):
    foot = name.endswith("_tibia")
    return ContactSensorCfg(prim_path=f"{ROBOT_PRIM}/{path}", history_length=3,
        update_period=contract.PHYSICS_DT_S, track_pose=foot, track_contact_points=foot,
        max_contact_data_count_per_prim=16 if foot else None,
        filter_prim_paths_expr=[GROUND] if foot else [])


@configclass
class HexapodMkiiFourbarV1EnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.
    decimation = contract.DECIMATION
    action_space = 18
    observation_space = contract.OBSERVATION_DIM
    state_space = 0
    command_frame = contract.COMMAND_FRAME
    action_scale = contract.ACTION_SCALE_RAD
    stand_action_scale = 1.0
    axis_command_active_threshold = .01
    processed_joint_target_slew_limit_rad_per_20ms = contract.SLEW_RAD_PER_20MS
    reset_joint_jitter_rad = 0.0
    standing_only = False
    command_lin_vel_x_range_mps = (-.15, .15)
    command_lin_vel_y_range_mps = (-.15, .15)
    command_yaw_rate_range_rad_s = (-.30, .30)
    stand_command_fraction = .20
    command_hold_time_s = 4.
    nominal_height_m = KINEMATICS["nominal_height_m"]
    distal_foot_min_y_m = .155
    closure_coordinate_termination_rad = .02
    kinematics_path = str(ROOT / contract.KINEMATICS_PATH)
    sim = SimulationCfg(dt=contract.PHYSICS_DT_S, render_interval=contract.DECIMATION,
        physics=PhysxCfg(gpu_max_rigid_patch_count=2**20),
        physics_material=sim_utils.RigidBodyMaterialCfg(friction_combine_mode="multiply",
            restitution_combine_mode="multiply", static_friction=1., dynamic_friction=1., restitution=0.))
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(friction_combine_mode="multiply",
            restitution_combine_mode="multiply", static_friction=1., dynamic_friction=1., restitution=0.),
        debug_vis=False)
    scene = InteractiveSceneCfg(num_envs=32, env_spacing=2., replicate_physics=True)
    robot = ArticulationCfg(prim_path=ROBOT_PRIM,
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH, activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,
                retain_accelerations=False, linear_damping=0., angular_damping=0.,
                max_linear_velocity=20., max_angular_velocity=50., max_depenetration_velocity=1.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,
                solver_position_iteration_count=32, solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., KINEMATICS["reset_root_height_m"]),
            joint_pos=KINEMATICS["default_joint_positions_rad"], joint_vel={".*": 0.}),
        soft_joint_pos_limit_factor=contract.SOFT_LIMIT_FACTOR,
        actuators={"motors": make_rs05_v2_cfg(list(contract.ACTIVE_JOINT_NAMES),
                                           physics_dt_s=contract.PHYSICS_DT_S, assumed_bus_voltage_v=48.)})
    body_contact_sensors = {name: _sensor(name, path) for name, path in KINEMATICS["body_paths"].items()}


@configclass
class HexapodMkiiFourbarV1PPORunnerCfg(HexapodPPORunnerCfg):
    experiment_name = "hexapod_mkii_fourbar_v1_flat_direct"
    max_iterations = 100
    save_interval = 10
