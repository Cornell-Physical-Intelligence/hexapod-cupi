"""Direct task configuration for the approved mass-corrected RS05 robot.

The timing, solver counts, material, action path and observation widths repeat
the accepted prototype recipe: 400 Hz physics, eight substeps per 50 Hz control
step, 32 position solver iterations and no velocity iteration, external forces
every iteration, friction 1.0 with no restitution, a 0.35 rad action scale and
a 0.040 rad slew bound per control step.

The soft joint limit factor is 1.0 because the action path clamps to the exact
URDF limits itself. This config registers nothing and starts no simulation.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from ...actuators.rs05_paper_walk import make_rs05_paper_walk_cfg
from ...assets.articulation import articulation_cfg_from_spec
from ...assets.mkii_rs05 import (
    MKII_RS05_ASSET,
    MKII_RS05_CANONICAL_JOINT_NAMES,
    MKII_RS05_JOINT_LIMITS_RAD,
    MKII_RS05_TOE_LOCAL_POINTS_M,
    MKII_RS05_USD_SHA256,
)
from ...ppo_cfg import HexapodPPORunnerCfg
from .math import (
    ACTION_SCALE_RAD,
    CRITIC_OBSERVATION_WIDTH,
    DECIMATION,
    ENV_SPACING_M,
    EPISODE_SECONDS,
    PHYSICS_DT_S,
    POLICY_OBSERVATION_WIDTH,
    SOLVER_POSITION_ITERATIONS,
    SOLVER_VELOCITY_ITERATIONS,
    TARGET_SLEW_RAD,
)

ROBOT_PRIM = "/World/envs/env_.*/Robot"
GROUND_PLANE_COLLISION_PATH = "/World/ground/terrain/GroundPlane/CollisionPlane"
#: The prepared USD keeps all 19 links as direct children of the robot prim.
BODY_NAMES = (MKII_RS05_ASSET.root_link,) + tuple(
    name for group in MKII_RS05_ASSET.leg_link_names for name in group
)

HEXAPOD_MKII_RS05_CFG = articulation_cfg_from_spec(
    MKII_RS05_ASSET,
    actuator_cfg=make_rs05_paper_walk_cfg(MKII_RS05_ASSET.all_joints),
    solver_position_iterations=SOLVER_POSITION_ITERATIONS,
    solver_velocity_iterations=SOLVER_VELOCITY_ITERATIONS,
    # The prepared asset authors self-collision, and the accepted native
    # readback confirms it stays enabled.
    enabled_self_collisions=True,
    # The action path clamps to the exact URDF limits, so the simulator keeps
    # the full declared travel.
    soft_joint_pos_limit_factor=1.0,
)


def _contact_sensor(name):
    foot = name.endswith("_tibia")
    return ContactSensorCfg(
        prim_path=f"{ROBOT_PRIM}/{name}",
        history_length=3,
        update_period=PHYSICS_DT_S,
        track_pose=foot,
        track_contact_points=foot,
        max_contact_data_count_per_prim=16 if foot else None,
        filter_prim_paths_expr=[GROUND_PLANE_COLLISION_PATH] if foot else [],
    )


@configclass
class HexapodMkiiRs05FlatEnvCfg(DirectRLEnvCfg):
    """Flat-ground velocity task on the approved direct-drive robot."""

    episode_length_s = EPISODE_SECONDS
    decimation = DECIMATION
    action_space = 18
    observation_space = POLICY_OBSERVATION_WIDTH
    state_space = CRITIC_OBSERVATION_WIDTH
    action_scale = ACTION_SCALE_RAD
    target_slew_rad_per_control = TARGET_SLEW_RAD

    #: Per-leg joint order of every observation, reward and capture channel.
    canonical_joint_names = MKII_RS05_CANONICAL_JOINT_NAMES
    #: Articulation order the environment requires at construction.
    expected_runtime_joint_names = MKII_RS05_ASSET.runtime_joint_names
    body_names = BODY_NAMES
    #: Exact per-joint URDF travel the action path clamps against.
    joint_limits_rad = MKII_RS05_JOINT_LIMITS_RAD
    toe_local_points_m = MKII_RS05_TOE_LOCAL_POINTS_M
    reset_root_height_m = MKII_RS05_ASSET.reset_root_height_m
    nominal_height_m = MKII_RS05_ASSET.nominal_height_m
    asset_usd_sha256 = MKII_RS05_USD_SHA256

    #: Command ranges, forward and lateral metres per second and yaw radians
    #: per second in navigation axes. The defaults hold the prototype's fixed
    #: pilot command, so a run reproduces it until a curriculum is approved.
    command_forward_range_mps = (0.10, 0.10)
    command_left_range_mps = (0.0, 0.0)
    command_yaw_rate_range_rad_s = (0.0, 0.0)
    command_hold_time_s = 4.0
    stand_command_fraction = 0.0
    standing_only = False

    #: Set by the admission runner. ``None`` records nothing.
    diagnostic_capture_dir: str | None = None
    diagnostic_geometry_path: str | None = None
    diagnostic_geometry_extrema_path: str | None = None

    sim: SimulationCfg = SimulationCfg(
        dt=PHYSICS_DT_S,
        render_interval=DECIMATION,
        gravity=(0.0, 0.0, -9.81),
        physics=PhysxCfg(gpu_max_rigid_patch_count=2**20, enable_external_forces_every_iteration=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=ENV_SPACING_M, replicate_physics=True
    )
    robot = HEXAPOD_MKII_RS05_CFG.replace(prim_path=ROBOT_PRIM)
    body_contact_sensors = {name: _contact_sensor(name) for name in BODY_NAMES}


@configclass
class HexapodMkiiRs05PPORunnerCfg(HexapodPPORunnerCfg):
    """Separate experiment directory. This definition launches nothing."""

    experiment_name = "hexapod_mkii_rs05_flat_direct"


__all__ = [
    "HEXAPOD_MKII_RS05_CFG",
    "HexapodMkiiRs05FlatEnvCfg",
    "HexapodMkiiRs05PPORunnerCfg",
]
