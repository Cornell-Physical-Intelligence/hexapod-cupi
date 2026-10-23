"""Declare the walking task as one manager-based configuration on the robot, floor and simulation of the Direct task.

Each reward term is one component of the recorded reward with weight 1; a zero weight removes that component.
The policy and critic groups emit the unscaled 231- and 234-value inputs; ``rl.manager_wrapper`` applies the
recorded input recipe.
"""

import json
from dataclasses import MISSING
from pathlib import Path

import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_physx.sim.spawners.materials import RigidBodyMaterialCfg

from locomotion.omniwalk.env_cfg import make_config
from locomotion.omniwalk.manager_based import terms
from locomotion.robots.hexapod import JOINT_NAMES, USD_SHA256, verify_assets

FLOOR_MATERIAL = RigidBodyMaterialCfg(
    static_friction=1.0,
    dynamic_friction=1.0,
    restitution=0.0,
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
)


@configclass
class RobotModelCfg:
    """The approved model files and reference stance that the native terms read."""

    model_path: str = MISSING
    geometry_path: str = MISSING
    reference_metadata: dict = {}
    neutral: list = MISSING
    reset_height_m: float = MISSING
    nominal_height_m: float = MISSING
    output: str | None = None


@configclass
class WalkingSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    floor = AssetBaseCfg(
        prim_path=terms.FLOOR,
        collision_group=-1,
        spawn=terms.RecordedFloorCfg(physics_material=FLOOR_MATERIAL),
    )
    # 54 contacts per body hold at least the Direct task's 1024 per robot.
    floor_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        filter_prim_paths_expr=[terms.FLOOR + "/mesh"],
        max_contact_data_count_per_prim=54,
    )


@configclass
class ActionsCfg:
    joint_targets = terms.SlewLimitedJointTargetsCfg()


@configclass
class CommandsCfg:
    velocity = terms.BankCommandCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        proprio = ObsTerm(func=terms.proprio_frame, history_length=5)
        command = ObsTerm(func=terms.held_command)
        executed_action = ObsTerm(func=terms.executed_action)

    @configclass
    class CriticCfg(PolicyCfg):
        linear_velocity_nav = ObsTerm(func=terms.linear_velocity_nav)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


def component(name):
    return RewTerm(func=terms.reward_component, weight=1.0, params={"name": name})


@configclass
class RewardsCfg:
    linear_tracking = component("linear_tracking")
    yaw_tracking = component("yaw_tracking")
    gait_schedule = component("gait_schedule")
    swing_travel = component("swing_travel")
    vertical_velocity = component("vertical_velocity")
    roll_pitch_rate = component("roll_pitch_rate")
    yaw_rate = component("yaw_rate")
    planar_rate = component("planar_rate")
    tilt = component("tilt")
    joint_torque = component("joint_torque")
    joint_acceleration = component("joint_acceleration")
    target_rate = component("target_rate")
    height = component("height")
    collisions = component("collisions")
    torque_limit = component("torque_limit")
    over_rating = component("over_rating")
    joint_margin = component("joint_margin")
    action_limit = component("action_limit")
    quiet_joint_rate = component("quiet_joint_rate")
    quiet_target_motion = component("quiet_target_motion")
    quiet_contact = component("quiet_contact")
    quiet_strain = component("quiet_strain")
    termination = component("termination")
    schedule_yaw_gate = component("schedule_yaw_gate")
    stride_tracking = component("stride_tracking")
    stride_length = component("stride_length")


@configclass
class TerminationsCfg:
    physical_end = DoneTerm(func=terms.physical_end)
    time_out = DoneTerm(func=terms.control_timeout, params={"episode_seconds": 20.0}, time_out=True)


@configclass
class EventsCfg:
    floor_material = EventTerm(func=terms.bind_floor_material, mode="prestartup")
    reset_to_stance = EventTerm(func=terms.reset_to_stance, mode="reset")
    restart_reward_memory = EventTerm(func=terms.restart_reward_memory, mode="reset")


@configclass
class RecordersCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    substep_loads = RecorderTermCfg(class_type=terms.SubstepLoads400Hz)


@configclass
class WalkingEnvCfg(ManagerBasedRLEnvCfg):
    scene: WalkingSceneCfg = MISSING
    robot_model: RobotModelCfg = MISSING
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    recorders: RecordersCfg = RecordersCfg()
    decimation: int = 8
    episode_length_s: float = 20.0


def make_env_cfg(
    config,
    asset,
    model_path,
    geometry_path,
    output,
    *,
    reference_metadata=None,
    command_seed=None,
    evaluation=False,
):
    """The configuration for one ``EnvConfig`` on the verified asset, with the Direct task's robot and simulation.

    ``evaluation`` keeps the caller's commands and resets no row inside a step, as the Direct task's
    evaluator expects; telemetry still reports terminations and timeouts.
    """
    if config.floor != "isaaclab_mesh":
        raise ValueError("The manager-based prototype builds the recorded mesh floor alone")
    verify_assets(asset, model_path)
    if json.loads(Path(geometry_path).read_text()).get("asset_robot_sha256") != USD_SHA256:
        raise ValueError("Diagnostic geometry belongs to another asset")
    metadata = reference_metadata or {}
    # The Direct task passes the float32 neutral to its articulation; round the same way.
    neutral = torch.tensor(
        metadata.get("nominal_joint_position_rad", [0.0] * 18), dtype=torch.float32
    ).tolist()
    reset_height = float(metadata.get("reset_root_height_m", config.reset_height_m))
    direct, robot = make_config(config, asset, zip(JOINT_NAMES, neutral), reset_height, output)
    cfg = WalkingEnvCfg(
        scene=WalkingSceneCfg(
            num_envs=config.num_envs,
            env_spacing=config.spacing_m,
            replicate_physics=False,
            filter_collisions=True,
            robot=robot,
        ),
        robot_model=RobotModelCfg(
            model_path=str(model_path),
            geometry_path=str(geometry_path),
            reference_metadata=metadata,
            neutral=neutral,
            reset_height_m=reset_height,
            nominal_height_m=float(metadata.get("root_height_m", reset_height)),
            output=str(output),
        ),
        sim=direct.sim,
        seed=config.seed,
        episode_length_s=config.episode_seconds,
    )
    cfg.scene.floor.spawn.contact_offset_m = config.floor_contact_offset_m
    cfg.scene.floor.spawn.rest_offset_m = config.floor_rest_offset_m
    cfg.actions.joint_targets.scale = config.action_scale_rad
    cfg.actions.joint_targets.slew = config.target_slew_rad
    cfg.commands.velocity.seed = config.seed if command_seed is None else command_seed
    cfg.terminations.time_out.params["episode_seconds"] = config.episode_seconds
    if evaluation:
        cfg.commands.velocity.resample = False
        cfg.commands.velocity.initial = (
            config.command_forward_mps,
            config.command_left_mps,
            config.command_yaw_rad_s,
        )
        cfg.terminations.physical_end = cfg.terminations.time_out = None
    return cfg


def make_env(config, asset, model_path, geometry_path, output, **options):
    return ManagerBasedRLEnv(
        make_env_cfg(config, asset, model_path, geometry_path, output, **options)
    )
