"""Declare the environment settings and build the Isaac Lab Direct and robot configurations."""
from dataclasses import asdict, dataclass
from pathlib import Path
import math

from locomotion.omni.robot import JOINT_NAMES, LEGS


@dataclass(frozen=True)
class EnvConfig:
    num_envs: int = 32
    spacing_m: float = 2.0
    physics_dt: float = 0.0025
    decimation: int = 8
    episode_seconds: float = 20.0
    action_scale_rad: float = 0.35
    target_slew_rad: float = 0.040
    reset_height_m: float = 0.08161108940839767
    command_forward_mps: float = 0.10
    command_left_mps: float = 0.0
    command_yaw_rad_s: float = 0.0
    seed: int = 20260914
    record_motion_features: bool = False
    render: bool = False
    device: str = "cuda:0"

    def __post_init__(self):
        if type(self.num_envs) is not int or not 1 <= self.num_envs <= 1024:
            raise ValueError("Pilot supports 1 through 1024 replicas")
        if self.physics_dt != 0.0025 or self.decimation != 8 or self.target_slew_rad != 0.040:
            raise ValueError("Canonical 400/50Hz servo and target slew are fixed")
        if self.spacing_m < 2 or self.device != "cuda:0":
            raise ValueError("Native CUDA pilot with at least 2m spacing required")
        if not all(math.isfinite(x) for x in (self.spacing_m, self.episode_seconds,
                       self.action_scale_rad, self.reset_height_m, self.command_forward_mps,
                       self.command_left_mps, self.command_yaw_rad_s)):
            raise ValueError("Configuration contains a nonfinite value")
        if not 0 < self.action_scale_rad <= 0.5 or not 0 < self.episode_seconds <= 60:
            raise ValueError("Invalid bounded pilot action scale or episode length")

    @property
    def control_dt(self):
        return self.physics_dt * self.decimation

    def declaration(self):
        return {**asdict(self), "schema": "canonical_paper_walk_experiment_v1",
                "actor_width": 231, "critic_width": 234, "amp_width": 61,
                "joint_names": list(JOINT_NAMES), "legs": list(LEGS)}


def make_config(cfg, asset, neutral, reset_height, output):
    from isaaclab.assets import ArticulationCfg
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, UsdFileCfg, ArticulationRootPropertiesCfg
    from isaaclab.utils import configclass
    from isaaclab_physx.physics import PhysxCfg
    from locomotion.omni.actuator import MotorCfg

    @configclass
    class NativeConfig(DirectRLEnvCfg):
        action_space = 18
        observation_space = 231
        state_space = 234
        decimation = 8
        episode_length_s = 20.0
        scene = InteractiveSceneCfg(num_envs=32, env_spacing=2.0, replicate_physics=False,
                                    filter_collisions=True)
        sim = SimulationCfg(dt=.0025, render_interval=8,
                            physics=PhysxCfg(enable_external_forces_every_iteration=True))

    visualizers = []
    if cfg.render:
        from isaaclab_visualizers.kit import KitVisualizerCfg
        visualizers = [KitVisualizerCfg()]
    result = NativeConfig(seed=cfg.seed, episode_length_s=cfg.episode_seconds,
        scene=InteractiveSceneCfg(num_envs=cfg.num_envs, env_spacing=cfg.spacing_m,
                                  replicate_physics=False, filter_collisions=True),
        sim=SimulationCfg(dt=cfg.physics_dt, device=cfg.device, render_interval=cfg.decimation,
            physics=PhysxCfg(enable_external_forces_every_iteration=True),
            visualizer_cfgs=visualizers, log_dir=str(Path(output) / "isaac_logs")))
    robot = ArticulationCfg(prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(usd_path=str((Path(asset) / "robot.usda").resolve()),
            activate_contact_sensors=True,
            articulation_props=ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                solver_position_iteration_count=32, solver_velocity_iteration_count=0)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., reset_height),
            joint_pos=dict(neutral), joint_vel={".*": 0.}),
        actuators={"motor": MotorCfg()})
    return result, robot
