"""Pinned Isaac Lab 3.0 configuration for the unqualified direct candidate."""
from pathlib import Path
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg, ArticulationRootPropertiesCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from .direct_motor import DirectMotorCfg

ISAACLAB_REVISION = "ffff603eafc6b74264a5261cc0183d6a65390d78"


@configclass
class DirectConfig(DirectRLEnvCfg):
    action_space = 18
    observation_space = 231
    state_space = 234
    decimation = 8
    episode_length_s = 20.0
    scene = InteractiveSceneCfg(num_envs=32, env_spacing=2.0, replicate_physics=False,
                                filter_collisions=False)
    sim = SimulationCfg(dt=.0025, render_interval=8,
                        physics=PhysxCfg(enable_external_forces_every_iteration=True))


def make_config(cfg, asset, neutral, reset_height, output):
    visualizers = []
    if cfg.render:
        from isaaclab_visualizers.kit import KitVisualizerCfg
        visualizers = [KitVisualizerCfg()]
    result = DirectConfig(seed=cfg.seed, episode_length_s=cfg.episode_seconds,
        scene=InteractiveSceneCfg(num_envs=cfg.num_envs, env_spacing=cfg.spacing_m,
                                  replicate_physics=False, filter_collisions=False),
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
        actuators={"motor": DirectMotorCfg()})
    return result, robot
