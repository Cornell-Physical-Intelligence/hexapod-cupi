"""Explicit experimental canonical-robot allocation; no historical admission claim."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math

LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
JOINT_NAMES = tuple(f"{leg}_{joint}" for leg in LEGS for joint in ("coxa_yaw", "femur_pitch", "tibia_pitch"))
BODY_NAMES = ("body",) + tuple(f"{leg}_{body}" for leg in LEGS for body in ("coxa", "femur", "tibia"))
URDF_SHA256 = "9492fde54c50170e940b49ccb3037d7b413743a5b77bc1d8707a539d44349e78"
MODEL_SHA256 = "7126935b35398d2a7f03271335a8a1b8f2059354af2b19d9474f22b8e73e4881"
USD_SHA256 = "3be98420adc7923923757d6557d8492d5930f84128c1b1366b9c6db1bb03207c"
MASS_KG = 7.466088235225788
KD = (0.44197696391810243, 0.2456657243160222, 0.10552064613953134,
      0.4419789704596685, 0.24566676283633715, 0.10552131615792909,
      0.441977634689341, 0.24566456721764315, 0.10552100524837532,
      0.4419765017453795, 0.245667092769639, 0.10552130774083751,
      0.44197471269634825, 0.24566700905274597, 0.10552141334046235,
      0.4419772543481437, 0.24566767616660323, 0.10552122889741367)


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
                "joint_names": list(JOINT_NAMES), "legs": list(LEGS),
                "standing_admission": False, "stage2_complete": False,
                "prior_standing_failure_superseded": False,
                "authorization": "User explicitly requested experimental paper-based PPO training on the confirmed model.",
                "telemetry_scope": "Compact native body-force telemetry; no exact toe/shaft patch classification or formal standing admission."}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_assets(asset, model_path):
    asset, model_path = Path(asset), Path(model_path)
    if sha(asset / "robot.usda") != USD_SHA256 or sha(model_path) != MODEL_SHA256:
        raise ValueError("Canonical USD/model identity differs")
    model = json.loads(model_path.read_text())
    # The JSON retains its earlier CAD import's source hash; bind the actual
    # corrected URDF shipped with this exact USD asset instead.
    if sha(asset / "source" / "source.urdf") != URDF_SHA256:
        raise ValueError("Canonical URDF identity differs")
    if {x["name"] for x in model["links"]} != set(BODY_NAMES) or {x["name"] for x in model["joints"]} != set(JOINT_NAMES):
        raise ValueError("Canonical 19-body/18-joint topology differs")
    if abs(sum(x["mass"] for x in model["links"]) - MASS_KG) > 1e-9:
        raise ValueError("Canonical mass ledger differs")
    return model


ISAACLAB_REVISION = "ffff603eafc6b74264a5261cc0183d6a65390d78"


def motor_damping(joint_names):
    """Resolve exact damping gains in the native joint order."""
    if len(joint_names) != 18 or set(joint_names) != set(JOINT_NAMES):
        raise ValueError("Motor requires the canonical 18 named joints")
    return [KD[JOINT_NAMES.index(name)] for name in joint_names]


def make_config(cfg, asset, neutral, reset_height, output):
    from isaaclab.assets import ArticulationCfg
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, UsdFileCfg, ArticulationRootPropertiesCfg
    from isaaclab.utils import configclass
    from isaaclab_physx.physics import PhysxCfg
    from .motor import MotorCfg

    @configclass
    class NativeConfig(DirectRLEnvCfg):
        action_space = 18
        observation_space = 231
        state_space = 234
        decimation = 8
        episode_length_s = 20.0
        scene = InteractiveSceneCfg(num_envs=32, env_spacing=2.0, replicate_physics=False,
                                    filter_collisions=False)
        sim = SimulationCfg(dt=.0025, render_interval=8,
                            physics=PhysxCfg(enable_external_forces_every_iteration=True))

    visualizers = []
    if cfg.render:
        from isaaclab_visualizers.kit import KitVisualizerCfg
        visualizers = [KitVisualizerCfg()]
    result = NativeConfig(seed=cfg.seed, episode_length_s=cfg.episode_seconds,
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
        actuators={"motor": MotorCfg()})
    return result, robot
