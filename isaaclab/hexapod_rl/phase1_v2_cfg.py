"""Safer second-pass configuration for Phase 1 forward walking."""

from isaaclab.utils.configclass import configclass

from .env_cfg import HexapodFlatEnvCfg
from .ppo_cfg import HexapodPPORunnerCfg


@configclass
class HexapodPhase1V2EnvCfg(HexapodFlatEnvCfg):
    """Reduce raw position demand and explicitly protect heading and falls."""

    action_scale = 0.20
    lin_vel_reward_scale = 4.0
    yaw_rate_reward_scale = 1.0
    yaw_rate_tracking_std_rad_s = 0.20
    fall_penalty = -1.0


@configclass
class HexapodPhase1V2PPORunnerCfg(HexapodPPORunnerCfg):
    experiment_name = "hexapod_robstride_phase1_forward_v2_direct"


__all__ = ["HexapodPhase1V2EnvCfg", "HexapodPhase1V2PPORunnerCfg"]
