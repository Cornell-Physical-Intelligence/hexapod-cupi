"""Phase 1 torque-aware, heading-stable walking fine-tune configuration."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from .phase1_v2_cfg import HexapodPhase1V2EnvCfg, HexapodPhase1V2PPORunnerCfg


@configclass
class HexapodPhase1V3EnvCfg(HexapodPhase1V2EnvCfg):
    """Fine-tune the strongest v2 gait against measured drift and clipping."""

    # Preserve v2 action semantics so model_350 can be resumed safely.
    action_scale = 0.20
    lin_vel_reward_scale = 3.5
    yaw_rate_reward_scale = 2.0
    yaw_rate_tracking_std_rad_s = 0.10
    rated_torque_excess_reward_scale = -0.12
    torque_saturation_reward_scale = -0.5


@configclass
class HexapodPhase1V3PPORunnerCfg(HexapodPhase1V2PPORunnerCfg):
    """Low-entropy PPO continuation for conservative gait refinement."""

    max_iterations = 200
    save_interval = 25
    experiment_name = "hexapod_robstride_phase1_forward_v3_direct"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )


__all__ = ["HexapodPhase1V3EnvCfg", "HexapodPhase1V3PPORunnerCfg"]
