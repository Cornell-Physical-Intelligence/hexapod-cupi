"""Phase 1 heading-only fine-tune from the strongest v2 walking checkpoint."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from .phase1_v2_cfg import HexapodPhase1V2EnvCfg, HexapodPhase1V2PPORunnerCfg


@configclass
class HexapodPhase1V4EnvCfg(HexapodPhase1V2EnvCfg):
    """Tighten yaw-rate tracking without changing the learned speed/torque tradeoff."""

    action_scale = 0.20
    lin_vel_reward_scale = 4.0
    yaw_rate_reward_scale = 2.0
    yaw_rate_tracking_std_rad_s = 0.10
    rated_torque_excess_reward_scale = -0.03
    torque_saturation_reward_scale = 0.0


@configclass
class HexapodPhase1V4PPORunnerCfg(HexapodPhase1V2PPORunnerCfg):
    """Very low-step, low-entropy heading refinement."""

    max_iterations = 75
    save_interval = 15
    experiment_name = "hexapod_robstride_phase1_forward_v4_direct"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=1.0,
    )


__all__ = ["HexapodPhase1V4EnvCfg", "HexapodPhase1V4PPORunnerCfg"]
