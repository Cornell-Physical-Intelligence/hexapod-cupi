"""RSL-RL PPO configuration for the flat-ground curriculum."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class HexapodPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1200
    save_interval = 50
    experiment_name = "hexapod_robstride_flat_direct"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        # With Kp=30 and a 1.6 N*m continuous cap, std=0.7 would make initial
        # exploration nearly bang-bang. This stays below saturation for most
        # first-step samples while PPO learns coordinated motion.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.15),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
