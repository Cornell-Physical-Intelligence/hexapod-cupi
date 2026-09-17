"""Stage2G: biomechanically grounded insect-gait rework.

Stage2C's speed-conditioned five/four/three-foot support floor and heavy
deck-stillness costs force a statically supported shuffle.  Stage2G replaces
that structure with the alternating-tripod organization real insects use at
speed: a per-environment gait clock paced by commanded speed, expected
stance/swing windows per tripod, and a swing-apex clearance target, with the
stillness terms relaxed enough that a natural stepping rhythm is admissible.
Safety terms (RS05 torque costs and the 5.5 N*m raw-demand termination) are
inherited unchanged.

Two arms:

* ``HexapodStage2GInsectGaitEnvCfg`` — from-scratch arm.  Adds the two-value
  gait-phase observation (68-dim policy input) and wider action authority so
  the policy can learn swing-speed leg motion.
* ``HexapodStage2GInsectGaitAdaptEnvCfg`` — adaptation arm.  Keeps the exact
  66-dim Stage2C observation and action interface so the immutable current
  best checkpoint loads model-only; only the reward landscape changes.
"""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from .phase2_cfg import (
    HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg,
    HexapodPhase2RecoveryStage2CStabilizedForwardPPORunnerCfg,
)


@configclass
class HexapodStage2GInsectGaitEnvCfg(
    HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg
):
    """From-scratch alternating-tripod gait task with a phase-clock input."""

    observation_space = 68
    include_gait_phase_observation = True

    # Alternating-tripod shaping.  The phase reward is the dominant gait-shape
    # term; clearance is secondary so the policy prefers rhythm first, then
    # clean swings.
    gait_phase_contact_reward_scale = 1.5
    gait_duty_factor = 0.5
    gait_cycles_per_meter = 8.0
    gait_min_frequency_hz = 1.2
    gait_max_frequency_hz = 3.0
    swing_clearance_reward_scale = 0.75
    swing_clearance_target_m = 0.030
    swing_clearance_tolerance_m = 0.015

    # A tripod keeps exactly three feet planted; the old five/four/three-foot
    # schedule is the structural cause of the conservative shuffle.  Keep a
    # three-foot floor purely as a safety bias.
    speed_conditioned_support_targets = False
    support_contact_target = 3.0
    support_shortfall_reward_scale = -0.25
    # The fixed-target air-time bonus stays off; the phase reward supersedes
    # it with per-tripod timing.
    feet_air_time_reward_scale = 0.0

    # Relax deck stillness enough for a natural stepping rhythm.  Running
    # insects bounce; a perfectly frozen deck and a lively tripod gait are
    # mutually exclusive objectives.
    deck_stability_reward_scale = 1.5
    z_vel_reward_scale = -4.0
    flat_orientation_reward_scale = -15.0
    base_height_reward_scale = -400.0

    # Swing-speed authority: 2.5 Hz swings need faster joint targets than the
    # 0.04 rad / 20 ms playback limiter allows.  4 rad/s remains far below the
    # 55 rad/s actuator model and every torque safety term is unchanged.
    action_scale = 0.25
    processed_joint_target_slew_limit_rad_per_20ms = 0.08


@configclass
class HexapodStage2GInsectGaitAdaptEnvCfg(HexapodStage2GInsectGaitEnvCfg):
    """Adaptation arm: Stage2C-compatible 66-dim interface, new rewards only."""

    observation_space = 66
    include_gait_phase_observation = False
    # Keep the deployed action interface so the current-best checkpoint's
    # behavior transfers; the clock still paces the reward internally.
    action_scale = 0.20
    processed_joint_target_slew_limit_rad_per_20ms = 0.06


@configclass
class HexapodStage2GInsectGaitPPORunnerCfg(
    HexapodPhase2RecoveryStage2CStabilizedForwardPPORunnerCfg
):
    """From-scratch PPO recipe (Phase-1-proven exploration settings)."""

    max_iterations = 1500
    save_interval = 50
    experiment_name = "hexapod_robstride_stage2g_insect_gait_direct"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
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
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HexapodStage2GInsectGaitAdaptPPORunnerCfg(
    HexapodPhase2RecoveryStage2CStabilizedForwardPPORunnerCfg
):
    """Model-only-resume adaptation from the immutable Stage2C best."""

    max_iterations = 400
    save_interval = 25
    experiment_name = "hexapod_robstride_stage2g_insect_gait_adapt_direct"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.10,
        entropy_coef=3.0e-4,
        num_learning_epochs=3,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.5,
    )


__all__ = [
    "HexapodStage2GInsectGaitEnvCfg",
    "HexapodStage2GInsectGaitAdaptEnvCfg",
    "HexapodStage2GInsectGaitPPORunnerCfg",
    "HexapodStage2GInsectGaitAdaptPPORunnerCfg",
]
