"""Stage2E joystick and command-transition curriculum.

Three short stages bridge the accepted Stage2D C5 pure-y policy to a real
navigation-frame x/y/yaw joystick.  They preserve deterministic anchor
buckets, introduce reverse-x at 0.02--0.05, then 0.04--0.10, then 0.08--0.20
m/s, and shorten fixed command holds from eight to five to three seconds.
The environment lineage directly retains the Stage2C lower stance, deck
stability objective, processed-target limiter, and RS05 termination gates.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from .phase2_cfg import HexapodVelocityCommandCfg
from .phase2d_cfg import (
    HexapodPhase2RecoveryStage2DC5EnvCfg,
    HexapodPhase2RecoveryStage2DC5PPORunnerCfg,
)


@configclass
class HexapodStage2EE0CommandCfg(HexapodVelocityCommandCfg):
    """E0: preserve C5 anchors while acquiring a very small reverse step."""

    sampling_mode: str = "stage2e_joystick_transitions"
    resampling_time_range_s: tuple[float, float] = (8.0, 8.0)
    bucket_counts: tuple[int, int, int, int, int, int, int] = (
        4, 8, 4, 4, 4, 12, 4
    )
    bucket_stride: int = 13

    standing_probability: float = 0.10
    longitudinal_only_probability: float = 0.20
    lateral_only_probability: float = 0.20
    yaw_only_probability: float = 0.10
    combined_probability: float = 0.40

    lateral_anchor_abs_range: tuple[float, float] = (0.08, 0.10)
    forward_anchor_range: tuple[float, float] = (0.20, 0.30)
    yaw_anchor_forward_range: tuple[float, float] = (0.14, 0.20)
    yaw_anchor_abs_range: tuple[float, float] = (0.20, 0.28)
    joystick_forward_range: tuple[float, float] = (0.08, 0.22)
    joystick_reverse_abs_range: tuple[float, float] = (0.02, 0.05)
    joystick_lateral_abs_range: tuple[float, float] = (0.04, 0.10)
    joystick_yaw_abs_range: tuple[float, float] = (0.10, 0.24)

    lin_vel_x_range: tuple[float, float] = (-0.05, 0.30)
    lin_vel_y_range: tuple[float, float] = (-0.10, 0.10)
    ang_vel_z_range: tuple[float, float] = (-0.28, 0.28)


@configclass
class HexapodStage2EE1CommandCfg(HexapodStage2EE0CommandCfg):
    """E1: double reverse exposure/range and shorten command holds."""

    resampling_time_range_s: tuple[float, float] = (5.0, 5.0)
    bucket_counts: tuple[int, int, int, int, int, int, int] = (
        4, 8, 4, 4, 4, 8, 8
    )

    standing_probability: float = 0.10
    longitudinal_only_probability: float = 0.20
    lateral_only_probability: float = 0.20
    yaw_only_probability: float = 0.10
    combined_probability: float = 0.40

    joystick_forward_range: tuple[float, float] = (0.05, 0.26)
    joystick_reverse_abs_range: tuple[float, float] = (0.04, 0.10)
    yaw_anchor_forward_range: tuple[float, float] = (0.05, 0.10)
    joystick_lateral_abs_range: tuple[float, float] = (0.04, 0.10)
    joystick_yaw_abs_range: tuple[float, float] = (0.08, 0.26)
    lin_vel_x_range: tuple[float, float] = (-0.10, 0.30)


@configclass
class HexapodStage2EE2CommandCfg(HexapodStage2EE1CommandCfg):
    """E2: terminal joystick envelope with frequent direction transitions."""

    resampling_time_range_s: tuple[float, float] = (3.0, 3.0)
    bucket_counts: tuple[int, int, int, int, int, int, int] = (
        4, 8, 4, 4, 4, 8, 8
    )

    joystick_forward_range: tuple[float, float] = (0.03, 0.30)
    joystick_reverse_abs_range: tuple[float, float] = (0.08, 0.20)
    yaw_anchor_forward_range: tuple[float, float] = (0.0, 0.0)
    joystick_lateral_abs_range: tuple[float, float] = (0.03, 0.10)
    joystick_yaw_abs_range: tuple[float, float] = (0.06, 0.28)
    lin_vel_x_range: tuple[float, float] = (-0.20, 0.30)


@configclass
class HexapodPhase2RecoveryStage2EE0EnvCfg(
    HexapodPhase2RecoveryStage2DC5EnvCfg
):
    """Shared reverse-safe joystick shaping on the accepted C5 body contract."""

    # C5's threshold is 0.05 m/s because its active y commands start at 0.08.
    # E0 deliberately introduces 0.02 m/s reverse commands, so lower only the
    # axis activity threshold and give x the same sign-aware acquisition form.
    axis_command_active_threshold = 0.015
    longitudinal_signed_progress_reward_scale = 6.0
    longitudinal_normalized_error_penalty_scale = 2.0
    longitudinal_normalized_error_cap = 2.0
    yaw_signed_progress_reward_scale = 4.0

    velocity_command: HexapodStage2EE0CommandCfg = HexapodStage2EE0CommandCfg()
    command_lin_vel_x_range_mps = velocity_command.lin_vel_x_range
    command_lin_vel_y_range_mps = velocity_command.lin_vel_y_range
    command_yaw_rate_range_rad_s = velocity_command.ang_vel_z_range
    stand_command_fraction = velocity_command.standing_probability


@configclass
class HexapodPhase2RecoveryStage2EE1EnvCfg(
    HexapodPhase2RecoveryStage2EE0EnvCfg
):
    """E1 environment: medium reverse range and five-second transitions."""

    velocity_command: HexapodStage2EE1CommandCfg = HexapodStage2EE1CommandCfg()
    command_lin_vel_x_range_mps = velocity_command.lin_vel_x_range
    command_lin_vel_y_range_mps = velocity_command.lin_vel_y_range
    command_yaw_rate_range_rad_s = velocity_command.ang_vel_z_range


@configclass
class HexapodPhase2RecoveryStage2EE2EnvCfg(
    HexapodPhase2RecoveryStage2EE1EnvCfg
):
    """E2 environment: final x/y/yaw envelope and three-second transitions."""

    velocity_command: HexapodStage2EE2CommandCfg = HexapodStage2EE2CommandCfg()
    command_lin_vel_x_range_mps = velocity_command.lin_vel_x_range
    command_lin_vel_y_range_mps = velocity_command.lin_vel_y_range
    command_yaw_rate_range_rad_s = velocity_command.ang_vel_z_range


@configclass
class HexapodPhase2RecoveryStage2EE0PPORunnerCfg(
    HexapodPhase2RecoveryStage2DC5PPORunnerCfg
):
    """Conservative checkpoint-dense PPO shared by the joystick bridge."""

    max_iterations = 30
    save_interval = 5
    experiment_name = (
        "hexapod_robstride_phase2_recovery_stage2e_e0_joystick_bridge_direct"
    )
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.12),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.08,
        entropy_coef=1.0e-4,
        num_learning_epochs=3,
        num_mini_batches=8,
        learning_rate=4.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.5,
    )


@configclass
class HexapodPhase2RecoveryStage2EE1PPORunnerCfg(
    HexapodPhase2RecoveryStage2EE0PPORunnerCfg
):
    max_iterations = 35
    experiment_name = (
        "hexapod_robstride_phase2_recovery_stage2e_e1_reverse_bridge_direct"
    )


@configclass
class HexapodPhase2RecoveryStage2EE2PPORunnerCfg(
    HexapodPhase2RecoveryStage2EE1PPORunnerCfg
):
    max_iterations = 45
    experiment_name = "hexapod_robstride_phase2_recovery_stage2e_e2_joystick_direct"


__all__ = [
    "HexapodPhase2RecoveryStage2EE0EnvCfg",
    "HexapodPhase2RecoveryStage2EE0PPORunnerCfg",
    "HexapodPhase2RecoveryStage2EE1EnvCfg",
    "HexapodPhase2RecoveryStage2EE1PPORunnerCfg",
    "HexapodPhase2RecoveryStage2EE2EnvCfg",
    "HexapodPhase2RecoveryStage2EE2PPORunnerCfg",
    "HexapodStage2EE0CommandCfg",
    "HexapodStage2EE1CommandCfg",
    "HexapodStage2EE2CommandCfg",
]
