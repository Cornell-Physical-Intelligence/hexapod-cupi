"""Stage 2D oblique-to-pure-lateral acquisition homotopy.

The six short stages retain the accepted Stage 2C low, stable stance and
hardware limits while gradually rotating the dominant command from forward
motion toward pure navigation-frame lateral motion.  Each stage keeps one
fixed-bucket command for the full episode; only the oblique heading and active
threshold change between stages.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from .phase2_cfg import (
    HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg,
    HexapodPhase2RecoveryStage2CStabilizedForwardPPORunnerCfg,
    HexapodVelocityCommandCfg,
)


@configclass
class HexapodStage2DC0CommandCfg(HexapodVelocityCommandCfg):
    """C0: small lateral component on an accepted-speed forward gait."""

    sampling_mode: str = "stage2d_oblique_homotopy"
    resampling_time_range_s: tuple[float, float] = (30.0, 30.0)

    # These probability aliases document the sampler's exact fixed allocation.
    # The Stage2D sampler itself assigns deterministic global-index buckets.
    standing_probability: float = 0.0
    longitudinal_only_probability: float = 0.20
    lateral_only_probability: float = 0.0
    yaw_only_probability: float = 0.10
    combined_probability: float = 0.70

    oblique_forward_range: tuple[float, float] = (0.22, 0.26)
    oblique_lateral_abs_range: tuple[float, float] = (0.02, 0.04)
    forward_anchor_range: tuple[float, float] = (0.20, 0.30)
    yaw_anchor_forward_range: tuple[float, float] = (0.20, 0.26)
    yaw_anchor_abs_range: tuple[float, float] = (0.20, 0.28)

    lin_vel_x_range: tuple[float, float] = (0.20, 0.30)
    lin_vel_y_range: tuple[float, float] = (-0.04, 0.04)
    ang_vel_z_range: tuple[float, float] = (-0.28, 0.28)


@configclass
class HexapodStage2DC1CommandCfg(HexapodStage2DC0CommandCfg):
    """C1: increase lateral authority while retaining strong forward motion."""

    oblique_forward_range: tuple[float, float] = (0.20, 0.24)
    oblique_lateral_abs_range: tuple[float, float] = (0.04, 0.06)
    lin_vel_x_range: tuple[float, float] = (0.20, 0.30)
    lin_vel_y_range: tuple[float, float] = (-0.06, 0.06)


@configclass
class HexapodStage2DC2CommandCfg(HexapodStage2DC1CommandCfg):
    """C2: make lateral motion substantial while forward motion still leads."""

    oblique_forward_range: tuple[float, float] = (0.14, 0.18)
    oblique_lateral_abs_range: tuple[float, float] = (0.06, 0.08)
    lin_vel_x_range: tuple[float, float] = (0.14, 0.30)
    lin_vel_y_range: tuple[float, float] = (-0.08, 0.08)


@configclass
class HexapodStage2DC3CommandCfg(HexapodStage2DC2CommandCfg):
    """C3: cross into a lateral-dominant oblique heading."""

    oblique_forward_range: tuple[float, float] = (0.08, 0.12)
    oblique_lateral_abs_range: tuple[float, float] = (0.08, 0.10)
    lin_vel_x_range: tuple[float, float] = (0.08, 0.30)
    lin_vel_y_range: tuple[float, float] = (-0.10, 0.10)


@configclass
class HexapodStage2DC4CommandCfg(HexapodStage2DC3CommandCfg):
    """C4: retain only a small forward component before pure lateral motion."""

    oblique_forward_range: tuple[float, float] = (0.03, 0.06)
    oblique_lateral_abs_range: tuple[float, float] = (0.08, 0.10)
    lin_vel_x_range: tuple[float, float] = (0.03, 0.30)


@configclass
class HexapodStage2DC5CommandCfg(HexapodStage2DC4CommandCfg):
    """C5: pure signed navigation-frame lateral acquisition."""

    oblique_forward_range: tuple[float, float] = (0.0, 0.0)
    oblique_lateral_abs_range: tuple[float, float] = (0.08, 0.10)
    lin_vel_x_range: tuple[float, float] = (0.0, 0.30)


@configclass
class HexapodPhase2RecoveryStage2DC0EnvCfg(
    HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg
):
    """Shared Stage2D acquisition objective at the C0 command heading."""

    gate_axis_rewards_by_command = True
    axis_command_active_threshold = 0.010
    lin_vel_x_reward_scale = 4.0
    # A Gaussian centered on tiny signed y commands rewards zero response too
    # strongly.  The EMA signed-progress and normalized error terms below
    # provide the acquisition gradient instead.
    lin_vel_y_reward_scale = 0.0
    lateral_signed_progress_reward_scale = 8.0
    navigation_lateral_velocity_ema_tau_s = 0.22
    lateral_normalized_error_penalty_scale = 2.0
    lateral_normalized_error_cap = 2.0
    inactive_lateral_velocity_reward_scale = -0.50
    processed_joint_target_slew_limit_rad_per_20ms = 0.04

    velocity_command: HexapodStage2DC0CommandCfg = HexapodStage2DC0CommandCfg()
    command_lin_vel_x_range_mps = velocity_command.lin_vel_x_range
    command_lin_vel_y_range_mps = velocity_command.lin_vel_y_range
    command_yaw_rate_range_rad_s = velocity_command.ang_vel_z_range
    stand_command_fraction = 0.0


@configclass
class HexapodPhase2RecoveryStage2DC1EnvCfg(
    HexapodPhase2RecoveryStage2DC0EnvCfg
):
    """C1 environment: stronger signed lateral component."""

    axis_command_active_threshold = 0.018
    velocity_command: HexapodStage2DC1CommandCfg = HexapodStage2DC1CommandCfg()
    command_lin_vel_x_range_mps = velocity_command.lin_vel_x_range
    command_lin_vel_y_range_mps = velocity_command.lin_vel_y_range
    command_yaw_rate_range_rad_s = velocity_command.ang_vel_z_range


@configclass
class HexapodPhase2RecoveryStage2DC2EnvCfg(
    HexapodPhase2RecoveryStage2DC1EnvCfg
):
    """C2 environment: balanced acquisition bridge."""

    axis_command_active_threshold = 0.030
    velocity_command: HexapodStage2DC2CommandCfg = HexapodStage2DC2CommandCfg()
    command_lin_vel_x_range_mps = velocity_command.lin_vel_x_range
    command_lin_vel_y_range_mps = velocity_command.lin_vel_y_range
    command_yaw_rate_range_rad_s = velocity_command.ang_vel_z_range


@configclass
class HexapodPhase2RecoveryStage2DC3EnvCfg(
    HexapodPhase2RecoveryStage2DC2EnvCfg
):
    """C3 environment: lateral-dominant oblique acquisition."""

    axis_command_active_threshold = 0.040
    velocity_command: HexapodStage2DC3CommandCfg = HexapodStage2DC3CommandCfg()
    command_lin_vel_x_range_mps = velocity_command.lin_vel_x_range
    command_lin_vel_y_range_mps = velocity_command.lin_vel_y_range
    command_yaw_rate_range_rad_s = velocity_command.ang_vel_z_range


@configclass
class HexapodPhase2RecoveryStage2DC4EnvCfg(
    HexapodPhase2RecoveryStage2DC3EnvCfg
):
    """C4 environment: near-lateral acquisition."""

    axis_command_active_threshold = 0.045
    velocity_command: HexapodStage2DC4CommandCfg = HexapodStage2DC4CommandCfg()
    command_lin_vel_x_range_mps = velocity_command.lin_vel_x_range
    command_lin_vel_y_range_mps = velocity_command.lin_vel_y_range
    command_yaw_rate_range_rad_s = velocity_command.ang_vel_z_range


@configclass
class HexapodPhase2RecoveryStage2DC5EnvCfg(
    HexapodPhase2RecoveryStage2DC4EnvCfg
):
    """C5 environment: true pure-y acquisition endpoint."""

    axis_command_active_threshold = 0.050
    velocity_command: HexapodStage2DC5CommandCfg = HexapodStage2DC5CommandCfg()
    command_lin_vel_x_range_mps = velocity_command.lin_vel_x_range
    command_lin_vel_y_range_mps = velocity_command.lin_vel_y_range
    command_yaw_rate_range_rad_s = velocity_command.ang_vel_z_range


@configclass
class HexapodPhase2RecoveryStage2DC0PPORunnerCfg(
    HexapodPhase2RecoveryStage2CStabilizedForwardPPORunnerCfg
):
    """Conservative checkpoint-dense PPO shared by the Stage2D homotopy."""

    max_iterations = 25
    save_interval = 5
    experiment_name = "hexapod_robstride_phase2_recovery_stage2d_c0_direct"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.14),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.10,
        entropy_coef=3.0e-4,
        num_learning_epochs=3,
        num_mini_batches=8,
        learning_rate=6.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=0.5,
    )


@configclass
class HexapodPhase2RecoveryStage2DC1PPORunnerCfg(
    HexapodPhase2RecoveryStage2DC0PPORunnerCfg
):
    experiment_name = "hexapod_robstride_phase2_recovery_stage2d_c1_direct"


@configclass
class HexapodPhase2RecoveryStage2DC2PPORunnerCfg(
    HexapodPhase2RecoveryStage2DC1PPORunnerCfg
):
    experiment_name = "hexapod_robstride_phase2_recovery_stage2d_c2_direct"


@configclass
class HexapodPhase2RecoveryStage2DC3PPORunnerCfg(
    HexapodPhase2RecoveryStage2DC2PPORunnerCfg
):
    experiment_name = "hexapod_robstride_phase2_recovery_stage2d_c3_direct"


@configclass
class HexapodPhase2RecoveryStage2DC4PPORunnerCfg(
    HexapodPhase2RecoveryStage2DC3PPORunnerCfg
):
    experiment_name = "hexapod_robstride_phase2_recovery_stage2d_c4_direct"


@configclass
class HexapodPhase2RecoveryStage2DC5PPORunnerCfg(
    HexapodPhase2RecoveryStage2DC4PPORunnerCfg
):
    max_iterations = 40
    experiment_name = "hexapod_robstride_phase2_recovery_stage2d_c5_pure_y_direct"


__all__ = [
    "HexapodPhase2RecoveryStage2DC0EnvCfg",
    "HexapodPhase2RecoveryStage2DC0PPORunnerCfg",
    "HexapodPhase2RecoveryStage2DC1EnvCfg",
    "HexapodPhase2RecoveryStage2DC1PPORunnerCfg",
    "HexapodPhase2RecoveryStage2DC2EnvCfg",
    "HexapodPhase2RecoveryStage2DC2PPORunnerCfg",
    "HexapodPhase2RecoveryStage2DC3EnvCfg",
    "HexapodPhase2RecoveryStage2DC3PPORunnerCfg",
    "HexapodPhase2RecoveryStage2DC4EnvCfg",
    "HexapodPhase2RecoveryStage2DC4PPORunnerCfg",
    "HexapodPhase2RecoveryStage2DC5EnvCfg",
    "HexapodPhase2RecoveryStage2DC5PPORunnerCfg",
    "HexapodStage2DC0CommandCfg",
    "HexapodStage2DC1CommandCfg",
    "HexapodStage2DC2CommandCfg",
    "HexapodStage2DC3CommandCfg",
    "HexapodStage2DC4CommandCfg",
    "HexapodStage2DC5CommandCfg",
]
