"""Opt-in Phase 2 omnidirectional command and runner configurations.

The profiles expose ``velocity_command`` as the canonical Phase 2 contract. Keeping
that contract in a separate module lets legacy Phase 1 registrations retain their
original body-frame, reset-only command behavior.
"""

from __future__ import annotations

import math

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from .asset_cfg import FEMUR_JOINTS, HEXAPOD_CFG, TIBIA_JOINTS
from .phase1_v2_cfg import HexapodPhase1V2EnvCfg, HexapodPhase1V2PPORunnerCfg
from .phase1_v5_cfg import HexapodPhase1V5EnvCfg, HexapodPhase1V5PPORunnerCfg


@configclass
class HexapodVelocityCommandCfg:
    """Velocity-command sampling profile for the DirectRLEnv."""

    lin_vel_x_range: tuple[float, float] = (-0.20, 0.55)
    lin_vel_y_range: tuple[float, float] = (-0.20, 0.20)
    ang_vel_z_range: tuple[float, float] = (-0.50, 0.50)
    resampling_time_range_s: tuple[float, float] = (6.0, 10.0)

    standing_probability: float = 0.10
    longitudinal_only_probability: float = 0.25
    lateral_only_probability: float = 0.15
    yaw_only_probability: float = 0.15
    combined_probability: float = 0.35

    def __post_init__(self):
        ranges = {
            "lin_vel_x_range": self.lin_vel_x_range,
            "lin_vel_y_range": self.lin_vel_y_range,
            "ang_vel_z_range": self.ang_vel_z_range,
            "resampling_time_range_s": self.resampling_time_range_s,
        }
        for name, value_range in ranges.items():
            if len(value_range) != 2 or not all(math.isfinite(value) for value in value_range):
                raise ValueError(f"{name} must contain two finite values, got {value_range!r}")
            if value_range[0] > value_range[1]:
                raise ValueError(f"{name} must be ordered low-to-high, got {value_range!r}")
        if self.resampling_time_range_s[0] <= 0.0:
            raise ValueError("resampling_time_range_s must be strictly positive")

        probabilities = (
            self.standing_probability,
            self.longitudinal_only_probability,
            self.lateral_only_probability,
            self.yaw_only_probability,
            self.combined_probability,
        )
        if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
            raise ValueError(f"Command-mixture probabilities must be in [0, 1], got {probabilities!r}")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(f"Command-mixture probabilities must sum to 1.0, got {sum(probabilities)}")


@configclass
class HexapodStage1RecoveryCommandCfg(HexapodVelocityCommandCfg):
    """Episode-stable 55/20/20/5 gait-preserving recovery buckets."""

    sampling_mode: str = "stage1_recovery"
    lin_vel_x_range: tuple[float, float] = (0.20, 0.35)
    lin_vel_y_range: tuple[float, float] = (-0.07, 0.07)
    ang_vel_z_range: tuple[float, float] = (-0.15, 0.15)
    # Longer than the 20-second episode, so a bucket keeps one command until
    # reset. Later recovery stages can deliberately introduce transitions.
    resampling_time_range_s: tuple[float, float] = (30.0, 30.0)

    standing_probability: float = 0.0
    longitudinal_only_probability: float = 0.55
    lateral_only_probability: float = 0.20
    yaw_only_probability: float = 0.20
    combined_probability: float = 0.05

    forward_range: tuple[float, float] = (0.20, 0.35)
    moving_forward_range: tuple[float, float] = (0.20, 0.32)
    lateral_abs_range: tuple[float, float] = (0.03, 0.07)
    yaw_abs_range: tuple[float, float] = (0.06, 0.15)
    combined_forward_range: tuple[float, float] = (0.22, 0.30)
    combined_lateral_abs_range: tuple[float, float] = (0.03, 0.06)
    combined_yaw_abs_range: tuple[float, float] = (0.06, 0.12)


@configclass
class HexapodStage2RecoveryCommandCfg(HexapodVelocityCommandCfg):
    """Episode-stable 40/30/30 forward/yaw/lateral acquisition buckets."""

    sampling_mode: str = "stage2_recovery"
    lin_vel_x_range: tuple[float, float] = (0.20, 0.34)
    lin_vel_y_range: tuple[float, float] = (-0.14, 0.14)
    ang_vel_z_range: tuple[float, float] = (-0.30, 0.30)
    resampling_time_range_s: tuple[float, float] = (30.0, 30.0)

    standing_probability: float = 0.0
    longitudinal_only_probability: float = 0.40
    lateral_only_probability: float = 0.30
    yaw_only_probability: float = 0.30
    combined_probability: float = 0.0

    forward_range: tuple[float, float] = (0.22, 0.34)
    moving_forward_range: tuple[float, float] = (0.20, 0.30)
    lateral_abs_range: tuple[float, float] = (0.08, 0.14)
    yaw_abs_range: tuple[float, float] = (0.18, 0.30)


@configclass
class HexapodStage2BLateralAcquisitionCommandCfg(HexapodVelocityCommandCfg):
    """40/40/10/10 forward/pure-y/forward+y/forward+yaw buckets."""

    sampling_mode: str = "stage2b_lateral_acquisition"
    lin_vel_x_range: tuple[float, float] = (0.0, 0.32)
    lin_vel_y_range: tuple[float, float] = (-0.12, 0.12)
    ang_vel_z_range: tuple[float, float] = (-0.28, 0.28)
    resampling_time_range_s: tuple[float, float] = (30.0, 30.0)

    standing_probability: float = 0.0
    longitudinal_only_probability: float = 0.40
    lateral_only_probability: float = 0.40
    yaw_only_probability: float = 0.10
    combined_probability: float = 0.10

    forward_range: tuple[float, float] = (0.22, 0.32)
    lateral_only_abs_range: tuple[float, float] = (0.08, 0.12)
    combined_forward_range: tuple[float, float] = (0.18, 0.24)
    combined_lateral_abs_range: tuple[float, float] = (0.06, 0.10)
    yaw_forward_range: tuple[float, float] = (0.20, 0.26)
    yaw_abs_range: tuple[float, float] = (0.20, 0.28)


@configclass
class HexapodStage2CStabilizedForwardCommandCfg(HexapodVelocityCommandCfg):
    """Lower-stance forward curriculum with deliberate stationary anchors."""

    sampling_mode: str = "mixture"
    lin_vel_x_range: tuple[float, float] = (0.16, 0.32)
    lin_vel_y_range: tuple[float, float] = (0.0, 0.0)
    ang_vel_z_range: tuple[float, float] = (0.0, 0.0)
    # Long holds let the optimizer see complete gait cycles while the occasional
    # transition prevents a policy that is only stable immediately after reset.
    resampling_time_range_s: tuple[float, float] = (6.0, 10.0)

    standing_probability: float = 0.20
    longitudinal_only_probability: float = 0.80
    lateral_only_probability: float = 0.0
    yaw_only_probability: float = 0.0
    combined_probability: float = 0.0


@configclass
class HexapodPhase2RecoveryStage1EnvCfg(HexapodPhase1V5EnvCfg):
    """First recovery stage anchored to the accepted anatomical-forward gait."""

    command_frame: str = "navigation"
    axiswise_velocity_rewards = True
    lin_vel_x_reward_scale = 3.5
    lin_vel_y_reward_scale = 2.0
    lin_vel_x_tracking_std_mps = 0.18
    lin_vel_y_tracking_std_mps = 0.12
    yaw_rate_reward_scale = 2.0
    yaw_rate_tracking_std_rad_s = 0.25

    # RS05-aware protection. The stochastic-policy clamp in the recovery
    # launcher is the primary protection; these costs reject sustained raw PD
    # demand without erasing the accepted gait.
    rated_torque_excess_reward_scale = -0.06
    torque_saturation_reward_scale = -0.20
    fall_penalty = -5.0

    # The original forward gait is safe but visibly bobs and rocks the deck.
    # Penalize only vertical translation and roll/pitch motion here: commanded
    # yaw remains free, and the increases are deliberately modest so steering
    # acquisition still dominates the objective.  The lower-femur stance is
    # evaluated separately before changing the checkpoint's physical reset pose.
    z_vel_reward_scale = -2.5
    ang_vel_reward_scale = -0.30
    flat_orientation_reward_scale = -2.5
    base_height_reward_scale = -20.0

    velocity_command: HexapodStage1RecoveryCommandCfg = (
        HexapodStage1RecoveryCommandCfg()
    )
    command_lin_vel_x_range_mps = velocity_command.lin_vel_x_range
    command_lin_vel_y_range_mps = velocity_command.lin_vel_y_range
    command_yaw_rate_range_rad_s = velocity_command.ang_vel_z_range
    stand_command_fraction = 0.0


@configclass
class HexapodPhase2RecoveryStage1PPORunnerCfg(HexapodPhase1V5PPORunnerCfg):
    """Low-KL, bounded-exploration PPO for Stage 1 recovery."""

    max_iterations = 125
    save_interval = 25
    experiment_name = "hexapod_robstride_phase2_recovery_stage1_direct"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.10),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.10,
        entropy_coef=1.0e-4,
        num_learning_epochs=3,
        num_mini_batches=8,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.002,
        max_grad_norm=0.5,
    )


@configclass
class HexapodPhase2RecoveryStage2EnvCfg(HexapodPhase2RecoveryStage1EnvCfg):
    """Force balanced signed y/yaw acquisition while retaining forward gait."""

    gate_axis_rewards_by_command = True
    axis_command_active_threshold = 0.01
    lin_vel_y_reward_scale = 3.0
    lin_vel_y_tracking_std_mps = 0.12
    yaw_rate_reward_scale = 3.0
    yaw_rate_tracking_std_rad_s = 0.20
    lateral_signed_progress_reward_scale = 2.0
    yaw_signed_progress_reward_scale = 2.0
    inactive_lateral_velocity_reward_scale = -0.25
    inactive_yaw_rate_reward_scale = -0.10

    # Stage 1 telemetry drifted monotonically in vertical bob and roll/pitch.
    # These are 2--3x the Stage 1 coefficients, meaningful at the measured RMS
    # motion but still bounded well below the active-axis acquisition reward.
    z_vel_reward_scale = -6.0
    ang_vel_reward_scale = -0.75
    flat_orientation_reward_scale = -6.0
    base_height_reward_scale = -60.0

    # Dynamic A/B playback selected this halfway lower-body stance: it improves
    # Stage 1 stability and continuous-torque demand without approaching the
    # RS05 5.5 N*m simulation peak seen in the deeper candidate.  Build a new
    # articulation and InitialStateCfg; never mutate the inherited/shared
    # HEXAPOD_CFG dictionaries in place.
    robot = HEXAPOD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=HEXAPOD_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.198),
            joint_pos={
                **HEXAPOD_CFG.init_state.joint_pos,
                **{name: 0.50 for name in FEMUR_JOINTS},
                **{name: 2.170 for name in TIBIA_JOINTS},
            },
        )
    )
    nominal_height_m = 0.189

    velocity_command: HexapodStage2RecoveryCommandCfg = (
        HexapodStage2RecoveryCommandCfg()
    )
    command_lin_vel_x_range_mps = velocity_command.lin_vel_x_range
    command_lin_vel_y_range_mps = velocity_command.lin_vel_y_range
    command_yaw_rate_range_rad_s = velocity_command.ang_vel_z_range
    stand_command_fraction = 0.0


@configclass
class HexapodPhase2RecoveryStage2PPORunnerCfg(
    HexapodPhase2RecoveryStage1PPORunnerCfg
):
    """Fixed-rate, bounded PPO for signed steering-axis acquisition."""

    max_iterations = 150
    save_interval = 25
    experiment_name = "hexapod_robstride_phase2_recovery_stage2_direct"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=3.0e-4,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=0.75,
    )


@configclass
class HexapodPhase2RecoveryStage2BLateralEnvCfg(
    HexapodPhase2RecoveryStage2EnvCfg
):
    """Acquire signed lateral gait without sacrificing the safe lower stance."""

    # The first Stage 2 screen showed that forward+y commands let the warm-start
    # policy ignore y.  Half of this fleet now receives an active lateral
    # command, with 40% receiving vx=0.  Strong dense y gradients are paired
    # with tighter RS05-demand costs so PPO cannot buy tracking with raw torque.
    lin_vel_x_reward_scale = 4.0
    lin_vel_y_reward_scale = 6.0
    # With balanced +/-0.10 commands, a narrow Gaussian makes an unconditional
    # one-sided drift score better than zero response.  A 0.18 width reverses
    # that shortcut while the signed-progress term supplies the dense
    # command-conditional gradient needed to acquire both directions.
    lin_vel_y_tracking_std_mps = 0.18
    lateral_signed_progress_reward_scale = 5.0
    yaw_rate_reward_scale = 4.0
    yaw_rate_tracking_std_rad_s = 0.20
    yaw_signed_progress_reward_scale = 3.0
    inactive_lateral_velocity_reward_scale = -0.50
    inactive_yaw_rate_reward_scale = -0.20
    rated_torque_excess_reward_scale = -0.12
    torque_saturation_reward_scale = -0.35
    fall_penalty = -8.0
    terminate_on_computed_torque_demand_nm = 5.5
    terminate_on_computed_torque_demand_duration_s = 0.10
    torque_demand_termination_grace_s = 0.50

    velocity_command: HexapodStage2BLateralAcquisitionCommandCfg = (
        HexapodStage2BLateralAcquisitionCommandCfg()
    )
    command_lin_vel_x_range_mps = velocity_command.lin_vel_x_range
    command_lin_vel_y_range_mps = velocity_command.lin_vel_y_range
    command_yaw_rate_range_rad_s = velocity_command.ang_vel_z_range
    stand_command_fraction = 0.0


@configclass
class HexapodPhase2RecoveryStage2BLateralPPORunnerCfg(
    HexapodPhase2RecoveryStage2PPORunnerCfg
):
    """Checkpoint-dense conservative PPO for the Stage 2B lateral branch."""

    max_iterations = 100
    save_interval = 10
    experiment_name = "hexapod_robstride_phase2_recovery_stage2b_lateral_direct"
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
class HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg(
    HexapodPhase2RecoveryStage2EnvCfg
):
    """Lower, hardware-aware forward gait with deck steadiness as an objective."""

    # The exact model-25 A/B playback selected this deeper pose for the
    # forward-only adaptation stage.  At 0.20/0.30 m/s it lowered the moving
    # deck from 193.3 mm to 181.2 mm, improved all four deck-motion metrics,
    # and retained 1.28 N*m of raw-demand margin at the worst tested speed.
    # It is not automatically admitted to the later joystick task: mixed-axis
    # playback has its own RS05 and fall gates.
    robot = HEXAPOD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=HEXAPOD_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.185),
            joint_pos={
                **HEXAPOD_CFG.init_state.joint_pos,
                **{name: 0.60 for name in FEMUR_JOINTS},
                **{name: 2.2335 for name in TIBIA_JOINTS},
            },
        ),
    )
    # Standing settles slightly below the moving gait's deck target. The
    # inherited 0.20 action scale permits a 0.40-rad saturated sign flip, so
    # bound the final post-soft-limit target to 0.04 rad per 20 ms step.
    nominal_height_m = 0.181
    stand_nominal_height_m = 0.177
    stand_action_scale = 0.0
    processed_joint_target_slew_limit_rad_per_20ms = 0.04

    # Forward tracking must remain the dominant task, but unlike the earlier
    # recovery runs the stability terms are numerically large enough to affect
    # gait selection.  The bounded score prevents rare impacts from dwarfing
    # the locomotion signal while the L2 terms still distinguish very calm
    # candidates near the top of that score.
    gate_longitudinal_reward_by_command = True
    lin_vel_x_reward_scale = 4.5
    lin_vel_x_tracking_std_mps = 0.15
    deck_stability_reward_scale = 3.0
    deck_stability_vertical_velocity_scale_mps = 0.10
    deck_stability_roll_pitch_rate_scale_rad_s = 0.50
    deck_stability_projected_gravity_xy_scale = 0.035
    deck_stability_height_error_scale_m = 0.010
    z_vel_reward_scale = -12.0
    ang_vel_reward_scale = -1.5
    flat_orientation_reward_scale = -30.0
    base_height_reward_scale = -800.0

    # Keep a conservative five/four/three-foot support floor as speed rises,
    # smooth motor demand, and favor planted rather than skating feet.  This is
    # a minimum-contact safety bias, not an exact gait-phase target.  Air-time
    # is intentionally left disabled because its fixed 0.25 s target encourages
    # high, long swings that worsen body bob on this geometry.
    speed_conditioned_support_targets = True
    support_contact_low_speed_threshold_mps = 0.12
    support_contact_high_speed_threshold_mps = 0.24
    support_contact_target_low_speed = 5.0
    support_contact_target_medium_speed = 4.0
    support_contact_target_high_speed = 3.0
    support_shortfall_reward_scale = -0.50
    foot_slip_reward_scale = -0.50
    action_rate_reward_scale = -0.06
    joint_accel_reward_scale = -4.0e-7
    joint_torque_reward_scale = -2.0e-4
    joint_torque_slew_reward_scale = -0.01
    rated_torque_excess_reward_scale = -0.14
    torque_saturation_reward_scale = -0.40
    fall_penalty = -8.0
    terminate_on_computed_torque_demand_nm = 5.5
    terminate_on_computed_torque_demand_duration_s = 0.10
    torque_demand_termination_grace_s = 0.50

    velocity_command: HexapodStage2CStabilizedForwardCommandCfg = (
        HexapodStage2CStabilizedForwardCommandCfg()
    )
    command_lin_vel_x_range_mps = velocity_command.lin_vel_x_range
    command_lin_vel_y_range_mps = velocity_command.lin_vel_y_range
    command_yaw_rate_range_rad_s = velocity_command.ang_vel_z_range
    stand_command_fraction = velocity_command.standing_probability


@configclass
class HexapodPhase2RecoveryStage2CStabilizedForwardPPORunnerCfg(
    HexapodPhase2RecoveryStage2BLateralPPORunnerCfg
):
    """Conservative checkpoint-dense PPO for lower-body gait stabilization."""

    max_iterations = 120
    save_interval = 10
    experiment_name = "hexapod_robstride_phase2_recovery_stage2c_stable_forward_direct"
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
class HexapodPhase2WarmupEnvCfg(HexapodPhase1V2EnvCfg):
    """Conservative anatomical-navigation curriculum.

    Observation tensor shapes remain checkpoint-compatible, but a legacy body-X
    walking policy is not semantically a forward-navigation warm start.
    """

    command_frame: str = "navigation"
    yaw_rate_reward_scale = 2.0
    yaw_rate_tracking_std_rad_s = 0.15
    velocity_command: HexapodVelocityCommandCfg = HexapodVelocityCommandCfg(
        lin_vel_x_range=(-0.20, 0.55),
        lin_vel_y_range=(-0.20, 0.20),
        ang_vel_z_range=(-0.50, 0.50),
    )
    # Legacy aliases retained for tooling/config introspection.
    command_lin_vel_x_range_mps = velocity_command.lin_vel_x_range
    command_lin_vel_y_range_mps = velocity_command.lin_vel_y_range
    command_yaw_rate_range_rad_s = velocity_command.ang_vel_z_range
    stand_command_fraction = velocity_command.standing_probability


@configclass
class HexapodPhase2FinalEnvCfg(HexapodPhase1V2EnvCfg):
    """Full anatomical-navigation Phase 2 velocity-command ranges."""

    command_frame: str = "navigation"
    yaw_rate_reward_scale = 2.0
    yaw_rate_tracking_std_rad_s = 0.15
    velocity_command: HexapodVelocityCommandCfg = HexapodVelocityCommandCfg(
        lin_vel_x_range=(-0.40, 0.60),
        lin_vel_y_range=(-0.35, 0.35),
        ang_vel_z_range=(-0.75, 0.75),
    )
    # Legacy aliases retained for tooling/config introspection.
    command_lin_vel_x_range_mps = velocity_command.lin_vel_x_range
    command_lin_vel_y_range_mps = velocity_command.lin_vel_y_range
    command_yaw_rate_range_rad_s = velocity_command.ang_vel_z_range
    stand_command_fraction = velocity_command.standing_probability


@configclass
class HexapodPhase2WarmupPPORunnerCfg(HexapodPhase1V2PPORunnerCfg):
    """Moderate-rate joystick fine-tune from accepted anatomical Phase 1."""

    max_iterations = 500
    save_interval = 25
    experiment_name = "hexapod_robstride_phase2_omni_warmup_direct"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=1.0,
    )


@configclass
class HexapodPhase2FinalPPORunnerCfg(HexapodPhase1V2PPORunnerCfg):
    """Separate log/checkpoint namespace for the final Phase 2 range."""

    max_iterations = 300
    save_interval = 25
    experiment_name = "hexapod_robstride_phase2_omni_direct"
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


__all__ = [
    "HexapodPhase2RecoveryStage1EnvCfg",
    "HexapodPhase2RecoveryStage1PPORunnerCfg",
    "HexapodPhase2RecoveryStage2EnvCfg",
    "HexapodPhase2RecoveryStage2PPORunnerCfg",
    "HexapodPhase2RecoveryStage2BLateralEnvCfg",
    "HexapodPhase2RecoveryStage2BLateralPPORunnerCfg",
    "HexapodPhase2RecoveryStage2CStabilizedForwardEnvCfg",
    "HexapodPhase2RecoveryStage2CStabilizedForwardPPORunnerCfg",
    "HexapodPhase2FinalEnvCfg",
    "HexapodPhase2FinalPPORunnerCfg",
    "HexapodPhase2WarmupEnvCfg",
    "HexapodPhase2WarmupPPORunnerCfg",
    "HexapodStage1RecoveryCommandCfg",
    "HexapodStage2RecoveryCommandCfg",
    "HexapodStage2BLateralAcquisitionCommandCfg",
    "HexapodStage2CStabilizedForwardCommandCfg",
    "HexapodVelocityCommandCfg",
]
