"""Vectorized direct RL environment for straight-line hexapod locomotion."""

from __future__ import annotations

import math

import gymnasium as gym
import torch
import warp as wp

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .command_sampling import (
    active_axis_gaussian_tracking_reward,
    advance_resampling_timers,
    body_to_navigation_frame,
    normalized_signed_axis_progress,
    sample_stage1_recovery_commands,
    sample_stage2_recovery_commands,
    sample_stage2b_lateral_acquisition_commands,
    sample_stage2d_oblique_homotopy_commands,
    sample_stage2e_joystick_transition_commands,
    sample_uniform_intervals,
    sample_velocity_command_mixture,
)
from .env_cfg import LEG_LINK_NAMES, HexapodFlatEnvCfg
from .rewards.actions import (
    apply_command_conditioned_stand_action_scale,
    limit_processed_joint_target_slew,
)
from .rewards.axes import (
    bounded_inactive_yaw_rate_slew_cost,
    capped_normalized_axis_error,
    reset_safe_exponential_moving_average,
)
from .rewards.contact_wrench import (
    bounded_inactive_bilateral_longitudinal_contact_moment_cost,
    bounded_inactive_ground_contact_yaw_moment_cost,
    reset_batch_time_mean_and_p50,
)
from .rewards.gait import (
    advance_gait_phase,
    assign_tripod_pairs_from_foot_offsets,
    gait_phase_contact_reward,
    swing_clearance_reward,
    tripod_expected_stance,
)
from .rewards.stability import (
    normalized_deck_stability_reward,
    select_command_conditioned_nominal_height,
    select_support_contact_target,
)
from .rewards.torque import (
    applied_torque_slew_l2,
    max_joint_rated_torque_excess_l1,
    max_joint_rated_torque_excess_l2,
)

__all__ = [
    "HexapodEnv",
    "advance_gait_phase",
    "applied_torque_slew_l2",
    "apply_command_conditioned_stand_action_scale",
    "assign_tripod_pairs_from_foot_offsets",
    "bounded_inactive_bilateral_longitudinal_contact_moment_cost",
    "bounded_inactive_ground_contact_yaw_moment_cost",
    "bounded_inactive_yaw_rate_slew_cost",
    "capped_normalized_axis_error",
    "gait_phase_contact_reward",
    "limit_processed_joint_target_slew",
    "max_joint_rated_torque_excess_l1",
    "max_joint_rated_torque_excess_l2",
    "normalized_deck_stability_reward",
    "reset_batch_time_mean_and_p50",
    "reset_safe_exponential_moving_average",
    "select_command_conditioned_nominal_height",
    "select_support_contact_target",
    "swing_clearance_reward",
    "tripod_expected_stance",
]


class HexapodEnv(DirectRLEnv):
    cfg: HexapodFlatEnvCfg

    def __init__(self, cfg: HexapodFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._stand_action_scale = float(self.cfg.stand_action_scale)
        if (
            not math.isfinite(self._stand_action_scale)
            or not 0.0 <= self._stand_action_scale <= 1.0
        ):
            raise ValueError(
                "stand_action_scale must be finite and in [0, 1], "
                f"got {self.cfg.stand_action_scale!r}"
            )
        self._processed_actions = self._robot.data.default_joint_pos.torch.clone()
        self._previous_processed_joint_target = self._processed_actions.clone()
        self._has_previous_processed_joint_target = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._joint_target_slew_limited_fraction = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._processed_joint_target_slew_limit_rad_per_20ms = getattr(
            self.cfg, "processed_joint_target_slew_limit_rad_per_20ms", None
        )
        if self._processed_joint_target_slew_limit_rad_per_20ms is not None and (
            not math.isfinite(
                float(self._processed_joint_target_slew_limit_rad_per_20ms)
            )
            or float(self._processed_joint_target_slew_limit_rad_per_20ms) <= 0.0
        ):
            raise ValueError(
                "processed_joint_target_slew_limit_rad_per_20ms must be None "
                "or finite and positive, got "
                f"{self._processed_joint_target_slew_limit_rad_per_20ms!r}"
            )
        self._previous_applied_torque = torch.zeros_like(
            self._robot.data.applied_torque.torch
        )
        self._has_previous_applied_torque = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._previous_command_frame_yaw_rate = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._has_previous_command_frame_yaw_rate = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        # Insect-gait clock state.  The tripod split is resolved lazily from
        # the first reward step's base-frame foot geometry because leg names
        # do not encode physical sides on this robot.
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)
        self._tripod_a_mask: torch.Tensor | None = None
        self._gait_shaping_enabled = (
            self.cfg.gait_phase_contact_reward_scale != 0.0
            or self.cfg.swing_clearance_reward_scale != 0.0
        )
        self._velocity_command_cfg = getattr(self.cfg, "velocity_command", None)
        self._command_frame = getattr(self.cfg, "command_frame", "body")
        if self._command_frame not in {"body", "navigation"}:
            raise ValueError(
                "command_frame must be 'body' or 'navigation', "
                f"got {self._command_frame!r}"
            )
        self._navigation_lateral_velocity_ema_tau_s = getattr(
            self.cfg, "navigation_lateral_velocity_ema_tau_s", None
        )
        if self._navigation_lateral_velocity_ema_tau_s is not None:
            if (
                not math.isfinite(float(self._navigation_lateral_velocity_ema_tau_s))
                or float(self._navigation_lateral_velocity_ema_tau_s) <= 0.0
            ):
                raise ValueError(
                    "navigation_lateral_velocity_ema_tau_s must be None or finite "
                    "and positive, got "
                    f"{self._navigation_lateral_velocity_ema_tau_s!r}"
                )
            if self._command_frame != "navigation":
                raise ValueError(
                    "navigation lateral-velocity EMA requires command_frame='navigation'"
                )
        self._navigation_lateral_velocity_ema = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._has_navigation_lateral_velocity_ema = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        for name in (
            "longitudinal_normalized_error_penalty_scale",
            "lateral_normalized_error_penalty_scale",
        ):
            error_penalty_scale = float(getattr(self.cfg, name, 0.0))
            if not math.isfinite(error_penalty_scale) or error_penalty_scale < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative, "
                    f"got {error_penalty_scale!r}"
                )
        self._command_resampling_time_left_s = (
            torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            if self._velocity_command_cfg is not None
            else None
        )
        self._command_resample_epoch = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._episode_elapsed_s = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self._torque_demand_excess_duration_s = torch.zeros_like(
            self._episode_elapsed_s
        )
        self._moving_command_elapsed_s = torch.zeros_like(self._episode_elapsed_s)
        self._forward_metric_sums = {
            key: torch.zeros_like(self._episode_elapsed_s)
            for key in (
                "commanded_velocity_x_mps",
                "achieved_velocity_x_mps",
                "abs_velocity_error_x_mps",
                "tracking_within_tolerance_fraction",
            )
        }
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in (
                "track_lin_vel_xy_exp",
                "track_lin_vel_x_exp",
                "track_lin_vel_y_exp",
                "track_ang_vel_z_exp",
                "longitudinal_signed_progress",
                "longitudinal_normalized_error_penalty",
                "lateral_signed_progress",
                "lateral_normalized_error_penalty",
                "yaw_signed_progress",
                "inactive_lin_vel_y_l2",
                "inactive_ang_vel_z_l2",
                "inactive_yaw_rate_slew",
                "inactive_ground_contact_yaw_moment",
                "inactive_bilateral_longitudinal_contact_moment",
                "alive",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "joint_torques_l2",
                "rated_torque_excess_l2",
                "max_joint_rated_torque_excess_l2",
                "max_joint_rated_torque_excess_l1",
                "torque_saturation_cost",
                "joint_acc_l2",
                "action_rate_l2",
                "joint_torque_slew_l2",
                "feet_air_time",
                "foot_slip_l2",
                "undesired_contacts",
                "flat_orientation_l2",
                "base_height_l2",
                "joint_limits_l2",
                "support_shortfall",
                "deck_stability",
                "gait_phase_contact",
                "swing_clearance",
                "fall_penalty",
                "torque_saturation_fraction",
                "navigation_lateral_velocity_ema_mps",
                "normalized_longitudinal_error",
                "normalized_lateral_error",
                "joint_target_slew_limited_fraction",
                "ground_contact_yaw_moment_abs_nm",
                "distal_friction_xy_force_n",
                "bilateral_longitudinal_contact_moment_nm",
            )
        }
        self._episode_metric_names = {
            "torque_saturation_fraction",
            "navigation_lateral_velocity_ema_mps",
            "normalized_longitudinal_error",
            "normalized_lateral_error",
            "joint_target_slew_limited_fraction",
            "ground_contact_yaw_moment_abs_nm",
            "distal_friction_xy_force_n",
            "bilateral_longitudinal_contact_moment_nm",
        }

        expected_foot_names = [tibia for _, _, tibia in LEG_LINK_NAMES]
        self._feet_body_ids, foot_names = self._robot.find_bodies(
            expected_foot_names, preserve_order=True
        )
        body_counts = {
            "base": len(self._base_contact_sensor.body_names),
            "feet": sum(len(sensor.body_names) for sensor in self._feet_contact_sensors),
            "coxae": len(self._coxa_contact_sensor.body_names),
            "femurs": sum(len(sensor.body_names) for sensor in self._femur_contact_sensors),
        }
        foot_filter_counts = [
            sensor.contact_view.filter_count for sensor in self._feet_contact_sensors
        ]
        if (
            body_counts != {"base": 1, "feet": 6, "coxae": 6, "femurs": 6}
            or foot_names != expected_foot_names
            or foot_filter_counts != [1] * 6
        ):
            raise RuntimeError(
                f"Unexpected contact-body layout: {body_counts}, "
                f"robot_feet={foot_names}, plane_filters={foot_filter_counts}"
            )
        ground_wrench_reward_requested = (
            self.cfg.inactive_ground_contact_yaw_moment_reward_scale != 0.0
            or self.cfg.inactive_bilateral_longitudinal_contact_moment_reward_scale
            != 0.0
        )
        if ground_wrench_reward_requested and any(
            not sensor.cfg.track_friction_forces
            or sensor.data.friction_forces_w is None
            for sensor in self._feet_contact_sensors
        ):
            raise RuntimeError(
                "Ground-wrench rewards require track_friction_forces=True "
                "on every foot contact sensor"
            )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._base_contact_sensor = ContactSensor(self.cfg.base_contact_sensor)
        self._coxa_contact_sensor = ContactSensor(self.cfg.coxa_contact_sensor)
        self._feet_contact_sensors = [ContactSensor(cfg) for cfg in self.cfg.feet_contact_sensors]
        self._femur_contact_sensors = [ContactSensor(cfg) for cfg in self.cfg.femur_contact_sensors]
        self.scene.sensors["base_contact_sensor"] = self._base_contact_sensor
        self.scene.sensors["coxa_contact_sensor"] = self._coxa_contact_sensor
        for index, sensor in enumerate(self._feet_contact_sensors):
            self.scene.sensors[f"foot_{index}_contact_sensor"] = sensor
        for index, sensor in enumerate(self._femur_contact_sensors):
            self.scene.sensors[f"femur_{index}_contact_sensor"] = sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.78, 0.82))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        clipped_actions = actions.clone().clamp(-1.0, 1.0)
        self._actions = apply_command_conditioned_stand_action_scale(
            clipped_actions,
            self._commands,
            stand_action_scale=self._stand_action_scale,
            command_active_threshold=self.cfg.axis_command_active_threshold,
        )
        targets = (
            self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos.torch
        )
        soft_limits = self._robot.data.soft_joint_pos_limits.torch
        clamped_targets = torch.clamp(
            targets, soft_limits[:, :, 0], soft_limits[:, :, 1]
        )
        (
            self._processed_actions,
            self._joint_target_slew_limited_fraction,
        ) = limit_processed_joint_target_slew(
            clamped_targets,
            self._previous_processed_joint_target,
            self._has_previous_processed_joint_target,
            max_delta_rad_per_20ms=(
                self._processed_joint_target_slew_limit_rad_per_20ms
            ),
            step_dt=self.step_dt,
        )
        if self._processed_joint_target_slew_limit_rad_per_20ms is not None:
            self._previous_processed_joint_target.copy_(self._processed_actions)
            self._has_previous_processed_joint_target.fill_(True)

    def _apply_action(self):
        self._robot.set_joint_position_target_index(target=self._processed_actions)

    def _vector_in_command_frame(self, body_vector: torch.Tensor) -> torch.Tensor:
        """Express a body-frame vector in the frame used by policy commands."""

        if self._command_frame == "navigation":
            return body_to_navigation_frame(body_vector)
        return body_vector

    def _get_observations(self) -> dict:
        root_lin_vel = self._vector_in_command_frame(self._robot.data.root_lin_vel_b.torch)
        root_ang_vel = self._vector_in_command_frame(self._robot.data.root_ang_vel_b.torch)
        projected_gravity = self._vector_in_command_frame(
            self._robot.data.projected_gravity_b.torch
        )
        observation_terms = [
            root_lin_vel,
            root_ang_vel,
            projected_gravity,
            self._commands,
            self._robot.data.joint_pos.torch - self._robot.data.default_joint_pos.torch,
            self._robot.data.joint_vel.torch,
            self._actions,
        ]
        if self.cfg.include_gait_phase_observation:
            gait_angle = 2.0 * math.pi * self._gait_phase
            observation_terms.append(
                torch.stack((torch.sin(gait_angle), torch.cos(gait_angle)), dim=-1)
            )
        observations = torch.cat(observation_terms, dim=-1)
        self._previous_actions = self._actions.clone()
        return {"policy": observations}

    def _get_foot_contact_state(
        self,
        *,
        include_ground_wrench: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        """Return contact masks/speed and, optionally, ground-wrench inputs.

        The default three-tensor return preserves the validation/debug API.
        Reward integration opts into contact points, full reaction forces, and
        per-foot planar friction magnitudes from the contact sensors.
        """
        contact_points_w = []
        sensor_positions_w = []
        sensor_quaternions_w = []
        normal_forces_w = []
        friction_forces_w = []
        for sensor in self._feet_contact_sensors:
            data = sensor.data
            contact_points_w.append(data.contact_pos_w.torch[:, 0, 0])
            sensor_positions_w.append(data.pos_w.torch[:, 0])
            sensor_quaternions_w.append(data.quat_w.torch[:, 0])
            normal_forces_w.append(data.force_matrix_w.torch[:, 0, 0])
            if include_ground_wrench:
                if data.friction_forces_w is None:
                    raise RuntimeError(
                        "Ground-wrench input requested without foot friction-force tracking"
                    )
                friction_forces_w.append(data.friction_forces_w.torch[:, 0, 0])

        contact_points_w = torch.stack(contact_points_w, dim=1)
        sensor_positions_w = torch.stack(sensor_positions_w, dim=1)
        sensor_quaternions_w = torch.stack(sensor_quaternions_w, dim=1)
        normal_forces_w = torch.stack(normal_forces_w, dim=1)

        finite_point = torch.isfinite(contact_points_w).all(dim=-1)
        safe_contact_points_w = torch.where(
            finite_point.unsqueeze(-1), contact_points_w, sensor_positions_w
        )
        contact_points_l = math_utils.quat_apply_inverse(
            sensor_quaternions_w, safe_contact_points_w - sensor_positions_w
        )
        plane_contact = finite_point & (
            torch.linalg.norm(normal_forces_w, dim=-1) > 1.0
        )
        distal_contact = plane_contact & (
            contact_points_l[:, :, 1] > self.cfg.distal_foot_min_y_m
        )
        shaft_contact = plane_contact & ~distal_contact

        link_positions_w = self._robot.data.body_link_pos_w.torch[:, self._feet_body_ids]
        link_linear_velocities_w = self._robot.data.body_link_lin_vel_w.torch[
            :, self._feet_body_ids
        ]
        link_angular_velocities_w = self._robot.data.body_link_ang_vel_w.torch[
            :, self._feet_body_ids
        ]
        contact_offsets_w = safe_contact_points_w - link_positions_w
        contact_velocities_w = link_linear_velocities_w + torch.cross(
            link_angular_velocities_w, contact_offsets_w, dim=-1
        )
        contact_speed_xy = torch.linalg.norm(contact_velocities_w[:, :, :2], dim=-1)
        if include_ground_wrench:
            friction_forces_w = torch.stack(friction_forces_w, dim=1)
            ground_reaction_forces_w = normal_forces_w + friction_forces_w
            friction_force_xy_n = torch.linalg.norm(
                friction_forces_w[:, :, :2], dim=-1
            )
            return (
                distal_contact,
                shaft_contact,
                contact_speed_xy,
                safe_contact_points_w,
                ground_reaction_forces_w,
                friction_force_xy_n,
            )
        return distal_contact, shaft_contact, contact_speed_xy

    def _get_rewards(self) -> torch.Tensor:
        root_lin_vel = self._vector_in_command_frame(self._robot.data.root_lin_vel_b.torch)
        root_vertical_velocity_w = self._robot.data.root_lin_vel_w.torch[:, 2]
        root_ang_vel = self._vector_in_command_frame(self._robot.data.root_ang_vel_b.torch)
        projected_gravity = self._vector_in_command_frame(
            self._robot.data.projected_gravity_b.torch
        )
        lateral_velocity_for_shaping = root_lin_vel[:, 1]
        lateral_velocity_ema_metric = torch.zeros_like(lateral_velocity_for_shaping)
        if self._navigation_lateral_velocity_ema_tau_s is not None:
            filtered_lateral_velocity = reset_safe_exponential_moving_average(
                root_lin_vel[:, 1],
                self._navigation_lateral_velocity_ema,
                self._has_navigation_lateral_velocity_ema,
                step_dt=self.step_dt,
                tau_s=self._navigation_lateral_velocity_ema_tau_s,
            )
            self._navigation_lateral_velocity_ema.copy_(filtered_lateral_velocity)
            self._has_navigation_lateral_velocity_ema.fill_(True)
            lateral_velocity_for_shaping = filtered_lateral_velocity
            lateral_velocity_ema_metric = filtered_lateral_velocity
        lin_vel_axis_error = torch.square(
            self._commands[:, :2] - root_lin_vel[:, :2]
        )
        lin_vel_error = torch.sum(lin_vel_axis_error, dim=1)
        yaw_rate_error = torch.square(self._commands[:, 2] - root_ang_vel[:, 2])
        lateral_command_active = (
            torch.abs(self._commands[:, 1])
            > self.cfg.axis_command_active_threshold
        )
        yaw_command_active = (
            torch.abs(self._commands[:, 2])
            > self.cfg.axis_command_active_threshold
        )
        joint_torque = self._robot.data.applied_torque.torch
        computed_torque = self._robot.data.computed_torque.torch
        torque_excess = torch.relu(torch.abs(computed_torque) - self.cfg.rated_torque_nm)
        torque_saturation_fraction = torch.mean(
            (torch.abs(computed_torque) > self.cfg.rated_torque_nm).float(), dim=1
        )

        first_contact = torch.cat(
            [sensor.compute_first_contact(self.step_dt).torch for sensor in self._feet_contact_sensors],
            dim=1,
        )
        last_air_time = torch.cat(
            [sensor.data.last_air_time.torch for sensor in self._feet_contact_sensors], dim=1
        )
        air_time = torch.sum((last_air_time - 0.25) * first_contact, dim=1)
        air_time *= (
            torch.linalg.norm(self._commands[:, :2], dim=1)
            > self.cfg.moving_command_threshold_mps
        )

        foot_contact_points_w = None
        foot_ground_forces_w = None
        foot_friction_force_xy_n = None
        ground_wrench_reward_enabled = (
            self.cfg.inactive_ground_contact_yaw_moment_reward_scale != 0.0
            or self.cfg.inactive_bilateral_longitudinal_contact_moment_reward_scale
            != 0.0
        )
        if ground_wrench_reward_enabled:
            (
                foot_contact,
                shaft_contact,
                foot_speed_xy,
                foot_contact_points_w,
                foot_ground_forces_w,
                foot_friction_force_xy_n,
            ) = self._get_foot_contact_state(include_ground_wrench=True)
        else:
            # Preserve the exact legacy contact-state path while the new term
            # is dormant, including the historical three-tensor interface.
            foot_contact, shaft_contact, foot_speed_xy = (
                self._get_foot_contact_state()
            )
        foot_slip = torch.sum(
            torch.square(foot_speed_xy) * foot_contact.float(), dim=1
        )
        commanded_planar_speed = torch.linalg.norm(self._commands[:, :2], dim=1)
        support_contact_target = select_support_contact_target(
            commanded_planar_speed,
            scalar_target=self.cfg.support_contact_target,
            speed_conditioning_enabled=self.cfg.speed_conditioned_support_targets,
            low_speed_threshold_mps=self.cfg.support_contact_low_speed_threshold_mps,
            high_speed_threshold_mps=self.cfg.support_contact_high_speed_threshold_mps,
            low_speed_target=self.cfg.support_contact_target_low_speed,
            medium_speed_target=self.cfg.support_contact_target_medium_speed,
            high_speed_target=self.cfg.support_contact_target_high_speed,
        )
        support_shortfall = torch.relu(
            support_contact_target - torch.sum(foot_contact.float(), dim=1)
        )

        gait_phase_contact = None
        swing_clearance = None
        if self._gait_shaping_enabled:
            if self._tripod_a_mask is None:
                root_pos_w = self._robot.data.root_pos_w.torch[0]
                root_quat_w = self._robot.data.root_quat_w.torch[0]
                foot_pos_w = self._robot.data.body_link_pos_w.torch[
                    0, self._feet_body_ids
                ]
                foot_offsets_b = math_utils.quat_apply_inverse(
                    root_quat_w.unsqueeze(0).expand(6, 4),
                    foot_pos_w - root_pos_w.unsqueeze(0),
                )
                self._tripod_a_mask = assign_tripod_pairs_from_foot_offsets(
                    foot_offsets_b
                )
            gait_moving = (
                commanded_planar_speed > self.cfg.moving_command_threshold_mps
            )
            self._gait_phase = advance_gait_phase(
                self._gait_phase,
                commanded_planar_speed,
                gait_moving,
                step_dt=self.step_dt,
                cycles_per_meter=self.cfg.gait_cycles_per_meter,
                min_frequency_hz=self.cfg.gait_min_frequency_hz,
                max_frequency_hz=self.cfg.gait_max_frequency_hz,
            )
            expected_stance = tripod_expected_stance(
                self._gait_phase,
                self._tripod_a_mask,
                duty_factor=self.cfg.gait_duty_factor,
                sharpness=self.cfg.gait_phase_transition_sharpness,
            )
            gait_moving_weight = gait_moving.to(expected_stance.dtype)
            gait_phase_contact = (
                gait_phase_contact_reward(foot_contact, expected_stance)
                * gait_moving_weight
            )
            foot_link_pos_w = self._robot.data.body_link_pos_w.torch[
                :, self._feet_body_ids
            ]
            foot_link_quat_w = self._robot.data.body_link_quat_w.torch[
                :, self._feet_body_ids
            ]
            pad_offset_l = torch.zeros_like(foot_link_pos_w)
            pad_offset_l[:, :, 1] = self.cfg.swing_clearance_pad_offset_y_m
            pad_pos_w = foot_link_pos_w + math_utils.quat_apply(
                foot_link_quat_w, pad_offset_l
            )
            swing_clearance = (
                swing_clearance_reward(
                    pad_pos_w[:, :, 2],
                    expected_stance,
                    target_m=self.cfg.swing_clearance_target_m,
                    tolerance_m=self.cfg.swing_clearance_tolerance_m,
                )
                * gait_moving_weight
            )

        joint_torque_slew = applied_torque_slew_l2(
            joint_torque,
            self._previous_applied_torque,
            self._has_previous_applied_torque,
        )
        self._previous_applied_torque.copy_(joint_torque)
        self._has_previous_applied_torque.fill_(True)

        inactive_yaw_rate_slew = None
        if self.cfg.inactive_yaw_rate_slew_reward_scale != 0.0:
            inactive_yaw_rate_slew = bounded_inactive_yaw_rate_slew_cost(
                root_ang_vel[:, 2],
                self._previous_command_frame_yaw_rate,
                self._has_previous_command_frame_yaw_rate,
                yaw_command_active,
                reference_rad_s_per_step=(
                    self.cfg.inactive_yaw_rate_slew_reference_rad_s_per_step
                ),
            )
            self._previous_command_frame_yaw_rate.copy_(root_ang_vel[:, 2])
            # Active-yaw samples must not become history for a later inactive
            # sample; this masks the command-transition boundary as well.
            self._has_previous_command_frame_yaw_rate.copy_(~yaw_command_active)

        straight_gait_active = (
            commanded_planar_speed > self.cfg.moving_command_threshold_mps
        ) & ~yaw_command_active
        ground_contact_yaw_moment_abs_nm = torch.zeros_like(
            commanded_planar_speed
        )
        distal_friction_xy_force_n = torch.zeros_like(commanded_planar_speed)
        if ground_wrench_reward_enabled:
            assert foot_friction_force_xy_n is not None
            distal_contact_count = torch.sum(foot_contact.float(), dim=1)
            distal_friction_xy_force_n = torch.where(
                straight_gait_active,
                torch.sum(
                    foot_friction_force_xy_n * foot_contact.float(), dim=1
                )
                / torch.clamp(distal_contact_count, min=1.0),
                torch.zeros_like(commanded_planar_speed),
            )

        inactive_ground_contact_yaw_moment = None
        if self.cfg.inactive_ground_contact_yaw_moment_reward_scale != 0.0:
            assert foot_contact_points_w is not None
            assert foot_ground_forces_w is not None
            (
                inactive_ground_contact_yaw_moment,
                ground_contact_yaw_moment_abs_nm,
            ) = (
                bounded_inactive_ground_contact_yaw_moment_cost(
                    foot_contact_points_w,
                    foot_ground_forces_w,
                    self._robot.data.root_pos_w.torch,
                    foot_contact,
                    commanded_planar_speed > self.cfg.moving_command_threshold_mps,
                    yaw_command_active,
                    reference_nm=(
                        self.cfg.inactive_ground_contact_yaw_moment_reference_nm
                    ),
                )
            )

        inactive_bilateral_longitudinal_contact_moment = None
        bilateral_longitudinal_contact_moment_nm = torch.zeros_like(
            commanded_planar_speed
        )
        if (
            self.cfg.inactive_bilateral_longitudinal_contact_moment_reward_scale
            != 0.0
        ):
            assert foot_contact_points_w is not None
            assert foot_ground_forces_w is not None
            root_positions_w = self._robot.data.root_pos_w.torch
            root_quaternions_w = self._robot.data.root_quat_w.torch[:, None, :].expand(
                -1, foot_contact_points_w.shape[1], -1
            )
            contact_offsets_b = math_utils.quat_apply_inverse(
                root_quaternions_w,
                foot_contact_points_w - root_positions_w[:, None, :],
            )
            ground_forces_b = math_utils.quat_apply_inverse(
                root_quaternions_w, foot_ground_forces_w
            )
            contact_offsets_command_frame = self._vector_in_command_frame(
                contact_offsets_b
            )
            ground_forces_command_frame = self._vector_in_command_frame(
                ground_forces_b
            )
            (
                inactive_bilateral_longitudinal_contact_moment,
                bilateral_longitudinal_contact_moment_nm,
            ) = bounded_inactive_bilateral_longitudinal_contact_moment_cost(
                contact_offsets_command_frame,
                ground_forces_command_frame,
                foot_contact,
                torch.abs(self._commands[:, 0])
                > self.cfg.moving_command_threshold_mps,
                yaw_command_active,
                reference_nm=(
                    self.cfg.inactive_bilateral_longitudinal_contact_moment_reference_nm
                ),
            )

        nominal_height = select_command_conditioned_nominal_height(
            commanded_planar_speed,
            moving_nominal_height_m=self.cfg.nominal_height_m,
            stand_nominal_height_m=self.cfg.stand_nominal_height_m,
            moving_command_threshold_mps=self.cfg.moving_command_threshold_mps,
        )
        base_height_error = (
            self._robot.data.root_pos_w.torch[:, 2] - nominal_height
        )
        deck_stability = normalized_deck_stability_reward(
            root_vertical_velocity_w,
            root_ang_vel[:, :2],
            projected_gravity[:, :2],
            base_height_error,
            vertical_velocity_scale_mps=(
                self.cfg.deck_stability_vertical_velocity_scale_mps
            ),
            roll_pitch_rate_scale_rad_s=(
                self.cfg.deck_stability_roll_pitch_rate_scale_rad_s
            ),
            projected_gravity_xy_scale=(
                self.cfg.deck_stability_projected_gravity_xy_scale
            ),
            height_error_scale_m=self.cfg.deck_stability_height_error_scale_m,
        )

        coxa_forces = self._coxa_contact_sensor.data.net_forces_w_history.torch
        femur_forces = torch.cat(
            [sensor.data.net_forces_w_history.torch for sensor in self._femur_contact_sensors],
            dim=2,
        )
        undesired_contact = torch.cat(
            (
                torch.linalg.norm(coxa_forces, dim=-1).max(dim=1)[0] > 1.0,
                torch.linalg.norm(femur_forces, dim=-1).max(dim=1)[0] > 1.0,
                shaft_contact,
            ),
            dim=1,
        )
        undesired_contact_count = torch.sum(undesired_contact, dim=1)

        soft_limits = self._robot.data.soft_joint_pos_limits.torch
        below_limit = torch.relu(soft_limits[:, :, 0] - self._robot.data.joint_pos.torch)
        above_limit = torch.relu(self._robot.data.joint_pos.torch - soft_limits[:, :, 1])
        joint_limit_error = torch.sum(below_limit + above_limit, dim=1)

        if self.cfg.axiswise_velocity_rewards:
            track_lin_vel_xy = torch.zeros(self.num_envs, device=self.device)
            if self.cfg.gate_longitudinal_reward_by_command:
                track_lin_vel_x = active_axis_gaussian_tracking_reward(
                    self._commands[:, 0],
                    root_lin_vel[:, 0],
                    tracking_std=self.cfg.lin_vel_x_tracking_std_mps,
                    reward_scale=self.cfg.lin_vel_x_reward_scale,
                    active_threshold=self.cfg.axis_command_active_threshold,
                )
            else:
                track_lin_vel_x = torch.exp(
                    -lin_vel_axis_error[:, 0]
                    / self.cfg.lin_vel_x_tracking_std_mps**2
                ) * self.cfg.lin_vel_x_reward_scale
            if self.cfg.gate_axis_rewards_by_command:
                track_lin_vel_y = active_axis_gaussian_tracking_reward(
                    self._commands[:, 1],
                    root_lin_vel[:, 1],
                    tracking_std=self.cfg.lin_vel_y_tracking_std_mps,
                    reward_scale=self.cfg.lin_vel_y_reward_scale,
                    active_threshold=self.cfg.axis_command_active_threshold,
                )
            else:
                track_lin_vel_y = torch.exp(
                    -lin_vel_axis_error[:, 1]
                    / self.cfg.lin_vel_y_tracking_std_mps**2
                ) * self.cfg.lin_vel_y_reward_scale
        else:
            track_lin_vel_xy = (
                torch.exp(-lin_vel_error / self.cfg.lin_vel_tracking_std_mps**2)
                * self.cfg.lin_vel_reward_scale
            )
            track_lin_vel_x = torch.zeros(self.num_envs, device=self.device)
            track_lin_vel_y = torch.zeros(self.num_envs, device=self.device)

        if self.cfg.gate_axis_rewards_by_command:
            track_ang_vel_z = active_axis_gaussian_tracking_reward(
                self._commands[:, 2],
                root_ang_vel[:, 2],
                tracking_std=self.cfg.yaw_rate_tracking_std_rad_s,
                reward_scale=self.cfg.yaw_rate_reward_scale,
                active_threshold=self.cfg.axis_command_active_threshold,
            )
        else:
            track_ang_vel_z = torch.exp(
                -yaw_rate_error / self.cfg.yaw_rate_tracking_std_rad_s**2
            ) * self.cfg.yaw_rate_reward_scale

        longitudinal_signed_progress = normalized_signed_axis_progress(
            self._commands[:, 0],
            root_lin_vel[:, 0],
            active_threshold=self.cfg.axis_command_active_threshold,
        ) * self.cfg.longitudinal_signed_progress_reward_scale
        normalized_longitudinal_error = capped_normalized_axis_error(
            self._commands[:, 0],
            root_lin_vel[:, 0],
            active_threshold=self.cfg.axis_command_active_threshold,
            error_cap=self.cfg.longitudinal_normalized_error_cap,
        )
        longitudinal_normalized_error_penalty = (
            -normalized_longitudinal_error
            * self.cfg.longitudinal_normalized_error_penalty_scale
        )
        lateral_signed_progress = normalized_signed_axis_progress(
            self._commands[:, 1],
            lateral_velocity_for_shaping,
            active_threshold=self.cfg.axis_command_active_threshold,
        ) * self.cfg.lateral_signed_progress_reward_scale
        normalized_lateral_error = capped_normalized_axis_error(
            self._commands[:, 1],
            lateral_velocity_for_shaping,
            active_threshold=self.cfg.axis_command_active_threshold,
            error_cap=self.cfg.lateral_normalized_error_cap,
        )
        lateral_normalized_error_penalty = (
            -normalized_lateral_error
            * self.cfg.lateral_normalized_error_penalty_scale
        )
        yaw_signed_progress = normalized_signed_axis_progress(
            self._commands[:, 2],
            root_ang_vel[:, 2],
            active_threshold=self.cfg.axis_command_active_threshold,
        ) * self.cfg.yaw_signed_progress_reward_scale

        rewards = {
            "track_lin_vel_xy_exp": track_lin_vel_xy,
            "track_lin_vel_x_exp": track_lin_vel_x,
            "track_lin_vel_y_exp": track_lin_vel_y,
            "track_ang_vel_z_exp": track_ang_vel_z,
            "longitudinal_signed_progress": longitudinal_signed_progress,
            "longitudinal_normalized_error_penalty": (
                longitudinal_normalized_error_penalty
            ),
            "lateral_signed_progress": lateral_signed_progress,
            "lateral_normalized_error_penalty": lateral_normalized_error_penalty,
            "yaw_signed_progress": yaw_signed_progress,
            "inactive_lin_vel_y_l2": (
                torch.square(root_lin_vel[:, 1])
                * (~lateral_command_active).float()
                * self.cfg.inactive_lateral_velocity_reward_scale
            ),
            "inactive_ang_vel_z_l2": (
                torch.square(root_ang_vel[:, 2])
                * (~yaw_command_active).float()
                * self.cfg.inactive_yaw_rate_reward_scale
            ),
            "alive": torch.ones(self.num_envs, device=self.device) * self.cfg.alive_reward_scale,
            "lin_vel_z_l2": torch.square(root_lin_vel[:, 2]) * self.cfg.z_vel_reward_scale,
            "ang_vel_xy_l2": torch.sum(torch.square(root_ang_vel[:, :2]), dim=1)
            * self.cfg.ang_vel_reward_scale,
            "joint_torques_l2": torch.sum(torch.square(joint_torque), dim=1)
            * self.cfg.joint_torque_reward_scale,
            "rated_torque_excess_l2": torch.sum(torch.square(torque_excess), dim=1)
            * self.cfg.rated_torque_excess_reward_scale,
            "torque_saturation_cost": torque_saturation_fraction
            * self.cfg.torque_saturation_reward_scale,
            "joint_acc_l2": torch.sum(torch.square(self._robot.data.joint_acc.torch), dim=1)
            * self.cfg.joint_accel_reward_scale,
            "action_rate_l2": torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
            * self.cfg.action_rate_reward_scale,
            "joint_torque_slew_l2": joint_torque_slew
            * self.cfg.joint_torque_slew_reward_scale,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale,
            "foot_slip_l2": foot_slip * self.cfg.foot_slip_reward_scale,
            "undesired_contacts": undesired_contact_count * self.cfg.undesired_contact_reward_scale,
            "flat_orientation_l2": torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
            * self.cfg.flat_orientation_reward_scale,
            "base_height_l2": torch.square(
                base_height_error
            )
            * self.cfg.base_height_reward_scale,
            "joint_limits_l2": joint_limit_error * self.cfg.joint_limit_reward_scale,
            "support_shortfall": support_shortfall * self.cfg.support_shortfall_reward_scale,
            "deck_stability": deck_stability * self.cfg.deck_stability_reward_scale,
        }
        # Keep the default reward tensor exactly unchanged: unlike a
        # multiply-by-zero path, this also stays dormant for non-finite input.
        if self.cfg.gait_phase_contact_reward_scale != 0.0:
            assert gait_phase_contact is not None
            rewards["gait_phase_contact"] = (
                gait_phase_contact * self.cfg.gait_phase_contact_reward_scale
            )
        if self.cfg.swing_clearance_reward_scale != 0.0:
            assert swing_clearance is not None
            rewards["swing_clearance"] = (
                swing_clearance * self.cfg.swing_clearance_reward_scale
            )
        if self.cfg.max_joint_rated_torque_excess_reward_scale != 0.0:
            rewards["max_joint_rated_torque_excess_l2"] = (
                max_joint_rated_torque_excess_l2(
                    computed_torque,
                    rated_torque_nm=self.cfg.rated_torque_nm,
                )
                * self.cfg.max_joint_rated_torque_excess_reward_scale
            )
        if self.cfg.max_joint_rated_torque_excess_l1_reward_scale != 0.0:
            rewards["max_joint_rated_torque_excess_l1"] = (
                max_joint_rated_torque_excess_l1(
                    computed_torque,
                    rated_torque_nm=self.cfg.rated_torque_nm,
                )
                * self.cfg.max_joint_rated_torque_excess_l1_reward_scale
            )
        if self.cfg.inactive_yaw_rate_slew_reward_scale != 0.0:
            assert inactive_yaw_rate_slew is not None
            rewards["inactive_yaw_rate_slew"] = (
                inactive_yaw_rate_slew
                * self.cfg.inactive_yaw_rate_slew_reward_scale
            )
        if self.cfg.inactive_ground_contact_yaw_moment_reward_scale != 0.0:
            assert inactive_ground_contact_yaw_moment is not None
            rewards["inactive_ground_contact_yaw_moment"] = (
                inactive_ground_contact_yaw_moment
                * self.cfg.inactive_ground_contact_yaw_moment_reward_scale
            )
        if (
            self.cfg.inactive_bilateral_longitudinal_contact_moment_reward_scale
            != 0.0
        ):
            assert inactive_bilateral_longitudinal_contact_moment is not None
            rewards["inactive_bilateral_longitudinal_contact_moment"] = (
                inactive_bilateral_longitudinal_contact_moment
                * self.cfg.inactive_bilateral_longitudinal_contact_moment_reward_scale
            )
        scaled = {name: value * self.step_dt for name, value in rewards.items()}
        for name, value in scaled.items():
            self._episode_sums[name] += value
        # A fall is an event, not a per-second cost. Multiplying this by
        # ``step_dt`` would make the configured penalty fifty times weaker.
        fall_penalty = self.cfg.fall_penalty * self.reset_terminated.float()
        self._episode_sums["fall_penalty"] += fall_penalty
        self._episode_sums["torque_saturation_fraction"] += (
            torque_saturation_fraction * self.step_dt
        )
        self._episode_sums["navigation_lateral_velocity_ema_mps"] += (
            lateral_velocity_ema_metric * self.step_dt
        )
        self._episode_sums["normalized_longitudinal_error"] += (
            normalized_longitudinal_error * self.step_dt
        )
        self._episode_sums["normalized_lateral_error"] += (
            normalized_lateral_error * self.step_dt
        )
        self._episode_sums["joint_target_slew_limited_fraction"] += (
            self._joint_target_slew_limited_fraction * self.step_dt
        )
        self._episode_sums["ground_contact_yaw_moment_abs_nm"] += (
            ground_contact_yaw_moment_abs_nm * self.step_dt
        )
        self._episode_sums["distal_friction_xy_force_n"] += (
            distal_friction_xy_force_n * self.step_dt
        )
        self._episode_sums["bilateral_longitudinal_contact_moment_nm"] += (
            bilateral_longitudinal_contact_moment_nm * self.step_dt
        )
        self._episode_elapsed_s += self.step_dt

        # Metrics are accumulated only while a movement command is active, so
        # future stand-command episodes cannot dilute forward tracking quality.
        commanded_forward_velocity = self._commands[:, 0]
        achieved_forward_velocity = torch.nan_to_num(
            root_lin_vel[:, 0], nan=0.0, posinf=0.0, neginf=0.0
        )
        moving_command = (
            torch.abs(commanded_forward_velocity) > self.cfg.moving_command_threshold_mps
        )
        moving_command_float = moving_command.float()
        self._moving_command_elapsed_s += moving_command_float * self.step_dt
        self._forward_metric_sums["commanded_velocity_x_mps"] += (
            commanded_forward_velocity * moving_command_float * self.step_dt
        )
        self._forward_metric_sums["achieved_velocity_x_mps"] += (
            achieved_forward_velocity * moving_command_float * self.step_dt
        )
        forward_velocity_error = torch.abs(
            commanded_forward_velocity - achieved_forward_velocity
        )
        self._forward_metric_sums["abs_velocity_error_x_mps"] += (
            forward_velocity_error * moving_command_float * self.step_dt
        )
        self._forward_metric_sums["tracking_within_tolerance_fraction"] += (
            (forward_velocity_error < self.cfg.forward_tracking_tolerance_mps).float()
            * moving_command_float
            * self.step_dt
        )
        # DirectRLEnv computes rewards before observations.  Resampling here
        # makes the old command own this transition's reward and exposes the
        # replacement command in the immediately following observation.
        self._resample_commands_if_due()
        return torch.sum(torch.stack(tuple(scaled.values())), dim=0) + fall_penalty

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_forces = self._base_contact_sensor.data.net_forces_w_history.torch
        base_contact = torch.linalg.norm(base_forces, dim=-1).max(dim=1)[0][:, 0] > 5.0
        too_low = self._robot.data.root_pos_w.torch[:, 2] < 0.055
        upside_down = self._robot.data.projected_gravity_b.torch[:, 2] > -0.45
        torque_demand_limit = getattr(
            self.cfg, "terminate_on_computed_torque_demand_nm", None
        )
        if torque_demand_limit is None:
            excessive_torque_demand = torch.zeros_like(base_contact)
            self._torque_demand_excess_duration_s.zero_()
        else:
            excessive_torque_now = torch.any(
                torch.abs(self._robot.data.computed_torque.torch)
                > float(torque_demand_limit),
                dim=1,
            )
            grace_s = float(
                getattr(self.cfg, "torque_demand_termination_grace_s", 0.0)
            )
            excessive_torque_now &= self._episode_elapsed_s >= grace_s
            self._torque_demand_excess_duration_s = torch.where(
                excessive_torque_now,
                self._torque_demand_excess_duration_s + self.step_dt,
                torch.zeros_like(self._torque_demand_excess_duration_s),
            )
            required_duration_s = float(
                getattr(
                    self.cfg,
                    "terminate_on_computed_torque_demand_duration_s",
                    0.0,
                )
            )
            excessive_torque_demand = (
                excessive_torque_now
                if required_duration_s <= 0.0
                else self._torque_demand_excess_duration_s
                >= required_duration_s
            )
        return (
            base_contact | too_low | upside_down | excessive_torque_demand,
            time_out,
        )

    def _sample_commands(self, env_ids: torch.Tensor):
        """Sample commands, preserving the exact legacy Phase 1 path."""

        if self._velocity_command_cfg is None:
            # Indexing with ``self._commands[env_ids, column]`` is advanced
            # indexing and returns a copy. Sample a contiguous block first.
            sampled_commands = torch.empty(
                (len(env_ids), self._commands.shape[1]),
                dtype=self._commands.dtype,
                device=self.device,
            )
            sampled_commands[:, 0].uniform_(*self.cfg.command_lin_vel_x_range_mps)
            sampled_commands[:, 1].uniform_(*self.cfg.command_lin_vel_y_range_mps)
            sampled_commands[:, 2].uniform_(*self.cfg.command_yaw_rate_range_rad_s)
            stand_mask = (
                torch.rand(len(env_ids), device=self.device)
                < self.cfg.stand_command_fraction
            )
            sampled_commands[stand_mask] = 0.0
        else:
            command_cfg = self._velocity_command_cfg
            sampling_mode = getattr(command_cfg, "sampling_mode", "mixture")
            if sampling_mode == "stage1_recovery":
                sampled_commands, _ = sample_stage1_recovery_commands(
                    env_ids,
                    forward_range=command_cfg.forward_range,
                    moving_forward_range=command_cfg.moving_forward_range,
                    lateral_abs_range=command_cfg.lateral_abs_range,
                    yaw_abs_range=command_cfg.yaw_abs_range,
                    combined_forward_range=command_cfg.combined_forward_range,
                    combined_lateral_abs_range=command_cfg.combined_lateral_abs_range,
                    combined_yaw_abs_range=command_cfg.combined_yaw_abs_range,
                    dtype=self._commands.dtype,
                )
            elif sampling_mode == "stage2_recovery":
                sampled_commands, _ = sample_stage2_recovery_commands(
                    env_ids,
                    forward_range=command_cfg.forward_range,
                    moving_forward_range=command_cfg.moving_forward_range,
                    lateral_abs_range=command_cfg.lateral_abs_range,
                    yaw_abs_range=command_cfg.yaw_abs_range,
                    dtype=self._commands.dtype,
                )
            elif sampling_mode == "stage2b_lateral_acquisition":
                sampled_commands, _ = sample_stage2b_lateral_acquisition_commands(
                    env_ids,
                    forward_range=command_cfg.forward_range,
                    lateral_only_abs_range=command_cfg.lateral_only_abs_range,
                    combined_forward_range=command_cfg.combined_forward_range,
                    combined_lateral_abs_range=command_cfg.combined_lateral_abs_range,
                    yaw_forward_range=command_cfg.yaw_forward_range,
                    yaw_abs_range=command_cfg.yaw_abs_range,
                    dtype=self._commands.dtype,
                )
            elif sampling_mode == "stage2d_oblique_homotopy":
                sampled_commands, _ = sample_stage2d_oblique_homotopy_commands(
                    env_ids,
                    oblique_forward_range=command_cfg.oblique_forward_range,
                    oblique_lateral_abs_range=(
                        command_cfg.oblique_lateral_abs_range
                    ),
                    forward_anchor_range=command_cfg.forward_anchor_range,
                    yaw_anchor_forward_range=(
                        command_cfg.yaw_anchor_forward_range
                    ),
                    yaw_anchor_abs_range=command_cfg.yaw_anchor_abs_range,
                    dtype=self._commands.dtype,
                )
            elif sampling_mode == "stage2e_joystick_transitions":
                sampled_commands, _ = sample_stage2e_joystick_transition_commands(
                    env_ids,
                    resample_epochs=self._command_resample_epoch[env_ids],
                    bucket_counts=command_cfg.bucket_counts,
                    bucket_stride=command_cfg.bucket_stride,
                    lateral_anchor_abs_range=(
                        command_cfg.lateral_anchor_abs_range
                    ),
                    forward_anchor_range=command_cfg.forward_anchor_range,
                    yaw_anchor_forward_range=(
                        command_cfg.yaw_anchor_forward_range
                    ),
                    yaw_anchor_abs_range=command_cfg.yaw_anchor_abs_range,
                    joystick_forward_range=command_cfg.joystick_forward_range,
                    joystick_reverse_abs_range=(
                        command_cfg.joystick_reverse_abs_range
                    ),
                    joystick_lateral_abs_range=(
                        command_cfg.joystick_lateral_abs_range
                    ),
                    joystick_yaw_abs_range=(
                        command_cfg.joystick_yaw_abs_range
                    ),
                    dtype=self._commands.dtype,
                )
            elif sampling_mode == "mixture":
                sampled_commands, _ = sample_velocity_command_mixture(
                    len(env_ids),
                    lin_vel_x_range=command_cfg.lin_vel_x_range,
                    lin_vel_y_range=command_cfg.lin_vel_y_range,
                    ang_vel_z_range=command_cfg.ang_vel_z_range,
                    category_probabilities=(
                        command_cfg.standing_probability,
                        command_cfg.longitudinal_only_probability,
                        command_cfg.lateral_only_probability,
                        command_cfg.yaw_only_probability,
                        command_cfg.combined_probability,
                    ),
                    device=self.device,
                    dtype=self._commands.dtype,
                )
            else:
                raise ValueError(f"Unsupported velocity-command sampling mode: {sampling_mode!r}")
        self._commands[env_ids] = sampled_commands

    def _reset_command_resampling_timers(self, env_ids: torch.Tensor):
        if self._command_resampling_time_left_s is None:
            return
        self._command_resampling_time_left_s[env_ids] = sample_uniform_intervals(
            len(env_ids),
            self._velocity_command_cfg.resampling_time_range_s,
            device=self.device,
            dtype=self._command_resampling_time_left_s.dtype,
        )

    def _resample_commands_if_due(self):
        if self._command_resampling_time_left_s is None:
            return
        active = ~self.reset_buf
        updated, due = advance_resampling_timers(
            self._command_resampling_time_left_s, active, self.step_dt
        )
        self._command_resampling_time_left_s = updated
        due_env_ids = torch.nonzero(due, as_tuple=False).squeeze(-1)
        if len(due_env_ids) == 0:
            return
        self._command_resample_epoch[due_env_ids] += 1
        self._sample_commands(due_env_ids)
        next_intervals = sample_uniform_intervals(
            len(due_env_ids),
            self._velocity_command_cfg.resampling_time_range_s,
            device=self.device,
            dtype=self._command_resampling_time_left_s.dtype,
        )
        # Retain the sub-step overshoot so repeated hold times do not drift.
        self._command_resampling_time_left_s[due_env_ids] += next_intervals

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        episode_elapsed_s = self._episode_elapsed_s[env_ids].clone()
        moving_command_elapsed_s = self._moving_command_elapsed_s[env_ids].clone()
        forward_metric_sums = {
            name: values[env_ids].clone()
            for name, values in self._forward_metric_sums.items()
        }
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_applied_torque[env_ids] = 0.0
        self._has_previous_applied_torque[env_ids] = False
        self._previous_command_frame_yaw_rate[env_ids] = 0.0
        self._has_previous_command_frame_yaw_rate[env_ids] = False
        self._previous_processed_joint_target[env_ids] = 0.0
        self._has_previous_processed_joint_target[env_ids] = False
        self._joint_target_slew_limited_fraction[env_ids] = 0.0
        self._navigation_lateral_velocity_ema[env_ids] = 0.0
        self._has_navigation_lateral_velocity_ema[env_ids] = False
        self._torque_demand_excess_duration_s[env_ids] = 0.0
        self._command_resample_epoch[env_ids] = 0
        if self._gait_shaping_enabled:
            # Randomized initial phase decorrelates the fleet's gait clocks;
            # the seeded evaluator makes playback reproducible regardless.
            self._gait_phase[env_ids] = torch.rand(
                len(env_ids), device=self.device
            )

        self._sample_commands(env_ids)
        self._reset_command_resampling_timers(env_ids)

        joint_pos = self._robot.data.default_joint_pos.torch[env_ids]
        joint_pos += torch.empty_like(joint_pos).uniform_(-0.03, 0.03)
        if self._processed_joint_target_slew_limit_rad_per_20ms is not None:
            # Anchor the first limited target to the new episode's actual
            # randomized reset pose, never to a stale target from the prior
            # episode.
            self._previous_processed_joint_target[env_ids] = joint_pos
            self._has_previous_processed_joint_target[env_ids] = True
        joint_vel = self._robot.data.default_joint_vel.torch[env_ids]
        default_root_pose = self._robot.data.default_root_pose.torch[env_ids]
        default_root_vel = self._robot.data.default_root_vel.torch[env_ids]
        default_root_pose[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        self._robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        self.extras["log"] = {}
        bilateral_reward_scale = float(
            self.cfg.inactive_bilateral_longitudinal_contact_moment_reward_scale
        )
        if bilateral_reward_scale != 0.0:
            # The raw metric is already integrated independently above.  The
            # bounded Cauchy cost is recovered from its existing scaled reward
            # integral, so telemetry cannot perturb the reward tensor or
            # dynamics.  Percentiles are over per-environment episode means in
            # this reset batch, matching the existing Episode_Metric mean.
            _, bilateral_moment_p50 = reset_batch_time_mean_and_p50(
                self._episode_sums[
                    "bilateral_longitudinal_contact_moment_nm"
                ][env_ids],
                episode_elapsed_s,
                min_elapsed_s=self.step_dt,
            )
            bilateral_cost_mean, bilateral_cost_p50 = reset_batch_time_mean_and_p50(
                self._episode_sums[
                    "inactive_bilateral_longitudinal_contact_moment"
                ][env_ids]
                / bilateral_reward_scale,
                episode_elapsed_s,
                min_elapsed_s=self.step_dt,
            )
            self.extras["log"][
                "Episode_Metric/bilateral_longitudinal_contact_moment_nm_p50"
            ] = bilateral_moment_p50
            self.extras["log"][
                "Episode_Metric/"
                "bilateral_longitudinal_contact_moment_cauchy_cost_mean"
            ] = bilateral_cost_mean
            self.extras["log"][
                "Episode_Metric/"
                "bilateral_longitudinal_contact_moment_cauchy_cost_p50"
            ] = bilateral_cost_p50
        for name, values in self._episode_sums.items():
            category = (
                "Episode_Metric"
                if name in self._episode_metric_names
                else "Episode_Reward"
            )
            if name in self._episode_metric_names:
                elapsed = torch.clamp(episode_elapsed_s, min=self.step_dt)
                logged_value = torch.mean(values[env_ids] / elapsed)
            else:
                logged_value = torch.mean(values[env_ids]) / self.max_episode_length_s
            self.extras["log"][f"{category}/{name}"] = logged_value
            values[env_ids] = 0.0

        valid_movement_episode = moving_command_elapsed_s > 0.0
        valid_movement_count = torch.clamp(
            valid_movement_episode.float().sum(), min=1.0
        )
        safe_movement_elapsed_s = torch.clamp(
            moving_command_elapsed_s, min=self.step_dt
        )
        forward_metric_means = {
            name: values / safe_movement_elapsed_s
            for name, values in forward_metric_sums.items()
        }
        for name, values in forward_metric_means.items():
            self.extras["log"][f"Forward/{name}"] = torch.sum(
                torch.where(valid_movement_episode, values, 0.0)
            ) / valid_movement_count

        commanded_integral = forward_metric_sums["commanded_velocity_x_mps"]
        achieved_integral = forward_metric_sums["achieved_velocity_x_mps"]
        safe_commanded_integral = torch.where(
            torch.abs(commanded_integral) > 1.0e-6,
            commanded_integral,
            torch.ones_like(commanded_integral),
        )
        tracking_ratio = torch.clamp(
            achieved_integral / safe_commanded_integral, min=-1.0, max=2.0
        )
        self.extras["log"]["Forward/tracking_ratio_clipped"] = torch.sum(
            torch.where(valid_movement_episode, tracking_ratio, 0.0)
        ) / valid_movement_count
        self.extras["log"]["Forward/moving_command_time_fraction"] = torch.mean(
            torch.where(
                episode_elapsed_s > 0.0,
                moving_command_elapsed_s
                / torch.clamp(episode_elapsed_s, min=self.step_dt),
                0.0,
            )
        )

        self._episode_elapsed_s[env_ids] = 0.0
        self._moving_command_elapsed_s[env_ids] = 0.0
        for values in self._forward_metric_sums.values():
            values[env_ids] = 0.0
        self.extras["log"]["Episode_Termination/fallen"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
