"""Flat omnidirectional direct-joint PPO task, isolated from archived rewards."""
import torch
from isaaclab.utils import math as math_utils
from hexapod_rl.env import HexapodEnv
from hexapod_rl.env import limit_processed_joint_target_slew
from omni_flat_math import ObservationHistory, sample_commands, slew_commands, tracking_terms


def quiet_stand_terms(commands, joint_velocity, target, previous_target, step_dt):
    """Penalize motion only during zero/near-zero commands; retain feedback control."""
    standing = (commands[:, :2].norm(dim=-1) <= .03) & (commands[:, 2].abs() <= .05)
    return {"stand_joint_velocity": joint_velocity.square().mean(-1) * standing,
            "stand_target_velocity": ((target - previous_target) / step_dt).square().mean(-1) * standing}


class OmniFlatEnv(HexapodEnv):
    def __init__(self, cfg, render_mode=None, *, evaluation=False, **kwargs):
        self.omni_evaluation = evaluation
        super().__init__(cfg, render_mode, **kwargs)
        self.omni_step = 0
        self.omni_targets = torch.zeros_like(self._commands)
        self.omni_timer = torch.zeros(self.num_envs, device=self.device)
        self.omni_history = ObservationHistory(self.num_envs, 5, 63, self.device)
        self.omni_previous_action = torch.zeros_like(self._actions)
        self.omni_previous_velocity = torch.zeros_like(self._actions)
        self.omni_previous_target = self._previous_processed_joint_target.clone()
        self.omni_filtered_target = self._previous_processed_joint_target.clone()
        self.omni_noise = torch.zeros(self.num_envs, 63, device=self.device)
        self.omni_noise_step = -1
        self.omni_sums = {}

    def _sample_commands(self, env_ids):
        # Called by the inherited physical reset. Initial command is stationary,
        # then the next observations receive a bounded ramp toward the target.
        self._commands[env_ids] = 0
        if hasattr(self, "omni_targets"):
            self.omni_targets[env_ids] = (0 if self.omni_evaluation else
                                          sample_commands(len(env_ids), self.device))
            self.omni_timer[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(3., 6.)

    def set_evaluation_targets(self, targets):
        if not self.omni_evaluation:
            raise RuntimeError("External evaluation commands cannot override a training run")
        self.omni_targets.copy_(targets)

    def _pre_physics_step(self, actions):
        self.omni_raw_policy_action = actions.clone()
        if getattr(self, "omni_diagnostic_enabled", False):
            self.omni_diagnostic_start_position = self._robot.data.root_pos_w.torch.clone()
            self.omni_diagnostic_start_quaternion = self._robot.data.root_quat_w.torch.clone()
        self.omni_step += 1
        self.omni_previous_action.copy_(self._actions)
        self.omni_previous_velocity.copy_(self._robot.data.joint_vel.torch)
        # Inherited reset anchors this limiter state to the new physical pose.
        self.omni_previous_target.copy_(self._previous_processed_joint_target)
        super()._pre_physics_step(actions)
        if self.cfg.omni_target_filter_time_constant_s > 0:
            from omni_action_filter import filter_joint_targets
            # Low-pass the desired position before applying the existing hard
            # slew bound. Keep actor action history unchanged for this controlled
            # actuator-response comparison; no torque gain or cap is changed.
            limits = self._robot.data.soft_joint_pos_limits.torch
            requested = (self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos.torch).clamp(
                limits[:, :, 0], limits[:, :, 1])
            self.omni_filtered_target.copy_(filter_joint_targets(
                requested, self.omni_filtered_target, self.step_dt, self.cfg.omni_target_filter_time_constant_s))
            self._processed_actions, self._joint_target_slew_limited_fraction = limit_processed_joint_target_slew(
                self.omni_filtered_target, self.omni_previous_target, self._has_previous_processed_joint_target,
                max_delta_rad_per_20ms=self._processed_joint_target_slew_limit_rad_per_20ms, step_dt=self.step_dt)
            self._previous_processed_joint_target.copy_(self._processed_actions)

    def _get_observations(self):
        d = self._robot.data
        current = torch.cat((
            self._vector_in_command_frame(d.root_ang_vel_b.torch) * .25,
            self._vector_in_command_frame(d.projected_gravity_b.torch),
            self._commands * torch.tensor([5., 5., 2.5], device=self.device),
            d.joint_pos.torch - d.default_joint_pos.torch,
            d.joint_vel.torch * .05, self._actions), -1)
        if self.omni_noise_step != self.omni_step:
            scales = torch.zeros(63, device=self.device)
            scales[:3] = .00375; scales[3:6] = .01
            scales[9:27] = .005; scales[27:45] = .0025
            self.omni_noise = torch.randn_like(current) * scales * self.cfg.omni_observation_noise_scale
            self.omni_noise_step = self.omni_step
        actor = self.omni_history.observe(current + self.omni_noise, self.omni_step)
        # Velocity truth helps the critic only. The actor must infer motion from
        # IMU/encoder/action history; no contact truth, root pose, or phase enters it.
        critic = torch.cat((actor, self._vector_in_command_frame(d.root_lin_vel_b.torch) * 5.), -1)
        return {"policy": actor, "critic": critic}

    def _get_rewards(self):
        d = self._robot.data
        v = self._vector_in_command_frame(d.root_lin_vel_b.torch)
        w = self._vector_in_command_frame(d.root_ang_vel_b.torch)
        distal, shaft, slip = self._get_foot_contact_state()
        coxa = torch.linalg.vector_norm(self._coxa_contact_sensor.data.net_forces_w_history.torch, dim=-1).amax(1) > 1
        femur = torch.cat([torch.linalg.vector_norm(s.data.net_forces_w_history.torch, dim=-1).amax(1) > 1
                           for s in self._femur_contact_sensors], 1)
        nonfoot = torch.cat((coxa, femur, shaft), 1).float().sum(-1)
        moving = (self._commands[:, :2].norm(dim=-1) > .03) | (self._commands[:, 2].abs() > .05)
        terms = tracking_terms(v[:, :2], w[:, 2], self._commands)
        excess = ((d.computed_torque.torch.abs() - 1.6).clamp_min(0) / 1.6).clamp(max=10)
        first_contact = torch.cat([s.compute_first_contact(self.step_dt).torch for s in self._feet_contact_sensors], 1)
        air = torch.cat([s.data.last_air_time.torch for s in self._feet_contact_sensors], 1)
        limits = d.soft_joint_pos_limits.torch
        lo = (limits[:, :, 0] + .03 - d.joint_pos.torch).clamp_min(0)
        hi = (d.joint_pos.torch - limits[:, :, 1] + .03).clamp_min(0)
        terms.update({
            "vertical_velocity": d.root_lin_vel_w.torch[:, 2].square(),
            "roll_pitch_rate": w[:, :2].square().sum(-1),
            "tilt": d.projected_gravity_b.torch[:, :2].square().sum(-1),
            "height": (d.root_pos_w.torch[:, 2] - self.cfg.nominal_height_m).square(),
            "torque": d.applied_torque.torch.square().mean(-1),
            "torque_excess": excess.square().mean(-1),
            "worst_torque_excess": excess.amax(-1),
            "saturation": (d.computed_torque.torch.abs() > 1.6).float().mean(-1),
            "positive_power": (d.applied_torque.torch * d.joint_vel.torch).clamp_min(0).sum(-1),
            "action_rate": (self._actions - self.omni_previous_action).square().sum(-1),
            "joint_acceleration": ((d.joint_vel.torch - self.omni_previous_velocity) / self.step_dt).square().sum(-1),
            "slip": (slip.square() * distal).sum(-1),
            "nonfoot": nonfoot,
            "joint_limits": (lo + hi).sum(-1),
            "airtime": ((air - .20).clamp(-.20, .25) * first_contact).sum(-1) * moving,
            "stand_posture": ((d.joint_pos.torch - d.default_joint_pos.torch).square().mean(-1)) * ~moving,
        })
        terms.update(quiet_stand_terms(self._commands, d.joint_vel.torch, self._processed_actions,
                                      self.omni_previous_target, self.step_dt))
        from omni_repair_training import stand_raw_action_cost
        terms["stand_raw_action"] = stand_raw_action_cost(self._commands, self.omni_raw_policy_action)
        weights = self.cfg.omni_reward_weights
        reward = sum(terms[k] * weight for k, weight in weights.items()) * self.step_dt
        # This event penalty is not multiplied by dt: termination must not pay
        # by evading the remainder of a difficult command.
        reward -= 3. * self.reset_terminated.float()
        if getattr(self, "omni_diagnostic_enabled", False):
            from omni_diagnostics import capture_step
            capture_step(self, terms)
        if getattr(self, "omni_measurements_enabled", False):
            from omni_flat_evaluation import measurements
            self.omni_measurement_sample = {key: value.clone() for key, value in measurements(self).items()}
            self.omni_measurement_age_s = self._episode_elapsed_s.clone() + self.step_dt
            self.omni_measurement_position = d.root_pos_w.torch.clone()
        self._episode_elapsed_s += self.step_dt
        for name, value in terms.items():
            if name not in self.omni_sums:
                self.omni_sums[name] = torch.zeros_like(value)
            self.omni_sums[name] += value * self.step_dt
        if not self.omni_evaluation:
            self.omni_timer -= self.step_dt
            ids = torch.where(self.omni_timer <= 0)[0]
            if len(ids):
                self.omni_targets[ids] = sample_commands(len(ids), self.device)
                self.omni_timer[ids] = torch.empty(len(ids), device=self.device).uniform_(3., 6.)
        # Commands advance after scoring this action, before the next observation.
        self._commands.copy_(slew_commands(self._commands, self.omni_targets, self.step_dt))
        return reward

    def _reset_idx(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        elapsed = self._episode_elapsed_s[ids].clamp_min(self.step_dt).clone()
        summaries = {name: (value[ids] / elapsed).mean() for name, value in getattr(self, "omni_sums", {}).items()}
        super()._reset_idx(env_ids)
        if hasattr(self, "omni_history"):
            self.omni_history.reset(ids)
            self.omni_previous_action[ids] = 0
            self.omni_previous_velocity[ids] = 0
            self.omni_filtered_target[ids] = self._previous_processed_joint_target[ids]
            for value in self.omni_sums.values():
                value[ids] = 0
            # Avoid misleading archived forward-only telemetry in this task.
            self.extras["log"] = {"Omni/" + k: v for k, v in summaries.items()}
            self.extras["log"]["Omni/terminations"] = self.reset_terminated[ids].sum()


def configure_omni(cfg, overrides=None):
    cfg.observation_space = 315
    cfg.state_space = 318
    cfg.include_gait_phase_observation = False
    cfg.gait_phase_contact_reward_scale = 0.
    cfg.swing_clearance_reward_scale = 0.
    cfg.velocity_command = None
    cfg.command_frame = "navigation"
    cfg.action_scale = .50
    cfg.stand_action_scale = 1.
    cfg.processed_joint_target_slew_limit_rad_per_20ms = .06
    cfg.omni_target_filter_time_constant_s = 0.
    cfg.omni_observation_noise_scale = 1.
    cfg.terminate_on_computed_torque_demand_nm = 5.5
    cfg.terminate_on_computed_torque_demand_duration_s = .10
    cfg.torque_demand_termination_grace_s = .50
    cfg.omni_reward_weights = {
        "linear_tracking": 4., "yaw_tracking": 2., "linear_progress": 1.5, "yaw_progress": .75,
        "vertical_velocity": -2., "roll_pitch_rate": -.15, "tilt": -2., "height": -25.,
        "torque": -.03, "torque_excess": -.30, "worst_torque_excess": -.10, "saturation": -.25,
        "positive_power": -.015, "action_rate": -.025, "joint_acceleration": -2.5e-7,
        "slip": -.5, "nonfoot": -2., "joint_limits": -2., "airtime": .5, "stand_posture": -1.,
        # Opt-in repair terms preserve the frozen controller's reward defaults.
        "stand_joint_velocity": 0., "stand_target_velocity": 0., "stand_raw_action": 0.,
    }
    if overrides is not None:
        from omni_diagnostics import apply_omni_overrides
        apply_omni_overrides(cfg, overrides)
