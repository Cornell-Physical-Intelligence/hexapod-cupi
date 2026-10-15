"""Sample commands, apply the walking reward and report metrics around selected episode resets."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math

import torch

from locomotion.omni.commands import TaskConfig, command_bank
from locomotion.omni.frames import inverse_rotate, navigation
from locomotion.omni.reward import (CONTROL_DT_S, REWARD_CONFIG, REWARD_VERSION, SCHEMA, StrideWindow,
                                    displacement_velocity, measured_reward, scheduled_swing)

METRIC_COMMAND_CLASSES = ("zero", "linear_0.025", "linear_0.05", "yaw", "arc", "other")
METRIC_TERMINATION_REASONS = ("height", "tilt", "joint_limit", "timeout")
METRIC_SLEW_TOLERANCE_RAD = 2e-6


class TrainingTask:
    """Own command attribution, reward memory and explicit selected resets."""

    def __init__(self, env, config=None, output_dir=None, reward_config=None):
        chosen = REWARD_CONFIG if reward_config is None else reward_config
        self.gate_window = StrideWindow(env.num_envs, chosen.gate_window_controls, 1, env.device)
        self.stride_window = StrideWindow(env.num_envs, chosen.stride_window_controls, 3, env.device)
        self.liftoff_world = torch.zeros(env.num_envs, 6, 3, device=env.device)
        self.lifted = torch.zeros(env.num_envs, 6, dtype=torch.bool, device=env.device)

        task_config = TaskConfig() if config is None else config
        self.reward_config = REWARD_CONFIG if reward_config is None else reward_config
        self.reward_config.validate()
        n, device = env.num_envs, env.device
        self.previous_joint_velocity = env.current["dq"].detach().clone().to(device)
        self.previous_pose = env.current["root"].detach().clone().to(device)
        self.previous_toe_world = env.current["toe_world"].detach().clone().to(device)
        self.window = StrideWindow(n, self.reward_config.tracking_window_controls, 3, device)
        self.air_time = torch.zeros(n, 6, device=device)
        self.in_contact = torch.ones(n, 6, dtype=torch.bool, device=device)
        self.command_age = torch.zeros(n, dtype=torch.long, device=device)
        self.gait_totals = {}

        self.env = env
        self.config = task_config
        self.config.validate()
        if env.cfg.control_dt != .02 or env.cfg.decimation != 8:
            raise ValueError("Task requires the unchanged 50/400Hz environment")
        self.bank = torch.tensor(command_bank(self.config), dtype=torch.float32, device=env.device)
        self.rng = torch.Generator(device="cpu").manual_seed(self.config.seed)
        self.remaining_controls = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.previous_target = env.held.detach().clone()
        self.command_draws = 0
        self.zero_command_draws = 0
        self.controls_completed = 0
        self.last_components = {}
        self.last_held_command = None
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.metric_totals, self.metric_interval = {}, {}
        self.metric_rows = self.metric_interval_rows = 0
        self.metric_component_names = None
        self.metric_joint_names = tuple(env.joint_names)
        if len(self.metric_joint_names) != 18 or len(set(self.metric_joint_names)) != 18:
            raise ValueError("Target-slew reporting requires 18 unique named joints")
        self.metric_target_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.nominal_height = float(env.reference_metadata.get("root_height_m", env.reset_height))
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "task_definition.json"
            content = json.dumps(self.declaration(), indent=2, allow_nan=False) + "\n"
            if path.exists() and path.read_text() != content:
                raise ValueError("Existing task definition differs")
            path.write_text(content)

    def reset(self, indices=None):
        selected = self._indices(indices)
        output = self.env.reset(selected)
        self._resample(selected)
        self.previous_target[selected] = self.env.held[selected]
        self.metric_target_valid[selected] = False
        output = self._next_command_observation(output)
        self.previous_joint_velocity[selected] = self.env.current["dq"][selected].to(self.previous_joint_velocity)
        self.previous_pose[selected] = self.env.current["root"][selected].to(self.previous_pose)
        self.previous_toe_world[selected] = self.env.current["toe_world"][selected].detach().to(self.previous_toe_world)
        self.window.restart(selected, self.env.commands[selected])
        self.command_age[selected] = 0
        self.air_time[selected] = 0
        self.in_contact[selected] = True

        self.gate_window.restart(selected, self.env.commands[selected])
        self.stride_window.restart(selected, self.env.commands[selected])
        self.lifted[selected] = False
        return output

    def step(self, action):
        if bool((self.remaining_controls <= 0).any()):
            raise RuntimeError("Call task.reset before collecting a rollout")
        command = self.env.commands.detach().clone()
        output = self.env.step(action)
        reward, components = self._reward(command, output["terminated"])
        metric_previous_target = self.previous_target
        self.previous_target = self.env.telemetry["joint_target_rad"].detach().clone()
        self.last_held_command = command
        self.last_components = {key: value.detach().clone() for key, value in components.items()}
        self.remaining_controls -= 1
        done = output["terminated"] | output["truncated"]
        # Terminal observations remain attributed to their final command. The
        # learner supplies an explicit selected reset before collecting again.
        expired = ((self.remaining_controls == 0) & ~done).nonzero(as_tuple=False).squeeze(-1)
        self._resample(expired)
        self.controls_completed += 1
        self._metrics(command, output, reward, components, previous_target=metric_previous_target)
        result = self._next_command_observation(output)
        result["reward"] = reward
        return result

    def reward_telemetry(self, command):
        velocity = displacement_velocity(self.env.telemetry["root_pose_xyzw"].to(self.previous_pose),
                                                 self.previous_pose)
        stride_yaw = self.gate_window.update(velocity[:, 2:], command)[:, 0]
        stride = self.stride_window.update(velocity, command)
        # Liftoff and touchdown by the same contact threshold as v4, read before v4 advances its contact state.
        t = self.env.telemetry
        contact = torch.linalg.vector_norm(t["tibia_floor_force_world_n"], dim=-1).to(self.lifted.device) > self.reward_config.contact_force_n
        world = t["toe_xyz_world"].detach().to(self.liftoff_world)
        liftoff = ~contact & self.in_contact.to(contact.device)
        landed = contact & ~self.in_contact.to(contact.device) & self.lifted
        pose = t["root_pose_xyzw"].to(self.liftoff_world)
        travel = navigation(inverse_rotate(pose[:, None, 3:].expand(-1, 6, -1), world - self.liftoff_world))[..., :2]
        self.liftoff_world = torch.where(liftoff[..., None], world, self.liftoff_world)
        self.lifted = (self.lifted | liftoff) & ~(contact & ~self.in_contact.to(contact.device))

        t = self.env.telemetry
        pose = t["root_pose_xyzw"].to(self.previous_pose)
        changed = (command.to(self.window.command) != self.window.command).any(-1)
        self.command_age = torch.where(changed.to(self.command_age.device), torch.zeros_like(self.command_age), self.command_age) + 1
        tracked = self.window.update(displacement_velocity(pose, self.previous_pose), command)
        contact = torch.linalg.vector_norm(t["tibia_floor_force_world_n"], dim=-1) > self.reward_config.contact_force_n
        touchdown = contact & ~self.in_contact
        completed = self.air_time.clone()
        self.air_time = torch.where(contact, torch.zeros_like(self.air_time), self.air_time + CONTROL_DT_S)
        self.in_contact = contact
        # Toe velocity over the floor, expressed in the body's navigation axes.
        world = t["toe_xyz_world"].detach().to(self.previous_toe_world)
        body_axes = pose[:, None, 3:].expand(-1, 6, -1)
        toe_velocity = navigation(inverse_rotate(body_axes, (world - self.previous_toe_world) / CONTROL_DT_S))[..., :2]
        self.previous_toe_world = world.clone()
        q = t["joint_position_rad"].to(contact.device)
        # The control just completed started at episode step minus one.
        base = {**t, "tracked_velocity_nav": tracked, "previous_joint_velocity_rad_s": self.previous_joint_velocity,
                "previous_joint_target_rad": self.previous_target, "touchdown": touchdown, "air_time_s": completed,
                "toe_velocity_nav": toe_velocity, "toe_xyz_nav": navigation(t["toe_xyz_body"].detach())[..., :2],
                "scheduled_swing": scheduled_swing(self.env.episode_steps - 1, self.reward_config).to(contact.device),
                "joint_limit_margin_rad": torch.minimum(q - self.env.lower.to(q), self.env.upper.to(q) - q),
                "executed_action": (t["joint_target_rad"].to(q) - self.env.neutral.to(q)) / self.env.cfg.action_scale_rad,
                "command_age_controls": self.command_age.clone()}
        return {**base, "gate_yaw_rate_rad_s": stride_yaw, "stride_velocity_nav": stride,
                "stride_touchdown": landed, "stride_displacement_nav": torch.where(landed[..., None], travel, torch.zeros_like(travel))}

    def _reward(self, command, terminated):
        telemetry = self.reward_telemetry(command)
        reward, components = measured_reward(telemetry, command, terminated, self.reward_config,
                                                         self.nominal_height)
        self.previous_joint_velocity = self.env.telemetry["joint_velocity_rad_s"].detach().clone().to(self.previous_joint_velocity)
        self.previous_pose = self.env.telemetry["root_pose_xyzw"].detach().clone().to(self.previous_pose)
        self._gait_metrics(command, telemetry)
        return reward, components

    def _resample(self, indices):
        if not len(indices):
            return
        size = len(indices)
        bank_indices = torch.randint(len(self.bank), (size,), generator=self.rng)
        values = self.bank[bank_indices.to(self.env.device)].clone()
        # An integer quota avoids a finite pilot accidentally seeing no stops.
        before = math.ceil(self.config.zero_command_fraction*self.command_draws)
        after = math.ceil(self.config.zero_command_fraction*(self.command_draws+size))
        zero_count = after-before
        selected_zero = torch.randperm(size, generator=self.rng)[:zero_count].to(self.env.device)
        values[selected_zero] = 0
        durations = torch.randint(100, 251, (size,), generator=self.rng, dtype=torch.long).to(self.env.device)
        self.env.commands[indices] = values
        self.remaining_controls[indices] = durations
        self.command_draws += size
        self.zero_command_draws += zero_count
        share = self.reward_config.turn_draw_fraction
        if share and len(indices):
            chosen = (torch.rand(len(indices), generator=self.rng) < share).to(self.env.device)
            chosen &= (self.env.commands[indices] != 0).any(-1)
            sign = (torch.randint(0, 2, (len(indices),), generator=self.rng) * 2 - 1).to(self.env.commands)
            turns = torch.zeros(len(indices), 3, device=self.env.device, dtype=self.env.commands.dtype)
            turns[:, 2] = sign * self.config.yaw_rate_rad_s
            self.env.commands[indices[chosen]] = turns[chosen]

    def _indices(self, indices):
        result = torch.arange(self.env.num_envs, device=self.env.device) if indices is None else torch.as_tensor(indices, dtype=torch.long, device=self.env.device)
        if result.ndim != 1 or bool((result < 0).any()) or bool((result >= self.env.num_envs).any()) or len(result.unique()) != len(result):
            raise ValueError("Invalid command/reset row indices")
        return result

    def _next_command_observation(self, output):
        obs, critic = output["obs"].clone(), output["critic"].clone()
        if obs.shape != (self.env.num_envs, 231) or critic.shape != (self.env.num_envs, 234):
            raise ValueError("Task actor/critic ABI differs")
        if not torch.equal(critic[:, :231], obs):
            raise ValueError("Task received inconsistent actor/critic prefix")
        obs[:, 210:213] = self.env.commands
        critic[:, :231] = obs
        return {**output, "obs": obs, "critic": critic}

    def _metrics(self, command, output, reward, components, *, previous_target=None):
        t = self.env.telemetry
        actual = t["linear_velocity_nav"][:, :2]
        error = actual-command[:, :2]
        yaw_error = t["angular_velocity_body"][:, 2]-command[:, 2]
        requested_speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
        moving = requested_speed > 1e-8
        projection = (actual*command[:, :2]).sum(-1)/requested_speed.clamp_min(1e-8)
        zero = (command == 0).all(-1)
        turning = command[:, 2] != 0
        masks = [zero,
            moving & ~turning & torch.isclose(requested_speed, requested_speed.new_tensor(.025), rtol=0, atol=1e-6),
            moving & ~turning & torch.isclose(requested_speed, requested_speed.new_tensor(.05), rtol=0, atol=1e-6),
            ~moving & turning, moving & turning]
        masks.append(~torch.stack(masks).any(0))
        classes = torch.stack(masks).to(torch.float64)
        names = tuple(components)
        if self.metric_component_names is not None and names != self.metric_component_names:
            raise ValueError("Reward metric component names changed during an attempt")
        if reward.shape != (len(command),) or any(value.shape != reward.shape for value in components.values()):
            raise ValueError("Reward metric rows differ")
        component_values = torch.stack(tuple(components.values()), dim=-1).detach().to(torch.float64)
        reward_values = reward.detach().to(torch.float64)
        nonfoot = t["other_body_force_max_400hz"] > 1.
        # These are the existing env.py termination predicates, using its exact
        # returned gravity Z rather than recomputing orientation from a quaternion.
        terminated, truncated = output["terminated"], output["truncated"]
        height = t["root_pose_xyzw"][:, 2] < .045
        tilt = output["obs"][:, 173] > -math.cos(.85)
        q = t["joint_position_rad"]
        joint_limit = ((q < self.env.lower-2e-6) | (q > self.env.upper+2e-6)).any(-1)
        physical_reasons = torch.stack((height, tilt, joint_limit), dim=-1)
        reason_rows = torch.cat((physical_reasons & terminated[:, None], truncated[:, None]), dim=-1)
        actual_delta = (t["joint_target_rad"].to(torch.float64)
            - (self.previous_target if previous_target is None else previous_target).to(torch.float64)).abs()
        valid = self.metric_target_valid[:, None]
        slew_limit = float(self.env.cfg.target_slew_rad)
        at_limit = (actual_delta-slew_limit).abs() <= METRIC_SLEW_TOLERANCE_RAD
        # Reduce in float64 so long cumulative runs do not lose small component
        # contributions. These tensors never enter the learning objective.
        values = {"signed_navigation_velocity_error_mps": error.sum(0),
            "absolute_navigation_velocity_error_mps": error.abs().sum(0),
            "achieved_planar_speed_mps": torch.linalg.vector_norm(actual, dim=-1).sum(),
            "requested_planar_speed_mps": requested_speed.sum(),
            "signed_command_direction_speed_mps": projection.sum(),
            "moving_command_rows": moving.sum(), "signed_yaw_error_rad_s": yaw_error.sum(),
            "absolute_yaw_error_rad_s": yaw_error.abs().sum(),
            "requested_saturation_fraction": t["saturation_count_400hz"].sum()/(8*18),
            "terminations": output["terminated"].sum(), "truncations": output["truncated"].sum(),
            "zero_command_rows": zero.sum(), "nonfoot_event_rows": nonfoot.sum(),
            "_moving_projection_sum": projection[moving].to(torch.float64).sum(),
            "_reward_sum": reward_values.sum(),
            "_component_sums": component_values.sum(0),
            "_component_absolute_sums": component_values.abs().sum(0),
            "_class_rows": classes.sum(-1),
            "_class_reward_sums": classes @ reward_values,
            "_class_component_sums": classes @ component_values,
            "_class_component_absolute_sums": classes @ component_values.abs(),
            "_class_requested_speed_sums": classes @ requested_speed.to(torch.float64),
            "_class_projection_sums": classes @ (projection*moving).to(torch.float64),
            "_class_moving_rows": classes @ moving.to(torch.float64),
            "_class_nonfoot_rows": classes @ nonfoot.to(torch.float64),
            "_termination_reason_rows": reason_rows.sum(0),
            "_termination_multiple_reason_rows": (terminated & (physical_reasons.sum(-1)>1)).sum(),
            "_termination_unattributed_rows": (terminated & ~physical_reasons.any(-1)).sum(),
            "_termination_flag_mismatch_rows": (terminated != physical_reasons.any(-1)).sum(),
            "_slew_at_limit_rows": (at_limit & valid).sum(0),
            "_slew_beyond_limit_rows": ((actual_delta > slew_limit+METRIC_SLEW_TOLERANCE_RAD) & valid).sum(0),
            "_slew_valid_rows": self.metric_target_valid.sum(),
            "_slew_excluded_rows": (~self.metric_target_valid).sum(),
            "_zero_joint_rate_square_sums": t["joint_velocity_rad_s"][zero].to(torch.float64).square().sum(0)}
        if not bool(torch.isfinite(torch.cat([value.reshape(-1).to(torch.float64) for value in values.values()])).all()):
            raise FloatingPointError("Nonfinite task reporting metrics")
        self.metric_component_names = names
        for dest in (self.metric_totals, self.metric_interval):
            for key, value in values.items():
                dest[key] = dest.get(key, torch.zeros_like(value)) + value.detach()
        self.metric_rows += len(command)
        self.metric_interval_rows += len(command)
        self.metric_target_valid[:] = True

    def _gait_metrics(self, command, telemetry):
        """Reporting accumulators for stepping; these values never enter the learning objective."""
        moving = (command != 0).any(-1)
        tracked = telemetry["tracked_velocity_nav"].to(torch.float64)
        speed = torch.linalg.vector_norm(command[:, :2], dim=-1).to(torch.float64)
        along = (tracked[:, :2] * command[:, :2].to(torch.float64)).sum(-1) / speed.clamp_min(1e-8)
        translating = speed > 1e-8
        values = {"moving_rows": moving.sum().to(torch.float64),
            "moving_foot_contact_rows": (self.in_contact & moving[:, None]).sum(0).to(torch.float64),
            "moving_touchdowns": (telemetry["touchdown"] & moving[:, None]).sum(0).to(torch.float64),
            "moving_air_time_s": (telemetry["air_time_s"] * (telemetry["touchdown"] & moving[:, None])).sum(0).to(torch.float64),
            "translating_rows": translating.sum().to(torch.float64),
            "tracked_speed_ratio_sum": (along[translating] / speed[translating]).sum()}
        for key, value in values.items():
            self.gait_totals[key] = self.gait_totals.get(key, torch.zeros_like(value)) + value

    def status(self, *, reset_interval=False):
        def summary(values, rows):
            counts = {"moving_command_rows", "terminations", "truncations", "zero_command_rows", "nonfoot_event_rows"}
            report = {"environment_controls": rows, **{key: (value if key in counts else value/max(1, rows)).cpu().tolist()
                for key, value in values.items() if not key.startswith("_")}}
            if not rows:
                return report
            raw = {key: value.detach().cpu() for key, value in values.items() if key.startswith("_")}
            moving_rows, zero_rows = report["moving_command_rows"], report["zero_command_rows"]
            names = self.metric_component_names
            report.update(
                moving_signed_command_direction_speed_mps=float(raw["_moving_projection_sum"])/moving_rows if moving_rows else None,
                nonfoot_event_fraction=report["nonfoot_event_rows"]/rows,
                task_reward_sum=float(raw["_reward_sum"]), task_reward_mean=float(raw["_reward_sum"])/rows,
                reward_components={name: {"sum": float(raw["_component_sums"][i]),
                    "absolute_sum": float(raw["_component_absolute_sums"][i]),
                    "mean": float(raw["_component_sums"][i])/rows,
                    "mean_absolute": float(raw["_component_absolute_sums"][i])/rows} for i, name in enumerate(names)},
                command_classes={})
            for i, name in enumerate(METRIC_COMMAND_CLASSES):
                count = int(raw["_class_rows"][i])
                moving_count = int(raw["_class_moving_rows"][i])
                report["command_classes"][name] = {"environment_controls": count, "time_fraction": count/rows,
                    "task_reward_mean": float(raw["_class_reward_sums"][i])/count if count else None,
                    "reward_component_means": {key: float(raw["_class_component_sums"][i, j])/count if count else None
                        for j, key in enumerate(names)},
                    "reward_component_mean_absolute": {key: float(raw["_class_component_absolute_sums"][i, j])/count if count else None
                        for j, key in enumerate(names)},
                    "requested_planar_speed_mps": float(raw["_class_requested_speed_sums"][i])/count if count else None,
                    "moving_command_rows": moving_count,
                    "moving_signed_command_direction_speed_mps": float(raw["_class_projection_sums"][i])/moving_count if moving_count else None,
                    "nonfoot_event_fraction": float(raw["_class_nonfoot_rows"][i])/count if count else None}
            rms = (raw["_zero_joint_rate_square_sums"]/zero_rows).sqrt() if zero_rows else None
            report.update(zero_hold_joint_rate_rms_rad_s=None if rms is None else rms.tolist(),
                zero_hold_worst_joint_rate_rms_rad_s=None if rms is None else float(rms.max()),
                zero_hold_all_joint_rate_rms_rad_s=None if rms is None else float(rms.square().mean().sqrt()),
                diagnostic_scope="Time-weighted 50Hz control endpoints; nonfoot events use each control's maximum 400Hz body/coxa/femur floor force >1N. These are reporting metrics, not Stage2 admission.")
            slew_rows = int(raw["_slew_valid_rows"])
            report["termination_reasons"] = {"rows": {name:int(raw["_termination_reason_rows"][i])
                for i,name in enumerate(METRIC_TERMINATION_REASONS)},
                "multiple_physical_reason_rows":int(raw["_termination_multiple_reason_rows"]),
                "unattributed_termination_rows":int(raw["_termination_unattributed_rows"]),
                "termination_flag_mismatch_rows":int(raw["_termination_flag_mismatch_rows"]),
                "scope":"Counts returned terminal rows with each measured predicate; reasons may overlap. Timeout counts returned truncations separately."}
            report["actual_target_slew"] = {"environment_controls":slew_rows,
                "excluded_first_or_reset_rows":int(raw["_slew_excluded_rows"]),
                "limit_rad_per_control":float(self.env.cfg.target_slew_rad),
                "occupancy_tolerance_rad":METRIC_SLEW_TOLERANCE_RAD,
                "by_joint": {name:{"at_limit_rows":int(raw["_slew_at_limit_rows"][i]),
                    "at_limit_fraction":float(raw["_slew_at_limit_rows"][i])/slew_rows if slew_rows else None,
                    "beyond_limit_rows":int(raw["_slew_beyond_limit_rows"][i])} for i,name in enumerate(self.metric_joint_names)}}
            return report
        result = {"schema": SCHEMA, "reward_version": REWARD_VERSION, "controls_completed": self.controls_completed,
                "command_draws": self.command_draws, "zero_command_draws": self.zero_command_draws,
                "zero_draw_fraction": self.zero_command_draws/max(1, self.command_draws),
                "remaining_controls": self.remaining_controls.detach().cpu().tolist(),
                "commands": self.env.commands.detach().cpu().tolist(), "stage2_complete": False,
                "cumulative_metrics": summary(self.metric_totals, self.metric_rows),
                "interval_metrics": summary(self.metric_interval, self.metric_interval_rows)}
        if reset_interval:
            self.metric_interval, self.metric_interval_rows = {}, 0

        totals = {key: value.detach().cpu() for key, value in self.gait_totals.items()}
        if totals:
            rows, touchdowns = float(totals["moving_rows"]), totals["moving_touchdowns"]
            translating = float(totals["translating_rows"])
            result["interval_gait"] = {"moving_command_rows": rows,
                "foot_contact_fraction": (totals["moving_foot_contact_rows"] / max(1., rows)).tolist(),
                "touchdowns_per_foot_per_second": (touchdowns / max(1., rows) / CONTROL_DT_S).tolist(),
                "mean_air_time_s": (totals["moving_air_time_s"] / touchdowns.clamp_min(1.)).tolist(),
                "tracked_speed_ratio": float(totals["tracked_speed_ratio_sum"]) / translating if translating else None,
                "scope": "Moving-command controls in this interval; contact is the 50 Hz endpoint toe force above the "
                         "declared threshold. Reporting values, not acceptance."}
        if reset_interval:
            self.gait_totals = {}
        return result

    def declaration(self):
        return {"schema": SCHEMA, "config": asdict(self.config), "reward_version": REWARD_VERSION,
                "reward": {"version": REWARD_VERSION, "config": asdict(self.reward_config)},
                "nonzero_command_bank": self.bank.cpu().tolist(),
                "source_sha256": hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes()
                    for name in ("task.py", "reward.py", "commands.py"))).hexdigest(),
                "physics_changed": True, "stage2_gates_changed": False, "stage2_complete": False}

    def __getattr__(self, name):
        return getattr(self.env, name)
