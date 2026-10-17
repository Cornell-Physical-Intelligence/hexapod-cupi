"""Accumulate the training task's reporting metrics; no value enters the learning objective."""
import math

import torch

from locomotion.omniwalk.mdp.rewards import CONTROL_DT_S, REWARD_VERSION, SCHEMA

METRIC_COMMAND_CLASSES = ("zero", "linear_0.025", "linear_0.05", "yaw", "arc", "other")
METRIC_TERMINATION_REASONS = ("height", "tilt", "joint_limit", "timeout")
METRIC_SLEW_TOLERANCE_RAD = 2e-6


def scalars(value, prefix):
    """Return finite numeric leaves of a task status; drop lists, text and missing values."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(scalars(item, f'{prefix}/{key}'))
        return result
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return {prefix: value}
    return {}


class TaskMetrics:
    """Observe a TrainingTask and report its command, reward, termination, slew and gait metrics.

    The constructor binds this object as ``task.observer``. The task calls ``reset`` after it resamples the
    selected rows and ``step`` after each control. These are the points where the task updated these
    metrics before they moved here, so each reported value stays the same.
    """

    def __init__(self, task):
        env = task.env
        self.task, self.env = task, env
        self.metric_totals, self.metric_interval = {}, {}
        self.metric_rows = self.metric_interval_rows = 0
        self.metric_component_names = None
        self.metric_joint_names = tuple(env.joint_names)
        if len(self.metric_joint_names) != 18 or len(set(self.metric_joint_names)) != 18:
            raise ValueError("Target-slew reporting requires 18 unique named joints")
        self.metric_target_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.gait_totals = {}
        task.observer = self

    def reset(self, selected):
        self.metric_target_valid[selected] = False

    def step(self, command, output, reward, components, previous_target, telemetry):
        self._gait_metrics(command, telemetry)
        self._metrics(command, output, reward, components, previous_target)

    def _metrics(self, command, output, reward, components, previous_target):
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
        actual_delta = (t["joint_target_rad"].to(torch.float64) - previous_target.to(torch.float64)).abs()
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
            "moving_foot_contact_rows": (self.task.in_contact & moving[:, None]).sum(0).to(torch.float64),
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
        task = self.task
        result = {"schema": SCHEMA, "reward_version": REWARD_VERSION, "controls_completed": task.controls_completed,
                "command_draws": task.command_draws, "zero_command_draws": task.zero_command_draws,
                "zero_draw_fraction": task.zero_command_draws/max(1, task.command_draws),
                "remaining_controls": task.remaining_controls.detach().cpu().tolist(),
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
