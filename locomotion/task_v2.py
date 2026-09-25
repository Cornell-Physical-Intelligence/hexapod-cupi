"""Reward version 2: Liu et al. Table I task and penalty terms beside version 1.

``task.py`` keeps version 1 byte for byte (``test_recorded_parity`` freezes it),
so version 2 lives here. ``TrainingTaskV2`` reuses version 1's command sampling,
proximity guard and reporting and replaces only the reward. It adds per-replica
memory of the previous raw action, the previous joint velocity and, for the
stride kernel, a velocity window that restarts at each command change and reset.

``measured_reward_v2`` is pure: callers pass that memory in the telemetry, so
training, CPU tests and ``locomotion.reward_scorer`` evaluate one function.
Score it offline with ``--reward v2=locomotion.task_v2:scorer_default``, or
override defaults with ``--reward b=locomotion.task_v2:scorer_reward:tracking_kernel=scaled``.

The reviewer, James, approved the defaults on 2026-09-25: the decisions in
docs/REWARD_V2_TABLE1_AUDIT.md section 8 except the style term, which still
waits on the AMP owner. ``REVIEW`` records this in every task definition.
"""
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import math
from pathlib import Path

import torch

from .task import TaskConfig, TrainingTask, command_bank

REWARD_VERSION = "paper_table1_command_scaled_v2"
# Audit section 5 names the per-control command-scaled kernel B and its stride-mean form C.
KERNELS = {"scaled": "B", "stride": "C"}
CONTROL_DT_S = .02
SUBSTEPS = 8
ADAPTATIONS = (
    "Tracking uses exp(-||e||/sigma) with sigma = k * max(|c|, c_min), per component: the printed "
    "exponent has no minus sign, and the printed 0.15 scale pays a motionless robot 72 to 85 percent "
    "at this robot's commands (audit sections 2 and 4).",
    "The stride kernel tracks each replica's causal mean velocity over its last stride_controls "
    "controls, restarted at every command change and reset.",
    "Planar and vertical velocity are measured at the root-link origin, as version 1 does.",
    "||tau|| uses each joint's RMS applied torque over the control's eight physics substeps.",
    "||q_ddot|| is the joint-velocity change between consecutive 50 Hz controls divided by 0.02 s; "
    "the first control after a reset uses the reset state's joint velocity.",
    "The action rate uses the raw action passed to the environment, before clipping and the "
    "0.040 rad target limiter; the previous action is zero after a reset.",
    "n_collision counts one event when any non-tibia body presses the ground with more than 1 N "
    "during the control; the simulation does not report self-collisions.",
    "The torque limit applies to the largest requested torque over the eight substeps, before the "
    "motor cap, against the 1.6 N·m software cap; applied torque never exceeds that cap.",
    "The joint-velocity limit is the URDF limit, 50.27 rad/s, checked at each control's end.",
    "Continuous penalty weights follow paper_reward.CALIBRATION: together they average 0.8 of the "
    "mean tracking reward on seven recorded rollouts, in equal shares.",
)
OMITTED = (
    "Style r^s needs the AMP discriminator and the AMP owner's agreement on its inputs "
    "(docs/TRAINING.md Step 5).",
    "The foot contact-force limit: the paper states no limit and the reviewer has not set one.",
    "Version 1's height, tilt and termination terms: Table I has none.",
)
ACTUATOR = ("Unchanged from version 1: 18 actions at 50 Hz clipped to [-1, 1]; targets neutral + 0.35 a, "
            "clipped to joint limits and to +-0.040 rad per 20 ms control; tau_req = 12 (q_target - q) "
            "- K_D q_dot with K_D 0.442/0.246/0.106 N·m·s/rad; applied torque capped by the 48 V "
            "speed curve at 1.6 N·m and zero at 50.27 rad/s; 400 Hz physics, eight substeps. "
            "The paper's cascaded law tau = Kp2 (Kp1 (q_des - q) - q_dot) is not used.")
REVIEW = {"status": "approved", "reviewer": "James", "date": "2026-09-25",
          "approved": ["tracking kernel C", "k 0.4", "command floors 0.025 m/s and 0.15 rad/s",
                       "yaw extension of the criterion", "calibrated penalty weights"],
          "pending": ["style inputs with the AMP owner"],
          "source": "docs/REWARD_V2_TABLE1_AUDIT.md section 8"}


@dataclass(frozen=True)
class RewardV2Config:
    tracking_kernel: str = "stride"
    scale_k: float = .4
    minimum_speed_mps: float = .025
    minimum_yaw_rate_rad_s: float = .15
    stride_controls: int = 60
    linear_tracking_weight: float = 1.
    yaw_tracking_weight: float = .5
    vertical_velocity_weight: float = 137.
    roll_pitch_weight: float = 1.18
    torque_weight: float = .0691
    acceleration_weight: float = .000741
    action_rate_weight: float = .215
    collision_weight: float = .05
    torque_limit_weight: float = .05
    velocity_limit_weight: float = .5
    torque_limit_nm: float = 1.6
    velocity_limit_rad_s: float = 50.26548245743669
    collision_force_n: float = 1.

    def validate(self, task_config=None):
        if self.tracking_kernel not in KERNELS:
            raise ValueError(f"Reward v2 tracking kernel must be one of {tuple(KERNELS)}")
        for key, value in asdict(self).items():
            if isinstance(value, float) and (not math.isfinite(value) or value <= 0):
                raise ValueError("Positive finite reward v2 coefficient required: " + key)
        if type(self.stride_controls) is not int or self.stride_controls < 1:
            raise ValueError("The stride window needs a positive whole number of controls")
        # A motionless robot keeps exp(-1/k) of the tracking weight at any command at or above the floor.
        if self.scale_k >= 1/math.log(10):
            raise ValueError("k must stay below 1/ln(10) for a motionless robot to keep under 10 percent")
        if task_config is not None:
            bank = torch.tensor(command_bank(task_config))
            speed = torch.linalg.vector_norm(bank[:, :2], dim=-1)
            yaw = bank[:, 2].abs()
            if bool((speed[speed > 0] < self.minimum_speed_mps).any()) or bool((yaw[yaw > 0] < self.minimum_yaw_rate_rad_s).any()):
                raise ValueError("Command floors must not exceed the smallest nonzero bank command")


REWARD_V2_CONFIG = RewardV2Config()


def reward_declaration(config):
    return {"version": REWARD_VERSION, "config": asdict(config), "kernel_audit_label": KERNELS[config.tracking_kernel],
        "formulas": {
            "linear_tracking": "w_lin exp(-||v_xy - c_xy|| / (k max(||c_xy||, c_min_lin)))",
            "yaw_tracking": "w_yaw exp(-|w_z - c_yaw| / (k max(|c_yaw|, c_min_yaw)))",
            "tracked_velocity": {"scaled": "this control's endpoint velocity",
                                 "stride": "causal per-replica mean over the last stride_controls controls since the latest command change or reset"}[config.tracking_kernel],
            "vertical_velocity": "-w v_z^2", "roll_pitch_rate": "-w ||w_xy||",
            "joint_torque": "-w ||sqrt(sum_substeps tau^2 / 8)||", "joint_acceleration": "-w ||(dq - dq_prev) / 0.02||",
            "action_rate": "-w ||a - a_prev||", "collisions": "-w [max non-tibia floor force > 1 N]",
            "torque_limit": "-w ||max(|tau_requested| - 1.6, 0)||", "velocity_limit": "-w ||max(|dq| - 50.27, 0)||"},
        "stationary_translation_fraction_at_or_above_floor": math.exp(-1/config.scale_k),
        "adaptations": list(ADAPTATIONS), "omitted": list(OMITTED), "actuator": ACTUATOR, "review": REVIEW}


def measured_reward_v2(telemetry, commands, config=REWARD_V2_CONFIG):
    """Table I terms from one control's telemetry plus the caller's previous-control memory.

    ``previous_action`` and ``previous_joint_velocity_rad_s`` are always required;
    ``tracked_planar_velocity_nav`` and ``tracked_yaw_rate_rad_s`` only for the stride kernel.
    """
    n, device = commands.shape[0], commands.device
    def field(name, shape):
        value = telemetry.get(name)
        if not torch.is_tensor(value) or tuple(value.shape) != (n, *shape) or not bool(torch.isfinite(value).all()):
            raise ValueError("Missing/nonfinite reward v2 telemetry: " + name)
        return value.to(device, torch.float32)
    velocity = field("linear_velocity_nav", (3,))
    gyro = field("angular_velocity_body", (3,))
    rate = field("joint_velocity_rad_s", (18,))
    previous_rate = field("previous_joint_velocity_rad_s", (18,))
    action = field("action", (18,))
    previous_action = field("previous_action", (18,))
    torque_square = field("torque_square_sum_400hz", (18,))
    requested = field("requested_torque_abs_max_400hz", (18,))
    other_force = field("other_body_force_max_400hz", ())
    if not torch.equal(field("command", (3,)), commands):
        raise ValueError("Reward command differs from the native completed hold")
    if config.tracking_kernel == "stride":
        tracked_xy, tracked_yaw = field("tracked_planar_velocity_nav", (2,)), field("tracked_yaw_rate_rad_s", ())
    else:
        tracked_xy, tracked_yaw = velocity[:, :2], gyro[:, 2]
    c = config
    norm = torch.linalg.vector_norm
    linear_scale = c.scale_k * norm(commands[:, :2], dim=-1).clamp_min(c.minimum_speed_mps)
    yaw_scale = c.scale_k * commands[:, 2].abs().clamp_min(c.minimum_yaw_rate_rad_s)
    components = {
        "linear_tracking": c.linear_tracking_weight * torch.exp(-norm(tracked_xy - commands[:, :2], dim=-1) / linear_scale),
        "yaw_tracking": c.yaw_tracking_weight * torch.exp(-(tracked_yaw - commands[:, 2]).abs() / yaw_scale),
        "vertical_velocity": -c.vertical_velocity_weight * velocity[:, 2].square(),
        "roll_pitch_rate": -c.roll_pitch_weight * norm(gyro[:, :2], dim=-1),
        "joint_torque": -c.torque_weight * norm((torque_square / SUBSTEPS).sqrt(), dim=-1),
        "joint_acceleration": -c.acceleration_weight * norm((rate - previous_rate) / CONTROL_DT_S, dim=-1),
        "action_rate": -c.action_rate_weight * norm(action - previous_action, dim=-1),
        "collisions": -c.collision_weight * (other_force > c.collision_force_n).to(torch.float32),
        "torque_limit": -c.torque_limit_weight * norm((requested - c.torque_limit_nm).clamp_min(0), dim=-1),
        "velocity_limit": -c.velocity_limit_weight * norm((rate.abs() - c.velocity_limit_rad_s).clamp_min(0), dim=-1),
    }
    reward = sum(components.values())
    if not bool(torch.isfinite(reward).all()):
        raise FloatingPointError("Nonfinite reward v2")
    return reward, components


class StrideWindow:
    """Per-replica causal mean of the last ``window`` values since that replica's latest restart."""

    def __init__(self, replicas, window, width, device="cpu"):
        self.window = window
        self.values = torch.zeros(replicas, window, width, device=device)
        self.count = torch.zeros(replicas, dtype=torch.long, device=device)
        self.command = torch.zeros(replicas, 3, device=device)

    def restart(self, indices, command):
        self.values[indices] = 0
        self.count[indices] = 0
        self.command[indices] = command

    def update(self, values, command):
        changed = (command != self.command).any(-1)
        self.values[changed] = 0
        self.count[changed] = 0
        self.command = command.clone()
        self.values = torch.cat((self.values[:, 1:], values[:, None].to(self.values)), dim=1)
        self.count = (self.count + 1).clamp_max(self.window)
        recent = torch.arange(self.window, device=self.count.device) >= self.window - self.count[:, None]
        return (self.values * recent[..., None]).sum(1) / self.count[:, None]


class TrainingTaskV2(TrainingTask):
    """Version 1's command, guard and reporting task with reward version 2."""

    def __init__(self, env, config=None, output_dir=None, reward_config=None):
        task_config = TaskConfig() if config is None else config
        reward_config = REWARD_V2_CONFIG if reward_config is None else reward_config
        reward_config.validate(task_config)
        self.reward_config = reward_config
        n = env.num_envs
        self.previous_action = torch.zeros(n, 18, device=env.device)
        self.previous_joint_velocity = env.current["dq"].detach().clone().to(env.device)
        self.stride = StrideWindow(n, reward_config.stride_controls, 3, env.device)
        super().__init__(env, task_config, output_dir)

    def declaration(self):
        result = super().declaration()
        del result["quiet_cost_tail"]
        result.update(reward_version=REWARD_VERSION, reward=reward_declaration(self.reward_config),
            reward_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            unused_version1_reward_fields=[name for name in (f.name for f in fields(TaskConfig))
                                           if name.endswith(("_weight", "_variance", "_scale_rad_s", "_scale_rad", "_penalty"))])
        return result

    def reset(self, indices=None):
        output = super().reset(indices)
        selected = self._indices(indices)
        self.previous_action[selected] = 0
        self.previous_joint_velocity[selected] = self.env.current["dq"][selected].to(self.previous_joint_velocity)
        self.stride.restart(selected, self.env.commands[selected])
        return output

    def reward_telemetry(self, command):
        """Current telemetry plus this task's previous-control memory; advances the stride window."""
        t = self.env.telemetry
        tracked = self.stride.update(torch.cat((t["linear_velocity_nav"][:, :2], t["angular_velocity_body"][:, 2:]), -1), command)
        return {**t, "previous_action": self.previous_action, "previous_joint_velocity_rad_s": self.previous_joint_velocity,
                "tracked_planar_velocity_nav": tracked[:, :2], "tracked_yaw_rate_rad_s": tracked[:, 2]}

    def step(self, action):
        # Version 1's step with measured_reward_v2 in place of measured_reward.
        if bool((self.remaining_controls <= 0).any()):
            raise RuntimeError("Call task.reset before collecting a rollout")
        roots_before = self.env.current["root"][:, :3].clone()
        self.proximity.check(roots_before, "before_control",
            speeds=torch.linalg.vector_norm(self.env.current["linear"], dim=-1))
        command = self.env.commands.detach().clone()
        output = self.env.step(action)
        self.proximity.check(self.env.current["root"][:, :3], "after_control", previous=roots_before)
        reward, components = measured_reward_v2(self.reward_telemetry(command), command, self.reward_config)
        metric_previous_target = self.previous_target
        self.previous_target = self.env.telemetry["joint_target_rad"].detach().clone()
        self.previous_action = self.env.telemetry["action"].detach().clone().to(self.previous_action)
        self.previous_joint_velocity = self.env.telemetry["joint_velocity_rad_s"].detach().clone().to(self.previous_joint_velocity)
        self.last_held_command = command
        self.last_components = {key: value.detach().clone() for key, value in components.items()}
        self.remaining_controls -= 1
        done = output["terminated"] | output["truncated"]
        expired = ((self.remaining_controls == 0) & ~done).nonzero(as_tuple=False).squeeze(-1)
        self._resample(expired)
        self.controls_completed += 1
        self._metrics(command, output, reward, components, previous_target=metric_previous_target)
        result = self._next_command_observation(output)
        result["reward"] = reward
        return result

    def status(self, *, reset_interval=False):
        return {**super().status(reset_interval=reset_interval), "reward_version": REWARD_VERSION}


def reward_config(spec=""):
    """``RewardV2Config`` from ``key=value,...`` overrides of the defaults."""
    types = {f.name: f.type for f in fields(RewardV2Config)}
    options = {}
    for item in filter(None, spec.split(",")):
        key, separator, value = item.partition("=")
        if not separator or key not in types or key in options:
            raise ValueError(f"Reward v2 options are unique RewardV2Config key=value pairs, got {item!r}")
        options[key] = types[key](value)
    config = replace(REWARD_V2_CONFIG, **options)
    config.validate(TaskConfig())
    return config


def scorer_reward(spec=""):
    """``locomotion.reward_scorer`` factory: rebuilds the task's memory from a flattened trace."""
    config = reward_config(spec)

    def reward(telemetry, commands, previous_target, terminated, _config=None, nominal_height=None):
        from .paper_reward import stride_average
        telemetry = dict(telemetry)
        if config.tracking_kernel == "stride":
            if "replicas" not in telemetry:
                raise ValueError("Stride tracking needs the telemetry replica count")
            values = torch.cat((torch.as_tensor(telemetry["linear_velocity_nav"])[:, :2],
                                torch.as_tensor(telemetry["angular_velocity_body"])[:, 2:]), -1)
            tracked = stride_average(values, commands, int(telemetry["replicas"]), config.stride_controls)
            telemetry.update(tracked_planar_velocity_nav=tracked[:, :2], tracked_yaw_rate_rad_s=tracked[:, 2])
        return measured_reward_v2(telemetry, commands, config)
    reward.config = config
    return reward


scorer_default = scorer_reward()
