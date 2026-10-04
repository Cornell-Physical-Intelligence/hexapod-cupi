"""Reward version 4: tracking and penalties sized for exploration noise, plus a stepping term.

Versions 2 and 3 calibrated their penalty weights on deterministic rollouts and paid them on
rollouts that carry exploration noise. PPO then earned reward by quieting its own noise and never
moved (docs/REWARD_V2_TABLE1_AUDIT.md section 15). Version 4 keeps Table I's term list and changes
how each term is measured and weighted; ``DEPARTURES`` lists every change with its reason.

``measured_reward_v4`` is pure: the task passes its per-replica memory in the telemetry.
"""
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import math
from pathlib import Path

import torch

from .task import TaskConfig, TrainingTask
from .task_v2 import ACTUATOR, CONTROL_DT_S, SUBSTEPS, StrideWindow

REWARD_VERSION = "noise_calibrated_stepping_v4"
KERNELS = ("quadratic", "absolute", "exponential")
SCALES = ("command", "fixed")
RATED_TORQUE_NM = 1.6
DEPARTURES = (
    "Tracking error uses the mean displacement velocity over the last tracking_window_controls "
    "controls since the latest command change or reset, in the body frame. Table I tracks the "
    "instantaneous velocity. At this robot's 0.025 to 0.05 m/s commands, exploration shakes the "
    "body by about 0.035 m/s per control, which exceeds the command; a position difference removes "
    "that jitter and a short window keeps the reward close to the action that earned it.",
    "The quadratic kernel 1 - e^2 replaces exp(-|e| / 0.15). Its expectation under zero-mean "
    "velocity noise equals the noise-free value minus a constant, so the gain from walking does "
    "not shrink with noise. The printed exponent has no minus sign and its 0.15 m/s scale pays a "
    "motionless robot 72 to 85 percent at these commands (audit sections 2 and 4).",
    "The error scale follows the command (floors 0.025 m/s and 0.15 rad/s, as in version 2), so a "
    "motionless robot earns zero tracking reward on every moving command.",
    "A stepping term pays each touchdown its preceding air time minus a threshold, on moving "
    "commands. Table I has no such term: the paper's gait comes from its adversarial style reward, "
    "which vanilla PPO omits. The term reads measured foot forces and no reference motion.",
    "Each continuous penalty is a mean square against a declared scale. Table I prints unsquared "
    "norms, which charge exploration noise in first order. Weights are set on rollouts that carry "
    "the training noise, so the noise-driven total stays below a declared share of the tracking "
    "reward; version 2's weights were 15 to 34,550 times the printed values.",
    "The target-rate penalty reads the executed joint target after the action clamp and the "
    "0.040 rad limiter. Version 2 read the raw sample, which charged the policy for noise that "
    "never reached the joints.",
    "A height term keeps the plate near its nominal height. Table I has none; the 400 Hz gate "
    "counts tibia-shaft contact as non-foot contact and compact training telemetry cannot.",
    "Quiet terms penalize joint rate and target change under a zero command, as version 1 does, "
    "because the stop gates bound both.",
    "Any non-tibia floor contact above 1 N costs collision_weight per control; the simulation "
    "reports no self-collision.",
    "Table I has no termination term. PPO bootstraps zero after a termination, so termination_weight "
    "exceeds the discounted loss of the largest per-control reward.",
)
OMITTED = (
    "Style r^s: no discriminator and no demonstration enter vanilla PPO.",
    "The foot contact-force limit: the paper states no limit.",
    "Joint velocity limit: the 50.27 rad/s URDF limit is 8 times the fastest recorded joint speed.",
)


@dataclass(frozen=True)
class RewardV4Config:
    tracking_kernel: str = "quadratic"
    tracking_scale: str = "command"
    tracking_window_controls: int = 10
    scale_k: float = 1.
    minimum_speed_mps: float = .025
    minimum_yaw_rate_rad_s: float = .15
    fixed_speed_scale_mps: float = .05
    fixed_yaw_scale_rad_s: float = .2
    tracking_floor: float = 1.
    linear_tracking_weight: float = 1.
    yaw_tracking_weight: float = .5
    air_time_weight: float = 1.
    air_time_threshold_s: float = .1
    air_time_cap_s: float = .5
    contact_force_n: float = 1.
    vertical_velocity_weight: float = .1
    vertical_velocity_scale_mps: float = .04
    roll_pitch_weight: float = .05
    roll_pitch_scale_rad_s: float = .3
    torque_weight: float = .05
    acceleration_weight: float = .05
    acceleration_scale_rad_s2: float = 100.
    target_rate_weight: float = .05
    target_rate_scale_rad: float = .04
    height_weight: float = .1
    height_scale_m: float = .02
    collision_weight: float = 1.
    torque_limit_weight: float = .1
    quiet_joint_rate_weight: float = .2
    quiet_joint_rate_scale_rad_s: float = .5
    quiet_target_weight: float = .2
    quiet_target_scale_rad: float = .02
    termination_weight: float = 20.

    def validate(self):
        if self.tracking_kernel not in KERNELS or self.tracking_scale not in SCALES:
            raise ValueError("Unknown reward v4 tracking kernel or scale")
        if type(self.tracking_window_controls) is not int or not 1 <= self.tracking_window_controls <= 250:
            raise ValueError("The tracking window needs 1 to 250 controls")
        for key, value in asdict(self).items():
            if isinstance(value, float) and (not math.isfinite(value) or value < 0):
                raise ValueError("Nonnegative finite reward v4 coefficient required: " + key)
        scales = [key for key in asdict(self) if "_scale_" in key or key.startswith(("minimum_", "fixed_"))]
        if any(getattr(self, key) <= 0 for key in scales + ["scale_k", "air_time_cap_s", "contact_force_n"]):
            raise ValueError("Positive reward v4 scales required")
        if self.air_time_cap_s <= self.air_time_threshold_s:
            raise ValueError("The air-time cap must exceed its threshold")
        # A fall must cost more than the discounted loss of the largest per-control reward.
        ceiling = self.linear_tracking_weight + self.yaw_tracking_weight
        if self.termination_weight and self.termination_weight < 10 * ceiling:
            raise ValueError("The termination weight must reach ten times the tracking ceiling")


REWARD_V4_CONFIG = RewardV4Config()


def reward_declaration(config):
    return {"version": REWARD_VERSION, "config": asdict(config),
        "formulas": {
            "tracked_velocity": "mean over the last tracking_window_controls controls, since the latest command "
                "change or reset, of the body-frame root displacement and heading change per control divided by 0.02 s",
            "error": {"command": "e_lin = ||v - c_xy|| / (k max(||c_xy||, c_min_lin)); e_yaw = |w - c_yaw| / (k max(|c_yaw|, c_min_yaw))",
                      "fixed": "e_lin = ||v - c_xy|| / s_lin; e_yaw = |w - c_yaw| / s_yaw"}[config.tracking_scale],
            "kernel": {"quadratic": "max(1 - e^2, -floor)", "absolute": "max(1 - e, -floor)",
                       "exponential": "exp(-e^2)"}[config.tracking_kernel],
            "linear_tracking": "w_lin kernel(e_lin)", "yaw_tracking": "w_yaw kernel(e_yaw)",
            "air_time": "w sum_feet [touchdown this control] (min(t_air, cap) - threshold), moving commands only",
            "vertical_velocity": "-w (v_z / s)^2", "roll_pitch_rate": "-w mean((w_xy / s)^2)",
            "joint_torque": "-w mean(sum_substeps tau^2 / 8) / 1.6^2",
            "joint_acceleration": "-w mean(((dq - dq_prev) / 0.02 / s)^2)",
            "target_rate": "-w mean(((q_target - q_target_prev) / s)^2), executed targets",
            "height": "-w ((z - z_nominal) / s)^2",
            "collisions": "-w [max non-tibia floor force > 1 N]",
            "torque_limit": "-w mean(max(|tau_requested| - 1.6, 0) / 1.6)",
            "quiet_joint_rate": "-w [zero command] mean((dq / s)^2)",
            "quiet_target_motion": "-w [zero command] mean(((q_target - q_target_prev) / s)^2)",
            "termination": "-w [height, tilt or joint-limit termination this control]"},
        "departures_from_table_1": list(DEPARTURES), "omitted": list(OMITTED), "actuator": ACTUATOR,
        "review": {"status": "experimental", "basis": "docs/REWARD_V2_TABLE1_AUDIT.md section 15"}}


def heading(quaternion_xyzw):
    """World yaw of the body, from a unit XYZW quaternion."""
    x, y, z, w = quaternion_xyzw.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def displacement_velocity(pose, previous_pose):
    """Navigation-frame planar velocity and yaw rate from two consecutive root poses."""
    from .env import inverse_rotate, navigation
    planar = navigation(inverse_rotate(pose[:, 3:], (pose[:, :3] - previous_pose[:, :3]) / CONTROL_DT_S))[:, :2]
    turn = heading(pose[:, 3:]) - heading(previous_pose[:, 3:])
    turn = torch.atan2(torch.sin(turn), torch.cos(turn))
    return torch.cat((planar, (turn / CONTROL_DT_S)[:, None]), -1)


def tracking_errors(tracked, commands, config=REWARD_V4_CONFIG):
    """Dimensionless planar and yaw errors of already selected velocities."""
    c, norm = config, torch.linalg.vector_norm
    if c.tracking_scale == "command":
        linear = c.scale_k * norm(commands[:, :2], dim=-1).clamp_min(c.minimum_speed_mps)
        yaw = c.scale_k * commands[:, 2].abs().clamp_min(c.minimum_yaw_rate_rad_s)
    else:
        linear = torch.full_like(commands[:, 0], c.fixed_speed_scale_mps)
        yaw = torch.full_like(commands[:, 0], c.fixed_yaw_scale_rad_s)
    return norm(tracked[:, :2] - commands[:, :2], dim=-1) / linear, (tracked[:, 2] - commands[:, 2]).abs() / yaw


def kernel(error, config=REWARD_V4_CONFIG):
    if config.tracking_kernel == "quadratic":
        return (1 - error.square()).clamp_min(-config.tracking_floor)
    if config.tracking_kernel == "absolute":
        return (1 - error).clamp_min(-config.tracking_floor)
    return torch.exp(-error.square())


def measured_reward_v4(telemetry, commands, terminated, config=REWARD_V4_CONFIG, nominal_height=None):
    """Version 4 terms from one control's telemetry plus the caller's memory.

    Memory fields: ``tracked_velocity_nav`` (planar velocity and yaw rate, already averaged),
    ``previous_joint_velocity_rad_s``, ``previous_joint_target_rad``, ``touchdown`` (six booleans)
    and ``air_time_s`` (the air time each foot had completed before this control's contact).
    """
    n, device = commands.shape[0], commands.device
    def field(name, shape, dtype=torch.float32):
        value = telemetry.get(name)
        if not torch.is_tensor(value) or tuple(value.shape) != (n, *shape) or not bool(torch.isfinite(value.to(torch.float32)).all()):
            raise ValueError("Missing/nonfinite reward v4 telemetry: " + name)
        return value.to(device, dtype)
    velocity = field("linear_velocity_nav", (3,))
    gyro = field("angular_velocity_body", (3,))
    pose = field("root_pose_xyzw", (7,))
    rate = field("joint_velocity_rad_s", (18,))
    previous_rate = field("previous_joint_velocity_rad_s", (18,))
    target = field("joint_target_rad", (18,))
    previous_target = field("previous_joint_target_rad", (18,))
    torque_square = field("torque_square_sum_400hz", (18,))
    requested = field("requested_torque_abs_max_400hz", (18,))
    other_force = field("other_body_force_max_400hz", ())
    tracked = field("tracked_velocity_nav", (3,))
    touchdown = field("touchdown", (6,), torch.bool)
    air_time = field("air_time_s", (6,))
    if not torch.equal(field("command", (3,)), commands):
        raise ValueError("Reward command differs from the native completed hold")
    terminated = torch.as_tensor(terminated, device=device)
    if tuple(terminated.shape) != (n,) or terminated.dtype != torch.bool:
        raise ValueError("Reward v4 needs one boolean termination flag per row")
    if nominal_height is None:
        raise ValueError("Reward v4 needs the nominal plate height")
    c = config
    quiet = (commands == 0).all(-1).to(torch.float32)
    moving = 1 - quiet
    linear_error, yaw_error = tracking_errors(tracked, commands, c)
    target_step = target - previous_target
    stride = (air_time.clamp_max(c.air_time_cap_s) - c.air_time_threshold_s) * touchdown
    components = {
        "linear_tracking": c.linear_tracking_weight * kernel(linear_error, c),
        "yaw_tracking": c.yaw_tracking_weight * kernel(yaw_error, c),
        "air_time": c.air_time_weight * moving * stride.sum(-1),
        "vertical_velocity": -c.vertical_velocity_weight * (velocity[:, 2] / c.vertical_velocity_scale_mps).square(),
        "roll_pitch_rate": -c.roll_pitch_weight * (gyro[:, :2] / c.roll_pitch_scale_rad_s).square().mean(-1),
        "joint_torque": -c.torque_weight * (torque_square / SUBSTEPS).mean(-1) / RATED_TORQUE_NM**2,
        "joint_acceleration": -c.acceleration_weight * ((rate - previous_rate) / CONTROL_DT_S / c.acceleration_scale_rad_s2).square().mean(-1),
        "target_rate": -c.target_rate_weight * (target_step / c.target_rate_scale_rad).square().mean(-1),
        "height": -c.height_weight * ((pose[:, 2] - nominal_height) / c.height_scale_m).square(),
        "collisions": -c.collision_weight * (other_force > c.contact_force_n).to(torch.float32),
        "torque_limit": -c.torque_limit_weight * ((requested - RATED_TORQUE_NM).clamp_min(0) / RATED_TORQUE_NM).mean(-1),
        "quiet_joint_rate": -c.quiet_joint_rate_weight * quiet * (rate / c.quiet_joint_rate_scale_rad_s).square().mean(-1),
        "quiet_target_motion": -c.quiet_target_weight * quiet * (target_step / c.quiet_target_scale_rad).square().mean(-1),
        "termination": -c.termination_weight * terminated.to(torch.float32),
    }
    reward = sum(components.values())
    if not bool(torch.isfinite(reward).all()):
        raise FloatingPointError("Nonfinite reward v4")
    return reward, components


class TrainingTaskV4(TrainingTask):
    """Version 1's command, guard and reporting task with reward version 4."""

    reward_config = REWARD_V4_CONFIG

    def __init__(self, env, config=None, output_dir=None, reward_config=None):
        task_config = TaskConfig() if config is None else config
        self.reward_config = type(self).reward_config if reward_config is None else reward_config
        self.reward_config.validate()
        n, device = env.num_envs, env.device
        self.previous_joint_velocity = env.current["dq"].detach().clone().to(device)
        self.previous_pose = env.current["root"].detach().clone().to(device)
        self.window = StrideWindow(n, self.reward_config.tracking_window_controls, 3, device)
        self.air_time = torch.zeros(n, 6, device=device)
        self.in_contact = torch.ones(n, 6, dtype=torch.bool, device=device)
        self.gait_totals = {}
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
        self.previous_joint_velocity[selected] = self.env.current["dq"][selected].to(self.previous_joint_velocity)
        self.previous_pose[selected] = self.env.current["root"][selected].to(self.previous_pose)
        self.window.restart(selected, self.env.commands[selected])
        self.air_time[selected] = 0
        self.in_contact[selected] = True
        return output

    def reward_telemetry(self, command):
        """Current telemetry plus this task's memory; advances the tracking window and foot timers."""
        t = self.env.telemetry
        pose = t["root_pose_xyzw"].to(self.previous_pose)
        tracked = self.window.update(displacement_velocity(pose, self.previous_pose), command)
        contact = torch.linalg.vector_norm(t["tibia_floor_force_world_n"], dim=-1) > self.reward_config.contact_force_n
        touchdown = contact & ~self.in_contact
        completed = self.air_time.clone()
        self.air_time = torch.where(contact, torch.zeros_like(self.air_time), self.air_time + CONTROL_DT_S)
        self.in_contact = contact
        return {**t, "tracked_velocity_nav": tracked, "previous_joint_velocity_rad_s": self.previous_joint_velocity,
                "previous_joint_target_rad": self.previous_target, "touchdown": touchdown, "air_time_s": completed}

    def step(self, action):
        # Version 1's step with measured_reward_v4 in place of measured_reward.
        if bool((self.remaining_controls <= 0).any()):
            raise RuntimeError("Call task.reset before collecting a rollout")
        roots_before = self.env.current["root"][:, :3].clone()
        self.proximity.check(roots_before, "before_control",
            speeds=torch.linalg.vector_norm(self.env.current["linear"], dim=-1))
        command = self.env.commands.detach().clone()
        output = self.env.step(action)
        self.proximity.check(self.env.current["root"][:, :3], "after_control", previous=roots_before)
        telemetry = self.reward_telemetry(command)
        reward, components = measured_reward_v4(telemetry, command, output["terminated"], self.reward_config,
                                                self.nominal_height)
        metric_previous_target = self.previous_target
        self.previous_target = self.env.telemetry["joint_target_rad"].detach().clone()
        self.previous_joint_velocity = self.env.telemetry["joint_velocity_rad_s"].detach().clone().to(self.previous_joint_velocity)
        self.previous_pose = self.env.telemetry["root_pose_xyzw"].detach().clone().to(self.previous_pose)
        self.last_held_command = command
        self.last_components = {key: value.detach().clone() for key, value in components.items()}
        self.remaining_controls -= 1
        done = output["terminated"] | output["truncated"]
        expired = ((self.remaining_controls == 0) & ~done).nonzero(as_tuple=False).squeeze(-1)
        self._resample(expired)
        self.controls_completed += 1
        self._metrics(command, output, reward, components, previous_target=metric_previous_target)
        self._gait_metrics(command, telemetry)
        result = self._next_command_observation(output)
        result["reward"] = reward
        return result

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
        result = {**super().status(reset_interval=reset_interval), "reward_version": REWARD_VERSION}
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


def reward_config(spec=""):
    """``RewardV4Config`` from ``key=value,...`` overrides of the defaults."""
    types = {f.name: f.type for f in fields(RewardV4Config)}
    options = {}
    for item in filter(None, spec.split(",")):
        key, separator, value = item.partition("=")
        if not separator or key not in types or key in options:
            raise ValueError(f"Reward v4 options are unique RewardV4Config key=value pairs, got {item!r}")
        options[key] = types[key](value)
    config = replace(REWARD_V4_CONFIG, **options)
    config.validate()
    return config


def variant(spec):
    """A task class with overridden reward coefficients, for design comparisons."""
    return type("TrainingTaskV4Variant", (TrainingTaskV4,), {"reward_config": reward_config(spec)})
