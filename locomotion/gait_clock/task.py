"""Active c48_long walking reward, command sampling and selected episode resets.

The recorded training source is 1b4d67a1bf37d99a08953034c6df150c1b549c69.
Historical reward implementations remain in Git and the frozen test archive.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import torch
from locomotion.gait_clock.control import inverse_rotate, navigation

SCHEMA = "canonical_paper_walking_task_v1"
REWARD_VERSION = "noise_sized_tripod_schedule_v4_yaw_gate_experimental"
CONTROL_DT_S = .02
SUBSTEPS = 8
RATED_TORQUE_NM = 1.6
TRIPOD_A = (True, False, True, False, True, False)
METRIC_COMMAND_CLASSES = ("zero", "linear_0.025", "linear_0.05", "yaw", "arc", "other")
METRIC_TERMINATION_REASONS = ("height", "tilt", "joint_limit", "timeout")
METRIC_SLEW_TOLERANCE_RAD = 2e-6


@dataclass(frozen=True)
class TaskConfig:
    seed: int = 20260914
    hold_min_seconds: float = 2.
    hold_max_seconds: float = 5.
    zero_command_fraction: float = .15
    speed_levels_mps: tuple = (.025, .05)
    maximum_speed_mps: float = .05
    yaw_rate_rad_s: float = .2
    arc_speed_mps: float = .04
    arc_yaw_rate_rad_s: float = .15

    def validate(self):
        if (self.hold_min_seconds, self.hold_max_seconds, self.zero_command_fraction) != (2., 5., .15):
            raise ValueError("Command holds and zero quota differ from the recorded task")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Nonnegative integer seed required")
        if (self.speed_levels_mps != (.025, .05) or self.maximum_speed_mps not in self.speed_levels_mps
                or self.yaw_rate_rad_s != .2 or self.arc_speed_mps != .04 or self.arc_yaw_rate_rad_s != .15):
            raise ValueError("Command bank differs from the recorded task")


def command_bank(config):
    """Nonzero navigation-frame commands; zero has a separate reserved quota."""
    config.validate()
    result = []
    for speed in config.speed_levels_mps:
        if speed <= config.maximum_speed_mps:
            for bearing in range(8):
                angle = bearing * math.pi / 4
                result.append([speed * math.cos(angle), speed * math.sin(angle), 0.])
    result.extend([[0., 0., -config.yaw_rate_rad_s], [0., 0., config.yaw_rate_rad_s]])
    if config.arc_speed_mps <= config.maximum_speed_mps:
        result.extend([[config.arc_speed_mps, 0., -config.arc_yaw_rate_rad_s],
                       [config.arc_speed_mps, 0., config.arc_yaw_rate_rad_s]])
    return result

@dataclass(frozen=True)
class RewardConfig:
    tracking_window_controls: int = 10
    fixed_speed_scale_mps: float = .05
    fixed_yaw_scale_rad_s: float = .2
    tracking_floor: float = 1.
    linear_tracking_weight: float = 1.
    yaw_tracking_weight: float = .5
    schedule_weight: float = 2.
    schedule_period_controls: int = 60
    schedule_swing_fraction: float = .4
    schedule_load_n: float = 12.2
    swing_travel_weight: float = 1.
    swing_travel_clip: float = 6.
    swing_travel_full: float = 6.
    contact_force_n: float = 1.
    vertical_velocity_weight: float = .3
    vertical_velocity_scale_mps: float = .04
    roll_pitch_weight: float = .05
    roll_pitch_scale_rad_s: float = .3
    yaw_rate_weight: float = .1
    yaw_rate_scale_rad_s: float = .1
    planar_rate_weight: float = .3
    tilt_weight: float = .5
    tilt_scale_rad: float = .0873
    torque_weight: float = .05
    acceleration_weight: float = .05
    acceleration_scale_rad_s2: float = 100.
    target_rate_weight: float = .05
    target_rate_scale_rad: float = .04
    height_weight: float = .2
    height_scale_m: float = .02
    collision_weight: float = 1.
    torque_limit_weight: float = .5
    over_rating_weight: float = 10.
    over_rating_onset_nm: float = 1.4
    joint_margin_weight: float = 1.
    joint_margin_rad: float = .1
    action_limit_weight: float = 6.
    action_limit_onset: float = .5
    quiet_joint_rate_weight: float = .5
    quiet_joint_rate_scale_rad_s: float = .5
    quiet_target_weight: float = .5
    quiet_target_scale_rad: float = .02
    quiet_contact_weight: float = 3.
    quiet_load_n: float = 12.2
    quiet_strain_weight: float = 10.
    quiet_grace_controls: int = 50
    termination_weight: float = 22.

    yaw_gate_scale_rad_s: float = .1
    gate_window_controls: int = 120
    turn_draw_fraction: float = .3
    yaw_gate_min_command_rad_s: float = .18
    yaw_gate_bonus: float = .25
    yaw_gate_floor: float = .5
    stride_tracking_weight: float = .3
    stride_window_controls: int = 60
    stride_length_weight: float = 2.

    def validate(self):
        for key, low in (("tracking_window_controls", 1), ("schedule_period_controls", 10), ("quiet_grace_controls", 0)):
            if type(getattr(self, key)) is not int or not low <= getattr(self, key) <= 250:
                raise ValueError(f"{key} needs {low} to 250 controls")
        for key, value in asdict(self).items():
            if isinstance(value, float) and (not math.isfinite(value) or value < 0):
                raise ValueError("Nonnegative finite reward v4 coefficient required: " + key)
        positive = [key for key in asdict(self) if "_scale_" in key] + [
            "schedule_load_n", "swing_travel_full", "swing_travel_clip", "contact_force_n", "joint_margin_rad",
            "quiet_load_n"]
        if any(getattr(self, key) <= 0 for key in positive):
            raise ValueError("Positive reward v4 scales required")
        if not 0 < self.schedule_swing_fraction <= .5:
            raise ValueError("Each tripod swings for at most half the schedule period")
        if not 0 < self.action_limit_onset < 1:
            raise ValueError("The action-limit onset lies inside the action range")
        if not 0 < self.over_rating_onset_nm < RATED_TORQUE_NM:
            raise ValueError("The over-rating onset lies below 1.6 N.m")
        # A fall must cost more than the discounted loss of the largest per-control tracking reward.
        if self.termination_weight < 10 * (self.linear_tracking_weight + self.yaw_tracking_weight):
            raise ValueError("The termination weight must reach ten times the tracking ceiling")
        if not math.isfinite(self.yaw_gate_scale_rad_s) or self.yaw_gate_scale_rad_s <= 0:
            raise ValueError("The yaw gate needs a positive finite scale")
        if type(self.gate_window_controls) is not int or not 1 <= self.gate_window_controls <= 250:
            raise ValueError("gate_window_controls needs 1 to 250 controls")
        if not 0 <= self.turn_draw_fraction <= 1:
            raise ValueError("turn_draw_fraction is a share from 0 to 1")
        for key in ("yaw_gate_min_command_rad_s", "yaw_gate_bonus"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(key + " must be finite and nonnegative")
        if not 0 <= self.yaw_gate_floor <= 1:
            raise ValueError("yaw_gate_floor is a share of the schedule pay from 0 to 1")
        if not math.isfinite(self.stride_tracking_weight) or self.stride_tracking_weight < 0:
            raise ValueError("stride_tracking_weight must be finite and nonnegative")
        if type(self.stride_window_controls) is not int or not 1 <= self.stride_window_controls <= 250:
            raise ValueError("stride_window_controls needs 1 to 250 controls")
        if not math.isfinite(self.stride_length_weight) or self.stride_length_weight < 0:
            raise ValueError("stride_length_weight must be finite and nonnegative")
        ceiling = ((self.linear_tracking_weight + self.yaw_tracking_weight) * (1 + self.stride_tracking_weight)
                   + 6 * self.stride_length_weight / self.schedule_period_controls)
        if self.termination_weight < 10 * ceiling:
            raise ValueError("The termination weight must reach ten times the tracking ceiling, stride term included")


REWARD_CONFIG = RewardConfig()

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

def heading(quaternion_xyzw):
    """World yaw of the body, from a unit XYZW quaternion."""
    x, y, z, w = quaternion_xyzw.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def displacement_velocity(pose, previous_pose):
    """Navigation-frame planar velocity and yaw rate from two consecutive root poses."""
    planar = navigation(inverse_rotate(pose[:, 3:], (pose[:, :3] - previous_pose[:, :3]) / CONTROL_DT_S))[:, :2]
    turn = heading(pose[:, 3:]) - heading(previous_pose[:, 3:])
    turn = torch.atan2(torch.sin(turn), torch.cos(turn))
    return torch.cat((planar, (turn / CONTROL_DT_S)[:, None]), -1)

def scheduled_swing(episode_steps, config=REWARD_CONFIG):
    """Six booleans per replica: the feet that the contact schedule wants off the floor at this control."""
    fraction = (episode_steps.to(torch.float32) % config.schedule_period_controls) / config.schedule_period_controls
    first = torch.tensor(TRIPOD_A, device=episode_steps.device)
    swing_a = fraction < config.schedule_swing_fraction
    swing_b = (fraction >= .5) & (fraction < .5 + config.schedule_swing_fraction)
    return torch.where(first, swing_a[:, None], swing_b[:, None])

def tracking_errors(tracked, commands, config=REWARD_CONFIG):
    """Dimensionless planar and yaw errors of already selected velocities."""
    speed, turn = torch.linalg.vector_norm(commands[:, :2], dim=-1), commands[:, 2].abs()
    linear = torch.where(speed > 0, speed, torch.full_like(speed, config.fixed_speed_scale_mps))
    yaw = torch.where(turn > 0, turn, torch.full_like(turn, config.fixed_yaw_scale_rad_s))
    return (torch.linalg.vector_norm(tracked[:, :2] - commands[:, :2], dim=-1) / linear,
            (tracked[:, 2] - commands[:, 2]).abs() / yaw)

def kernel(error, config=REWARD_CONFIG):
    return (1 - error.square()).clamp_min(-config.tracking_floor)

def _base_reward(telemetry, commands, terminated, config=REWARD_CONFIG, nominal_height=None):
    """Version 4 terms from one control's telemetry plus the caller's memory.

    Memory fields: ``tracked_velocity_nav`` (planar velocity and yaw rate, already averaged),
    ``previous_joint_velocity_rad_s``, ``previous_joint_target_rad``, ``scheduled_swing`` (six
    booleans), ``toe_velocity_nav`` (planar toe velocity over the floor, in body axes),
    ``toe_xyz_nav`` (planar toe position in the body frame) and ``joint_limit_margin_rad``.
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
    demand = field("computed_torque_nm", (18,)).abs()
    other_force = field("other_body_force_max_400hz", ())
    toe_force = torch.linalg.vector_norm(field("tibia_floor_force_world_n", (6, 3)), dim=-1)
    tracked = field("tracked_velocity_nav", (3,))
    wanted_swing = field("scheduled_swing", (6,), torch.bool).to(torch.float32)
    toe_velocity = field("toe_velocity_nav", (6, 2))
    toe = field("toe_xyz_nav", (6, 2))
    margin = field("joint_limit_margin_rad", (18,))
    executed = field("executed_action", (18,)).abs()
    # A record without the command age scores every control as settled.
    age = telemetry.get("command_age_controls")
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
    still = quiet if age is None else quiet * (age.to(device) > c.quiet_grace_controls).to(torch.float32)
    linear_error, yaw_error = tracking_errors(tracked, commands, c)
    target_step = target - previous_target
    airborne = (toe_force <= c.contact_force_n).to(torch.float32)
    # A foot that carries less than its even share of the weight counts as partly lifted.
    unloaded = (1 - toe_force / c.schedule_load_n).clamp(0, 1)
    # Commanded body velocity at each toe: translation plus yaw rate times the lever arm.
    wanted = commands[:, None, :2] + commands[:, None, 2:] * torch.stack((-toe[..., 1], toe[..., 0]), -1)
    ratio = (toe_velocity * wanted).sum(-1) / wanted.square().sum(-1).clamp_min(1e-8)
    travel = (ratio.clamp(-c.swing_travel_clip, c.swing_travel_clip) * airborne).sum(-1) / c.swing_travel_full
    x, y, z, w = (pose[:, 3:] / torch.linalg.vector_norm(pose[:, 3:], dim=-1, keepdim=True)).unbind(-1)
    gravity_xy_square = 4 * ((x*z - w*y).square() + (y*z + w*x).square())
    speed = torch.linalg.vector_norm(commands[:, :2], dim=-1)
    planar = torch.linalg.vector_norm(velocity[:, :2] - commands[:, :2], dim=-1) / speed.clamp_min(1e-8) * (speed > 0)
    over = ((demand - c.over_rating_onset_nm) / (RATED_TORQUE_NM - c.over_rating_onset_nm)).clamp(0, 1).mean(-1)
    components = {
        "linear_tracking": c.linear_tracking_weight * kernel(linear_error, c),
        "yaw_tracking": c.yaw_tracking_weight * kernel(yaw_error, c),
        "gait_schedule": c.schedule_weight * moving * (unloaded * (2 * wanted_swing - 1)).sum(-1) / 3,
        "swing_travel": c.swing_travel_weight * moving * travel.clamp_max(1.),
        "vertical_velocity": -c.vertical_velocity_weight * (velocity[:, 2] / c.vertical_velocity_scale_mps).square(),
        "roll_pitch_rate": -c.roll_pitch_weight * (gyro[:, :2] / c.roll_pitch_scale_rad_s).square().mean(-1),
        "yaw_rate": -c.yaw_rate_weight * ((gyro[:, 2] - commands[:, 2]) / c.yaw_rate_scale_rad_s).square(),
        "planar_rate": -c.planar_rate_weight * planar.square(),
        "tilt": -c.tilt_weight * gravity_xy_square / c.tilt_scale_rad**2,
        "joint_torque": -c.torque_weight * (torque_square / SUBSTEPS).mean(-1) / RATED_TORQUE_NM**2,
        "joint_acceleration": -c.acceleration_weight * ((rate - previous_rate) / CONTROL_DT_S / c.acceleration_scale_rad_s2).square().mean(-1),
        "target_rate": -c.target_rate_weight * (target_step / c.target_rate_scale_rad).square().mean(-1),
        "height": -c.height_weight * ((pose[:, 2] - nominal_height) / c.height_scale_m).square(),
        "collisions": -c.collision_weight * (other_force > c.contact_force_n).to(torch.float32),
        "torque_limit": -c.torque_limit_weight * ((requested - RATED_TORQUE_NM).clamp_min(0) / RATED_TORQUE_NM).mean(-1),
        "over_rating": -c.over_rating_weight * over,
        "joint_margin": -c.joint_margin_weight * ((c.joint_margin_rad - margin) / c.joint_margin_rad).clamp(0, 1).square().sum(-1),
        "action_limit": -c.action_limit_weight * ((executed - c.action_limit_onset).clamp_min(0) / (1 - c.action_limit_onset)).square().mean(-1),
        "quiet_joint_rate": -c.quiet_joint_rate_weight * still * (rate / c.quiet_joint_rate_scale_rad_s).square().mean(-1),
        "quiet_target_motion": -c.quiet_target_weight * still * (target_step / c.quiet_target_scale_rad).square().mean(-1),
        "quiet_contact": -c.quiet_contact_weight * quiet * (1 - toe_force / c.quiet_load_n).clamp(0, 1).mean(-1),
        "quiet_strain": -c.quiet_strain_weight * quiet * (torque_square[:, 0::3] / SUBSTEPS).mean(-1) / RATED_TORQUE_NM**2,
        "termination": -c.termination_weight * terminated.to(torch.float32),
    }
    reward = sum(components.values())
    if not bool(torch.isfinite(reward).all()):
        raise FloatingPointError("Nonfinite reward v4")
    return reward, components

def yaw_gate(tracked_yaw_rate, commands, config=REWARD_CONFIG):
    """One gate per row in [0, 1]: 1 when the windowed yaw rate matches the command."""
    scale = commands[:, 2].abs().clamp_min(config.yaw_gate_scale_rad_s)
    error = (tracked_yaw_rate - commands[:, 2]).abs() / scale
    return (1 - error.square()).clamp(0, 1)

def measured_reward(telemetry, commands, terminated, config=REWARD_CONFIG, nominal_height=None):
    """Reward v4 plus ``schedule_yaw_gate``; the v4 components are returned unchanged.

    ``gate_yaw_rate_rad_s`` is the caller's stride-mean heading rate, one value per row.
    ``stride_velocity_nav`` is the caller's ``stride_window_controls`` mean of planar velocity and
    yaw rate; it is read only when ``stride_tracking_weight`` is positive.
    """
    reward, components = _base_reward(telemetry, commands, terminated, config, nominal_height)
    device = commands.device
    stride_yaw = telemetry.get("gate_yaw_rate_rad_s")
    if (not torch.is_tensor(stride_yaw) or tuple(stride_yaw.shape) != (len(commands),)
            or not bool(torch.isfinite(stride_yaw).all())):
        raise ValueError("Missing/nonfinite yaw-gate telemetry: gate_yaw_rate_rad_s")
    toe_force = torch.linalg.vector_norm(telemetry["tibia_floor_force_world_n"].to(device, torch.float32), dim=-1)
    unloaded = (1 - toe_force / config.schedule_load_n).clamp(0, 1)
    swing = telemetry["scheduled_swing"].to(device, torch.bool).to(torch.float32)
    turning = (commands[:, 2] != 0) & (commands[:, 2].abs() >= config.yaw_gate_min_command_rad_s)
    active = turning
    earned = config.schedule_weight * active.to(torch.float32) * (unloaded * swing).sum(-1) / 3
    gate = yaw_gate(stride_yaw.to(device, torch.float32), commands, config)
    pay = config.yaw_gate_floor + (1 + config.yaw_gate_bonus - config.yaw_gate_floor) * gate
    components["schedule_yaw_gate"] = (pay - 1) * earned
    reward = reward + components["schedule_yaw_gate"]
    if config.stride_tracking_weight > 0:
        stride = telemetry.get("stride_velocity_nav")
        if (not torch.is_tensor(stride) or tuple(stride.shape) != (len(commands), 3)
                or not bool(torch.isfinite(stride).all())):
            raise ValueError("Missing/nonfinite stride-tracking telemetry: stride_velocity_nav")
        linear_error, yaw_error = tracking_errors(stride.to(device, torch.float32), commands, config)
        moving = (commands != 0).any(-1).to(torch.float32)
        components["stride_tracking"] = config.stride_tracking_weight * moving * (
            config.linear_tracking_weight * kernel(linear_error, config)
            + config.yaw_tracking_weight * kernel(yaw_error, config))
        reward = reward + components["stride_tracking"]
    if config.stride_length_weight > 0:
        landed = telemetry.get("stride_touchdown")
        travel = telemetry.get("stride_displacement_nav")
        if (not torch.is_tensor(landed) or tuple(landed.shape) != (len(commands), 6) or not torch.is_tensor(travel)
                or tuple(travel.shape) != (len(commands), 6, 2) or not bool(torch.isfinite(travel).all())):
            raise ValueError("Missing/nonfinite stride-length telemetry: stride_touchdown, stride_displacement_nav")
        toe = telemetry["toe_xyz_nav"].to(device, torch.float32)
        wanted = commands[:, None, :2] + commands[:, None, 2:] * torch.stack((-toe[..., 1], toe[..., 0]), -1)
        period = config.schedule_period_controls * CONTROL_DT_S
        ratio = (travel.to(device, torch.float32) * wanted).sum(-1) / (wanted.square().sum(-1).clamp_min(1e-8) * period)
        moving = (commands != 0).any(-1).to(torch.float32)
        paid = landed.to(device, torch.bool).to(torch.float32) * ratio.clamp(0, 1).square()
        components["stride_length"] = config.stride_length_weight * moving * paid.sum(-1)
        reward = reward + components["stride_length"]
    if not bool(torch.isfinite(reward).all()):
        raise FloatingPointError("Nonfinite yaw-gated reward v4")
    return reward, components

class TrainingTask:
    """Own command attribution, reward memory and explicit selected resets."""

    def __getattr__(self, name):
        return getattr(self.env, name)

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

    def declaration(self):
        return {"schema": SCHEMA, "config": asdict(self.config), "reward_version": REWARD_VERSION,
                "reward": {"version": REWARD_VERSION, "config": asdict(self.reward_config)},
                "nonzero_command_bank": self.bank.cpu().tolist(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "physics_changed": True, "stage2_gates_changed": False, "stage2_complete": False}
