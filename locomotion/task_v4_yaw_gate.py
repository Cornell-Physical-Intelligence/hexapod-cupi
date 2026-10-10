"""Experimental reward v4 variant: pay the contact schedule only while the body turns as commanded.

The v4 policies that train on the full command bank step in rhythm yet turn at 10 to 45 percent of
a commanded 0.2 rad/s, and many translation probes fail on heading drift alone. The gait schedule
(weight 2) is v4's largest positive term and pays a correct tripod rhythm whatever the body's
turn rate. This variant adds one term, ``schedule_yaw_gate``, that takes back part of that pay
when the body's stride-mean yaw rate misses its command:

    gate = clip(1 - e^2, 0, 1),  e = |w_stride - c_yaw| / max(|c_yaw|, yaw_gate_scale_rad_s)
    pay = yaw_gate_floor + (1 + yaw_gate_bonus - yaw_gate_floor) * gate
    schedule_yaw_gate = (pay - 1) * w_schedule * active * sum_feet(u * swing) / 3

``active`` marks the commands the gate covers. With ``yaw_gate_on="turn"``, the default, it covers
commands with a nonzero yaw rate (turns in place and arcs) and leaves translations to reward v4.
A surrogate run that gated every moving command walked slower and fell more at update 2000: no
policy holds its heading early in training, so the gate withheld schedule pay from every command
while the gait formed. ``yaw_gate_on="all"`` restores that version. ``yaw_gate_min_command_rad_s``
narrows the covered commands to those turning at least that fast; 0.18 keeps the 0.2 rad/s turns
in place and leaves the 0.15 rad/s arcs to reward v4.

``yaw_gate_bonus`` turns the gate from a withholding into a reward: on covered commands the
schedule pays ``gate * (1 + bonus)`` of its v4 value, so a turn that matches its command earns
``1 + bonus`` times v4's schedule pay, stepping without turning earns none, and the break-even gate
is ``1 / (1 + bonus)``. The default 0 keeps the withholding-only term.

``yaw_gate_floor`` keeps that share of v4's schedule pay on a covered command whatever the turn,
so a direction that does not turn yet still earns while it improves; the pay rises linearly with
the gate from the floor to ``1 + bonus``. Two gated surrogate runs turned well in one direction
and poorly in the other, consistent with the weak direction losing its schedule pay. The
default 0 keeps the earlier behaviour.

``u`` and ``swing`` are v4's unloaded share and scheduled swing. The term removes only the
positive part of ``gait_schedule``: unloaded stance feet keep their full charge, so a yaw error
never makes breaking the schedule cheaper. On a turn command a robot that steps without turning
loses all of its schedule pay. With ``yaw_gate_on="all"`` a drift of 0.065 rad/s on a translation
command, the failing probes' level, loses 42 percent of it at the default scale.

``w_stride`` is the mean heading rate over each replica's last ``gate_window_controls`` controls
(60, one schedule period) since its latest command change or reset. A first version read v4's
10-control tracking window: shorter than a stride, it followed the body's yaw wobble within each
stride and took about 40 percent of the schedule pay from replayed walks with no mean drift. A
full-stride mean cancels that wobble and keeps drift and a missing turn.

``turn_draw_fraction`` turns that share of the moving command draws into a turn in place at
version 1's yaw rate (0.2 rad/s), with a random sign; it draws from the task's generator only when
nonzero, so the default leaves earlier runs' command sequences unchanged. The 20-command bank holds
two turns in place, so a policy otherwise sees one in ten moving commands as a pure turn.
``yaw_gate_on="none"`` switches the gate off to test turn-heavy sampling alone.

``stride_tracking_weight`` adds a second tracking term on a longer window, for moving commands:

    stride_tracking = s * (w_lin * max(1 - e_lin^2, -floor) + w_yaw * max(1 - e_yaw^2, -floor))

with v4's errors, scales and kernel applied to the mean displacement velocity over the last
``stride_window_controls`` controls (60, one schedule period) since the latest command change or
reset. v4's 10-control window averages less than a stride and pays a burst within each stride;
a full-stride mean pays the progress the robot keeps. The kernel peaks at the command, so the
term pays reaching the commanded speed and not exceeding it. v4's rule that a fall costs ten
times the tracking ceiling covers this term too, so a weight above 1/3 needs a higher
``termination_weight``. The default 0 adds nothing.

Every other term is reward v4 unchanged. Train or score it with ``variant`` on the surrogate first
(locomotion/surrogate/README.md); it is not a reviewed reward version.
"""
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import math
from pathlib import Path

import torch

from . import task_v4

REWARD_VERSION = "noise_sized_tripod_schedule_v4_yaw_gate_experimental"


@dataclass(frozen=True)
class RewardV4YawGateConfig(task_v4.RewardV4Config):
    yaw_gate_scale_rad_s: float = .1
    gate_window_controls: int = 60
    yaw_gate_on: str = "turn"
    turn_draw_fraction: float = 0.
    yaw_gate_min_command_rad_s: float = 0.
    yaw_gate_bonus: float = 0.
    yaw_gate_floor: float = 0.
    stride_tracking_weight: float = 0.
    stride_window_controls: int = 60

    def validate(self):
        super().validate()
        if not math.isfinite(self.yaw_gate_scale_rad_s) or self.yaw_gate_scale_rad_s <= 0:
            raise ValueError("The yaw gate needs a positive finite scale")
        if type(self.gate_window_controls) is not int or not 1 <= self.gate_window_controls <= 250:
            raise ValueError("gate_window_controls needs 1 to 250 controls")
        if self.yaw_gate_on not in ("turn", "all", "none"):
            raise ValueError("yaw_gate_on is turn, all or none")
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
        ceiling = (self.linear_tracking_weight + self.yaw_tracking_weight) * (1 + self.stride_tracking_weight)
        if self.termination_weight < 10 * ceiling:
            raise ValueError("The termination weight must reach ten times the tracking ceiling, stride term included")


REWARD_V4_YAW_GATE_CONFIG = RewardV4YawGateConfig()


def yaw_gate(tracked_yaw_rate, commands, config=REWARD_V4_YAW_GATE_CONFIG):
    """One gate per row in [0, 1]: 1 when the windowed yaw rate matches the command."""
    scale = commands[:, 2].abs().clamp_min(config.yaw_gate_scale_rad_s)
    error = (tracked_yaw_rate - commands[:, 2]).abs() / scale
    return (1 - error.square()).clamp(0, 1)


def measured_reward_v4_yaw_gate(telemetry, commands, terminated, config=REWARD_V4_YAW_GATE_CONFIG, nominal_height=None):
    """Reward v4 plus ``schedule_yaw_gate``; the v4 components are returned unchanged.

    ``gate_yaw_rate_rad_s`` is the caller's stride-mean heading rate, one value per row.
    ``stride_velocity_nav`` is the caller's ``stride_window_controls`` mean of planar velocity and
    yaw rate; it is read only when ``stride_tracking_weight`` is positive.
    """
    reward, components = task_v4.measured_reward_v4(telemetry, commands, terminated, config, nominal_height)
    device = commands.device
    stride_yaw = telemetry.get("gate_yaw_rate_rad_s")
    if (not torch.is_tensor(stride_yaw) or tuple(stride_yaw.shape) != (len(commands),)
            or not bool(torch.isfinite(stride_yaw).all())):
        raise ValueError("Missing/nonfinite yaw-gate telemetry: gate_yaw_rate_rad_s")
    toe_force = torch.linalg.vector_norm(telemetry["tibia_floor_force_world_n"].to(device, torch.float32), dim=-1)
    unloaded = (1 - toe_force / config.schedule_load_n).clamp(0, 1)
    swing = telemetry["scheduled_swing"].to(device, torch.bool).to(torch.float32)
    turning = (commands[:, 2] != 0) & (commands[:, 2].abs() >= config.yaw_gate_min_command_rad_s)
    active = {"turn": turning, "all": (commands != 0).any(-1),
              "none": torch.zeros(len(commands), dtype=torch.bool, device=device)}[config.yaw_gate_on]
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
        linear_error, yaw_error = task_v4.tracking_errors(stride.to(device, torch.float32), commands, config)
        moving = (commands != 0).any(-1).to(torch.float32)
        components["stride_tracking"] = config.stride_tracking_weight * moving * (
            config.linear_tracking_weight * task_v4.kernel(linear_error, config)
            + config.yaw_tracking_weight * task_v4.kernel(yaw_error, config))
        reward = reward + components["stride_tracking"]
    if not bool(torch.isfinite(reward).all()):
        raise FloatingPointError("Nonfinite yaw-gated reward v4")
    return reward, components


class TrainingTaskV4YawGate(task_v4.TrainingTaskV4):
    """Reward v4's task with the yaw-gated contact schedule."""

    reward_config = REWARD_V4_YAW_GATE_CONFIG

    def __init__(self, env, config=None, output_dir=None, reward_config=None):
        chosen = type(self).reward_config if reward_config is None else reward_config
        self.gate_window = task_v4.StrideWindow(env.num_envs, chosen.gate_window_controls, 1, env.device)
        self.stride_window = task_v4.StrideWindow(env.num_envs, chosen.stride_window_controls, 3, env.device)
        super().__init__(env, config, output_dir, reward_config)

    def _resample(self, indices):
        """Reward v4's draw, then a declared share of the moving draws becomes a turn in place."""
        super()._resample(indices)
        share = self.reward_config.turn_draw_fraction
        if share and len(indices):
            chosen = (torch.rand(len(indices), generator=self.rng) < share).to(self.env.device)
            chosen &= (self.env.commands[indices] != 0).any(-1)
            sign = (torch.randint(0, 2, (len(indices),), generator=self.rng) * 2 - 1).to(self.env.commands)
            turns = torch.zeros(len(indices), 3, device=self.env.device, dtype=self.env.commands.dtype)
            turns[:, 2] = sign * self.config.yaw_rate_rad_s
            self.env.commands[indices[chosen]] = turns[chosen]

    def reset(self, indices=None):
        output = super().reset(indices)
        selected = self._indices(indices)
        self.gate_window.restart(selected, self.env.commands[selected])
        self.stride_window.restart(selected, self.env.commands[selected])
        return output

    def reward_telemetry(self, command):
        """v4's telemetry plus the stride-mean heading rate and velocity; reads the pose before v4 advances it."""
        velocity = task_v4.displacement_velocity(self.env.telemetry["root_pose_xyzw"].to(self.previous_pose),
                                                 self.previous_pose)
        stride_yaw = self.gate_window.update(velocity[:, 2:], command)[:, 0]
        stride = self.stride_window.update(velocity, command)
        return {**super().reward_telemetry(command), "gate_yaw_rate_rad_s": stride_yaw, "stride_velocity_nav": stride}

    def _reward(self, command, terminated):
        # TrainingTaskV4._reward with the gated reward in place of measured_reward_v4.
        telemetry = self.reward_telemetry(command)
        reward, components = measured_reward_v4_yaw_gate(telemetry, command, terminated, self.reward_config,
                                                         self.nominal_height)
        self.previous_joint_velocity = self.env.telemetry["joint_velocity_rad_s"].detach().clone().to(self.previous_joint_velocity)
        self.previous_pose = self.env.telemetry["root_pose_xyzw"].detach().clone().to(self.previous_pose)
        self._gait_metrics(command, telemetry)
        return reward, components

    def declaration(self):
        result = super().declaration()
        result["reward_version"] = REWARD_VERSION
        result["reward_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["reward_base_source_sha256"] = hashlib.sha256(Path(task_v4.__file__).read_bytes()).hexdigest()
        reward = result["reward"]
        reward["version"] = REWARD_VERSION
        reward["formulas"]["schedule_yaw_gate"] = (
            "(floor + (1 + yaw_gate_bonus - floor) * clip(1 - e^2, 0, 1) - 1) * w_schedule * sum over scheduled-swing "
            "feet of u / 3 with floor = yaw_gate_floor, "
            "on commands with |c_yaw| >= yaw_gate_min_command_rad_s and nonzero (yaw_gate_on=turn) or every moving "
            "command (yaw_gate_on=all), "
            "with e = |stride-mean yaw rate - c_yaw| / max(|c_yaw|, yaw_gate_scale_rad_s); the stride mean covers the "
            "last gate_window_controls controls since the latest command change or reset")
        if self.reward_config.stride_tracking_weight > 0:
            reward["formulas"]["stride_tracking"] = (
                "stride_tracking_weight * moving * (w_lin max(1 - e_lin^2, -floor) + w_yaw max(1 - e_yaw^2, -floor)), "
                "v4's errors of the mean displacement velocity over the last stride_window_controls controls since "
                "the latest command change or reset")
        reward["review"] = {"status": "experimental", "basis": "locomotion/task_v4_yaw_gate.py docstring"}
        return result

    def status(self, *, reset_interval=False):
        return {**super().status(reset_interval=reset_interval), "reward_version": REWARD_VERSION}


def reward_config(spec=""):
    """``RewardV4YawGateConfig`` from ``key=value,...`` overrides of the defaults."""
    types = {f.name: f.type for f in fields(RewardV4YawGateConfig)}
    options = {}
    for item in filter(None, spec.split(",")):
        key, separator, value = item.partition("=")
        if not separator or key not in types or key in options:
            raise ValueError(f"Yaw-gate options are unique RewardV4YawGateConfig key=value pairs, got {item!r}")
        options[key] = types[key](value)
    config = replace(REWARD_V4_YAW_GATE_CONFIG, **options)
    config.validate()
    return config


def variant(spec):
    """A task class with overridden coefficients, e.g. ``variant(forward_draw_fraction=0)``."""
    return type("TrainingTaskV4YawGateVariant", (TrainingTaskV4YawGate,), {"reward_config": reward_config(spec)})
