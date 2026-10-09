"""Experimental reward v4 variant: pay the contact schedule only while the body turns as commanded.

The v4 policies that train on the full command bank step in rhythm yet turn at 10 to 45 percent of
a commanded 0.2 rad/s, and many translation probes fail on heading drift alone. The gait schedule
(weight 2) is v4's largest positive term and pays a correct tripod rhythm whatever the body's
turn rate. This variant adds one term, ``schedule_yaw_gate``, that takes back part of that pay
when the body's stride-mean yaw rate misses its command:

    gate = clip(1 - e^2, 0, 1),  e = |w_stride - c_yaw| / max(|c_yaw|, yaw_gate_scale_rad_s)
    schedule_yaw_gate = -(1 - gate) * w_schedule * active * sum_feet(u * swing) / 3

``active`` marks the commands the gate covers. With ``yaw_gate_on="turn"``, the default, it covers
commands with a nonzero yaw rate (turns in place and arcs) and leaves translations to reward v4.
A surrogate run that gated every moving command walked slower and fell more at update 2000: no
policy holds its heading early in training, so the gate withheld schedule pay from every command
while the gait formed. ``yaw_gate_on="all"`` restores that version.

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

    def validate(self):
        super().validate()
        if not math.isfinite(self.yaw_gate_scale_rad_s) or self.yaw_gate_scale_rad_s <= 0:
            raise ValueError("The yaw gate needs a positive finite scale")
        if type(self.gate_window_controls) is not int or not 1 <= self.gate_window_controls <= 250:
            raise ValueError("gate_window_controls needs 1 to 250 controls")
        if self.yaw_gate_on not in ("turn", "all"):
            raise ValueError("yaw_gate_on is turn or all")


REWARD_V4_YAW_GATE_CONFIG = RewardV4YawGateConfig()


def yaw_gate(tracked_yaw_rate, commands, config=REWARD_V4_YAW_GATE_CONFIG):
    """One gate per row in [0, 1]: 1 when the windowed yaw rate matches the command."""
    scale = commands[:, 2].abs().clamp_min(config.yaw_gate_scale_rad_s)
    error = (tracked_yaw_rate - commands[:, 2]).abs() / scale
    return (1 - error.square()).clamp(0, 1)


def measured_reward_v4_yaw_gate(telemetry, commands, terminated, config=REWARD_V4_YAW_GATE_CONFIG, nominal_height=None):
    """Reward v4 plus ``schedule_yaw_gate``; the v4 components are returned unchanged.

    ``gate_yaw_rate_rad_s`` is the caller's stride-mean heading rate, one value per row.
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
    active = (commands[:, 2] != 0) if config.yaw_gate_on == "turn" else (commands != 0).any(-1)
    earned = config.schedule_weight * active.to(torch.float32) * (unloaded * swing).sum(-1) / 3
    gate = yaw_gate(stride_yaw.to(device, torch.float32), commands, config)
    components["schedule_yaw_gate"] = -(1 - gate) * earned
    reward = reward + components["schedule_yaw_gate"]
    if not bool(torch.isfinite(reward).all()):
        raise FloatingPointError("Nonfinite yaw-gated reward v4")
    return reward, components


class TrainingTaskV4YawGate(task_v4.TrainingTaskV4):
    """Reward v4's task with the yaw-gated contact schedule."""

    reward_config = REWARD_V4_YAW_GATE_CONFIG

    def __init__(self, env, config=None, output_dir=None, reward_config=None):
        window = (type(self).reward_config if reward_config is None else reward_config).gate_window_controls
        self.gate_window = task_v4.StrideWindow(env.num_envs, window, 1, env.device)
        super().__init__(env, config, output_dir, reward_config)

    def reset(self, indices=None):
        output = super().reset(indices)
        self.gate_window.restart(self._indices(indices), self.env.commands[self._indices(indices)])
        return output

    def reward_telemetry(self, command):
        """v4's telemetry plus the stride-mean heading rate; reads the pose before v4 advances it."""
        turn = task_v4.displacement_velocity(self.env.telemetry["root_pose_xyzw"].to(self.previous_pose),
                                             self.previous_pose)[:, 2:]
        stride_yaw = self.gate_window.update(turn, command)[:, 0]
        return {**super().reward_telemetry(command), "gate_yaw_rate_rad_s": stride_yaw}

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
            "-(1 - clip(1 - e^2, 0, 1)) * w_schedule * sum over scheduled-swing feet of u / 3, on commands with a "
            "nonzero yaw rate (yaw_gate_on=turn) or every moving command (yaw_gate_on=all), "
            "with e = |stride-mean yaw rate - c_yaw| / max(|c_yaw|, yaw_gate_scale_rad_s); the stride mean covers the "
            "last gate_window_controls controls since the latest command change or reset")
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
