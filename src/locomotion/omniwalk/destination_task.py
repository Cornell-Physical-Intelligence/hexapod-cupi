"""Experiment ``destination_v1``: pay every control for staying near the pose the command would have reached.

The canonical task and reward stay unchanged. ``DestinationRewardConfig`` adds four fields to
``RewardConfig`` and ``DestinationTrainingTask`` follows each command's destination
(``mdp/destination.py``):

    destination = w * exp(-d / destination_distance_m)        exponential
                  w * 1 / (1 + (d / scale) ** power)         rational
                  w * max(0, 1 - d / scale)                  linear

every control, for the planar distance ``d`` from the destination; a fall forfeits it, so the fall-cost
ceiling counts the weight. ``command_controls`` holds every command that many controls instead of the
draw of 100 to 250. The experiment also gives the actor and critic the gap to the destination
(``locomotion.rl.destination_wrapper``).

``destination_power`` sets the rational kernel's exponent and stays None for the other kernels.
``EXPERIMENTS`` names the reviewed settings. ``destination_v1`` is the surrogate result of
2026-10-17: started from the PR model, every command 20 s, the exponential kernel at 3 cm and weight
1, termination 32 (ten times the 3.15 tracking ceiling). ``destination_v2`` keeps all of it but
halves the pay every centimetre, to train on from the native ``destination_v1`` checkpoint.
``destination_v3`` keeps v2 but concentrates the pay at the destination with the rational kernel,
to train on from the native ``destination_v2`` checkpoint.
"""

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from locomotion.omniwalk.mdp.destination import (
    advance_destination,
    destination_distance,
    planar_pose,
)
from locomotion.omniwalk.mdp.rewards import RewardConfig
from locomotion.omniwalk.task import TrainingTask

REWARD_VERSION = "noise_sized_tripod_schedule_v4_yaw_gate_destination_experimental"


@dataclass(frozen=True)
class DestinationRewardConfig(RewardConfig):
    destination_weight: float = 0.0
    destination_distance_m: float = 0.1
    destination_kernel: str = "linear"
    destination_power: float | None = None
    command_controls: int = 0

    def validate(self):
        super().validate()
        if self.destination_kernel not in ("linear", "exponential", "rational"):
            raise ValueError("destination_kernel is linear, exponential or rational")
        if (self.destination_kernel == "rational") != (self.destination_power is not None) or (
            self.destination_power is not None
            and not (math.isfinite(self.destination_power) and self.destination_power > 0)
        ):
            raise ValueError("The rational kernel, and only it, needs a positive finite power")
        if not math.isfinite(self.destination_distance_m) or self.destination_distance_m <= 0:
            raise ValueError("destination_distance_m must be positive and finite")
        if type(self.command_controls) is not int or not (
            self.command_controls == 0 or 10 <= self.command_controls <= 5000
        ):
            raise ValueError("command_controls is 0 (the drawn durations) or 10 to 5000 controls")
        ceiling = (
            (self.linear_tracking_weight + self.yaw_tracking_weight)
            * (1 + self.stride_tracking_weight)
            + 6 * self.stride_length_weight / self.schedule_period_controls
            + self.destination_weight
        )
        if self.termination_weight < 10 * ceiling:
            raise ValueError(
                "The termination weight must reach ten times the tracking ceiling, "
                "destination term included"
            )


EXPERIMENTS = {
    "destination_v1": {
        "reward": DestinationRewardConfig(
            termination_weight=32.0,
            destination_weight=1.0,
            destination_distance_m=0.03,
            destination_kernel="exponential",
            command_controls=1000,
        ),
        "destination_input": "ghost",
        "episode_seconds": 20.0,
    },
}
# destination_v2 trains on from destination_v1 with the pay concentrated near the destination: it
# halves every centimetre (1 at 0, 0.71 at 0.5 cm, 0.5 at 1 cm, 0.25 at 2 cm, 0.03 at 5 cm).
EXPERIMENTS["destination_v2"] = {
    **EXPERIMENTS["destination_v1"],
    "reward": replace(
        EXPERIMENTS["destination_v1"]["reward"], destination_distance_m=0.01 / math.log(2)
    ),
}
# destination_v3 trains on from destination_v2 with the pay concentrated at the destination: the
# rational kernel at 3.7 mm and power 1.2 (1 at 0, 0.83 at 1 mm, 0.62 at 2.5 mm, 0.41 at 5 mm,
# 0.23 at 1 cm, 0.12 at 2 cm, 0.04 at 5 cm).
EXPERIMENTS["destination_v3"] = {
    **EXPERIMENTS["destination_v2"],
    "reward": replace(
        EXPERIMENTS["destination_v2"]["reward"],
        destination_kernel="rational",
        destination_distance_m=0.0037,
        destination_power=1.2,
    ),
}


class DestinationTrainingTask(TrainingTask):
    """The canonical task plus each command's destination and the per-control destination pay."""

    def __init__(self, env, config=None, output_dir=None, reward_config=None):
        chosen = EXPERIMENTS["destination_v1"]["reward"] if reward_config is None else reward_config
        if not isinstance(chosen, DestinationRewardConfig):
            raise ValueError("The destination task needs a DestinationRewardConfig")
        self.destination = torch.zeros(env.num_envs, 3, device=env.device)
        self.command_started = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.command_elapsed = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        super().__init__(env, config, output_dir, chosen)

    def _resample(self, indices):
        super()._resample(indices)
        if len(indices):
            self.command_started[indices] = True
            if self.reward_config.command_controls:
                self.remaining_controls[indices] = self.reward_config.command_controls

    def _advance_destination(self, command):
        """Start a new command's destination at the pose before its first control, then move it one control."""
        start = planar_pose(self.previous_pose).to(self.destination)
        self.destination = torch.where(self.command_started[:, None], start, self.destination)
        self.command_elapsed = (
            torch.where(
                self.command_started, torch.zeros_like(self.command_elapsed), self.command_elapsed
            )
            + 1
        )
        self.command_started[:] = False
        self.destination = advance_destination(self.destination, command)

    def _reward(self, command, terminated):
        self._advance_destination(command)
        reward, components, telemetry = super()._reward(command, terminated)
        config = self.reward_config
        if config.destination_weight > 0:
            scaled = (
                destination_distance(telemetry["root_pose_xyzw"], self.destination)
                / config.destination_distance_m
            )
            if config.destination_kernel == "exponential":
                closeness = torch.exp(-scaled)
            elif config.destination_kernel == "rational":
                closeness = 1 / (1 + scaled.pow(config.destination_power))
            else:
                closeness = (1 - scaled).clamp_min(0)
            alive = (~terminated).to(closeness.device).to(closeness)
            components["destination"] = (config.destination_weight * alive * closeness).to(reward)
            reward = reward + components["destination"]
        return reward, components, telemetry

    def declaration(self):
        result = super().declaration()
        result["reward_version"] = result["reward"]["version"] = REWARD_VERSION
        here = Path(__file__).parent
        result["experiment_source_sha256"] = {
            name: hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in ("destination_task.py", "mdp/destination.py")
        }
        return result


def experiment_task(name):
    """The task class of a named experiment, for the surrogate's --task module:factory(text)."""
    chosen = EXPERIMENTS[name]["reward"]

    class ExperimentTask(DestinationTrainingTask):
        def __init__(self, env, config=None, output_dir=None, reward_config=None):
            super().__init__(
                env, config, output_dir, chosen if reward_config is None else reward_config
            )

    ExperimentTask.__name__ = ExperimentTask.__qualname__ = "ExperimentTask"
    return ExperimentTask


def replace_reward(name, **changes):
    """A named experiment's reward with fields changed, for sensitivity runs."""
    return replace(EXPERIMENTS[name]["reward"], **changes)
