"""Experiment ``destination_v1``: pay every control for staying near the pose the command would have reached.

The canonical task and reward stay unchanged. ``DestinationRewardConfig`` adds four fields to
``RewardConfig`` and ``DestinationTrainingTask`` follows each command's destination
(``mdp/destination.py``):

    destination = w * exp(-d / destination_distance_m)      (or w * max(0, 1 - d / scale), linear)

every control, for the planar distance ``d`` from the destination; a fall forfeits it, so the fall-cost
ceiling counts the weight. ``command_controls`` holds every command that many controls instead of the
draw of 100 to 250. The experiment also gives the actor and critic the gap to the destination
(``locomotion.rl.destination_wrapper``).

``EXPERIMENTS`` names the reviewed settings. ``destination_v1`` is the surrogate result of
2026-10-17: started from the PR model, every command 20 s, the exponential kernel at 3 cm and weight
1, termination 32 (ten times the 3.15 tracking ceiling).
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
    command_controls: int = 0

    def validate(self):
        super().validate()
        if self.destination_kernel not in ("linear", "exponential"):
            raise ValueError("destination_kernel is linear or exponential")
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
            closeness = (
                torch.exp(-scaled)
                if config.destination_kernel == "exponential"
                else (1 - scaled).clamp_min(0)
            )
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
