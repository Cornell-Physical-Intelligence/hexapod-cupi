"""Experiment ``destination_v1``: pay every control for staying near the pose the command would have reached.

The canonical task and reward stay unchanged. ``DestinationRewardConfig`` adds its fields to
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

Two zero-command fields leave moving commands alone. ``destination_quiet="flat"`` pays the full weight
while the command is zero, so standing still never trades joint motion for millimetres of ghost
distance; ``destination_quiet="off"`` pays nothing while the command is zero. ``hold_weight`` follows each joint's running mean over ``hold_smoothing_controls`` (an
exponential mean restarted when a command begins), which removes the zero-mean exploration jitter of
training but keeps slow drift. The mean is anchored ``hold_grace_controls`` into a zero command and
from then on the task charges

    hold = -hold_weight * mean_j clamp((|mean_j - anchor_j| - hold_deadband_rad) / hold_span_rad, 0, 1)

linear in the drift, like the evaluator's joint range check that opens 2 s into a zero command. A
fall forfeits the destination pay and the hold charge. None leaves each field off.

``EXPERIMENTS`` names the reviewed settings. ``destination_v1`` is the surrogate result of
2026-10-17: started from the PR model, every command 20 s, the exponential kernel at 3 cm and weight
1, termination 32 (ten times the 3.15 tracking ceiling). ``destination_v2`` keeps all of it but
halves the pay every centimetre, to train on from the native ``destination_v1`` checkpoint.
``destination_v3`` keeps v2 but concentrates the pay at the destination with the rational kernel,
to train on from the native ``destination_v2`` checkpoint. ``destination_v4`` keeps v3's kernel
without the hold charge and pays nothing for the destination under a zero command, also from the
native ``destination_v2`` checkpoint. ``destination_v5`` is v4 that pays the destination under a zero
command too: v3's kernel without either zero-command change. ``destination_v6`` is v5 that also
trains stopping: ``stop_fraction`` of the moving commands drawn last ``stop_after_min_controls`` to
``stop_after_max_controls`` controls and give way to a zero command for the rest of the
``command_controls``, so a walk is followed by a stop as in the evaluator's walk-then-stop probe.
``destination_v7`` is v5 with a heading term of the same form as the rational destination pay,

    heading = heading_weight / (1 + (|wrap(heading_d - heading)| / heading_width_rad) ** heading_power)

every control for the gap to the destination's heading, which starts at the body's heading and turns
only with the commanded yaw rate; a fall forfeits it and the fall-cost ceiling counts its weight.
``destination_v8`` is v7 with every joint penalty of the canonical reward doubled (``JOINT_PENALTIES``).
``destination_v9`` is v8 with v6's stops after a walk, for a short fine-tune from the native v8
checkpoint at v8's learning rate and final action deviation.
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
    destination_quiet: str | None = None
    hold_weight: float | None = None
    hold_grace_controls: int | None = None
    hold_deadband_rad: float | None = None
    hold_span_rad: float | None = None
    hold_smoothing_controls: int | None = None
    stop_fraction: float | None = None
    stop_after_min_controls: int | None = None
    stop_after_max_controls: int | None = None
    heading_weight: float | None = None
    heading_width_rad: float | None = None
    heading_power: float | None = None
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
        stop = (self.stop_fraction, self.stop_after_min_controls, self.stop_after_max_controls)
        if any(value is not None for value in stop) and (
            any(value is None for value in stop)
            or not math.isfinite(self.stop_fraction)
            or not 0 < self.stop_fraction <= 1
            or type(self.stop_after_min_controls) is not int
            or type(self.stop_after_max_controls) is not int
            or not 10 <= self.stop_after_min_controls <= self.stop_after_max_controls
            or self.stop_after_max_controls >= self.command_controls
        ):
            raise ValueError(
                "Stops need a fraction in (0, 1] and 10 to fewer than command_controls controls "
                "of walking first"
            )
        heading = (self.heading_weight, self.heading_width_rad, self.heading_power)
        if any(value is not None for value in heading) and (
            any(value is None for value in heading)
            or not all(math.isfinite(value) and value > 0 for value in heading)
        ):
            raise ValueError("The heading pay needs a positive finite weight, width and power")
        if self.destination_quiet not in (None, "flat", "off"):
            raise ValueError("destination_quiet is None (track the ghost), flat or off")
        hold = (
            self.hold_weight,
            self.hold_grace_controls,
            self.hold_deadband_rad,
            self.hold_span_rad,
            self.hold_smoothing_controls,
        )
        if any(value is not None for value in hold) and (
            any(value is None for value in hold)
            or not all(math.isfinite(value) for value in hold)
            or self.hold_weight <= 0
            or type(self.hold_grace_controls) is not int
            or not 1 <= self.hold_grace_controls <= 1000
            or self.hold_deadband_rad < 0
            or self.hold_span_rad <= 0
            or type(self.hold_smoothing_controls) is not int
            or not 1 <= self.hold_smoothing_controls <= self.hold_grace_controls
        ):
            raise ValueError(
                "The hold charge needs a positive weight, 1 to 1000 grace controls, "
                "a nonnegative deadband and a positive span and 1 to grace smoothing controls"
            )
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
            + (self.heading_weight or 0.0)
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
# 0.23 at 1 cm, 0.12 at 2 cm, 0.04 at 5 cm). A zero command pays the full destination weight and,
# from 2 s in, charges up to 2 per control for joints whose 1 s running mean drifts 5 mrad to
# 50 mrad from where it stood (exploration noise alone moves standing joints about 20 mrad).
EXPERIMENTS["destination_v3"] = {
    **EXPERIMENTS["destination_v2"],
    "reward": replace(
        EXPERIMENTS["destination_v2"]["reward"],
        destination_kernel="rational",
        destination_distance_m=0.0037,
        destination_power=1.2,
        destination_quiet="flat",
        hold_weight=2.0,
        hold_grace_controls=100,
        hold_deadband_rad=0.005,
        hold_span_rad=0.045,
        hold_smoothing_controls=50,
    ),
}
# destination_v4 keeps destination_v3's rational kernel without the hold charge and pays nothing for
# the destination while the command is zero; it also trains on from destination_v2.
EXPERIMENTS["destination_v4"] = {
    **EXPERIMENTS["destination_v3"],
    "reward": replace(
        EXPERIMENTS["destination_v3"]["reward"],
        destination_quiet="off",
        hold_weight=None,
        hold_grace_controls=None,
        hold_deadband_rad=None,
        hold_span_rad=None,
        hold_smoothing_controls=None,
    ),
}
# destination_v5 is destination_v4 that pays the destination under a zero command as well: v3's
# rational kernel with neither zero-command change, also from destination_v2.
EXPERIMENTS["destination_v5"] = {
    **EXPERIMENTS["destination_v4"],
    "reward": replace(EXPERIMENTS["destination_v4"]["reward"], destination_quiet=None),
}
# destination_v6 is destination_v5 that also trains stopping: a fifth of the moving commands last
# 4 s to 12 s and then give way to a zero command for the rest of the 20 s.
EXPERIMENTS["destination_v6"] = {
    **EXPERIMENTS["destination_v5"],
    "reward": replace(
        EXPERIMENTS["destination_v5"]["reward"],
        stop_fraction=0.2,
        stop_after_min_controls=200,
        stop_after_max_controls=600,
    ),
}
# destination_v7 is destination_v5 with a heading pay shaped like its destination pay: 1 at the
# destination's heading, 0.5 at 1 deg, 0.3 at 2 deg, 0.13 at 5 deg. Termination 42 keeps ten times
# the 4.15 tracking ceiling.
EXPERIMENTS["destination_v7"] = {
    **EXPERIMENTS["destination_v5"],
    "reward": replace(
        EXPERIMENTS["destination_v5"]["reward"],
        heading_weight=1.0,
        heading_width_rad=math.radians(1.0),
        heading_power=1.2,
        termination_weight=42.0,
    ),
}
# The canonical reward's joint penalties: torque, acceleration, target steps, torque requests over
# the rating, demand near the rating, joint-limit margin, action range, and the zero-command joint
# rate, target motion and coxa strain.
JOINT_PENALTIES = (
    "torque_weight",
    "acceleration_weight",
    "target_rate_weight",
    "torque_limit_weight",
    "over_rating_weight",
    "joint_margin_weight",
    "action_limit_weight",
    "quiet_joint_rate_weight",
    "quiet_target_weight",
    "quiet_strain_weight",
)
# destination_v8 is destination_v7 with every joint penalty doubled, also from destination_v2.
EXPERIMENTS["destination_v8"] = {
    **EXPERIMENTS["destination_v7"],
    "reward": replace(
        EXPERIMENTS["destination_v7"]["reward"],
        **{
            name: 2 * getattr(EXPERIMENTS["destination_v7"]["reward"], name)
            for name in JOINT_PENALTIES
        },
    ),
}
# destination_v9 is destination_v8 with destination_v6's stops: a fifth of the moving commands last
# 4 s to 12 s and then give way to a zero command for the rest of the 20 s.
EXPERIMENTS["destination_v9"] = {
    **EXPERIMENTS["destination_v8"],
    "reward": replace(
        EXPERIMENTS["destination_v8"]["reward"],
        stop_fraction=0.2,
        stop_after_min_controls=200,
        stop_after_max_controls=600,
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
        self.hold_anchor = torch.zeros(env.num_envs, 18, device=env.device)
        self.hold_mean = torch.zeros(env.num_envs, 18, device=env.device)
        self.stop_pending = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.stop_rest = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        super().__init__(env, config, output_dir, chosen)

    def reset(self, indices=None):
        # A reset draws a fresh command; a stop scheduled for the ended episode lapses.
        self.stop_pending[self._indices(indices)] = False
        return super().reset(indices)

    def _resample(self, indices):
        super()._resample(indices)
        if len(indices):
            self.command_started[indices] = True
            if self.reward_config.command_controls:
                self.remaining_controls[indices] = self.reward_config.command_controls
            if self.reward_config.stop_fraction is not None:
                self._schedule_stops(indices)

    def _schedule_stops(self, indices):
        """End a scheduled walk with a zero command; cut a share of fresh moving commands short."""
        config = self.reward_config
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.stop_pending.device)
        stopping = indices[self.stop_pending[indices]]
        self.env.commands[stopping] = 0
        self.remaining_controls[stopping] = self.stop_rest[stopping].to(self.remaining_controls)
        self.stop_pending[stopping] = False
        fresh = indices[~torch.isin(indices, stopping)]
        moving = (self.env.commands[fresh] != 0).any(-1).to(fresh.device)
        draw = torch.rand(len(fresh), generator=self.rng).to(fresh.device)
        walking = torch.randint(
            config.stop_after_min_controls,
            config.stop_after_max_controls + 1,
            (len(fresh),),
            generator=self.rng,
        ).to(fresh.device)
        chosen = fresh[moving & (draw < config.stop_fraction)]
        walking = walking[moving & (draw < config.stop_fraction)]
        self.remaining_controls[chosen] = walking.to(self.remaining_controls)
        self.stop_rest[chosen] = config.command_controls - walking
        self.stop_pending[chosen] = True

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
            quiet = (command == 0).all(-1).to(closeness.device)
            if config.destination_quiet == "flat":
                closeness = torch.where(quiet, torch.ones_like(closeness), closeness)
            elif config.destination_quiet == "off":
                closeness = torch.where(quiet, torch.zeros_like(closeness), closeness)
            alive = (~terminated).to(closeness.device).to(closeness)
            components["destination"] = (config.destination_weight * alive * closeness).to(reward)
            reward = reward + components["destination"]
        if config.hold_weight is not None:
            components["hold"] = self._hold(command, terminated).to(reward)
            reward = reward + components["hold"]
        if config.heading_weight is not None:
            actual = planar_pose(telemetry["root_pose_xyzw"]).to(self.destination)[:, 2]
            gap = torch.remainder(self.destination[:, 2] - actual + math.pi, 2 * math.pi) - math.pi
            closeness = 1 / (1 + (gap.abs() / config.heading_width_rad).pow(config.heading_power))
            alive = (~terminated).to(closeness.device).to(closeness)
            components["heading"] = (config.heading_weight * alive * closeness).to(reward)
            reward = reward + components["heading"]
        return reward, components, telemetry

    def _hold(self, command, terminated):
        """Anchor the joints' running mean a grace period into a zero command and charge its drift."""
        config = self.reward_config
        q = self.env.telemetry["joint_position_rad"].detach().to(self.hold_anchor)
        age = self.command_age.to(q.device)
        quiet = (command.to(q.device) == 0).all(-1)
        self.hold_mean = torch.where(
            (age <= 1)[:, None],
            q,
            self.hold_mean + (q - self.hold_mean) / config.hold_smoothing_controls,
        )
        anchor = quiet & (age == config.hold_grace_controls)
        self.hold_anchor = torch.where(anchor[:, None], self.hold_mean, self.hold_anchor)
        drift = (
            ((self.hold_mean - self.hold_anchor).abs() - config.hold_deadband_rad)
            / config.hold_span_rad
        ).clamp(0, 1)
        active = quiet & (age >= config.hold_grace_controls) & ~terminated.to(q.device)
        return -config.hold_weight * active.to(q) * drift.mean(-1)

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
