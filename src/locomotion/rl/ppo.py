"""Extend stock RSL-RL PPO with command bootstraps, a rate cap, bounded means and a deviation schedule."""

import copy
import math

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import resolve_callable
from torch import nn


class CommandBootstrapPPO(PPO):
    """Value a completed command hold at its post-action state with the held command."""

    def process_env_step(self, obs, rewards, dones, extras):
        switched = extras.get("command_time_outs")
        if switched is not None:
            with torch.no_grad():
                values = self.critic(extras["command_bootstrap_observation"]).squeeze(-1)
            rewards = rewards + self.gamma * values * switched.to(values)
            # Stock PPO uses pre-action values for timeouts. Exclude the holds handled above.
            extras = {**extras, "time_outs": extras["time_outs"] & ~switched}
        super().process_env_step(obs, rewards, dones, extras)


class CappedRatePPO(CommandBootstrapPPO):
    """Stock PPO whose learning rate never exceeds ``learning_rate_max``."""

    def __init__(self, *args, learning_rate_max, **kwargs):
        self.learning_rate_max = float(learning_rate_max)
        super().__init__(*args, **kwargs)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = min(float(value), self.learning_rate_max)


# The environment clips sampled actions before its physical target limits. PPO stores those raw
# samples and evaluates their Gaussian likelihoods. Only the mean uses tanh; this distribution
# does not squash sampled actions.
class BoundedMeanGaussian(GaussianDistribution):
    """Keep the policy mean in [-1, 1] for training and inference."""

    def update(self, mlp_output):
        super().update(torch.tanh(mlp_output))

    def deterministic_output(self, mlp_output):
        return torch.tanh(mlp_output)

    def as_deterministic_output_module(self):
        return nn.Tanh()


class CorrelatedBoundedMeanGaussian(BoundedMeanGaussian):
    """Bounded mean whose exploration noise persists across controls."""

    def __init__(self, output_dim, init_std=1.0, std_type="scalar", noise_correlation=0.0):
        super().__init__(output_dim, init_std=init_std, std_type=std_type)
        if not 0 <= noise_correlation < 1:
            raise ValueError("Noise correlation must lie in [0, 1)")
        self.noise_correlation = float(noise_correlation)
        self._noise = None

    def sample(self):
        mean, std = self.mean, self.std
        white = torch.randn_like(mean)
        if self._noise is None or self._noise.shape != white.shape:
            self._noise = white
        else:
            keep = self.noise_correlation
            self._noise = keep * self._noise + (1 - keep * keep) ** 0.5 * white
        return mean + std * self._noise


def load_actor(checkpoint, config, observations, device):
    """Load the recorded stock actor without constructing a learner."""
    settings = copy.deepcopy(config["actor"])
    model = resolve_callable(settings.pop("class_name"))
    target = torch.device(device)
    devices = (
        [target.index if target.index is not None else torch.cuda.current_device()]
        if target.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        actor = model(observations, config["obs_groups"], "actor", 18, **settings).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()
    return actor, checkpoint.get("infos")


class DeviationSchedule:
    """Hold the action deviation on a declared linear schedule; PPO no longer learns it."""

    def __init__(self, algorithm, start, final, updates):
        self.algorithm, self.start, self.final, self.updates = (
            algorithm,
            float(start),
            float(final),
            int(updates),
        )
        self.parameter = algorithm.actor.distribution.log_std_param
        self.parameter.requires_grad_(False)
        self.inner_update = algorithm.update
        self.completed = 0
        self.apply()

    def value(self):
        return self.start + (self.final - self.start) * min(1.0, self.completed / self.updates)

    def apply(self):
        with torch.no_grad():
            self.parameter.fill_(math.log(self.value()))

    def update(self):
        losses = self.inner_update()
        self.completed += 1
        self.apply()
        return losses
