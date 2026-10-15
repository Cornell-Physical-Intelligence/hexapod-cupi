"""Adapt command-boundary targets and cap the learning rate of stock RSL-RL PPO."""
import torch
from rsl_rl.algorithms import PPO


class CommandBootstrapPPO(PPO):
    """Value a completed command hold at its post-action state with the held command."""

    def process_env_step(self, obs, rewards, dones, extras):
        switched = extras.get('command_time_outs')
        if switched is not None:
            with torch.no_grad():
                values = self.critic(extras['command_bootstrap_observation']).squeeze(-1)
            rewards = rewards + self.gamma * values * switched.to(values)
            # Stock PPO uses pre-action values for timeouts. Exclude the holds handled above.
            extras = {**extras, 'time_outs': extras['time_outs'] & ~switched}
        super().process_env_step(obs, rewards, dones, extras)


class CappedRatePPO(CommandBootstrapPPO):
    """Stock PPO whose learning rate never exceeds ``learning_rate_max``."""

    def __init__(self, *args, learning_rate_max, **kwargs):
        self.learning_rate_max = float(learning_rate_max)
        super().__init__(*args, **kwargs)
        for group in self.optimizer.param_groups:
            group['lr'] = self.learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = min(float(value), self.learning_rate_max)


"""Bound the Gaussian mean while retaining raw samples for stock PPO.

The environment clips sampled actions before its physical target limits. PPO
stores those raw samples and evaluates their Gaussian likelihoods. Only the
mean uses tanh; this distribution does not squash sampled actions.
"""
import torch
from torch import nn

from rsl_rl.modules.distribution import GaussianDistribution


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

    def __init__(self, output_dim, init_std=1., std_type='scalar', noise_correlation=0.):
        super().__init__(output_dim, init_std=init_std, std_type=std_type)
        if not 0 <= noise_correlation < 1:
            raise ValueError('Noise correlation must lie in [0, 1)')
        self.noise_correlation = float(noise_correlation)
        self._noise = None

    def sample(self):
        mean, std = self.mean, self.std
        white = torch.randn_like(mean)
        if self._noise is None or self._noise.shape != white.shape:
            self._noise = white
        else:
            keep = self.noise_correlation
            self._noise = keep * self._noise + (1 - keep * keep) ** .5 * white
        return mean + std * self._noise
