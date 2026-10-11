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
    """Bounded mean whose exploration noise persists across controls.

    Each replica's noise follows a first-order autoregression with unit stationary variance. One
    sample keeps the Gaussian marginal that the PPO likelihood assumes; consecutive samples share
    the fraction ``noise_correlation`` of their noise. Independent noise at 50 Hz shakes the body
    and cannot hold a joint offset for the several controls that lifting a foot takes.
    """

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
