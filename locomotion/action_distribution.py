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
