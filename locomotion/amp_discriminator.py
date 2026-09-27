"""Adversarial motion-prior discriminator and tripod style reward (Liu et al., Eqs. (1)-(2), Table III).

``Discriminator`` scores AMP transitions (s_t, s_t+1) built from the kernel's
61-value AMP state; ``style_reward`` is Eq. (2); ``discriminator_loss`` is
Eq. (1). In the paper the discriminator trains against the current policy during
PPO, which needs the native trainer (docs/TRAINING.md Step 5). ``train`` fits a
frozen discriminator offline from recorded traces so the style term can be
inspected on existing rollouts; that is a diagnostic, not AMP training.

The style-reward factories for the scorer and viewer and the command that trains
a discriminator from traces live in ``locomotion.paper_reward``, for example
``--reward style=locomotion.paper_reward:paper_with_style:disc.pt``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# The 61-value AMP state (docs/TRAINING.md). PR #39 adds locomotion/amp.py with
# feature_contract(); once it merges, take this width from there.
AMP_WIDTH = 61
DECISIONS = (
    "Gradient penalty: Eq. (1) prints the gradient with respect to discriminator parameters; this "
    "follows the cited AMP method and penalizes the gradient with respect to prior transitions.",
    "The gradient-penalty coefficient is not stated; alpha_gp = 10.",
    "Hidden layers [1024, 512] follow Table III; ELU activations and a linear output are not stated there.",
    "Inputs are standardized with the motion-prior dataset's per-feature mean and standard deviation.",
    "The AMP state uses six 3D toe positions in the body frame and the root height above the flat floor.",
)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 20260925
    steps: int = 3000
    batch_size: int = 512
    learning_rate: float = 1e-4
    gradient_penalty: float = 10.
    hidden: tuple = (1024, 512)


class Discriminator(nn.Module):
    def __init__(self, mean, std, hidden=(1024, 512)):
        super().__init__()
        layers, width = [], 2 * AMP_WIDTH
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ELU()]
            width = size
        self.net = nn.Sequential(*layers, nn.Linear(width, 1))
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, transitions):
        return self.net((transitions - self.mean) / self.std).squeeze(-1)


def transitions(before, after):
    """Concatenate (s_t, s_t+1) AMP states into discriminator inputs."""
    before, after = torch.as_tensor(before, dtype=torch.float32), torch.as_tensor(after, dtype=torch.float32)
    if before.shape != after.shape or before.shape[-1] != AMP_WIDTH:
        raise ValueError(f"AMP states must be matching (..., {AMP_WIDTH}) arrays")
    return torch.cat((before, after), dim=-1).reshape(-1, 2 * AMP_WIDTH)


def trace_transitions(trace):
    return transitions(trace.data["amp_state_before"], trace.data["amp_state_after"])


def style_reward(scores):
    """Eq. (2): max(0, 1 - 0.25 (D - 1)^2), in [0, 1]."""
    return (1 - .25 * (scores - 1).square()).clamp_min(0)


def discriminator_loss(discriminator, prior, policy, gradient_penalty=10.):
    """Eq. (1): least-squares targets +1 for prior and -1 for policy transitions, plus a gradient penalty."""
    prior = prior.clone().requires_grad_(True)
    prior_scores = discriminator(prior)
    gradient, = torch.autograd.grad(prior_scores.sum(), prior, create_graph=True)
    terms = {"prior": (prior_scores - 1).square().mean(), "policy": (discriminator(policy) + 1).square().mean(),
             "gradient_penalty": .5 * gradient_penalty * gradient.square().sum(-1).mean()}
    return sum(terms.values()), {key: float(value.detach()) for key, value in terms.items()}


def train(prior, policy, config=TrainConfig()):
    """Fit a frozen discriminator: prior transitions against recorded policy transitions."""
    generator = torch.Generator().manual_seed(config.seed)
    torch.manual_seed(config.seed)
    prior, policy = torch.as_tensor(prior, dtype=torch.float32), torch.as_tensor(policy, dtype=torch.float32)
    std = prior.std(0).clamp_min(1e-3)
    discriminator = Discriminator(prior.mean(0), std, config.hidden)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate)
    history = []
    for step in range(config.steps):
        batch = lambda data: data[torch.randint(len(data), (config.batch_size,), generator=generator)]
        loss, terms = discriminator_loss(discriminator, batch(prior), batch(policy), config.gradient_penalty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == config.steps - 1:
            history.append({"step": step, **terms})
    return discriminator.eval(), history


def save(discriminator, path, metadata):
    torch.save({"state_dict": discriminator.state_dict(), "hidden": list(metadata["config"]["hidden"]),
                "metadata": metadata}, path)


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    discriminator = Discriminator(state["mean"], state["std"], tuple(checkpoint["hidden"]))
    discriminator.load_state_dict(state)
    return discriminator.eval(), checkpoint["metadata"]
