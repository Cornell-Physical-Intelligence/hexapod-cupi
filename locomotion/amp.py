"""Shared 61-value AMP feature contract, discriminator and style reward.

The feature contract (``CONTRACT``, ``extract_features``) is the dataset side of
issue #36: demonstrations and policies pack the same raw body-frame measurements
without commands or gait phase.

``Discriminator`` scores AMP transitions (s_t, s_t+1) built from the kernel's
61-value AMP state; ``style_reward`` is Eq. (2); ``discriminator_loss`` is
Eq. (1) (Liu et al., Table III). In the paper the discriminator trains against
the current policy during PPO, which needs the native trainer
(docs/TRAINING.md Step 5). ``train`` fits a frozen discriminator offline from
recorded traces so the style term can be inspected on existing rollouts; that
is a diagnostic, not AMP training.

Factories for the scorer and viewer take a checkpoint path, for example
``--reward style=locomotion.amp:paper_with_style:disc.pt``.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import paper_reward
from .env_config import JOINT_NAMES, LEGS

CONTRACT = {
    'schema': 'hexapod_amp_features_v1', 'width': 61, 'control_dt_s': .02,
    'joint_names': list(JOINT_NAMES), 'leg_order': list(LEGS),
    'body_axes': ['left', 'backward', 'up'],
    'fields': [
        {'name': 'joint_position', 'slice': [0, 18], 'unit': 'rad', 'frame': 'native_joint_absolute'},
        {'name': 'joint_velocity', 'slice': [18, 36], 'unit': 'rad/s', 'frame': 'native_joint'},
        {'name': 'linear_velocity', 'slice': [36, 39], 'unit': 'm/s', 'frame': 'body_root_origin'},
        {'name': 'angular_velocity', 'slice': [39, 42], 'unit': 'rad/s', 'frame': 'body'},
        {'name': 'height', 'slice': [42, 43], 'unit': 'm', 'frame': 'world_z_above_flat_zero_plane'},
        {'name': 'toe_position', 'slice': [43, 61], 'unit': 'm', 'frame': 'body_root_relative_xyz'},
    ],
    'normalization': 'none; learner normalization must preserve this raw contract',
    'interpretation': 'James retains six XYZ toe positions; six scalar heights would give 49 values.',
    'paper': 'https://arxiv.org/html/2511.03167v1#S3.SS1',
}

AMP_WIDTH = CONTRACT['width']
DECISIONS = (
    "Gradient penalty: Eq. (1) prints the gradient with respect to discriminator parameters; this "
    "follows the cited AMP method and penalizes the gradient with respect to prior transitions.",
    "The gradient-penalty coefficient is not stated; alpha_gp = 10.",
    "Hidden layers [1024, 512] follow Table III; ELU activations and a linear output are not stated there.",
    "Inputs are standardized with the motion-prior dataset's per-feature mean and standard deviation.",
    "The AMP state uses six 3D toe positions in the body frame and the root height above the flat floor.",
)


def feature_contract():
    return copy.deepcopy(CONTRACT)


def extract_features(state):
    """Pack native body-frame measurements without commands or gait phase."""
    batch = state['q'].shape[:-1]
    shapes = {'q': (18,), 'dq': (18,), 'linear': (3,), 'angular': (3,), 'toe_body': (6, 3)}
    for name, tail in shapes.items():
        if state[name].shape != batch+tail:
            raise ValueError('AMP field shape differs: '+name)
    if state['root'].shape not in (batch+(3,), batch+(7,)):
        raise ValueError('AMP root requires XYZ position or XYZ/XYZW pose')
    parts = (state['q'], state['dq'], state['linear'], state['angular'],
             state['root'][..., 2:3], state['toe_body'].reshape(batch+(18,)))
    if isinstance(state['q'], torch.Tensor):
        return torch.cat(parts, dim=-1)
    return np.concatenate(parts, axis=-1)


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


def _style_components(telemetry, discriminator):
    with torch.no_grad():
        scores = discriminator(transitions(telemetry["amp_state"], telemetry["next_amp_state"]))
    return {"style": style_reward(scores)}


def _with_style(base, path):
    discriminator, _ = load(path)

    def reward(telemetry, commands, previous_target, terminated, config=None, nominal_height=None):
        components = {}
        if base is not None:
            _, components = base(telemetry, commands, previous_target, terminated, config, nominal_height)
        components = {**components, **_style_components(telemetry, discriminator)}
        return sum(components.values()), components
    return reward


def style_only(path):
    """The style term alone, for inspecting the discriminator."""
    return _with_style(None, path)


def paper_with_style(path):
    """Table I task + style + printed penalties."""
    return _with_style(paper_reward.paper_reward, path)


def calibrated_with_style(path):
    """Table I task + style + calibrated penalties."""
    return _with_style(paper_reward.paper_reward_calibrated, path)


def main(argv=None):
    from . import reward_scorer
    parser = argparse.ArgumentParser(description="Train a frozen AMP discriminator from recorded traces.")
    parser.add_argument("--prior", nargs="+", required=True, help="Motion-prior control_trace.npz files.")
    parser.add_argument("--policy", nargs="+", required=True, help="Policy control_trace.npz files (negatives).")
    parser.add_argument("--output", type=Path, required=True, help="Discriminator checkpoint to write.")
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    args = parser.parse_args(argv)
    config = TrainConfig(steps=args.steps)
    prior_traces = [reward_scorer.load_trace(path) for path in args.prior]
    policy_traces = [reward_scorer.load_trace(path) for path in args.policy]
    prior = torch.cat([trace_transitions(trace) for trace in prior_traces])
    policy = torch.cat([trace_transitions(trace) for trace in policy_traces])
    discriminator, history = train(prior, policy, config)
    metadata = {"config": asdict(config), "decisions": list(DECISIONS), "history": history,
                "prior": [{"path": str(t.path), "sha256": t.sha256} for t in prior_traces],
                "policy": [{"path": str(t.path), "sha256": t.sha256} for t in policy_traces],
                "scope": "Frozen offline discriminator for inspecting recorded rollouts; AMP trains it against "
                         "the current policy during PPO."}
    save(discriminator, args.output, metadata)
    with torch.no_grad():
        summary = {"prior_style_mean": float(style_reward(discriminator(prior)).mean()),
                   "policy_style_mean": float(style_reward(discriminator(policy)).mean())}
    print(json.dumps({"output": str(args.output), **summary, "final": history[-1]}, indent=2))


if __name__ == "__main__":
    main()
