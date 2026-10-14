"""Liu et al. Table III actor and critic for RSL-RL PPO (docs/TRAINING.md Step 6).

The actor reads the five proprioception frames, the command and the previous
action from the existing 231-value policy observation. A velocity estimator
[64, 32] predicts base linear velocity from the five frames; a memory encoder
[512, 256, 128] compresses them into ``h_t``. The low-level network [256, 128,
64] receives the command, previous action, current frame, detached velocity
estimate and memory latent. The estimator is trained by supervised regression
against measured velocity; the memory encoder is trained by PPO.

The critic reads the same command, previous action and current frame plus an
8-value latent from the privileged encoder [64, 32] over the 42-value
privileged state. The paper's terrain encoder belongs to Step 8; the GPS-only
robot carries no terrain sensor, so the critic input omits ``l_t^e`` here.

Privileged inputs never reach the actor: its observation groups exclude the
privileged state and the velocity label, and ``test_paper_networks`` checks it.
"""
from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP

PROPRIO_WIDTH = 42
HISTORY_FRAMES = 5
HISTORY_WIDTH = PROPRIO_WIDTH * HISTORY_FRAMES
COMMAND_WIDTH = 3
ACTION_WIDTH = 18
VELOCITY_WIDTH = 3
PRIVILEGED_WIDTH = 42
PRIVILEGED_LATENT = 8
ESTIMATOR_HIDDEN = (64, 32)
MEMORY_HIDDEN = (512, 256, 128)
PRIVILEGED_HIDDEN = (64, 32)
ACTOR_HIDDEN = (256, 128, 64)
CRITIC_HIDDEN = (512, 256, 128)
ACTOR_GROUPS = ('proprio_history', 'command', 'previous_action')
CRITIC_GROUPS = ('proprio_history', 'command', 'previous_action', 'privileged')
DECISIONS = (
    "Table III gives no width for the memory latent h_t; this implementation uses 32.",
    "The privileged latent l_t^p is 8 values and the terrain latent l_t^e is omitted until Step 8.",
    "The velocity estimate enters the low-level actor detached; only the supervised loss trains the estimator.",
    "Observation normalization is RSL-RL's empirical normalizer over each model's concatenated groups.",
    "Hidden activations are ELU; the paper does not state activations.",
)


def _widths(obs, groups):
    return tuple(int(obs[group].shape[-1]) for group in groups)


class PaperActor(MLPModel):
    """Estimator, memory encoder and low-level actor from Table III."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=ACTOR_HIDDEN, activation='elu',
                 obs_normalization=True, distribution_cfg=None, memory_latent=32):
        if tuple(obs_groups[obs_set]) != ACTOR_GROUPS:
            raise ValueError(f"The paper actor reads exactly {ACTOR_GROUPS}, got {obs_groups[obs_set]}")
        if _widths(obs, ACTOR_GROUPS) != (HISTORY_WIDTH, COMMAND_WIDTH, ACTION_WIDTH):
            raise ValueError("Actor observation widths differ from five 42-value frames, command and action")
        if type(memory_latent) is not int or memory_latent < 1:
            raise ValueError("Memory latent width must be a positive integer")
        self._memory_latent = memory_latent
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         obs_normalization, distribution_cfg)
        self.estimator = MLP(HISTORY_WIDTH, VELOCITY_WIDTH, ESTIMATOR_HIDDEN, activation)
        self.memory = MLP(HISTORY_WIDTH, memory_latent, MEMORY_HIDDEN, activation)
        self.velocity_estimate = None

    def _get_latent_dim(self):
        return COMMAND_WIDTH + ACTION_WIDTH + PROPRIO_WIDTH + VELOCITY_WIDTH + self._memory_latent

    def get_latent(self, obs, masks=None, hidden_state=None):
        normalized = self.obs_normalizer(torch.cat([obs[group] for group in self.obs_groups], dim=-1))
        history = normalized[..., :HISTORY_WIDTH]
        command = normalized[..., HISTORY_WIDTH:HISTORY_WIDTH + COMMAND_WIDTH]
        previous_action = normalized[..., HISTORY_WIDTH + COMMAND_WIDTH:]
        current = history[..., -PROPRIO_WIDTH:]
        self.velocity_estimate = self.estimator(history)
        memory = self.memory(history)
        return torch.cat((command, previous_action, current, self.velocity_estimate.detach(), memory), dim=-1)

    def estimator_parameters(self):
        return list(self.estimator.parameters())

    def estimator_loss(self, obs, label_group='velocity_label'):
        """Supervised regression of the estimate against measured base velocity."""
        self.get_latent(obs)
        label = obs[label_group]
        if label.shape != self.velocity_estimate.shape:
            raise ValueError("Velocity label shape differs from the estimate")
        return torch.nn.functional.mse_loss(self.velocity_estimate, label)

    def as_jit(self):
        raise NotImplementedError("Paper actor export needs a composite wrapper; training only in Step 6")

    def as_onnx(self, verbose=False):
        raise NotImplementedError("Paper actor export needs a composite wrapper; training only in Step 6")


class PaperCritic(MLPModel):
    """Privileged encoder and critic from Table III without the terrain latent."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=CRITIC_HIDDEN, activation='elu',
                 obs_normalization=True, distribution_cfg=None):
        if tuple(obs_groups[obs_set]) != CRITIC_GROUPS:
            raise ValueError(f"The paper critic reads exactly {CRITIC_GROUPS}, got {obs_groups[obs_set]}")
        if _widths(obs, CRITIC_GROUPS) != (HISTORY_WIDTH, COMMAND_WIDTH, ACTION_WIDTH, PRIVILEGED_WIDTH):
            raise ValueError("Critic observation widths differ from history, command, action and privileged state")
        if distribution_cfg is not None:
            raise ValueError("The critic is deterministic")
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, None)
        self.privileged_encoder = MLP(PRIVILEGED_WIDTH, PRIVILEGED_LATENT, PRIVILEGED_HIDDEN, activation)

    def _get_latent_dim(self):
        return COMMAND_WIDTH + ACTION_WIDTH + PROPRIO_WIDTH + PRIVILEGED_LATENT

    def get_latent(self, obs, masks=None, hidden_state=None):
        normalized = self.obs_normalizer(torch.cat([obs[group] for group in self.obs_groups], dim=-1))
        start = HISTORY_WIDTH + COMMAND_WIDTH + ACTION_WIDTH
        history = normalized[..., :HISTORY_WIDTH]
        command = normalized[..., HISTORY_WIDTH:HISTORY_WIDTH + COMMAND_WIDTH]
        previous_action = normalized[..., HISTORY_WIDTH + COMMAND_WIDTH:start]
        latent = self.privileged_encoder(normalized[..., start:start + PRIVILEGED_WIDTH])
        return torch.cat((command, previous_action, history[..., -PROPRIO_WIDTH:], latent), dim=-1)

    def as_jit(self):
        raise NotImplementedError("The critic is not exported")

    def as_onnx(self, verbose=False):
        raise NotImplementedError("The critic is not exported")


def layer_widths(module):
    """Linear layer output widths in order, for architecture checks."""
    return [layer.out_features for layer in module.modules() if isinstance(layer, torch.nn.Linear)]


def actor_observation(policy_observation):
    """Split the 231-value policy observation into the actor's named groups."""
    if policy_observation.shape[-1] != HISTORY_WIDTH + COMMAND_WIDTH + ACTION_WIDTH:
        raise ValueError("Expected the 231-value policy observation")
    batch = policy_observation.shape[:-1]
    return TensorDict({'proprio_history': policy_observation[..., :HISTORY_WIDTH],
                       'command': policy_observation[..., HISTORY_WIDTH:HISTORY_WIDTH + COMMAND_WIDTH],
                       'previous_action': policy_observation[..., HISTORY_WIDTH + COMMAND_WIDTH:]},
                      batch_size=list(batch))
