"""Fresh-model PPO and adversarial motion priors for the canonical robot.

The simulator owns actual physics, position-target limits and terminal states.
This module never substitutes reference motion for a simulated policy rollout.
Paper-inspired pilot choices (including flat-ground critic and optimizer values)
are explicit in Config. Checkpoints restore learning state, not PhysX state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

OBS_WIDTH = 231
CRITIC_WIDTH = 234
AMP_WIDTH = 61
ACTION_WIDTH = 18
SCHEMA = "canonical_paper_walk_ppo_amp_v3"
AMP_FIELDS = ["joint_position:18", "joint_velocity:18", "body_linear_velocity:3",
              "body_angular_velocity:3", "base_height:1", "body_frame_toe_xyz:18"]


@dataclass
class Config:
    num_envs: int = 128
    rollout_steps: int = 24
    epochs: int = 5
    minibatches: int = 4
    learning_rate: float = 3e-4
    discriminator_learning_rate: float = 3e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    entropy_coefficient: float = .01
    value_coefficient: float = 1.
    estimator_coefficient: float = 1.
    style_coefficient: float = 1.
    gradient_penalty: float = 10.
    max_grad_norm: float = 1.
    initial_std: float = .4
    target_kl: float | None = None
    rollback_kl_steps: bool = False
    kl_backtrack_halvings: int = 3
    kl_backtrack_factor: float = .5
    freeze_actor_obs_normalizer: bool = False
    zero_accepted_update_limit: int = 3
    bc_optimizer: str = "shared"
    bc_learning_rate: float | None = None
    bc_velocity_coefficient: float = 0.
    bc_detach_velocity: bool = False
    seed: int = 20260914
    actor_hidden: tuple = (256, 128, 64)
    memory_hidden: tuple = (512, 256, 128)
    estimator_hidden: tuple = (64, 32)
    critic_hidden: tuple = (512, 256, 128)
    discriminator_hidden: tuple = (1024, 512)

    def validate(self):
        for key in ("num_envs", "rollout_steps", "epochs", "minibatches"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError("Positive integer required: " + key)
        if self.num_envs * self.rollout_steps < self.minibatches:
            raise ValueError("More minibatches than rollout samples")
        for key in ("learning_rate", "discriminator_learning_rate", "initial_std", "max_grad_norm"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError("Positive finite configuration required: " + key)
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("Invalid discount or GAE lambda")
        for key in ("target_kl", "bc_learning_rate"):
            value = getattr(self, key)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError("Positive finite configuration required: " + key)
        if self.bc_optimizer not in ("shared", "separate"):
            raise ValueError("BC optimizer must be shared or separate")
        if self.bc_optimizer == "separate" and self.bc_learning_rate is None:
            raise ValueError("Separate BC requires an explicit bc_learning_rate")
        if self.bc_optimizer == "shared" and self.bc_learning_rate not in (None, self.learning_rate):
            raise ValueError("Shared BC must use the PPO learning_rate")
        if (isinstance(self.bc_velocity_coefficient, bool) or not math.isfinite(self.bc_velocity_coefficient)
                or self.bc_velocity_coefficient < 0):
            raise ValueError("BC velocity coefficient must be nonnegative and finite")
        if type(self.bc_detach_velocity) is not bool:
            raise ValueError("BC velocity detach must be boolean")
        for key in ("rollback_kl_steps", "freeze_actor_obs_normalizer"):
            if type(getattr(self, key)) is not bool:
                raise ValueError("Boolean configuration required: " + key)
        if type(self.kl_backtrack_halvings) is not int or self.kl_backtrack_halvings < 0:
            raise ValueError("Backtracking halvings must be a nonnegative integer")
        if (isinstance(self.kl_backtrack_factor, bool) or not math.isfinite(self.kl_backtrack_factor)
                or not 0 < self.kl_backtrack_factor < 1):
            raise ValueError("Backtracking factor must be finite and between zero and one")
        if type(self.zero_accepted_update_limit) is not int or self.zero_accepted_update_limit <= 0:
            raise ValueError("Zero-accepted update limit must be a positive integer")
        if self.rollback_kl_steps and self.target_kl is None:
            raise ValueError("Rollback requires an explicit target_kl")


@torch.no_grad()
def diagonal_gaussian_kl(old_mean, old_std, new_mean, new_std):
    """KL(old || new): sum 18 action coordinates, return one value per state."""
    values = (old_mean, old_std, new_mean, new_std)
    if (not all(isinstance(x, torch.Tensor) for x in values)
            or old_mean.ndim != 2 or old_mean.shape[1] != ACTION_WIDTH
            or any(x.shape != old_mean.shape for x in values)
            or any(not bool(torch.isfinite(x).all()) for x in values)
            or bool((old_std <= 0).any()) or bool((new_std <= 0).any())):
        raise ValueError("Invalid Gaussian KL inputs")
    old_mean, old_std, new_mean, new_std = (x.to(torch.float64) for x in values)
    result = (torch.log(new_std / old_std)
              + (old_std.square() + (old_mean - new_mean).square()) / (2 * new_std.square()) - .5).sum(-1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Nonfinite Gaussian KL")
    return result.clamp_min(0.)


class RunningMeanStd(nn.Module):
    """Statistics change only between rollouts, never during a PPO ratio update."""
    def __init__(self, width):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))

    @torch.no_grad()
    def update(self, values):
        values = values.reshape(-1, self.mean.numel())
        if not len(values) or not torch.isfinite(values).all():
            raise ValueError("Empty or nonfinite normalizer input")
        batch_mean = values.mean(0)
        batch_var = values.var(0, unbiased=False)
        n = len(values)
        total = self.count + n
        delta = batch_mean - self.mean
        mean = self.mean + delta * (n / total).to(values.dtype)
        m2 = self.var * self.count + batch_var * n + delta.square() * self.count * n / total
        self.mean.copy_(mean)
        self.var.copy_((m2 / total).clamp_min(1e-8))
        self.count.copy_(total)

    def forward(self, values):
        return ((values - self.mean) / torch.sqrt(self.var + 1e-6)).clamp(-10., 10.)


def mlp(width, hidden, output=None):
    layers = []
    for size in hidden:
        layers.extend((nn.Linear(width, size), nn.ELU()))
        width = size
    if output is not None:
        layers.append(nn.Linear(width, output))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.obs_normalizer = RunningMeanStd(OBS_WIDTH)
        self.critic_normalizer = RunningMeanStd(CRITIC_WIDTH)
        self.estimator = mlp(210, config.estimator_hidden, 3)
        self.memory = mlp(210, config.memory_hidden)
        self.policy = mlp(42 + 3 + 18 + 3 + config.memory_hidden[-1], config.actor_hidden, 18)
        self.critic = mlp(CRITIC_WIDTH, config.critic_hidden, 1)
        self.log_std = nn.Parameter(torch.full((18,), math.log(config.initial_std)))
        # Fresh near-neutral mean; no historical actor or phase-driven controller.
        nn.init.normal_(self.policy[-1].weight, std=.01)
        nn.init.zeros_(self.policy[-1].bias)

    def actor(self, obs, *, detach_velocity=False):
        normalized = self.obs_normalizer(obs)
        history = normalized[:, :210]
        velocity = self.estimator(history)
        memory = self.memory(history)
        current = history[:, -42:]
        policy_velocity = velocity.detach() if detach_velocity else velocity
        mean = self.policy(torch.cat((current, normalized[:, 210:], policy_velocity, memory), -1))
        return mean, velocity

    def distribution(self, obs):
        mean, velocity = self.actor(obs)
        return Normal(mean, self.log_std.clamp(-5., 1.).exp().expand_as(mean)), velocity

    def value(self, critic):
        return self.critic(self.critic_normalizer(critic)).squeeze(-1)


class MotionPrior(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.normalizer = RunningMeanStd(AMP_WIDTH)
        self.discriminator = mlp(2 * AMP_WIDTH, config.discriminator_hidden, 1)

    def features(self, first, second):
        return torch.cat((self.normalizer(first), self.normalizer(second)), -1)

    @torch.no_grad()
    def style_reward(self, first, second):
        score = self.discriminator(self.features(first, second)).squeeze(-1)
        return (1. - .25 * (score - 1.).square()).clamp(0., 1.)

    def loss(self, expert_first, expert_second, policy_first, policy_second, penalty):
        expert = self.features(expert_first, expert_second).detach().requires_grad_(True)
        policy = self.features(policy_first, policy_second).detach()
        real = self.discriminator(expert).squeeze(-1)
        fake = self.discriminator(policy).squeeze(-1)
        real_loss = (real - 1.).square().mean()
        fake_loss = (fake + 1.).square().mean()
        # Differentiate with respect to normalized expert INPUTS, not parameters.
        input_gradient = torch.autograd.grad(real.sum(), expert, create_graph=True)[0]
        gp = input_gradient.square().sum(-1).mean()
        return real_loss + fake_loss + .5 * penalty * gp, {
            "expert_score": real.detach().mean(), "policy_score": fake.detach().mean(),
            "gradient_penalty": gp.detach(), "discriminator_expert_loss": real_loss.detach(),
            "discriminator_policy_loss": fake_loss.detach()}


def generalized_advantage(rewards, values, next_values, terminated, truncated, gamma, lam):
    """Timeouts bootstrap their terminal state but never carry GAE across reset."""
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[-1])
    for t in reversed(range(len(rewards))):
        bootstrap = (~terminated[t]).to(rewards.dtype)
        continuation = (~(terminated[t] | truncated[t])).to(rewards.dtype)
        delta = rewards[t] + gamma * bootstrap * next_values[t] - values[t]
        carry = delta + gamma * lam * continuation * carry
        advantage[t] = carry
    return advantage, advantage + values


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(exact_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(exact_equal(x, y) for x, y in zip(a, b))
    return a == b


class PPOLearner:
    """Environment ABI: reset(indices=None), step(actions), dict-shaped results.

    obs: [N,231]; critic: [N,234]; amp: [N,61]. Step additionally returns
    reward, terminated and truncated [N], and MUST return terminal observations
    before resetting. reset(indices) returns the full batch and changes only
    those rows. All states and rewards must come from native simulation.
    """
    def __init__(self, env, prior_path, output_dir, config=None, device="cuda:0"):
        self.config = Config() if config is None else config
        self.config.validate()
        self.device = torch.device(device)
        self.env = env
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.prior_path = Path(prior_path)
        self.prior_sha256 = file_sha(self.prior_path)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self.model = ActorCritic(self.config).to(self.device)
        self.amp = MotionPrior(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.discriminator_optimizer = torch.optim.Adam(self.amp.discriminator.parameters(), lr=self.config.discriminator_learning_rate)
        with np.load(self.prior_path, allow_pickle=False) as prior:
            self.expert_states = torch.as_tensor(prior["states"].copy(), dtype=torch.float32, device=self.device)
            self.expert_next_states = torch.as_tensor(prior["next_states"].copy(), dtype=torch.float32, device=self.device)
        if self.expert_states.ndim != 2 or self.expert_states.shape[1] != AMP_WIDTH or self.expert_states.shape != self.expert_next_states.shape or len(self.expert_states) < 2:
            raise ValueError("Prior must contain aligned states/next_states [M,61]")
        if not torch.isfinite(self.expert_states).all() or not torch.isfinite(self.expert_next_states).all():
            raise ValueError("Nonfinite reference motion")
        self.updates = 0
        self.transitions = 0
        self.optimizer_steps = 0
        self.discriminator_steps = 0
        self.episodes = 0
        self.bc_steps = 0
        self.consecutive_zero_accepted_updates = 0
        self.current = None
        self.last_rollout = None
        self.last_metrics = {}
        self.normalizer_policy_kl = None
        protocol = {"schema": SCHEMA, "config": asdict(self.config), "amp_fields": AMP_FIELDS,
                    "prior_sha256": self.prior_sha256, "learner_sha256": file_sha(__file__),
                    "actor_observation": "five oldest-to-newest proprio42 frames, command3, previous_action18",
                    "critic_observation": "actor observation plus privileged native body linear velocity3",
                    "critic_simplification": "flat-ground pilot; no terrain encoder",
                    "simulation_resume_supported": False, "behavior_cloning_default": False,
                    "training_requested_by_user": True, "execution_authorization_owner": "root dispatcher",
                    "physical_admission": False, "stage2_complete": False}
        (self.output / "learner_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    def _tensor(self, value, shape, dtype=torch.float32):
        if dtype == torch.bool and torch.as_tensor(value).dtype != torch.bool:
            raise ValueError("Native termination/truncation flags must be boolean")
        result = torch.as_tensor(value, dtype=dtype, device=self.device).detach()
        if tuple(result.shape) != shape or not torch.isfinite(result).all():
            raise ValueError(f"Malformed/nonfinite native tensor: {tuple(result.shape)} expected {shape}")
        return result

    def _state(self, data):
        n = self.config.num_envs
        result = {key: self._tensor(data[key], (n, width)).clone()
                  for key, width in (("obs", OBS_WIDTH), ("critic", CRITIC_WIDTH), ("amp", AMP_WIDTH))}
        if not torch.equal(result["obs"], result["critic"][:, :OBS_WIDTH]):
            raise ValueError("Critic prefix must equal actor observation")
        return result

    def _expert(self, size):
        index = torch.randint(len(self.expert_states), (size,), device=self.device)
        return self.expert_states[index], self.expert_next_states[index]

    @torch.no_grad()
    def _normalizers(self):
        probe = self.current["obs"] if self.config.target_kl is not None else None
        if probe is not None:
            before, _ = self.model.distribution(probe)
            old_mean, old_std = before.mean.clone(), before.stddev.clone()
        samples = self.current if self.last_rollout is None else self.last_rollout
        if not self.config.freeze_actor_obs_normalizer:
            self.model.obs_normalizer.update(samples["obs"])
        self.model.critic_normalizer.update(samples["critic"])
        first, second = self._expert(min(2048, len(self.expert_states)))
        self.amp.normalizer.update(torch.cat((first, second, samples["amp"].reshape(-1, AMP_WIDTH))))
        if probe is not None:
            after, _ = self.model.distribution(probe)
            self.normalizer_policy_kl = float(diagonal_gaussian_kl(old_mean, old_std,
                                                                  after.mean, after.stddev).mean())

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        distribution, _ = self.model.distribution(obs)
        return distribution.mean if deterministic else distribution.sample()

    @torch.no_grad()
    def collect(self):
        if self.current is None:
            self.current = self._state(self.env.reset())
        self._normalizers()
        keys = ("obs", "critic", "amp", "amp_next", "actions", "log_prob", "value", "next_value",
                "reward", "task_reward", "style_reward", "terminated", "truncated")
        if self.config.target_kl is not None:
            keys += ("old_action_mean", "old_action_std")
        rows = {key: [] for key in keys}
        n = self.config.num_envs
        for _ in range(self.config.rollout_steps):
            before = self.current
            distribution, _ = self.model.distribution(before["obs"])
            action = distribution.sample()
            value = self.model.value(before["critic"])
            output = self.env.step(action)
            terminal = self._state(output)
            reward = self._tensor(output["reward"], (n,))
            terminated = self._tensor(output["terminated"], (n,), torch.bool)
            truncated = self._tensor(output["truncated"], (n,), torch.bool)
            style = self.amp.style_reward(before["amp"], terminal["amp"])
            record = {**before, "amp_next": terminal["amp"], "actions": action,
                      "log_prob": distribution.log_prob(action).sum(-1), "value": value,
                      "next_value": self.model.value(terminal["critic"]),
                      "reward": reward + self.config.style_coefficient * style,
                      "task_reward": reward, "style_reward": style,
                      "terminated": terminated, "truncated": truncated}
            if self.config.target_kl is not None:
                record.update(old_action_mean=distribution.mean, old_action_std=distribution.stddev)
            for key in keys:
                rows[key].append(record[key].detach().clone())
            done = terminated | truncated
            self.current = terminal
            if done.any():
                indices = done.nonzero(as_tuple=False).squeeze(-1)
                reset = self._state(self.env.reset(indices))
                # A selective reset cannot corrupt another robot's observation history.
                if any(not torch.equal(reset[key][~done], terminal[key][~done]) for key in terminal):
                    raise ValueError("Selective reset changed an unaffected environment")
                self.current = reset
                self.episodes += len(indices)
        rollout = {key: torch.stack(value) for key, value in rows.items()}
        advantage, returns = generalized_advantage(rollout["reward"], rollout["value"], rollout["next_value"],
                                                  rollout["terminated"], rollout["truncated"], self.config.gamma, self.config.gae_lambda)
        rollout.update(advantage=advantage, returns=returns)
        self.transitions += self.config.num_envs * self.config.rollout_steps
        self.last_rollout = rollout
        return rollout

    @torch.no_grad()
    def policy_kl(self, data):
        """Deterministic full-rollout check; never changes statistics or RNG."""
        distribution, _ = self.model.distribution(data["obs"])
        return float(diagonal_gaussian_kl(data["old_action_mean"], data["old_action_std"],
                                         distribution.mean, distribution.stddev).mean())

    def _bounded_model_step(self, data):
        """Trial one fixed clipped gradient, retaining only an accepted Adam step."""
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        gradients = [None if p.grad is None else p.grad.detach().clone() for p in self.model.parameters()]
        rng = self._rng()

        def restore():
            self.model.load_state_dict(model_state, strict=True)
            # Adam can alias tensors supplied to load_state_dict. Never expose
            # the immutable snapshot itself to a subsequent optimizer step.
            self.optimizer.load_state_dict(deepcopy(optimizer_state))
            for parameter, gradient in zip(self.model.parameters(), gradients):
                parameter.grad = None if gradient is None else gradient.clone()
            self._restore_rng(rng)

        attempts = []
        try:
            for retry in range(self.config.kl_backtrack_halvings + 1):
                restore()
                rate = self.config.learning_rate * self.config.kl_backtrack_factor ** retry
                for group in self.optimizer.param_groups:
                    group['lr'] = rate
                self.optimizer.step()
                kl = self.policy_kl(data)
                if not math.isfinite(kl):
                    raise ValueError("Nonfinite trial policy KL")
                if not exact_equal(rng, self._rng()):
                    raise ValueError("Deterministic model trial changed RNG")
                accepted = kl <= self.config.target_kl
                attempts.append({'retry': retry, 'learning_rate': rate, 'kl': kl, 'accepted': accepted})
                if accepted:
                    return True, kl, attempts
        except Exception:
            restore()
            raise
        restore()
        return False, self.policy_kl(data), attempts

    def update(self, rollout):
        """Apply PPO/estimator losses with optional analytic KL controls.

        The crossing model step is retained; later model steps stop while the
        discriminator completes its schedule. This does not impose a hard KL
        cap or constrain between-rollout normalization changes. Explicit v3
        rollback rejects crossing trials; actor statistics can separately freeze.

        Fresh-source metric names correct two historical reporting errors:
        policy_loss now retains PPO (D losses have discriminator_ prefixes),
        and actor_grad_norm covers estimator, memory, policy and log_std only.
        model_grad_norm retains the old, mislabeled full-model norm. All three
        gradient norms are preclip minibatch means. Explained variance uses
        saved rollout values, not the critic after this update; constant returns
        make it undefined (JSON null). Return std uses population variance.
        """
        c = self.config
        data = {key: value.reshape(-1, *value.shape[2:]) for key, value in rollout.items()}
        advantage = data["advantage"]
        data["advantage"] = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        totals = {}
        ppo_batches = discriminator_batches = 0
        stopped = False
        kl_values = []
        kl_attempts = []
        model_proposals = 0
        kl_initial = None
        if c.target_kl is not None:
            for key in ("old_action_mean", "old_action_std"):
                if key not in data:
                    raise ValueError("Missing collection-time Gaussian reference")
            old = Normal(data["old_action_mean"], data["old_action_std"])
            stored_log_prob = old.log_prob(data["actions"]).sum(-1)
            if not torch.allclose(stored_log_prob, data["log_prob"], atol=2e-5, rtol=0):
                raise ValueError("Collection-time Gaussian/log-probability reference differs")
            kl_initial = self.policy_kl(data)
            if not math.isfinite(kl_initial) or kl_initial > 1e-6:
                raise ValueError("Stale collection-time policy reference")
        for _ in range(c.epochs):
            permutation = torch.randperm(len(advantage), device=self.device)
            for indices in torch.tensor_split(permutation, c.minibatches):
                if not stopped:
                    distribution, estimated_velocity = self.model.distribution(data["obs"][indices])
                    log_prob = distribution.log_prob(data["actions"][indices]).sum(-1)
                    log_ratio = log_prob - data["log_prob"][indices]
                    ratio = log_ratio.exp()
                    advantages = data["advantage"][indices]
                    policy_loss = -torch.minimum(ratio * advantages, ratio.clamp(1-c.clip, 1+c.clip) * advantages).mean()
                    value = self.model.value(data["critic"][indices])
                    old_value = data["value"][indices]
                    clipped_value = old_value + (value - old_value).clamp(-c.clip, c.clip)
                    target = data["returns"][indices]
                    value_loss = .5 * torch.maximum((value-target).square(), (clipped_value-target).square()).mean()
                    entropy = distribution.entropy().sum(-1).mean()
                    estimator_loss = (estimated_velocity - data["critic"][indices, -3:]).square().mean()
                    loss = policy_loss + c.value_coefficient * value_loss - c.entropy_coefficient * entropy + c.estimator_coefficient * estimator_loss
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite PPO loss")
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    actor_parameters = [self.model.log_std, *self.model.estimator.parameters(),
                                        *self.model.memory.parameters(), *self.model.policy.parameters()]
                    actor_grad_norm = torch.linalg.vector_norm(torch.stack([
                        torch.linalg.vector_norm(p.grad.detach()) for p in actor_parameters if p.grad is not None]))
                    critic_grad_norm = torch.linalg.vector_norm(torch.stack([
                        torch.linalg.vector_norm(p.grad.detach()) for p in self.model.critic.parameters() if p.grad is not None]))
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), c.max_grad_norm, error_if_nonfinite=True)
                    accepted = True
                    if c.rollback_kl_steps:
                        model_proposals += 1
                        accepted, kl, attempts = self._bounded_model_step(data)
                        kl_attempts.extend({'proposal': model_proposals, **attempt} for attempt in attempts)
                        stopped = not accepted
                        if accepted:
                            kl_values.append(kl)
                    else:
                        self.optimizer.step()
                    if accepted:
                        self.optimizer_steps += 1
                        ppo_batches += 1
                    metrics = {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy,
                               "estimator_loss": estimator_loss, "approx_kl": ((ratio-1)-log_ratio).mean(),
                               "actor_grad_norm": actor_grad_norm, "critic_grad_norm": critic_grad_norm,
                               "model_grad_norm": grad_norm,
                               "clip_fraction": ((ratio-1).abs() > c.clip).float().mean()}
                    if accepted:
                        for key, val in metrics.items():
                            totals[key] = totals.get(key, 0.) + float(val.detach())
                    if c.target_kl is not None and not c.rollback_kl_steps:
                        kl = self.policy_kl(data)
                        if not math.isfinite(kl):
                            raise ValueError("Nonfinite full-rollout policy KL")
                        kl_values.append(kl)
                        stopped = kl > c.target_kl
                # Preserve every discriminator update and RNG draw after model stop.
                expert, expert_next = self._expert(len(indices))
                d_loss, d_metrics = self.amp.loss(expert, expert_next, data["amp"][indices], data["amp_next"][indices], c.gradient_penalty)
                if not torch.isfinite(d_loss):
                    raise ValueError("Nonfinite AMP loss")
                self.discriminator_optimizer.zero_grad(set_to_none=True)
                d_loss.backward()
                nn.utils.clip_grad_norm_(self.amp.discriminator.parameters(), c.max_grad_norm, error_if_nonfinite=True)
                self.discriminator_optimizer.step()
                self.discriminator_steps += 1
                discriminator_batches += 1
                for key, val in {"discriminator_loss": d_loss, **d_metrics}.items():
                    totals[key] = totals.get(key, 0.) + float(val.detach())
        if c.rollback_kl_steps:
            for group in self.optimizer.param_groups:
                group['lr'] = c.learning_rate
        if ppo_batches == 0 and not c.rollback_kl_steps:
            raise ValueError("PPO update executed zero model batches")
        self.consecutive_zero_accepted_updates = (
            self.consecutive_zero_accepted_updates + 1 if c.rollback_kl_steps and not ppo_batches else 0)
        final_full_rollout_k3 = None
        if c.target_kl is not None:
            with torch.no_grad():
                final_distribution, _ = self.model.distribution(data["obs"])
                final_log_ratio = final_distribution.log_prob(data["actions"]).sum(-1) - data["log_prob"]
                final_k3 = ((final_log_ratio.exp() - 1) - final_log_ratio).mean()
                if not bool(torch.isfinite(final_k3)):
                    raise ValueError("Nonfinite final full-rollout sampled KL")
                final_full_rollout_k3 = float(final_k3)
        self.updates += 1
        discriminator_metrics = {"discriminator_loss", "expert_score", "policy_score", "gradient_penalty",
                                 "discriminator_expert_loss", "discriminator_policy_loss"}
        self.last_metrics = {key: value / (discriminator_batches if key in discriminator_metrics else ppo_batches)
                             for key, value in totals.items()}
        if not ppo_batches:
            for key in ('policy_loss', 'value_loss', 'entropy', 'estimator_loss', 'approx_kl',
                        'actor_grad_norm', 'critic_grad_norm', 'model_grad_norm', 'clip_fraction'):
                self.last_metrics[key] = None
        self.last_metrics.update(ppo_batches=ppo_batches, discriminator_batches=discriminator_batches,
                                 target_kl=c.target_kl, kl_initial=kl_initial,
                                 kl_final=kl_values[-1] if kl_values else (kl_initial if c.rollback_kl_steps else None),
                                 kl_max=max(kl_values) if kl_values else None,
                                 kl_max_step_increase=max([0.] + [value - previous for previous, value in
                                     zip([kl_initial] + kl_values[:-1], kl_values)]) if kl_values else None,
                                 kl_after_model_steps=kl_values, early_stopped=stopped,
                                 final_full_rollout_k3=final_full_rollout_k3,
                                 stop_reason=(('no_accepted_step' if not ppo_batches else 'backtracking_exhausted')
                                              if c.rollback_kl_steps and stopped else
                                              ('target_kl_crossed' if stopped else None)),
                                 learning_rate=self.optimizer.param_groups[0]["lr"],
                                 discriminator_learning_rate=self.discriminator_optimizer.param_groups[0]["lr"])
        if c.rollback_kl_steps:
            self.last_metrics.update(rollback_kl_steps=True, model_proposals=model_proposals,
                accepted_model_steps=ppo_batches, rejected_model_trials=sum(not x['accepted'] for x in kl_attempts),
                model_trial_attempts=kl_attempts,
                consecutive_zero_accepted_updates=self.consecutive_zero_accepted_updates,
                empirical_kl_scope='Mean whole-actor KL on the fixed collected rollout; no bound on physical motion or unseen states.')
        self.last_metrics.update(normalizer_policy_kl=self.normalizer_policy_kl,
                                 normalizer_policy_kl_samples=self.config.num_envs if c.target_kl is not None else 0,
                                 normalizer_policy_kl_scope="Same current observations and weights, before/after statistics update; reporting only")
        returns = data["returns"].detach()
        return_variance = returns.var(unbiased=False)
        explained_variance = (float(1. - (returns - data["value"].detach()).var(unbiased=False) / return_variance)
                              if float(return_variance) > 0. else None)
        self.last_metrics.update(update=self.updates, transitions=self.transitions, episodes=self.episodes,
                                 optimizer_steps=self.optimizer_steps, discriminator_steps=self.discriminator_steps,
                                 reward=float(rollout["reward"].mean()), task_reward=float(rollout["task_reward"].mean()),
                                 style_reward=float(rollout["style_reward"].mean()), action_std=float(self.model.log_std.detach().clamp(-5.,1.).exp().mean()),
                                 raw_action_abs_mean=float(rollout["actions"].abs().mean()),
                                 raw_action_clip_fraction=float((rollout["actions"].abs()>1).float().mean()),
                                 return_mean=float(returns.mean()), return_std=float(returns.std(unbiased=False)),
                                 explained_variance=explained_variance,
                                 terminated=int(rollout["terminated"].sum()), truncated=int(rollout["truncated"].sum()))
        return self.last_metrics

    @staticmethod
    def _rng():
        return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

    @staticmethod
    def _restore_rng(rng):
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])

    def save(self, path):
        path = Path(path)
        payload = {"schema": SCHEMA, "config": asdict(self.config), "prior_sha256": self.prior_sha256,
                   "learner_sha256": file_sha(__file__), "model": self.model.state_dict(), "amp": self.amp.state_dict(),
                   "optimizer": self.optimizer.state_dict(), "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                   "rng": self._rng(), "counters": {key: getattr(self, key) for key in ("updates", "transitions", "optimizer_steps", "discriminator_steps", "episodes", "bc_steps", "consecutive_zero_accepted_updates")},
                   "metrics": self.last_metrics, "simulation_resume_supported": False,
                   "physical_admission": False, "stage2_complete": False}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        # Real serialization readback includes model, normalizers, optimizers and RNG.
        reloaded = torch.load(path, map_location=self.device, weights_only=False)
        if not exact_equal(payload, reloaded):
            raise ValueError("Checkpoint serialization differs")
        # Reconstruct independent models and Adam states, then prove inference
        # equality. Preserve RNG so verification cannot perturb the next rollout.
        verification_rng = self._rng()
        try:
            model = ActorCritic(self.config).to(self.device)
            amp = MotionPrior(self.config).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            discriminator_optimizer = torch.optim.Adam(amp.discriminator.parameters(), lr=self.config.discriminator_learning_rate)
            model.load_state_dict(reloaded['model'], strict=True)
            amp.load_state_dict(reloaded['amp'], strict=True)
            optimizer.load_state_dict(reloaded['optimizer'])
            discriminator_optimizer.load_state_dict(reloaded['discriminator_optimizer'])
            for name, restored in (('model',model),('amp',amp),('optimizer',optimizer),('discriminator_optimizer',discriminator_optimizer)):
                if not exact_equal(restored.state_dict(),reloaded[name]):
                    raise ValueError('Independent checkpoint restore differs: '+name)
            probe = (self.current['obs'][:8] if self.current is not None else
                     torch.zeros((2,OBS_WIDTH),device=self.device))
            with torch.no_grad():
                if not torch.equal(self.model.actor(probe)[0],model.actor(probe)[0]):
                    raise ValueError('Checkpoint deterministic policy differs')
                first,second=self.expert_states[:8],self.expert_next_states[:8]
                if not torch.equal(self.amp.style_reward(first,second),amp.style_reward(first,second)):
                    raise ValueError('Checkpoint discriminator inference differs')
        finally:
            self._restore_rng(verification_rng)
        return {"path": str(path), "sha256": file_sha(path), "update": self.updates,
                "strict_serialization_readback": True, "strict_model_optimizer_restore": True,
                "deterministic_action_exact": True, "rng_preserved": True,
                "simulation_resume_supported": False}

    def load(self, path, *, restore_rng=True):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("schema") != SCHEMA or payload.get("config") != asdict(self.config) or payload.get("prior_sha256") != self.prior_sha256:
            raise ValueError("Checkpoint model/config/prior identity differs")
        if payload.get("learner_sha256") != file_sha(__file__):
            raise ValueError("Checkpoint learner source differs")
        counter_names = {"updates", "transitions", "optimizer_steps", "discriminator_steps", "episodes", "bc_steps", "consecutive_zero_accepted_updates"}
        if (set(payload.get('counters', {})) != counter_names
                or any(type(x) is not int or x < 0 for x in payload['counters'].values())):
            raise ValueError("Checkpoint counters differ")
        self.model.load_state_dict(payload["model"], strict=True)
        self.amp.load_state_dict(payload["amp"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.discriminator_optimizer.load_state_dict(payload["discriminator_optimizer"])
        for key, val in payload["counters"].items():
            setattr(self, key, val)
        self.last_metrics = payload["metrics"]
        if restore_rng:
            self._restore_rng(payload["rng"])
        self.current = None
        self.last_rollout = None
        for name in ("model", "amp", "optimizer", "discriminator_optimizer"):
            if not exact_equal(getattr(self, name).state_dict(), payload[name]):
                raise ValueError("Strict checkpoint restore differs: " + name)
        return payload

    def pretrain_bc(self, prior_path, steps, batch_size=512):
        """Fit explicit action pairs, optionally with BC-only velocity supervision.

        Caller must supply real reference history as observations[M,231] and
        normalized target actions[M,18]. No phase, history or actions are invented.
        Optional targets are native pre-hold body-origin navigation velocity.
        Detach affects only the estimated-velocity feature in this BC actor;
        ordinary PPO/evaluation routing stays unchanged. A separate BC Adam is
        discarded after fitting and never populates the fresh PPO Adam state.
        """
        if type(steps) is not int or steps <= 0 or type(batch_size) is not int or batch_size <= 0:
            raise ValueError("BC requires positive integer steps and batch_size")
        separate = self.config.bc_optimizer == "separate"
        if separate and (self.updates or self.optimizer_steps or self.optimizer.state):
            raise ValueError("Separate BC is allowed only before PPO with an empty PPO optimizer")
        with np.load(prior_path, allow_pickle=False) as prior:
            obs = torch.as_tensor(prior["observations"].copy(), dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(prior["actions"].copy(), dtype=torch.float32, device=self.device)
            velocity_targets = None
            if self.config.bc_velocity_coefficient > 0:
                if "velocity_targets_navigation_mps" not in prior:
                    raise ValueError("Velocity-supervised BC requires velocity_targets_navigation_mps")
                velocity_targets = torch.as_tensor(prior["velocity_targets_navigation_mps"].copy(),
                                                   dtype=torch.float32, device=self.device)
                if velocity_targets.shape != (len(obs), 3) or not bool(torch.isfinite(velocity_targets).all()):
                    raise ValueError("BC velocity targets must be finite aligned [M,3]")
                if "states" not in prior:
                    raise ValueError("Velocity-supervised BC requires native pre-hold states")
                states = torch.as_tensor(prior["states"].copy(), dtype=torch.float32, device=self.device)
                if states.shape != (len(obs), AMP_WIDTH) or not bool(torch.isfinite(states).all()):
                    raise ValueError("BC pre-hold states must be finite aligned [M,61]")
                expected = torch.stack((-states[:, 37], states[:, 36], states[:, 38]), -1)
                if not torch.equal(velocity_targets, expected):
                    raise ValueError("BC velocity targets differ from pre-hold native navigation velocity")
        if (obs.ndim != 2 or obs.shape[1] != OBS_WIDTH or not len(obs) or actions.shape != (len(obs),18)
                or not torch.isfinite(obs).all() or not torch.isfinite(actions).all()):
            raise ValueError("BC requires explicit finite observations[M,231], actions[M,18]")
        bc_optimizer = (torch.optim.Adam(self.model.parameters(), lr=self.config.bc_learning_rate)
                        if separate else self.optimizer)
        self.model.obs_normalizer.update(obs)
        for _ in range(steps):
            indices = torch.randint(len(obs), (min(batch_size,len(obs)),), device=self.device)
            mean, velocity = self.model.actor(obs[indices], detach_velocity=self.config.bc_detach_velocity)
            action_loss = (mean-actions[indices]).square().mean()
            velocity_loss = None if velocity_targets is None else (velocity-velocity_targets[indices]).square().mean()
            loss = action_loss if velocity_loss is None else action_loss + self.config.bc_velocity_coefficient * velocity_loss
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite BC loss")
            bc_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            report = (self.bc_steps + 1) % 50 == 0 or _ == steps - 1
            if report:
                def grad_norm(parameters):
                    gradients = [torch.linalg.vector_norm(p.grad.detach()) for p in parameters if p.grad is not None]
                    return float(torch.linalg.vector_norm(torch.stack(gradients))) if gradients else 0.
                metrics = {"bc_step":self.bc_steps + 1, "action_mse":float(action_loss.detach()),
                    "velocity_mse_m2_s2":None if velocity_loss is None else float(velocity_loss.detach()),
                    "total_loss":float(loss.detach()), "velocity_coefficient":self.config.bc_velocity_coefficient,
                    "detach_velocity_in_bc_actor":self.config.bc_detach_velocity,
                    "estimator_grad_norm_preclip":grad_norm(self.model.estimator.parameters()),
                    "policy_memory_grad_norm_preclip":grad_norm([*self.model.policy.parameters(), *self.model.memory.parameters()]),
                    "raw_action_clip_fraction":float((mean.detach().abs()>1).float().mean()),
                    "predicted_unlimited_target_step_over_0p04_fraction":float(
                        ((mean.detach()-obs[indices,213:]).abs()*.35>.04).float().mean()),
                    "scope":"Current sampled fitting batch before this step; target-step prediction excludes limiter and joint clamps; no native torque measurement"}
            total_grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm, error_if_nonfinite=True)
            if report:
                metrics.update(model_grad_norm_preclip=float(total_grad_norm),
                               gradient_clip_coefficient=float((self.config.max_grad_norm / (total_grad_norm + 1e-6)).clamp(max=1.)),
                               gradient_clip_threshold=self.config.max_grad_norm)
            bc_optimizer.step()
            self.bc_steps += 1
            if report:
                with (self.output/"bc_metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics, allow_nan=False)+"\n")
        if separate:
            bc_optimizer.zero_grad(set_to_none=True)
        receipt = {"explicitly_enabled":True, "steps":steps, "total_bc_steps":self.bc_steps,
            "dataset_sha256":file_sha(prior_path), "bc_optimizer":self.config.bc_optimizer,
            "bc_learning_rate":bc_optimizer.param_groups[0]["lr"], "bc_optimizer_discarded":separate,
            "ppo_optimizer_state_entries":len(self.optimizer.state), "ppo_learning_rate":self.optimizer.param_groups[0]["lr"],
            "bc_velocity_coefficient":self.config.bc_velocity_coefficient, "bc_detach_velocity":self.config.bc_detach_velocity,
            "velocity_target_convention":"Native pre-hold body-origin navigation m/s: [-states[:,37], states[:,36], states[:,38]]",
            "velocity_targets_sha256":None if velocity_targets is None else hashlib.sha256(
                velocity_targets.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            "velocity_targets_hash_format":"Contiguous float32 [M,3] row-major bytes",
            "privileged_velocity_is_actor_input":False, "is_physics_evidence":False}
        (self.output/"behavior_cloning.json").write_text(json.dumps(receipt,indent=2)+"\n")

    def train(self, updates, checkpoint_interval=25, callback=None):
        if type(updates) is not int or updates <= 0 or checkpoint_interval <= 0:
            raise ValueError("Positive update allocation/checkpoint interval required")
        started = time.monotonic()
        starting_transitions = self.transitions
        training_stop = None
        for _ in range(updates):
            metrics = self.update(self.collect())
            metrics["elapsed_s"] = time.monotonic()-started
            metrics["transitions_per_s"] = self.transitions/max(metrics["elapsed_s"],1e-6)
            metrics["allocation_new_transitions"] = self.transitions - starting_transitions
            metrics["allocation_new_transitions_per_s"] = metrics['allocation_new_transitions']/max(metrics['elapsed_s'],1e-6)
            with (self.output/"metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False)+"\n")
            if self.updates == 1 or self.updates % checkpoint_interval == 0:
                receipt = self.save(self.output/f"checkpoint_{self.updates:06d}.pt")
                (self.output/"latest_checkpoint.json").write_text(json.dumps(receipt,indent=2)+"\n")
            if (self.config.rollback_kl_steps
                    and self.consecutive_zero_accepted_updates >= self.config.zero_accepted_update_limit):
                training_stop = {'reason': 'consecutive_zero_accepted_updates',
                    'count': self.consecutive_zero_accepted_updates,
                    'limit': self.config.zero_accepted_update_limit,
                    'update': self.updates, 'transitions': self.transitions,
                    'scope': 'Learning diagnostic stop, not a physical admission gate.'}
            if callback is not None:
                if callback(metrics) is False:
                    break
            if training_stop is not None:
                break
        receipt = self.save(self.output/f"checkpoint_{self.updates:06d}.pt")
        if training_stop is not None:
            receipt['training_stop'] = training_stop
            (self.output/'training_stop.json').write_text(json.dumps({**training_stop, 'checkpoint': receipt}, indent=2)+'\n')
        (self.output/"latest_checkpoint.json").write_text(json.dumps(receipt,indent=2)+"\n")
        return receipt
