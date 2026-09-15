"""Fresh-model PPO and adversarial motion priors for the canonical robot.

The simulator owns actual physics, position-target limits and terminal states.
This module never substitutes reference motion for a simulated policy rollout.
Paper-inspired pilot choices (including flat-ground critic and optimizer values)
are explicit in Config. Checkpoints restore learning state, not PhysX state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
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
SCHEMA = "canonical_paper_walk_ppo_amp_v1"
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

    def actor(self, obs):
        normalized = self.obs_normalizer(obs)
        history = normalized[:, :210]
        velocity = self.estimator(history)
        memory = self.memory(history)
        current = history[:, -42:]
        mean = self.policy(torch.cat((current, normalized[:, 210:], velocity, memory), -1))
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
        self.current = None
        self.last_rollout = None
        self.last_metrics = {}
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
        samples = self.current if self.last_rollout is None else self.last_rollout
        self.model.obs_normalizer.update(samples["obs"])
        self.model.critic_normalizer.update(samples["critic"])
        first, second = self._expert(min(2048, len(self.expert_states)))
        self.amp.normalizer.update(torch.cat((first, second, samples["amp"].reshape(-1, AMP_WIDTH))))

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

    def update(self, rollout):
        """Update unchanged objectives; diagnostics describe this rollout/update.

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
        batches = 0
        for _ in range(c.epochs):
            permutation = torch.randperm(len(advantage), device=self.device)
            for indices in torch.tensor_split(permutation, c.minibatches):
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
                self.optimizer.step()
                self.optimizer_steps += 1
                expert, expert_next = self._expert(len(indices))
                d_loss, d_metrics = self.amp.loss(expert, expert_next, data["amp"][indices], data["amp_next"][indices], c.gradient_penalty)
                if not torch.isfinite(d_loss):
                    raise ValueError("Nonfinite AMP loss")
                self.discriminator_optimizer.zero_grad(set_to_none=True)
                d_loss.backward()
                nn.utils.clip_grad_norm_(self.amp.discriminator.parameters(), c.max_grad_norm, error_if_nonfinite=True)
                self.discriminator_optimizer.step()
                self.discriminator_steps += 1
                metrics = {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy,
                           "estimator_loss": estimator_loss, "discriminator_loss": d_loss,
                           "approx_kl": ((ratio-1)-log_ratio).mean(), "actor_grad_norm": actor_grad_norm,
                           "critic_grad_norm": critic_grad_norm, "model_grad_norm": grad_norm,
                           "clip_fraction": ((ratio-1).abs() > c.clip).float().mean(), **d_metrics}
                for key, val in metrics.items():
                    totals[key] = totals.get(key, 0.) + float(val.detach())
                batches += 1
        self.updates += 1
        self.last_metrics = {key: value / batches for key, value in totals.items()}
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
                   "rng": self._rng(), "counters": {key: getattr(self, key) for key in ("updates", "transitions", "optimizer_steps", "discriminator_steps", "episodes", "bc_steps")},
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
        """Optional, explicit BC of observation/action pairs; never default or physics.

        Caller must supply real reference history as observations[M,231] and
        normalized target actions[M,18]. No phase, history or actions are invented.
        """
        with np.load(prior_path, allow_pickle=False) as prior:
            obs = torch.as_tensor(prior["observations"].copy(), dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(prior["actions"].copy(), dtype=torch.float32, device=self.device)
        if obs.ndim != 2 or obs.shape[1] != OBS_WIDTH or actions.shape != (len(obs),18) or not torch.isfinite(obs).all() or not torch.isfinite(actions).all():
            raise ValueError("BC requires explicit finite observations[M,231], actions[M,18]")
        self.model.obs_normalizer.update(obs)
        for _ in range(steps):
            indices = torch.randint(len(obs), (min(batch_size,len(obs)),), device=self.device)
            mean, _ = self.model.actor(obs[indices])
            loss = (mean-actions[indices]).square().mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            self.bc_steps += 1
        (self.output/"behavior_cloning.json").write_text(json.dumps({"explicitly_enabled":True, "steps":steps,
            "dataset_sha256":file_sha(prior_path),"is_physics_evidence":False},indent=2)+"\n")

    def train(self, updates, checkpoint_interval=25, callback=None):
        if type(updates) is not int or updates <= 0 or checkpoint_interval <= 0:
            raise ValueError("Positive update allocation/checkpoint interval required")
        started = time.monotonic()
        for _ in range(updates):
            metrics = self.update(self.collect())
            metrics["elapsed_s"] = time.monotonic()-started
            metrics["transitions_per_s"] = self.transitions/max(metrics["elapsed_s"],1e-6)
            with (self.output/"metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False)+"\n")
            if self.updates == 1 or self.updates % checkpoint_interval == 0:
                receipt = self.save(self.output/f"checkpoint_{self.updates:06d}.pt")
                (self.output/"latest_checkpoint.json").write_text(json.dumps(receipt,indent=2)+"\n")
            if callback is not None:
                if callback(metrics) is False:
                    break
        receipt = self.save(self.output/f"checkpoint_{self.updates:06d}.pt")
        (self.output/"latest_checkpoint.json").write_text(json.dumps(receipt,indent=2)+"\n")
        return receipt
