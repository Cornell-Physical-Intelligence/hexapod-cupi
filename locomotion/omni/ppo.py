"""Extend stock RSL-RL PPO with command bootstraps, a rate cap, bounded means and update diagnostics."""
import copy
import math

import torch
from torch import nn

from rsl_rl.algorithms import PPO
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import resolve_callable


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


def load_actor(checkpoint, config, observations, device):
    """Load the recorded stock actor without constructing a learner."""
    settings = copy.deepcopy(config['actor'])
    model = resolve_callable(settings.pop('class_name'))
    target = torch.device(device)
    devices = [target.index if target.index is not None else torch.cuda.current_device()] if target.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        actor = model(observations, config['obs_groups'], 'actor', 18, **settings).to(device)
    actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
    actor.eval()
    return actor, checkpoint.get('infos')


def policy_change_metrics(actor, storage, clip_param):
    """Compare stored behavior with current Gaussian means without sampling or updating buffers."""
    with torch.no_grad():
        observations = storage.observations.flatten(0, 1)
        actions = storage.actions.flatten(0, 1)
        old = tuple(value.flatten(0, 1) for value in storage.distribution_params)
        mean = actor(observations)
        std = actor.distribution.log_std_param.exp().expand_as(mean)
        kl = actor.get_kl_divergence(old, (mean, std))
        log_prob = torch.distributions.Normal(mean, std).log_prob(actions).sum(-1)
        log_ratio = log_prob - storage.actions_log_prob.flatten()
        if not bool(torch.isfinite(kl).all() and torch.isfinite(log_ratio).all()):
            raise FloatingPointError('Nonfinite PPO policy comparison')
        return {'kl_mean': float(kl.mean()), 'kl_max': float(kl.max()),
            'log_ratio_abs_mean': float(log_ratio.abs().mean()),
            'ratio_outside_clip_fraction': float(((log_ratio < math.log(1 - clip_param))
                | (log_ratio > math.log(1 + clip_param))).float().mean())}


class DeviationSchedule:
    """Hold the action deviation on a declared linear schedule; PPO no longer learns it."""

    def __init__(self, algorithm, start, final, updates):
        self.algorithm, self.start, self.final, self.updates = algorithm, float(start), float(final), int(updates)
        self.parameter = algorithm.actor.distribution.log_std_param
        self.parameter.requires_grad_(False)
        self.inner_update = algorithm.update
        self.completed = 0
        self.apply()

    def value(self):
        return self.start + (self.final - self.start) * min(1., self.completed / self.updates)

    def apply(self):
        with torch.no_grad():
            self.parameter.fill_(math.log(self.value()))

    def update(self):
        losses = self.inner_update()
        self.completed += 1
        self.apply()
        return losses


class UpdateDiagnostics:
    """Record policy changes around the stock update; preserve its loss and optimizer."""

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.original_update = algorithm.update
        self.latest = None

    def update(self):
        algorithm = self.algorithm
        before = policy_change_metrics(algorithm.actor, algorithm.storage, algorithm.clip_param)
        rate_before = algorithm.learning_rate
        value = value_metrics(algorithm.storage)
        losses = self.original_update()
        self.latest = {'scope': 'same_collected_rollout_before_and_after_stock_ppo_update',
            'before': before, 'after': policy_change_metrics(algorithm.actor, algorithm.storage, algorithm.clip_param),
            'learning_rate_before': rate_before, 'learning_rate_after': algorithm.learning_rate,
            'value': value}
        return losses


def value_metrics(storage):
    """Score the critic on the collected rollout before the update changes it."""
    with torch.no_grad():
        returns, values = storage.returns.flatten(), storage.values.flatten()
        variance = returns.var()
        explained = float(1 - (returns - values).var() / variance) if float(variance) > 0 else None
        return {'explained_variance': explained, 'return_mean': float(returns.mean()),
                'return_std': float(returns.std()), 'value_mean': float(values.mean())}


def action_metrics(storage, joint_names):
    """Read the collected rollout; RSL-RL 5.0.1 clear resets its cursor only."""
    actions = storage.actions.detach()
    means = storage.distribution_params[0].detach()
    if actions.ndim != 3 or means.shape != actions.shape or len(joint_names) != actions.shape[-1]:
        raise ValueError('Action metrics require matching rollout samples, means and joint names')
    outside = (actions.abs() > 1).float().mean((0, 1)).cpu().tolist()
    near = (means.abs() >= .95).float().mean((0, 1)).cpu().tolist()
    maxima = means.abs().amax((0, 1)).cpu().tolist()
    return {'scope': 'last_collected_rollout_before_optimizer_update',
        'environment_controls': actions.shape[0] * actions.shape[1],
        'raw_sample_outside_bounds_fraction': sum(outside) / len(outside),
        'mean_abs_max': max(maxima), 'mean_near_bound_threshold': .95,
        'per_joint': {name: {'raw_sample_outside_bounds_fraction': outside[i],
            'mean_near_bound_fraction': near[i], 'mean_abs_max': maxima[i]}
            for i, name in enumerate(joint_names)}}
