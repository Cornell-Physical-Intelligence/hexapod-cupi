"""Measure PPO policy change, critic fit and action bounds around each update without changing the update."""
import math

import torch


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
