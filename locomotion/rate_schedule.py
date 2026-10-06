"""Adapt command-boundary targets and cap the learning rate of stock RSL-RL PPO.

The stock schedule raises the rate by 1.5 per mini-batch while the measured KL divergence stays
below half its target, up to 1e-2. A bounded action mean that saturates reports a small KL for a
large parameter step, so the schedule then climbs to its ceiling. This class lowers the ceiling
and leaves the update unchanged.
"""
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
