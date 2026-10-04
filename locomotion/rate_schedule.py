"""Cap the adaptive learning rate of stock RSL-RL PPO.

The stock schedule raises the rate by 1.5 per mini-batch while the measured KL divergence stays
below half its target, up to 1e-2. A bounded action mean that saturates reports a small KL for a
large parameter step, so the schedule then climbs to its ceiling. This class lowers the ceiling
and leaves the update unchanged.
"""
from rsl_rl.algorithms import PPO


class CappedRatePPO(PPO):
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
