"""Keep allocation bounds and completed-update behavior shared across entry points."""
import math

ALLOCATION_PROFILES = ('standard', 'flat_pilot_v1')
CLEANUP_MARGIN_SECONDS = 400
# Training stops at UPDATE_LIMIT; a training pack that opts in to extended updates may run to the larger limit.
UPDATE_LIMIT, EXTENDED_UPDATE_LIMIT = 2000, 5000


def update_limit(mode, extended):
    """The largest update count a native allocation accepts; only training can opt in to the extended limit."""
    if extended and mode != 'train':
        raise ValueError('Extended updates apply to training only')
    return EXTENDED_UPDATE_LIMIT if extended else UPDATE_LIMIT


def decay_updates(updates, decay):
    """Updates over which the action deviation falls to its final value; the whole run unless declared."""
    return updates if decay is None else decay


def validate_decay(mode, updates, decay):
    """A training decay length that ends after the run would never reach the final deviation."""
    if decay is not None and (type(decay) is not int or not 1 <= decay <= EXTENDED_UPDATE_LIMIT
                              or (mode == 'train' and decay > updates)):
        raise ValueError('The deviation decay length lies between 1 update and the training updates')


def validate_deadline(mode, profile, seconds):
    """Keep the standard bound; allow the named flat pilot a finite allocation."""
    if profile not in ALLOCATION_PROFILES:
        raise ValueError('Unknown allocation profile')
    if profile == 'flat_pilot_v1' and mode not in ('train', 'evaluate', 'diagnostic'):
        raise ValueError('The flat pilot deadline applies to training, evaluation and standing diagnostics')
    limit = 21600 if profile == 'flat_pilot_v1' else 6600
    if (type(seconds) not in (int, float) or not math.isfinite(seconds)
            or not 0 < seconds <= limit):
        raise ValueError('Native deadline exceeds the allocation profile')


def finish_training_update(update, updates, elapsed, deadline, checkpoint, save_loads):
    """Save the last complete update before a deadline failure."""
    expired = elapsed >= deadline
    if update % 50 == 0 or update == updates or expired:
        checkpoint(update)
        save_loads()
    if expired and update < updates:
        raise TimeoutError('Training allocation ended after a complete PPO update')


def collected_transitions(update, config, num_envs):
    """Environment controls collected through ``update``: each update gathers the configured rollout per replica."""
    return update * config['num_steps_per_env'] * num_envs




LEARNER_OPTION_DEFAULTS = {
    'observation_scaling': 'none', 'command_segments': 'continuous',
    'learning_rate_max': None, 'action_std': .15, 'action_noise_correlation': 0.,
    'action_std_final': None, 'gait_clock': 0, 'action_smoothing': 'none',
    'velocity_noise': 0., 'action_std_decay_updates': None, 'steps_per_env': 24,
}
