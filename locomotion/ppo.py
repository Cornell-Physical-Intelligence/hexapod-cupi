"""Adapt the admitted simulator to standard RSL-RL PPO without a motion prior."""
from __future__ import annotations

import math
import torch

# Fixed input scales, selected with observation_scaling='fixed'. Each proprioception frame holds
# body rates (rad/s), projected gravity, joint offsets (rad) and joint rates (rad/s); the command
# follows at columns 210:213 and the executed-target feature at 213:231. The scales bring each
# group to order one at this robot's commands (0.05 m/s, 0.2 rad/s) and action range (0.35 rad).
FRAME_SCALES = (2.,)*3 + (1.,)*3 + (1/.35,)*18 + (.5,)*18
COMMAND_SCALES = (20., 20., 5.)
ACTOR_SCALES = FRAME_SCALES*5 + COMMAND_SCALES + (1.,)*18
CRITIC_SCALES = ACTOR_SCALES + (20.,)*3
RATE_CLASS = 'locomotion.rate_schedule:CappedRatePPO'


def scale_observation(value, scaling, *, critic=False):
    """Apply the declared fixed scales to a 231-value actor or 234-value critic observation."""
    if scaling == 'none':
        return value
    if scaling != 'fixed':
        raise ValueError('Observation scaling must be none or fixed')
    scales = CRITIC_SCALES if critic else ACTOR_SCALES
    if value.shape[-1] != len(scales):
        raise ValueError('Observation width differs from the declared scales')
    return value * torch.as_tensor(scales, dtype=value.dtype, device=value.device)


def clock_features(episode_steps, period, command):
    """Sine and cosine of the gait phase while a motion command is held; zeros under a zero command.

    The phase is the episode control count modulo the declared period. A stationary input under a
    zero command lets the policy hold still without learning to ignore an oscillating one.
    """
    phase = 2 * math.pi * (episode_steps.to(torch.float32) % period) / period
    moving = (command != 0).any(-1, keepdim=True).to(torch.float32)
    return torch.stack((torch.sin(phase), torch.cos(phase)), -1) * moving.to(phase.device)


def smoothed_action(action, previous, episode_steps):
    """Mean of this action and the previous one; an episode's first control pairs with the neutral action.

    A two-control mean has no gain at half the control rate. An action that alternates between two
    values on consecutive controls then leaves the joint target unchanged.
    """
    fresh = (episode_steps == 0)[:, None].to(action.device)
    return .5 * (action + torch.where(fresh, torch.zeros_like(action), previous))


def ppo_config(seed, *, action_mean='unbounded', observation_normalization='empirical',
               observation_scaling='none', command_segments='continuous', learning_rate_max=None,
               action_std=.15, action_noise_correlation=0., action_std_final=None, gait_clock=0,
               action_smoothing='none', velocity_noise=0.):
    if action_mean not in ('unbounded', 'tanh'):
        raise ValueError('Action mean must be unbounded or tanh')
    if observation_normalization not in ('empirical', 'none'):
        raise ValueError('Observation normalization must be empirical or none')
    if observation_scaling not in ('none', 'fixed') or command_segments not in ('continuous', 'bootstrap'):
        raise ValueError('Observation scaling must be none or fixed; command segments continuous or bootstrap')
    if type(gait_clock) is not int or (gait_clock and not 10 <= gait_clock <= 250):
        raise ValueError('The gait clock period is zero or 10 to 250 controls')
    if action_smoothing not in ('none', 'mean2'):
        raise ValueError('Action smoothing must be none or mean2')
    if type(velocity_noise) is not float or not 0 <= velocity_noise <= 5:
        raise ValueError('The joint-velocity observation noise lies between 0 and 5 rad/s')
    if learning_rate_max is not None and not (type(learning_rate_max) is float and 1e-5 <= learning_rate_max <= 1e-2):
        raise ValueError('The learning-rate ceiling must lie inside the stock schedule range')
    if type(action_std) is not float or not .01 <= action_std <= 1.:
        raise ValueError('The initial action standard deviation must lie between 0.01 and 1')
    if type(action_noise_correlation) is not float or not 0 <= action_noise_correlation < 1 or (
            action_noise_correlation and action_mean != 'tanh'):
        raise ValueError('Correlated action noise needs a correlation in [0, 1) and the tanh mean')
    if action_std_final is not None and not (type(action_std_final) is float and .005 <= action_std_final <= action_std):
        raise ValueError('The final action deviation must lie between 0.005 and the initial deviation')
    distribution = ('GaussianDistribution' if action_mean == 'unbounded'
                    else 'locomotion.action_distribution:BoundedMeanGaussian')
    config = {
        'seed': seed, 'num_steps_per_env': 24, 'save_interval': 50,
        'obs_groups': {'actor': ['policy'], 'critic': ['critic']},
        'actor': {'class_name': 'MLPModel', 'hidden_dims': [256, 256, 128],
            'activation': 'elu', 'obs_normalization': observation_normalization == 'empirical',
            'distribution_cfg': {'class_name': distribution, 'init_std': action_std, 'std_type': 'log'}},
        'critic': {'class_name': 'MLPModel', 'hidden_dims': [256, 256, 128],
            'activation': 'elu', 'obs_normalization': observation_normalization == 'empirical'},
        'algorithm': {'class_name': 'PPO', 'value_loss_coef': 1., 'use_clipped_value_loss': True,
            'clip_param': .2, 'entropy_coef': .01, 'num_learning_epochs': 5,
            'num_mini_batches': 4, 'learning_rate': 1e-3, 'schedule': 'adaptive',
            'gamma': .99, 'lam': .95, 'desired_kl': .01, 'max_grad_norm': 1.,
            'rnd_cfg': None, 'symmetry_cfg': None},
        'logger': 'tensorboard', 'check_for_nan': True,
    }
    # Options appear only when selected, so earlier checkpoints keep their recorded configuration.
    if action_noise_correlation:
        config['actor']['distribution_cfg'].update(noise_correlation=action_noise_correlation,
            class_name='locomotion.action_distribution:CorrelatedBoundedMeanGaussian')
    if learning_rate_max is not None:
        config['algorithm'].update(class_name=RATE_CLASS, learning_rate_max=learning_rate_max,
                                   learning_rate=min(1e-3, learning_rate_max))
    if action_std_final is not None:
        config['exploration'] = {'action_std_final': action_std_final}
    wrapper = {key: value for key, value, default in (
        ('observation_scaling', observation_scaling, 'none'),
        ('command_segments', command_segments, 'continuous'), ('gait_clock', gait_clock, 0),
        ('action_smoothing', action_smoothing, 'none'), ('velocity_noise', velocity_noise, 0.)) if value != default}
    if wrapper:
        config['environment_wrapper'] = wrapper
    return config


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
    """Hold the action deviation on a declared linear schedule; PPO no longer learns it.

    A learned deviation rose in every retained run that moved, and those policies stood still
    once evaluation removed the noise. The schedule lowers the deviation as training proceeds,
    so the mean action has to produce the motion.
    """

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


class VanillaVecEnv:
    """Preserve terminal rewards and reset selected replicas before the next action.

    ``command_segments='bootstrap'`` ends the learner's return at each command change without a
    reset: the critic then values the held command alone and the next draw adds no return noise.
    """

    def __init__(self, task, tensor_dict=None, *, observation_scaling='none', command_segments='continuous',
                 gait_clock=0, action_smoothing='none', velocity_noise=0.):
        if tensor_dict is None:
            from tensordict import TensorDict
            tensor_dict = TensorDict
        if observation_scaling not in ('none', 'fixed') or command_segments not in ('continuous', 'bootstrap'):
            raise ValueError('Unknown observation scaling or command segment option')
        if action_smoothing not in ('none', 'mean2'):
            raise ValueError('Action smoothing must be none or mean2')
        self.action_smoothing, self.velocity_noise = action_smoothing, velocity_noise
        self.previous_action = torch.zeros(task.num_envs, 18, device=task.device)
        self.task, self.tensor_dict = task, tensor_dict
        self.observation_scaling, self.command_segments = observation_scaling, command_segments
        self.gait_clock = gait_clock
        self.num_envs, self.num_actions, self.device = task.num_envs, 18, task.device
        self.max_episode_length = round(task.cfg.episode_seconds / task.cfg.control_dt)
        self.cfg = {'physics': task.cfg.declaration(), 'task': task.declaration(),
                    'algorithm': 'RSL-RL PPO', 'motion_prior': False}
        self.current = task.reset()

    @property
    def episode_length_buf(self):
        return self.task.episode_steps

    def observations(self, output):
        policy = scale_observation(output['obs'], self.observation_scaling)
        critic = scale_observation(output['critic'], self.observation_scaling, critic=True)
        if self.velocity_noise:
            # The actor alone reads noisy joint velocities; evaluation and the critic read the measured values.
            scale = FRAME_SCALES[-1] if self.observation_scaling == 'fixed' else 1.
            policy = policy.clone()
            for frame in range(5):
                block = policy[:, 42 * frame + 24:42 * frame + 42]
                block += torch.randn_like(block) * self.velocity_noise * scale
        if self.gait_clock:
            # The phase restarts with each episode, so evaluation can rebuild it from the control count.
            clock = clock_features(self.task.episode_steps, self.gait_clock, output['obs'][:, 210:213]).to(policy)
            policy, critic = torch.cat((policy, clock), -1), torch.cat((critic, clock), -1)
        return self.tensor_dict({'policy': policy, 'critic': critic}, batch_size=[self.num_envs])

    def get_observations(self):
        return self.observations(self.current)

    def step(self, action):
        held = self.task.commands.clone() if self.command_segments == 'bootstrap' else None
        if self.action_smoothing == 'mean2':
            # PPO keeps the raw sample; the environment receives the two-control mean.
            raw = action.detach().clone()
            action = smoothed_action(action, self.previous_action.to(action), self.task.episode_steps)
            self.previous_action = raw
        output = self.task.step(action)
        rewards = output['reward'].clone()
        done = output['terminated'] | output['truncated']
        # A physical termination overrides a coincident episode timeout.
        extras = {'time_outs': output['truncated'] & ~output['terminated']}
        selected = done.nonzero(as_tuple=False).flatten()
        self.current = self.task.reset(selected) if len(selected) else output
        if held is not None:
            # The next command reaches the actor now; the return of the finished hold bootstraps here.
            switched = (output['obs'][:, 210:213] != held).any(-1) & ~done
            extras['time_outs'] = extras['time_outs'] | switched
            done = done | switched
        return self.get_observations(), rewards, done.long(), extras


class TrainingLoads:
    """Accumulate 400 Hz loads without storing a training trajectory in memory."""

    def __init__(self, env):
        self.env = env
        self.steps = 0
        self.windows = {}
        width = 1 + len(env.native_body_names) + 2 * len(env.joint_names)
        for name in ('full_training', 'commanded_locomotion_after_settle'):
            self.windows[name] = {
                'count': torch.zeros((), dtype=torch.int64, device=env.device),
                'sum': torch.zeros(width, dtype=torch.float64, device=env.device),
                'square': torch.zeros(width, dtype=torch.float64, device=env.device),
                'peak': torch.full((width,), -torch.inf, dtype=torch.float64, device=env.device),
            }

    def __call__(self, env, state, q, dq, requested, applied, ceiling, target, forces, substep, native_pre):
        values = torch.cat((forces[:, :, 2].sum(-1, keepdim=True),
            torch.linalg.vector_norm(forces, dim=-1), native_pre.abs(), requested.abs()), -1).double()
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError('Nonfinite training contact or torque sample')
        moving = (env.commands != 0).any(-1) & (env.episode_steps >= 100)
        for name, mask in (('full_training', torch.ones_like(moving)),
                           ('commanded_locomotion_after_settle', moving)):
            data = self.windows[name]
            weights = mask[:, None]
            data['count'] += mask.sum()
            data['sum'] += (values * weights).sum(0)
            data['square'] += (values.square() * weights).sum(0)
            data['peak'] = torch.maximum(data['peak'], values.masked_fill(~weights, -torch.inf).amax(0))
        self.steps += 1

    def report(self):
        names = (['total_vertical_support_n']
            + ['body_normal_resultant_n:'+name for name in self.env.native_body_names]
            + ['applied_abs_torque_nm:'+name for name in self.env.joint_names]
            + ['requested_abs_torque_nm:'+name for name in self.env.joint_names])
        windows = {}
        for name, data in self.windows.items():
            count = int(data['count'])
            windows[name] = {'samples_across_replicas': count, 'metrics': None}
            if count:
                mean = (data['sum']/count).cpu().tolist()
                rms = (data['square']/count).sqrt().cpu().tolist()
                peak = data['peak'].cpu().tolist()
                windows[name]['metrics'] = {key: {'mean': mean[i], 'rms': rms[i], 'peak': peak[i]}
                    for i, key in enumerate(names)}
        return {'schema': 'canonical_training_loads_v1', 'status': 'available',
            'physics_steps': self.steps, 'physics_hz': 400, 'windows': windows,
            'scope': 'Aggregate native normal forces by body; tibia values include toe and shaft contacts. '
                'Forces exclude friction. Applied torque uses native effort readback. '
                'The locomotion window starts after 2 seconds per episode and includes failed motion. '
                'Exact per-foot classification and percentiles come from the separate evaluation capture.'}
