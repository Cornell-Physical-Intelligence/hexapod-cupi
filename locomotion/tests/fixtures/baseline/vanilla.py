"""Adapt the admitted simulator to standard RSL-RL PPO without a motion prior."""
from __future__ import annotations

import torch


def ppo_config(seed):
    return {
        'seed': seed, 'num_steps_per_env': 24, 'save_interval': 50,
        'obs_groups': {'actor': ['policy'], 'critic': ['critic']},
        'actor': {'class_name': 'MLPModel', 'hidden_dims': [256, 256, 128],
            'activation': 'elu', 'obs_normalization': True,
            'distribution_cfg': {'class_name': 'GaussianDistribution', 'init_std': .15, 'std_type': 'log'}},
        'critic': {'class_name': 'MLPModel', 'hidden_dims': [256, 256, 128],
            'activation': 'elu', 'obs_normalization': True},
        'algorithm': {'class_name': 'PPO', 'value_loss_coef': 1., 'use_clipped_value_loss': True,
            'clip_param': .2, 'entropy_coef': .01, 'num_learning_epochs': 5,
            'num_mini_batches': 4, 'learning_rate': 1e-3, 'schedule': 'adaptive',
            'gamma': .99, 'lam': .95, 'desired_kl': .01, 'max_grad_norm': 1.,
            'rnd_cfg': None, 'symmetry_cfg': None},
        'logger': 'tensorboard', 'check_for_nan': True,
    }


class VanillaVecEnv:
    """Preserve terminal rewards and reset selected replicas before the next action."""

    def __init__(self, task, tensor_dict=None):
        if tensor_dict is None:
            from tensordict import TensorDict
            tensor_dict = TensorDict
        self.task, self.tensor_dict = task, tensor_dict
        self.num_envs, self.num_actions, self.device = task.num_envs, 18, task.device
        self.max_episode_length = round(task.cfg.episode_seconds / task.cfg.control_dt)
        self.cfg = {'physics': task.cfg.declaration(), 'task': task.declaration(),
                    'algorithm': 'RSL-RL PPO', 'motion_prior': False}
        self.current = task.reset()

    @property
    def episode_length_buf(self):
        return self.task.episode_steps

    def observations(self, output):
        return self.tensor_dict({'policy': output['obs'], 'critic': output['critic']},
                                batch_size=[self.num_envs])

    def get_observations(self):
        return self.observations(self.current)

    def step(self, action):
        output = self.task.step(action)
        rewards = output['reward'].clone()
        done = output['terminated'] | output['truncated']
        # A physical termination overrides a coincident episode timeout.
        extras = {'time_outs': output['truncated'] & ~output['terminated']}
        selected = done.nonzero(as_tuple=False).flatten()
        self.current = self.task.reset(selected) if len(selected) else output
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
