"""Accumulate force and torque measurements during training."""
import torch


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
