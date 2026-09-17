"""Prepare action-imitation initialization for a paired forward PPO pilot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SCHEMA = 'canonical_forward_example_ppo_v1'
ARMS = ('scratch', 'example')
COMMAND = (.05, 0., 0.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_protocol(path, expected_sha, *, arm, seed, updates, smoke=False):
    path = Path(path)
    if sha(path) != expected_sha:
        raise ValueError('Experiment protocol bytes differ')
    value = json.loads(path.read_text())
    if (value['schema'] != SCHEMA or arm not in ARMS or seed != value['seed']
            or value['arms'] != list(ARMS) or value['command'] != list(COMMAND)
            or value['num_envs'] != 128 or value['updates'] != 1200
            or updates != (2 if smoke else value['updates'])
            or value['evaluation_updates'] != [0, 300, 600, 1200]
            or value['bc'] != {'steps': 1000, 'batch_size': 256, 'learning_rate': .001,
                               'train_rows': [0, 700], 'validation_rows': [700, 1000]}):
        raise ValueError('Experiment settings differ from the declared pilot')
    data_path = path.parent/'demonstration.npz'
    if sha(data_path) != value['demonstration_sha256']:
        raise ValueError('Demonstration bytes differ')
    with np.load(data_path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in ('policy', 'critic', 'action')}
    for name, width in (('policy', 231), ('critic', 234), ('action', 18)):
        if (arrays[name].shape != (1000, width) or arrays[name].dtype != np.float32
                or not np.isfinite(arrays[name]).all()):
            raise ValueError('Invalid demonstration '+name)
    if (not np.array_equal(arrays['critic'][:, :231], arrays['policy'])
            or not np.allclose(arrays['policy'][:, 210:213], COMMAND, rtol=0, atol=1e-8)
            or np.max(np.abs(arrays['action'])) > 1):
        raise ValueError('Demonstration observation or action contract differs')
    return value, arrays


def forward_task(base):
    """Keep the admitted task reward and resets; replace its command sampler."""
    class ForwardTask(base):
        def _resample(self, indices):
            self.env.commands[indices] = self.env.commands.new_tensor(COMMAND)
            self.remaining_controls[indices] = 1000
            self.command_draws += len(indices)

        def declaration(self):
            result = super().declaration()
            result.update(schema='canonical_forward_only_task_v1',
                nonzero_command_bank=[list(COMMAND)], zero_sampling='No zero-command draws in this pilot.',
                hold_controls_inclusive=[1000, 1000],
                sampler_source_sha256=sha(__file__),
                experiment_scope='One fixed forward command; quiet and stop probes are diagnostics.')
            return result
    return ForwardTask


def initialize(runner, arrays, protocol, arm):
    """Change only actor MLP weights in the example arm before PPO starts."""
    from tensordict import TensorDict
    if arm not in ARMS or runner.alg.optimizer.state:
        raise ValueError('Initialization requires a fresh PPO optimizer and a declared arm')
    actor, critic = runner.alg.actor, runner.alg.critic
    device = next(actor.parameters()).device
    observations = TensorDict({name: torch.as_tensor(arrays[name], device=device)
        for name in ('policy', 'critic')}, batch_size=[1000])
    actions = torch.as_tensor(arrays['action'], device=device)
    training = slice(*protocol['bc']['train_rows'])
    validation = slice(*protocol['bc']['validation_rows'])
    initial_actor = state_digest(actor.state_dict())
    # Share input statistics so action imitation is the only treatment difference.
    actor.train(); critic.train()
    actor.update_normalization(observations[training])
    critic.update_normalization(observations[training])
    fixed = {k: v.detach().clone() for k, v in actor.state_dict().items() if not k.startswith('mlp.')}
    critic_digest = state_digest(critic.state_dict())
    actor.eval()
    def errors():
        with torch.inference_mode():
            squared = (actor(observations)-actions).square()
            return {'train_action_mse': float(squared[training].mean()),
                    'validation_action_mse': float(squared[validation].mean()),
                    'startup_action_mse': float(squared[:100].mean())}
    before = errors()
    sample_digest = hashlib.sha256()
    devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        if arm == 'example':
            optimizer = torch.optim.Adam(actor.mlp.parameters(), lr=protocol['bc']['learning_rate'])
            generator = torch.Generator(device='cpu').manual_seed(protocol['seed'])
            for _ in range(protocol['bc']['steps']):
                indices = torch.randint(training.start, training.stop,
                    (protocol['bc']['batch_size'],), generator=generator)
                sample_digest.update(indices.numpy().tobytes())
                indices = indices.to(device)
                loss = (actor(observations[indices])-actions[indices]).square().mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite imitation loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.mlp.parameters(), 1.)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    after = errors()
    if (any(not torch.equal(actor.state_dict()[k], v) for k, v in fixed.items())
            or state_digest(critic.state_dict()) != critic_digest or runner.alg.optimizer.state):
        raise ValueError('Imitation changed the critic, normalization, action variance or PPO optimizer')
    if not all(bool(torch.isfinite(v).all()) for v in actor.state_dict().values()):
        raise FloatingPointError('Nonfinite actor after initialization')
    actor.train()
    return {'arm': arm, 'initial_actor_sha256': initial_actor,
        'final_actor_sha256': state_digest(actor.state_dict()),
        'shared_actor_non_mlp_sha256': state_digest(fixed), 'shared_critic_sha256': critic_digest,
        'bc_updates': protocol['bc']['steps'] if arm == 'example' else 0,
        'bc_sample_indices_sha256': sample_digest.hexdigest() if arm == 'example' else None,
        'before': before, 'after': after, 'ppo_optimizer_empty': True,
        'ppo_transitions': 0, 'ppo_rng_preserved': True, 'stage2_complete': False}
