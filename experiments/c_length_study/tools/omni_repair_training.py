"""Explicit, checkpoint-bound PPO fine-tune initialization; no simulator imports."""
from __future__ import annotations

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import hashlib
import math
from pathlib import Path
import torch


def repair_options(value):
    required = {'checkpoint_sha256', 'exploration_std', 'entropy_coef', 'optimizer', 'learning_rate'}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError('omni.repair_training requires exactly the documented explicit fields')
    sha = value['checkpoint_sha256']
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('Expected lowercase checkpoint SHA256')
    for key, lo, hi in (('exploration_std', .05, .2), ('learning_rate', 1e-5, 1e-4)):
        v = value[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f'Invalid bounded {key}')
    if value['entropy_coef'] != 0 or isinstance(value['entropy_coef'], bool):
        raise ValueError('This comparison requires entropy_coef=0')
    if value['optimizer'] != 'reset':
        raise ValueError('This experiment explicitly resets optimizer moments')
    return dict(value)


def verified_checkpoint(path, options):
    options = repair_options(options)
    if path is None or hashlib.sha256(Path(path).read_bytes()).hexdigest() != options['checkpoint_sha256']:
        raise ValueError('Repair checkpoint missing or SHA256 mismatch; no resume performed')
    return options


def _equal_state(actual, expected, exclude=()):
    if set(actual) != set(expected):
        raise RuntimeError('Loaded model state keys differ from checkpoint')
    for key in actual:
        if key not in exclude and not torch.equal(actual[key].detach().cpu(), expected[key].detach().cpu()):
            raise RuntimeError(f'Loaded model/normalizer differs at {key}')


def load_repair_checkpoint(runner, path, options):
    """Preserve all learned tensors except exploration; reject incompatible APIs."""
    options = verified_checkpoint(path, options)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    actor = saved['actor_state_dict']; critic = saved['critic_state_dict']
    if actor.get('mlp.0.weight', torch.empty(0, 0)).shape[1:] != (315,) or critic.get('mlp.0.weight', torch.empty(0, 0)).shape[1:] != (318,):
        raise ValueError('Repair requires the exact 315 actor / 318 critic architecture')
    if not any('obs_normalizer' in k for k in actor) or not any('obs_normalizer' in k for k in critic):
        raise ValueError('Checkpoint must include both observation normalizers')
    if runner.alg.optimizer.state:
        raise RuntimeError('Repair requires a fresh optimizer before checkpoint load')
    cfg = {'actor': True, 'critic': True, 'optimizer': False, 'iteration': True, 'rnd': False}
    runner.load(str(path), load_cfg=cfg, strict=True, map_location=runner.device)
    _equal_state(runner.alg.actor.state_dict(), actor)
    _equal_state(runner.alg.critic.state_dict(), critic)
    dist = runner.alg.actor.distribution
    if getattr(dist, 'std_type', None) != 'scalar' or 'distribution.std_param' not in actor:
        raise ValueError('Expected verified RSL scalar Gaussian distribution')
    if dist.std_param.shape != (18,):
        raise ValueError('Expected 18 action standard deviations')
    before = dist.std_param.detach().cpu().tolist()
    with torch.no_grad():
        dist.std_param.fill_(options['exploration_std'])
    _equal_state(runner.alg.actor.state_dict(), actor, exclude={'distribution.std_param'})
    _equal_state(runner.alg.critic.state_dict(), critic)
    if runner.alg.optimizer.state:
        raise RuntimeError('Optimizer reset did not remain empty')
    runner.alg.learning_rate = options['learning_rate']
    runner.alg.entropy_coef = options['entropy_coef']
    for group in runner.alg.optimizer.param_groups:
        group['lr'] = options['learning_rate']
    return {**options, 'actor_and_critic_preserved_except_std': True,
            'observation_normalizers_preserved': True, 'optimizer_state_entries': len(runner.alg.optimizer.state),
            'checkpoint_iteration': saved['iter'], 'runner_iteration': runner.current_learning_iteration,
            'exploration_std_before': before, 'exploration_std_after': dist.std_param.detach().cpu().tolist(),
            'load_cfg': cfg, 'learning_rate_schedule': runner.alg.schedule,
            'stage2_complete': False}


def stand_raw_action_cost(commands, raw_action):
    """Mean square sampled normalized intent before clipping/slew; feedback stays active."""
    standing = (commands[:, :2].norm(dim=-1) <= .03) & (commands[:, 2].abs() <= .05)
    return raw_action.square().mean(-1) * standing
