"""Copy a named kernel and explicit inputs into a fresh native allocation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

from locomotion.backends.isaac.config import sha
from .sources import runtime_paths, write_runtime
from ..learning.config import ppo_config
from .validation import ALLOCATION_PROFILES, CLEANUP_MARGIN_SECONDS, update_limit, validate_decay, validate_deadline
from .spark_paths import LEGACY_ROOT, RUN_ROOTS, within_roots

ROOT = Path(__file__).resolve().parents[2]
# Standing diagnostics accept each count; training and probes require admitted counts.
REPLICA_COUNTS = (1, 32, 128, 512, 1024)
# Counts with standing admission and a program-lead budget for training and probes (512 opened 2026-10-11).
TRAINING_COUNTS = (128, 512)
REMOTE_ROOT = LEGACY_ROOT
LEARNER_OPTION_DEFAULTS = {'observation_scaling': 'none', 'command_segments': 'continuous',
                           'learning_rate_max': None, 'action_std': .15, 'action_noise_correlation': 0.,
                           'action_std_final': None, 'gait_clock': 0, 'action_smoothing': 'none',
                           'velocity_noise': 0., 'action_std_decay_updates': None,
                           'steps_per_env': 24}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(output, remote_root, *, mode='train', updates=512, seed=20260917,
            num_envs=None, inputs=None, eval_scope='focus', checkpoint=None, checkpoint_sha=None,
            checkpoint_declaration_sha=None, checkpoint_compatibility=None, checkpoint_compatibility_sha256=None,
            logger='tensorboard', wandb_project=None, wandb_mode='offline',
            action_mean='unbounded',
            observation_normalization='empirical', observation_scaling='none',
            command_segments='continuous', learning_rate_max=None, action_std=.15,
            action_noise_correlation=0., action_std_final=None, gait_clock=None, action_smoothing='none',
            velocity_noise=0., episode_seconds=20.,
            video_case=None, extended_updates=False, action_std_decay_updates=None, steps_per_env=24,
            allocation_profile='standard', max_wall_seconds=6200, bc_warmstart=False, root=ROOT):
    output, remote_root, root = map(Path, (output, remote_root, root))
    if gait_clock is None:
        gait_clock = 60 if mode == 'train' else 0
    if type(bc_warmstart) is not bool or (bc_warmstart and (
            mode != 'train'
            or gait_clock != 60)):
        raise ValueError('BC warmstart requires PPO training and the 60-control gait clock')
    if mode == 'train' and gait_clock != 60:
        raise ValueError('The active reward requires the 60-control gait clock')
    validate_deadline(mode, allocation_profile, max_wall_seconds)
    if type(max_wall_seconds) is not int:
        raise ValueError('Preparation requires an integer native deadline')
    if type(extended_updates) is not bool:
        raise ValueError('Extended updates is an explicit opt-in')
    limit = update_limit(mode, extended_updates)
    if (not within_roots(remote_root, RUN_ROOTS)
            or '..' in remote_root.parts or mode not in ('diagnostic', 'train', 'evaluate', 'probe')
            or type(updates) is not int or not 1 <= updates <= limit
            or type(seed) is not int or seed < 0):
        raise ValueError('Invalid native allocation')
    num_envs = (128 if mode in ('train', 'probe') else 1) if num_envs is None else num_envs
    if (num_envs not in REPLICA_COUNTS or (mode == 'evaluate' and num_envs != 1)
            or (mode in ('train', 'probe') and num_envs not in TRAINING_COUNTS) or eval_scope not in ('focus', 'probes', 'full')):
        raise ValueError('Invalid replica count for this mode')
    wandb_valid = (mode == 'train' and wandb_mode in ('offline', 'online')
                   and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', wandb_project or '') is not None)
    if ((logger == 'wandb' and not wandb_valid)
            or (logger == 'tensorboard' and (wandb_project is not None or wandb_mode != 'offline'))
            or logger not in ('tensorboard', 'wandb')):
        raise ValueError('W&B logging needs train mode, a project name and offline or online mode')
    if action_mean not in ('unbounded', 'tanh') or (action_mean != 'unbounded' and mode not in ('train', 'evaluate')):
        raise ValueError('Bounded action means apply to training and evaluation')
    if (observation_normalization not in ('empirical', 'none') or
            (observation_normalization != 'empirical' and (mode not in ('train', 'evaluate')))):
        raise ValueError('Observation normalization selection requires PPO training or evaluation')
    learner_options = {'observation_scaling': observation_scaling, 'command_segments': command_segments,
                       'learning_rate_max': learning_rate_max, 'action_std': action_std,
                       'action_noise_correlation': action_noise_correlation, 'action_std_final': action_std_final,
                       'gait_clock': gait_clock, 'action_smoothing': action_smoothing,
                       'velocity_noise': velocity_noise, 'action_std_decay_updates': action_std_decay_updates,
                       'steps_per_env': steps_per_env}
    selected_options = {key: value for key, value in learner_options.items() if value != LEARNER_OPTION_DEFAULTS[key]}
    try:
        ppo_config(seed, action_mean=action_mean, observation_normalization=observation_normalization,
                   **learner_options)
    except ValueError as error:
        raise ValueError('Observation scaling, command segments, the rate ceiling, the action deviation and '
                         'the noise correlation require PPO training or evaluation') from error
    if selected_options and (mode not in ('train', 'evaluate')):
        raise ValueError('Observation scaling, command segments, the rate ceiling, the action deviation and '
                         'the noise correlation require PPO training or evaluation')
    validate_decay(mode, updates, action_std_decay_updates)
    if video_case is not None and (mode != 'evaluate' or eval_scope == 'full' or not video_case.startswith('learning:')):
        raise ValueError('A video case applies to a learning-probe evaluation')
    if type(episode_seconds) is not float or (episode_seconds != 20. and (mode != 'train' or not 5. <= episode_seconds <= 20.)):
        raise ValueError('Episode length selection applies to training, between 5 and 20 seconds')
    supplied = (checkpoint is not None, checkpoint_sha is not None, checkpoint_declaration_sha is not None)
    if (mode == 'evaluate' and not all(supplied)) or (mode != 'evaluate' and any(supplied)):
        raise ValueError('Evaluation requires a checkpoint and its two file hashes')
    compatibility = (checkpoint_compatibility is not None, checkpoint_compatibility_sha256 is not None)
    if any(compatibility) and (mode != 'evaluate' or eval_scope != 'probes' or not all(compatibility)):
        raise ValueError('Compatibility requires evaluation and both receipt arguments')
    if all(compatibility):
        receipt = Path(checkpoint_compatibility)
        if (not within_roots(receipt, RUN_ROOTS) or '..' in receipt.parts
                or len(checkpoint_compatibility_sha256) != 64
                or set(checkpoint_compatibility_sha256) - set('0123456789abcdef')):
            raise ValueError('Invalid checkpoint compatibility path or hash')
    inputs = root/'configs/locomotion_spark.json' if inputs is None else Path(inputs)
    declared = json.loads(inputs.read_text())
    for key in ('asset', 'geometry_source', 'stance'):
        path = Path(declared[key])
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('Inputs must use explicit absolute paths')
    if declared['stance'] not in declared['input_files']:
        raise ValueError('The stance file needs a recorded hash')
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    package = source/'locomotion'
    package.mkdir(parents=True)
    for path in runtime_paths(root/'locomotion'):
        destination = package/path.relative_to(root/'locomotion')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    contracts = source/'contracts'
    contracts.mkdir()
    for name in ('__init__.py', 'release.py'):
        shutil.copy2(root/'contracts'/name, contracts/name)
    write_runtime(source)
    save(source/'FREEZE_SHA256.json', {p.relative_to(source).as_posix(): sha(p)
        for p in sorted(source.rglob('*')) if p.is_file()})
    freeze = sha(source/'FREEZE_SHA256.json')
    binding = {'schema': 'hexapod_locomotion_launch_v1', 'root_review_complete': False,
        'module': 'locomotion.runs.train',
        'mode': mode, 'source': str(remote_root/'source'), 'output': str(remote_root/'run'),
        'asset': declared['asset'], 'geometry_source': declared['geometry_source'],
        'prior': str(Path(declared['stance']).parent), 'input_files': declared['input_files'],
        'extra_mounts': declared.get('extra_mounts', []), 'source_freeze_sha256': freeze,
        'allocation_profile': allocation_profile, 'max_wall_seconds': max_wall_seconds,
        'max_seconds': max_wall_seconds + CLEANUP_MARGIN_SECONDS,
        'stage2_complete': False, 'physical_admission': False,
        'command_args': ['--mode', mode, '--asset', '/asset', '--model', '/asset/source/model.json',
            '--geometry', '/geometry_source/geometry/geometry.json',
            '--geometry-extrema', '/geometry_source/geometry/geometry_extrema.npz',
            '--stance', '/prior/'+Path(declared['stance']).name,
            '--num-envs', str(num_envs), '--source-freeze-sha256', freeze,
            '--max-wall-seconds', str(max_wall_seconds), '--headless', '--device', 'cuda:0']}
    if bc_warmstart:
        binding['command_args'] += ['--bc-warmstart']
    if allocation_profile != 'standard':
        binding['command_args'] += ['--allocation-profile', allocation_profile]
    if mode != 'diagnostic':
        binding['command_args'] += ['--standing-admission', '/admission/admission.json']
    if mode in ('train', 'evaluate'):
        binding['command_args'] += ['--updates', str(updates), '--seed', str(seed)]
    if extended_updates:
        binding['command_args'] += ['--extended-updates']
    if mode == 'probe':
        binding['command_args'] += ['--seed', str(seed)]
    if action_mean != 'unbounded':
        binding['command_args'] += ['--action-mean', action_mean]
    if observation_normalization != 'empirical':
        binding['command_args'] += ['--observation-normalization', observation_normalization]
    for key, value in selected_options.items():
        binding['command_args'] += ['--'+key.replace('_', '-'), str(value)]
    if logger == 'wandb':
        binding['command_args'] += ['--logger', 'wandb', '--wandb-project', wandb_project, '--wandb-mode', wandb_mode]
    if mode == 'evaluate':
        binding['command_args'] += ['--eval-scope', eval_scope]
    if video_case is not None:
        binding['command_args'] += ['--video-case', video_case]
    if episode_seconds != 20.:
        binding['command_args'] += ['--episode-seconds', str(episode_seconds)]
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if (not within_roots(checkpoint, RUN_ROOTS)
                or '..' in checkpoint.parts or any(len(h) != 64 or set(h)-set('0123456789abcdef')
                    for h in (checkpoint_sha, checkpoint_declaration_sha))):
            raise ValueError('Invalid checkpoint identity')
        binding['extra_mounts'].append([str(checkpoint.parent), '/checkpoint'])
        binding['input_files'][str(checkpoint)] = checkpoint_sha
        binding['input_files'][str(checkpoint.with_suffix('.json'))] = checkpoint_declaration_sha
        binding['command_args'] += ['--checkpoint', '/checkpoint/'+checkpoint.name, '--checkpoint-sha256', checkpoint_sha]
    if all(compatibility):
        binding['extra_mounts'].append([str(receipt.parent), '/compatibility'])
        binding['input_files'][str(receipt)] = checkpoint_compatibility_sha256
        binding['command_args'] += ['--checkpoint-compatibility', '/compatibility/'+receipt.name,
                                    '--checkpoint-compatibility-sha256', checkpoint_compatibility_sha256]
    save(output/'binding.json', binding)
    save(output/'PACK.json', {'remote_root': str(remote_root), 'source_freeze_sha256': freeze,
        'binding_sha256': sha(output/'binding.json'), 'mode': mode, 'seed': seed, 'reward_version': '4', 'reward_task': 'v4_yaw_gate',
        'action_mean': action_mean,
        'observation_normalization': observation_normalization, **selected_options,
        **({} if video_case is None else {'video_case': video_case}),
        **({} if episode_seconds == 20. else {'episode_seconds': episode_seconds}),
        **({'extended_updates': True} if extended_updates else {}),
        **({'bc_warmstart': True} if bc_warmstart else {}),
        'allocation_profile': allocation_profile, 'max_wall_seconds': max_wall_seconds,
        'updates': updates, 'stage2_complete': False, 'files': {
            p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--mode', choices=('diagnostic', 'train', 'evaluate', 'probe'), required=True)
    parser.add_argument('--bc-warmstart', action='store_true', help='Clone the scripted teacher before PPO with the active walking task.')
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--eval-scope', choices=['focus', 'probes', 'full'], default='focus')
    parser.add_argument('--updates', type=int, default=512)
    parser.add_argument('--extended-updates', action='store_true',
                        help='Accept up to 5000 training updates; without this opt-in the limit stays 2000.')
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha')
    parser.add_argument('--checkpoint-declaration-sha')
    parser.add_argument('--checkpoint-compatibility', type=Path)
    parser.add_argument('--checkpoint-compatibility-sha256')
    parser.add_argument('--logger', choices=['tensorboard', 'wandb'], default='tensorboard')
    parser.add_argument('--wandb-project')
    parser.add_argument('--wandb-mode', choices=['offline', 'online'], default='offline')
    parser.add_argument('--action-mean', choices=['unbounded', 'tanh'], default='unbounded')
    parser.add_argument('--observation-normalization', choices=['empirical', 'none'], default='empirical')
    parser.add_argument('--observation-scaling', choices=['none', 'fixed'], default='none')
    parser.add_argument('--command-segments', choices=['continuous', 'bootstrap'], default='continuous')
    parser.add_argument('--learning-rate-max', type=float)
    parser.add_argument('--action-std', type=float, default=.15)
    parser.add_argument('--action-noise-correlation', type=float, default=0.)
    parser.add_argument('--action-std-final', type=float)
    parser.add_argument('--action-std-decay-updates', type=int)
    parser.add_argument('--steps-per-env', type=int, default=24)
    parser.add_argument('--gait-clock', type=int)
    parser.add_argument('--action-smoothing', choices=['none', 'mean2'], default='none')
    parser.add_argument('--velocity-noise', type=float, default=0.)
    parser.add_argument('--episode-seconds', type=float, default=20.)
    parser.add_argument('--video-case')
    parser.add_argument('--allocation-profile', choices=ALLOCATION_PROFILES, default='standard')
    parser.add_argument('--max-wall-seconds', type=int, default=6200)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
