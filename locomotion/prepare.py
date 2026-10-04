"""Copy a named kernel and explicit inputs into a fresh native allocation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

from .env_config import sha
from .tripod_config import SWEEPS
from .train import ALLOCATION_PROFILES, CLEANUP_MARGIN_SECONDS, validate_deadline
from .spark_paths import LEGACY_ROOT, RUN_ROOTS, within_roots

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = LEGACY_ROOT
LEARNER_OPTION_DEFAULTS = {'observation_scaling': 'none', 'command_segments': 'continuous',
                           'learning_rate_max': None, 'action_std': .15, 'action_noise_correlation': 0.,
                           'action_std_final': None}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(output, remote_root, *, mode='train', updates=512, warmup_updates=None, seed=20260917,
            num_envs=None, inputs=None, eval_scope='focus', checkpoint=None, checkpoint_sha=None,
            checkpoint_declaration_sha=None, candidate=0, suite='screen',
            tripod_adaptation='paper', logger='tensorboard', wandb_project=None, wandb_mode='offline',
            reward_version='1', learner='ppo', networks='mlp', action_mean='unbounded',
            observation_normalization='empirical', observation_scaling='none',
            command_segments='continuous', learning_rate_max=None, action_std=.15,
            action_noise_correlation=0., action_std_final=None, video_case=None,
            allocation_profile='standard', max_wall_seconds=6200, root=ROOT):
    output, remote_root, root = map(Path, (output, remote_root, root))
    validate_deadline(mode, allocation_profile, max_wall_seconds)
    if type(max_wall_seconds) is not int:
        raise ValueError('Preparation requires an integer native deadline')
    if (not within_roots(remote_root, RUN_ROOTS)
            or '..' in remote_root.parts or mode not in ('diagnostic', 'train', 'evaluate', 'replay', 'tripod', 'throughput', 'probe')
            or type(updates) is not int or not 1 <= updates <= 2000
            or type(seed) is not int or seed < 0):
        raise ValueError('Invalid native allocation')
    num_envs = (128 if mode in ('train', 'probe') else 1) if num_envs is None else num_envs
    if (num_envs not in (1, 32, 128) or (mode in ('evaluate', 'replay', 'tripod') and num_envs != 1)
            or (mode in ('train', 'probe') and num_envs != 128) or eval_scope not in ('focus', 'probes', 'full')):
        raise ValueError('Invalid replica count for this mode')
    if (tripod_adaptation not in SWEEPS or (mode != 'tripod' and tripod_adaptation != 'paper')
            or type(candidate) is not int or candidate not in range(len(SWEEPS[tripod_adaptation]))
            or suite not in ('screen', 'qualification', 'clearance', 'forward_high', 'stops')):
        raise ValueError('Select a member of the declared tripod sweep and suite')
    if (mode == 'throughput') != (warmup_updates is not None) or (mode == 'throughput' and (
            type(warmup_updates) is not int or not 2 <= warmup_updates <= 10 or not 1 <= updates <= 20)):
        raise ValueError('A throughput profile needs 2-10 warmup and 1-20 measured updates per pass')
    wandb_valid = (mode == 'train' and wandb_mode in ('offline', 'online')
                   and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', wandb_project or '') is not None)
    if ((logger == 'wandb' and not wandb_valid)
            or (logger == 'tensorboard' and (wandb_project is not None or wandb_mode != 'offline'))
            or logger not in ('tensorboard', 'wandb')):
        raise ValueError('W&B logging needs train mode, a project name and offline or online mode')
    if reward_version not in ('1', '2', '3', '4') or (mode != 'train' and reward_version != '1'):
        raise ValueError('Reward version selection applies to training only')
    if action_mean not in ('unbounded', 'tanh') or (action_mean != 'unbounded' and mode not in ('train', 'evaluate')):
        raise ValueError('Bounded action means apply to training and evaluation')
    if (observation_normalization not in ('empirical', 'none') or
            (observation_normalization != 'empirical' and (learner != 'ppo' or mode not in ('train', 'evaluate')))):
        raise ValueError('Observation normalization selection requires PPO training or evaluation')
    learner_options = {'observation_scaling': observation_scaling, 'command_segments': command_segments,
                       'learning_rate_max': learning_rate_max, 'action_std': action_std,
                       'action_noise_correlation': action_noise_correlation, 'action_std_final': action_std_final}
    selected_options = {key: value for key, value in learner_options.items() if value != LEARNER_OPTION_DEFAULTS[key]}
    if (observation_scaling not in ('none', 'fixed') or command_segments not in ('continuous', 'bootstrap')
            or (learning_rate_max is not None and not (type(learning_rate_max) is float and 1e-5 <= learning_rate_max <= 1e-2))
            or type(action_std) is not float or not .01 <= action_std <= 1.
            or type(action_noise_correlation) is not float or not 0 <= action_noise_correlation < 1
            or (action_noise_correlation and action_mean != 'tanh')
            or (action_std_final is not None and not (type(action_std_final) is float and .005 <= action_std_final <= action_std))
            or (selected_options and (learner != 'ppo' or mode not in ('train', 'evaluate')))):
        raise ValueError('Observation scaling, command segments, the rate ceiling, the action deviation and '
                         'the noise correlation require PPO training or evaluation')
    if video_case is not None and (mode != 'evaluate' or eval_scope == 'full' or not video_case.startswith('learning:')):
        raise ValueError('A video case applies to a learning-probe evaluation')
    if (learner not in ('ppo', 'amp') or networks not in ('mlp', 'paper') or (networks == 'paper' and learner != 'amp')
            or (learner == 'amp' and mode not in ('train', 'evaluate'))):
        raise ValueError('The AMP learner trains or evaluates, and the paper networks need the AMP learner')
    supplied = (checkpoint is not None, checkpoint_sha is not None, checkpoint_declaration_sha is not None)
    if (mode == 'evaluate' and not all(supplied)) or (mode != 'evaluate' and any(supplied)):
        raise ValueError('Evaluation requires a checkpoint and its two file hashes')
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
    for path in sorted((root/'locomotion').glob('*.py')):
        shutil.copy2(path, package/path.name)
    contracts = source/'contracts'
    contracts.mkdir()
    for name in ('__init__.py', 'release.py'):
        shutil.copy2(root/'contracts'/name, contracts/name)
    if mode == 'replay':
        optional = source/'locomotion/priors'
        optional.mkdir(parents=True)
        for name in ('__init__.py', 'replay_native.py'):
            shutil.copy2(root/'locomotion/priors'/name, optional/name)
    if learner == 'amp':
        bank = 'locomotion/priors/datasets/amp_demonstrations_001'
        (source/bank).mkdir(parents=True)
        for name in ('manifest.json', 'review.json', 'transitions.npz'):
            shutil.copy2(root/bank/name, source/bank/name)
    save(source/'FREEZE_SHA256.json', {p.relative_to(source).as_posix(): sha(p)
        for p in sorted(source.rglob('*')) if p.is_file()})
    freeze = sha(source/'FREEZE_SHA256.json')
    binding = {'schema': 'hexapod_locomotion_launch_v1', 'root_review_complete': True,
        'module': ('locomotion.priors.replay_native' if mode == 'replay' else
                   'locomotion.tripod_evaluate' if mode == 'tripod' else
                   'locomotion.throughput' if mode == 'throughput' else 'locomotion.train'),
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
    if allocation_profile != 'standard':
        binding['command_args'] += ['--allocation-profile', allocation_profile]
    if mode != 'diagnostic':
        binding['command_args'] += ['--standing-admission', '/admission/admission.json']
    if mode in ('train', 'evaluate', 'throughput'):
        binding['command_args'] += ['--updates', str(updates), '--seed', str(seed)]
    if mode == 'probe':
        binding['command_args'] += ['--seed', str(seed)]
    if reward_version != '1':
        binding['command_args'] += ['--reward-version', reward_version]
    if action_mean != 'unbounded':
        binding['command_args'] += ['--action-mean', action_mean]
    if observation_normalization != 'empirical':
        binding['command_args'] += ['--observation-normalization', observation_normalization]
    for key, value in selected_options.items():
        binding['command_args'] += ['--'+key.replace('_', '-'), str(value)]
    if learner != 'ppo':
        binding['command_args'] += ['--learner', learner, '--networks', networks]
    if logger == 'wandb':
        binding['command_args'] += ['--logger', 'wandb', '--wandb-project', wandb_project, '--wandb-mode', wandb_mode]
    if mode == 'tripod':
        binding['command_args'] += ['--candidate', str(candidate), '--suite', suite, '--seed', str(seed),
                                   '--tripod-adaptation', tripod_adaptation]
    if mode == 'evaluate':
        binding['command_args'] += ['--eval-scope', eval_scope]
    if video_case is not None:
        binding['command_args'] += ['--video-case', video_case]
    if mode == 'throughput':
        binding['command_args'] += ['--warmup-updates', str(warmup_updates)]
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
    save(output/'binding.json', binding)
    save(output/'PACK.json', {'remote_root': str(remote_root), 'source_freeze_sha256': freeze,
        'binding_sha256': sha(output/'binding.json'), 'mode': mode, 'seed': seed, 'reward_version': reward_version,
        'learner': learner, 'networks': networks, 'action_mean': action_mean,
        'observation_normalization': observation_normalization, **selected_options,
        **({} if video_case is None else {'video_case': video_case}),
        'allocation_profile': allocation_profile, 'max_wall_seconds': max_wall_seconds,
        'updates': updates, 'stage2_complete': False, 'files': {
            p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--mode', choices=('diagnostic', 'train', 'evaluate', 'tripod', 'throughput', 'probe'), required=True)
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--eval-scope', choices=['focus', 'probes', 'full'], default='focus')
    parser.add_argument('--updates', type=int, default=512)
    parser.add_argument('--warmup-updates', type=int)
    parser.add_argument('--seed', type=int, default=20260917)
    parser.add_argument('--candidate', type=int, choices=range(4), default=0)
    parser.add_argument('--tripod-adaptation', choices=tuple(SWEEPS), default='paper')
    parser.add_argument('--suite', choices=['screen', 'qualification', 'clearance', 'forward_high', 'stops'], default='screen')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha')
    parser.add_argument('--checkpoint-declaration-sha')
    parser.add_argument('--logger', choices=['tensorboard', 'wandb'], default='tensorboard')
    parser.add_argument('--wandb-project')
    parser.add_argument('--wandb-mode', choices=['offline', 'online'], default='offline')
    parser.add_argument('--reward-version', choices=['1', '2', '3', '4'], default='1')
    parser.add_argument('--learner', choices=['ppo', 'amp'], default='ppo')
    parser.add_argument('--networks', choices=['mlp', 'paper'], default='mlp')
    parser.add_argument('--action-mean', choices=['unbounded', 'tanh'], default='unbounded')
    parser.add_argument('--observation-normalization', choices=['empirical', 'none'], default='empirical')
    parser.add_argument('--observation-scaling', choices=['none', 'fixed'], default='none')
    parser.add_argument('--command-segments', choices=['continuous', 'bootstrap'], default='continuous')
    parser.add_argument('--learning-rate-max', type=float)
    parser.add_argument('--action-std', type=float, default=.15)
    parser.add_argument('--action-noise-correlation', type=float, default=0.)
    parser.add_argument('--action-std-final', type=float)
    parser.add_argument('--video-case')
    parser.add_argument('--allocation-profile', choices=ALLOCATION_PROFILES, default='standard')
    parser.add_argument('--max-wall-seconds', type=int, default=6200)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
