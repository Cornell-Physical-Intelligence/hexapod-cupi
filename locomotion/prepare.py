"""Copy a named kernel and explicit inputs into a fresh native allocation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

from .env_config import sha
from .tripod_config import SWEEPS

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path('/home/orionh/HEXAPOD_runs/restart_20260914')


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(output, remote_root, *, mode='train', updates=512, seed=20260917,
            num_envs=None, inputs=None, eval_scope='focus', checkpoint=None, checkpoint_sha=None,
            checkpoint_declaration_sha=None, candidate=0, suite='screen',
            tripod_adaptation='paper', logger='tensorboard', wandb_project=None, wandb_mode='offline',
            root=ROOT):
    output, remote_root, root = map(Path, (output, remote_root, root))
    if (not remote_root.is_absolute() or REMOTE_ROOT not in remote_root.parents
            or '..' in remote_root.parts or mode not in ('diagnostic', 'train', 'evaluate', 'replay', 'tripod')
            or type(updates) is not int or not 1 <= updates <= 2000
            or type(seed) is not int or seed < 0):
        raise ValueError('Invalid native allocation')
    num_envs = (128 if mode == 'train' else 1) if num_envs is None else num_envs
    if (num_envs not in (1, 32, 128) or (mode in ('evaluate', 'replay', 'tripod') and num_envs != 1)
            or (mode == 'train' and num_envs != 128) or eval_scope not in ('focus', 'probes', 'full')):
        raise ValueError('Invalid replica count for this mode')
    if (tripod_adaptation not in SWEEPS or (mode != 'tripod' and tripod_adaptation != 'paper')
            or type(candidate) is not int or candidate not in range(len(SWEEPS[tripod_adaptation]))
            or suite not in ('screen', 'qualification', 'clearance', 'forward_high', 'stops')):
        raise ValueError('Select a member of the declared tripod sweep and suite')
    wandb_valid = (mode == 'train' and wandb_mode in ('offline', 'online')
                   and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', wandb_project or '') is not None)
    if ((logger == 'wandb' and not wandb_valid)
            or (logger == 'tensorboard' and (wandb_project is not None or wandb_mode != 'offline'))
            or logger not in ('tensorboard', 'wandb')):
        raise ValueError('W&B logging needs train mode, a project name and offline or online mode')
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
    save(source/'FREEZE_SHA256.json', {p.relative_to(source).as_posix(): sha(p)
        for p in sorted(source.rglob('*.py'))})
    freeze = sha(source/'FREEZE_SHA256.json')
    binding = {'schema': 'hexapod_locomotion_launch_v1', 'root_review_complete': True,
        'module': ('locomotion.priors.replay_native' if mode == 'replay' else
                   'locomotion.tripod_evaluate' if mode == 'tripod' else 'locomotion.train'),
        'mode': mode, 'source': str(remote_root/'source'), 'output': str(remote_root/'run'),
        'asset': declared['asset'], 'geometry_source': declared['geometry_source'],
        'prior': str(Path(declared['stance']).parent), 'input_files': declared['input_files'],
        'extra_mounts': declared.get('extra_mounts', []), 'source_freeze_sha256': freeze,
        'max_seconds': 6600, 'stage2_complete': False, 'physical_admission': False,
        'command_args': ['--mode', mode, '--asset', '/asset', '--model', '/asset/source/model.json',
            '--geometry', '/geometry_source/geometry/geometry.json',
            '--geometry-extrema', '/geometry_source/geometry/geometry_extrema.npz',
            '--stance', '/prior/'+Path(declared['stance']).name,
            '--num-envs', str(num_envs), '--source-freeze-sha256', freeze,
            '--max-wall-seconds', '6200', '--headless', '--device', 'cuda:0']}
    if mode != 'diagnostic':
        binding['command_args'] += ['--standing-admission', '/admission/admission.json']
    if mode in ('train', 'evaluate'):
        binding['command_args'] += ['--updates', str(updates), '--seed', str(seed)]
    if logger == 'wandb':
        binding['command_args'] += ['--logger', 'wandb', '--wandb-project', wandb_project, '--wandb-mode', wandb_mode]
    if mode == 'tripod':
        binding['command_args'] += ['--candidate', str(candidate), '--suite', suite, '--seed', str(seed),
                                   '--tripod-adaptation', tripod_adaptation]
    if mode == 'evaluate':
        binding['command_args'] += ['--eval-scope', eval_scope]
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if (not checkpoint.is_absolute() or REMOTE_ROOT not in checkpoint.parents
                or '..' in checkpoint.parts or any(len(h) != 64 or set(h)-set('0123456789abcdef')
                    for h in (checkpoint_sha, checkpoint_declaration_sha))):
            raise ValueError('Invalid checkpoint identity')
        binding['extra_mounts'].append([str(checkpoint.parent), '/checkpoint'])
        binding['input_files'][str(checkpoint)] = checkpoint_sha
        binding['input_files'][str(checkpoint.with_suffix('.json'))] = checkpoint_declaration_sha
        binding['command_args'] += ['--checkpoint', '/checkpoint/'+checkpoint.name, '--checkpoint-sha256', checkpoint_sha]
    save(output/'binding.json', binding)
    save(output/'PACK.json', {'remote_root': str(remote_root), 'source_freeze_sha256': freeze,
        'binding_sha256': sha(output/'binding.json'), 'mode': mode, 'seed': seed,
        'updates': updates, 'stage2_complete': False, 'files': {
            p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--mode', choices=('diagnostic', 'train', 'evaluate', 'tripod'), required=True)
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--eval-scope', choices=['focus', 'probes', 'full'], default='focus')
    parser.add_argument('--updates', type=int, default=512)
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
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
