"""Freeze a fresh standard-PPO allocation with unchanged admitted physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .model import ROOT, digest


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(output, remote_root, *, mode='train', updates=512, seed=20260917,
            checkpoint=None, checkpoint_sha=None, checkpoint_declaration_sha=None, root=ROOT):
    output, remote_root, root = Path(output), Path(remote_root), Path(root)
    allowed = Path('/home/orionh/HEXAPOD_runs/restart_20260914')
    if (not remote_root.is_absolute() or allowed not in remote_root.parents or '..' in remote_root.parts
            or mode not in ('train', 'evaluate') or type(updates) is not int or not 1 <= updates <= 2000
            or type(seed) is not int or seed < 0):
        raise ValueError('Invalid fresh allocation configuration')
    provided = (checkpoint is not None, checkpoint_sha is not None, checkpoint_declaration_sha is not None)
    if (mode == 'evaluate' and not all(provided)) or (mode == 'train' and any(provided)):
        raise ValueError('Evaluation requires a bound checkpoint; training starts from scratch')
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if (not checkpoint.is_absolute() or allowed not in checkpoint.parents or '..' in checkpoint.parts
                or any(len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
                       for value in (checkpoint_sha, checkpoint_declaration_sha))):
            raise ValueError('Invalid checkpoint identity')
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'; source.mkdir()
    for path in sorted((root/'experiments/paper_walk').glob('*.py')):
        shutil.copy2(path, source/('paper_train.py' if path.name == 'train.py' else path.name))
    for name in ('vanilla.py', 'vanilla_native.py', 'force_metrics.py'):
        shutil.copy2(root/'experiments/trajectory_optimization'/name,
                     source/('train.py' if name == 'vanilla_native.py' else name))
    save(source/'FREEZE_SHA256.json', {p.name: digest(p) for p in sorted(source.iterdir())})
    template = root/'artifacts/restart_2026-09-14/paper_walk_execution_001/results_startup_cold_refit_onset20_001/launch_binding.json'
    binding = json.loads(template.read_text())
    mounts = [pair for pair in binding['extra_mounts'] if pair[1] in ('/standing_one', '/standing_batch', '/admission')]
    retained = [binding['asset'], binding['geometry_source'], binding['prior'], *(p[0] for p in mounts)]
    bound = {name: value for name, value in binding['input_files'].items()
        if any(Path(base) in Path(name).parents for base in retained)}
    freeze = digest(source/'FREEZE_SHA256.json')
    binding.update(mode=mode, source=str(remote_root/'source'), output=str(remote_root/'run'),
        source_freeze_sha256=freeze, max_seconds=6600, extra_mounts=mounts, input_files=bound)
    binding['command_args'] = ['--mode', mode, '--asset', '/asset', '--model', '/asset/source/model.json',
        '--geometry', '/geometry_source/geometry/geometry.json',
        '--geometry-extrema', '/geometry_source/geometry/geometry_extrema.npz',
        '--prior', '/prior/tripod_prior.npz', '--prior-metadata', '/prior/prior_metadata.json',
        '--standing-admission', '/admission/admission.json', '--num-envs', '128' if mode == 'train' else '1',
        '--source-freeze-sha256', freeze, '--updates', str(updates), '--seed', str(seed),
        '--max-wall-seconds', '6200', '--headless', '--device', 'cuda:0']
    if checkpoint is not None:
        binding['extra_mounts'].append([str(checkpoint.parent), '/checkpoint'])
        binding['input_files'][str(checkpoint)] = checkpoint_sha
        binding['input_files'][str(checkpoint.with_suffix('.json'))] = checkpoint_declaration_sha
        binding['command_args'] += ['--checkpoint', '/checkpoint/'+checkpoint.name,
                                   '--checkpoint-sha256', checkpoint_sha]
    save(output/'binding.json', binding)
    save(output/'PACK.json', {'remote_root': str(remote_root), 'source_freeze_sha256': freeze,
        'binding_sha256': digest(output/'binding.json'), 'mode': mode, 'seed': seed,
        'updates': updates, 'motion_prior': False, 'prototype_source_unchanged': True,
        'stage2_complete': False, 'files': {p.relative_to(output).as_posix(): digest(p)
            for p in sorted(output.rglob('*')) if p.is_file()}})
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--updates', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20260917)
    args = parser.parse_args()
    binding = prepare(args.output, args.remote_root, updates=args.updates, seed=args.seed)
    print(json.dumps({'source_freeze_sha256': binding['source_freeze_sha256'], 'output': binding['output']}))


if __name__ == '__main__':
    main()
