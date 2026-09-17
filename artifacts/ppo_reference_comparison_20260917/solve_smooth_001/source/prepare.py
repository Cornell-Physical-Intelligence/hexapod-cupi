"""Pack a fresh native replay with unchanged physics and ownership controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .model import ROOT, digest
from .replay_native import validate_trajectory
from experiments.paper_walk.env_config import MODEL_SHA256


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def prepare(trajectory_directory, output, remote_root, root=ROOT):
    root, trajectory_directory, output = map(Path, (root, trajectory_directory, output))
    remote_root = Path(remote_root)
    allowed = Path('/home/orionh/HEXAPOD_runs/restart_20260914')
    if not remote_root.is_absolute() or allowed not in remote_root.parents or '..' in remote_root.parts:
        raise ValueError('Fresh canonical restart remote root required')
    trajectory = trajectory_directory/'trajectory.npz'
    validate_trajectory(trajectory, digest(trajectory), MODEL_SHA256)
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'; source.mkdir()
    for path in sorted((root/'experiments/paper_walk').glob('*.py')):
        shutil.copy2(path, source/('paper_train.py' if path.name == 'train.py' else path.name))
    shutil.copy2(Path(__file__).with_name('replay_native.py'), source/'train.py')
    shutil.copy2(Path(__file__).with_name('force_metrics.py'), source/'force_metrics.py')
    write(source/'FREEZE_SHA256.json', {p.name: digest(p) for p in sorted(source.iterdir())})
    inputs = output/'inputs'; inputs.mkdir()
    original = root/'artifacts/restart_2026-09-14/paper_walk_execution_001/prior_001'
    for name in ('tripod_prior.npz', 'prior_metadata.json'):
        shutil.copy2(original/name, inputs/name)
    optimized = inputs/'optimized'; optimized.mkdir()
    for name in ('INPUT.json', 'SOLVER.json', 'RESULT.json', 'trajectory.npz'):
        shutil.copy2(trajectory_directory/name, optimized/name)
    template = root/'artifacts/restart_2026-09-14/paper_walk_execution_001/results_startup_cold_refit_onset20_001/launch_binding.json'
    binding = json.loads(template.read_text())
    mounts = [pair for pair in binding['extra_mounts'] if pair[1] in ('/standing_one', '/standing_batch', '/admission')]
    retained_roots = [binding['asset'], binding['geometry_source'], *(p[0] for p in mounts)]
    bound = {name: value for name, value in binding['input_files'].items()
             if any(Path(base) in Path(name).parents for base in retained_roots)}
    for path in inputs.rglob('*'):
        if path.is_file():
            bound[str(remote_root/'inputs'/path.relative_to(inputs))] = digest(path)
    freeze_sha = digest(source/'FREEZE_SHA256.json')
    binding.update(mode='replay', source=str(remote_root/'source'), prior=str(remote_root/'inputs'),
        output=str(remote_root/'replay_001'), source_freeze_sha256=freeze_sha,
        max_seconds=1800, extra_mounts=mounts, input_files=bound)
    binding['command_args'] = ['--mode', 'replay', '--asset', '/asset', '--model', '/asset/source/model.json',
        '--geometry', '/geometry_source/geometry/geometry.json',
        '--geometry-extrema', '/geometry_source/geometry/geometry_extrema.npz',
        '--prior', '/prior/tripod_prior.npz', '--prior-metadata', '/prior/prior_metadata.json',
        '--standing-admission', '/admission/admission.json', '--num-envs', '1',
        '--trajectory', '/prior/optimized/trajectory.npz', '--trajectory-sha256', digest(trajectory),
        '--source-freeze-sha256', freeze_sha, '--max-wall-seconds', '1500', '--headless', '--device', 'cuda:0']
    write(output/'binding.json', binding)
    write(output/'PACK.json', {'remote_root': str(remote_root), 'source_freeze_sha256': freeze_sha,
        'binding_sha256': digest(output/'binding.json'), 'prototype_source_unchanged': True,
        'controller_kind': 'optimized_periodic_motor_targets', 'learned_policy': False,
        'stage2_complete': False, 'files': {p.relative_to(output).as_posix(): digest(p)
            for p in sorted(output.rglob('*')) if p.is_file()}})
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    args = parser.parse_args()
    binding = prepare(args.trajectory_directory, args.output, args.remote_root)
    print(json.dumps({'source': binding['source'], 'output': binding['output'],
                      'source_freeze_sha256': binding['source_freeze_sha256']}))


if __name__ == '__main__':
    main()
