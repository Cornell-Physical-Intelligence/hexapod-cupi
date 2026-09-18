"""Pack a fresh native replay with unchanged physics and ownership controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .model import ROOT, digest
from locomotion.prepare import prepare as prepare_kernel
from .replay_native import validate_trajectory
from locomotion.env_config import MODEL_SHA256


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def prepare(trajectory_directory, output, remote_root, root=ROOT, *, inputs=None):
    root, trajectory_directory, output = map(Path, (root, trajectory_directory, output))
    remote_root = Path(remote_root)
    allowed = Path('/home/orionh/HEXAPOD_runs/restart_20260914')
    if not remote_root.is_absolute() or allowed not in remote_root.parents or '..' in remote_root.parts:
        raise ValueError('Fresh canonical restart remote root required')
    trajectory = trajectory_directory/'trajectory.npz'
    validate_trajectory(trajectory, digest(trajectory), MODEL_SHA256)
    binding = prepare_kernel(output, remote_root, mode='replay', root=root, inputs=inputs)
    optimized = output/'trajectory'
    optimized.mkdir()
    for name in ('INPUT.json', 'SOLVER.json', 'RESULT.json', 'trajectory.npz'):
        shutil.copy2(trajectory_directory/name, optimized/name)
        binding['input_files'][str(remote_root/'trajectory'/name)] = digest(optimized/name)
    binding['extra_mounts'].append([str(remote_root/'trajectory'), '/realized_prior'])
    binding['command_args'] += ['--trajectory', '/realized_prior/trajectory.npz',
        '--trajectory-sha256', digest(trajectory)]
    binding['max_seconds'] = 1800
    deadline = binding['command_args'].index('--max-wall-seconds')+1
    binding['command_args'][deadline] = '1500'
    freeze_sha = binding['source_freeze_sha256']
    write(output/'binding.json', binding)
    write(output/'PACK.json', {'remote_root': str(remote_root), 'source_freeze_sha256': freeze_sha,
        'binding_sha256': digest(output/'binding.json'), 'kernel_source_unchanged': True,
        'controller_kind': 'optimized_periodic_motor_targets', 'learned_policy': False,
        'stage2_complete': False, 'files': {p.relative_to(output).as_posix(): digest(p)
            for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'PACK.json'}})
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    args = parser.parse_args()
    binding = prepare(args.trajectory_directory, args.output, args.remote_root, inputs=args.inputs)
    print(json.dumps({'source': binding['source'], 'output': binding['output'],
                      'source_freeze_sha256': binding['source_freeze_sha256']}))


if __name__ == '__main__':
    main()
