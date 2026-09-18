"""Verify canonical inputs and package a fresh portable copy without launching compute."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .env_config import sha, verify_assets
from .prepare import REMOTE_ROOT

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'robot/hexapod_mkii_updated_v1'


def check(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / MODEL / 'inputs.json').read_text())
    for name, expected in manifest['files'].items():
        path = root / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve()) or not path.is_file()
                or sha(path) != expected):
            raise ValueError('Canonical input bytes differ: ' + name)
    verify_assets(root / MODEL / 'usd', root / MODEL / 'model_rs05_mass_corrected.json')
    geometry = json.loads((root / MODEL / 'geometry/geometry.json').read_text())
    if sha(root / MODEL / 'geometry/geometry_extrema.npz') != geometry['extrema_sha256']:
        raise ValueError('Geometry extrema identity differs')
    return manifest


def declaration(remote_root, root=ROOT):
    manifest = check(root)
    remote = Path(remote_root)
    if not remote.is_absolute() or '..' in remote.parts or REMOTE_ROOT not in remote.parents:
        raise ValueError('Use a fresh directory under the guarded Spark restart root')
    files = {}
    for source in manifest['files']:
        relative = Path(source).relative_to(MODEL)
        if relative.parts[0] == 'usd':
            target = Path('asset') / Path(*relative.parts[1:])
        elif relative.parts[0] == 'geometry':
            target = Path('geometry_source') / relative
        elif relative.as_posix() == 'stance.json':
            target = Path('prior/stance.json')
        else:
            continue
        files[source] = target.as_posix()
    binding = {'asset': str(remote/'asset'), 'geometry_source': str(remote/'geometry_source'),
               'stance': str(remote/'prior/stance.json'), 'extra_mounts': [],
               'input_files': {str(remote/target): manifest['files'][source] for source, target in files.items()}}
    return binding, files


def pack(output, remote_root, root=ROOT):
    output, root = Path(output), Path(root)
    binding, files = declaration(remote_root, root)
    if output.resolve().is_relative_to((root/'robot').resolve()):
        raise ValueError('Input pack must be outside the approved robot directory')
    output.mkdir(parents=True, exist_ok=False)
    for source, target in files.items():
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/source, destination)
        if sha(destination) != binding['input_files'][str(Path(remote_root)/target)]:
            raise ValueError('Copied input bytes differ: ' + target)
    (output/'inputs.json').write_text(json.dumps(binding, indent=2)+'\n')
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check')
    create = commands.add_parser('pack')
    create.add_argument('--output', type=Path, required=True)
    create.add_argument('--remote-root', required=True)
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps({'files_verified': len(check()['files']), 'native_admission': False}))
    else:
        print(json.dumps(pack(args.output, args.remote_root), indent=2))


if __name__ == '__main__':
    main()
