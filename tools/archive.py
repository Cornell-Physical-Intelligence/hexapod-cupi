"""Resolve published history and restore a named path from its pinned Git tree."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


def index(root=ROOT):
    path = Path(root) / 'configs/archive.json'
    return json.loads(path.read_text()) if path.exists() else None


def safe_path(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or '..' in path.parts or '.git' in path.parts
            or path.as_posix() != value or value.startswith('-')):
        raise ValueError('Archive path must be a normalized repository-relative path')
    return path


def reference(value, root=ROOT):
    value = value.rstrip('/')
    data = index(root)
    if data is None or value not in data['references']:
        return None
    row = data['references'][value]
    safe_path(value)
    safe_path(row['path'])
    if (not re.fullmatch('[0-9a-f]{40}', row['commit'])
            or not re.fullmatch('[0-9a-f]{40}', row['git_oid'])
            or row['kind'] not in ('blob', 'tree')):
        raise ValueError('Invalid archive identity')
    return row


def url(value, root=ROOT):
    row = reference(value, root)
    if row is None:
        return None
    repository = index(root)['repository']
    return f"https://github.com/{repository}/{row['kind']}/{row['commit']}/" + quote(row['path'], safe='/')


def retained(value, root=ROOT):
    data = index(root)
    if data is None or value not in data.get('retained', {}):
        return None
    row = data['retained'][value]
    path = Path(root) / safe_path(row['path'])
    if (path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve())
            or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']):
        raise ValueError('Retained archive bytes changed: ' + value)
    return path


def check(root=ROOT, *, git_objects=False):
    data = index(root)
    if data is None:
        raise ValueError('Archive index is missing')
    for value in data['references']:
        row = reference(value, root)
        if git_objects:
            actual = subprocess.check_output(['git', 'rev-parse', row['commit']+':'+row['path']], cwd=root, text=True).strip()
            if actual != row['git_oid']:
                raise ValueError('Archive Git object differs: ' + value)
    for value in data['retained']:
        retained(value, root)
    if git_objects:
        actual = subprocess.check_output(['git', 'rev-parse', data['commit']+'^{tree}'], cwd=root, text=True).strip()
        if actual != data['tree']:
            raise ValueError('Archive snapshot tree differs')
    return {'references': len(data['references']), 'retained': len(data['retained']), 'git_objects_checked': git_objects}


def restore(value, destination, root=ROOT):
    safe_path(value)
    data = index(root)
    row = reference(value, root) or {'commit': data['commit'], 'path': value}
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Restore requires a fresh destination')
    # Inspect the complete archive before writing any members.
    with tempfile.TemporaryFile() as stream:
        subprocess.run(['git', 'archive', row['commit'], '--', row['path']], cwd=root, stdout=stream, check=True)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as bundle:
            members = bundle.getmembers()
            if not members:
                raise ValueError('Archive contains no matching path')
            for member in members:
                safe_path(member.name.rstrip('/'))
                if not (member.isfile() or member.isdir()):
                    raise ValueError('Restore accepts regular files and directories')
            destination.mkdir(parents=True)
            for member in members:
                target = destination / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as output, bundle.extractfile(member) as source:
                        for block in iter(lambda: source.read(1 << 20), b''):
                            output.write(block)
                    target.chmod(member.mode)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    verify = commands.add_parser('check')
    verify.add_argument('--git-objects', action='store_true')
    recover = commands.add_parser('restore')
    recover.add_argument('path')
    recover.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(check(git_objects=args.git_objects)))
    else:
        print(restore(args.path, args.destination))


if __name__ == '__main__':
    main()
