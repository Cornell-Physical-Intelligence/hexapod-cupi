"""Hash the runtime closure, including nested implementation modules."""
import hashlib
import json
from pathlib import Path

SCHEMA = 'hexapod_runtime_sources_v2'
MANIFEST = 'RUNTIME_SOURCES.json'
PACKAGES = ('core', 'backends', 'tasks', 'learning', 'evaluation', 'runs')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_paths(package):
    package = Path(package)
    paths = list(package.glob('*.py'))
    for name in PACKAGES:
        paths.extend((package/name).rglob('*.py'))
    return sorted(path for path in paths if 'tests' not in path.relative_to(package).parts
                  and '__pycache__' not in path.parts)


def source_files(package):
    package = Path(package)
    return {path.relative_to(package).as_posix(): digest(path) for path in runtime_paths(package)}


def physics_files(files):
    return {name: value for name, value in files.items()
            if name in ('env.py', 'env_config.py', 'amp.py')
            or name.startswith(('core/', 'backends/'))}


def runtime_manifest(source):
    source = Path(source)
    paths = sorted(source.rglob('*.py'))
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob('*')):
        raise ValueError('Runtime dependency uses a symbolic path')
    return {'schema': SCHEMA, 'files': {path.relative_to(source).as_posix(): digest(path)
                                      for path in paths if '__pycache__' not in path.parts}}


def write_runtime(source):
    value = runtime_manifest(source)
    (Path(source)/MANIFEST).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    return value


def verify_runtime(source):
    path = Path(source)/MANIFEST
    recorded = json.loads(path.read_text())
    if recorded != runtime_manifest(source):
        raise ValueError('Runtime dependency manifest differs')
    return recorded
