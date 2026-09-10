"""Verify, and optionally reconstruct, reference001 from exact velocity003."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
SOURCE_MANIFEST_SHA = 'a13f1534f9f802871a2cacbc0be603eae2d80738905381da40691f7cd4beff7c'
PARENT_MANIFEST_SHA = '00d4e07e0e31776e3d2faa0136a2b1386f79fd4940bf81d2f4a484155f81bb25'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def safe(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe manifest path: ' + name)
    path = root / relative
    if path.is_symlink():
        raise ValueError('Symlinks are not admitted: ' + name)
    return path


def verify(parent):
    manifest = HERE / 'source_identity/campaign_source_hashes.json'
    if sha(manifest) != SOURCE_MANIFEST_SHA:
        raise ValueError('Wrong reference source manifest')
    origin_path = HERE / 'source_identity/source_origin.json'
    expected = json.loads(manifest.read_text())
    if len(expected) != 923 or sha(origin_path) != expected['source_origin.json']:
        raise ValueError('Wrong reference source origin')
    origin = json.loads(origin_path.read_text())
    if sha(parent / 'campaign_source_hashes.json') != PARENT_MANIFEST_SHA:
        raise ValueError('Parent must be exact root-deployed velocity003')
    if sha(parent / 'source_origin.json') != origin['parent_source_origin_sha256']:
        raise ValueError('Parent origin mismatch')
    before = json.loads((parent / 'campaign_source_hashes.json').read_text())
    if len(before) != 909 or len(origin['overlays']) != 14:
        raise ValueError('Unexpected parent or overlay count')
    for name, digest in before.items():
        if sha(safe(parent, name)) != digest:
            raise ValueError('Parent file changed: ' + name)
    actual_names = {str(p.relative_to(parent)) for p in parent.rglob('*') if p.is_file()}
    if actual_names != set(before) | {'campaign_source_hashes.json'}:
        raise ValueError('Parent source has missing or extra files')
    composed = dict(before)
    for name, digest in origin['overlays'].items():
        if sha(safe(HERE / 'source_overlays', name)) != digest:
            raise ValueError('Overlay changed: ' + name)
        composed[name] = digest
    composed['source_origin.json'] = sha(origin_path)
    if composed != expected:
        raise ValueError('Composed source differs from the complete 923-file map')
    return before, origin, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='Optional fresh directory; default verifies without copying.')
    args = parser.parse_args()
    parent = args.parent.resolve()
    before, origin, expected = verify(parent)
    if args.out is not None:
        dest = args.out.resolve()
        if dest.exists() or dest == parent or parent in dest.parents:
            raise ValueError('Output must be fresh and outside the parent source')
        dest.mkdir(parents=True)
        for name in before:
            target = safe(dest, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(safe(parent, name), target)
        for name in origin['overlays']:
            target = safe(dest, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(safe(HERE / 'source_overlays', name), target)
        for name in ('source_origin.json', 'campaign_source_hashes.json'):
            shutil.copy2(HERE / 'source_identity' / name, dest / name)
        for name, digest in expected.items():
            if sha(safe(dest, name)) != digest:
                raise ValueError('Reconstructed copy failed hash check: ' + name)
    print(json.dumps(dict(passed=True, source_files=923, parent_files=909, overlays=14,
        source_manifest_sha256=SOURCE_MANIFEST_SHA,
        parent_manifest_sha256=PARENT_MANIFEST_SHA,
        reconstructed=str(args.out.resolve()) if args.out else None), indent=2))


if __name__ == '__main__':
    main()
