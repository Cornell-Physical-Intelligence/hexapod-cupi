"""Standard-library contract for passive canonical native inspection only."""
from pathlib import Path
import hashlib
import json

SCHEMA = 'canonical_native_inspection_v1'
ASSET_MAP_SHA256 = '4cf88f1a658c23e20ddf3f8ed7a08a100ce2ddd337e59605c6f611bc7570d858'
PHASE = 'inspection'
STEPS = 8
DT = 0.0025


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check_map(root, expected):
    root = Path(root).resolve()
    for name, digest in expected.items():
        p = root / name
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
            raise ValueError(f'Unsafe/missing input: {name}')
        if sha(p) != digest:
            raise ValueError(f'Changed input: {name}')
    return True


def verify_inputs(args):
    own = Path(__file__).resolve().parent
    if sha(own / 'ASSET_SHA256.json') != ASSET_MAP_SHA256:
        raise ValueError('Asset map is not the pinned canonical inventory')
    freeze = read(own / 'FREEZE_SHA256.json')
    check_map(own, freeze)
    actual_own = {str(p.relative_to(own)) for p in own.rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and p.name != 'FREEZE_SHA256.json'}
    if actual_own != set(freeze):
        raise ValueError('Unexpected inspector payload')
    asset = Path(args.asset).resolve()
    expected = read(own / 'ASSET_SHA256.json')
    if {str(p.relative_to(asset)) for p in asset.rglob('*') if p.is_file()} != set(expected):
        raise ValueError('Asset inventory differs from exact nine-file canonical bundle')
    check_map(asset, expected)
    output = Path(args.output).resolve()
    if output.is_relative_to(asset) or output.is_relative_to(own):
        raise ValueError('Output must be outside immutable inputs')
    return {'schema': SCHEMA, 'inspector_freeze_sha256': sha(own/'FREEZE_SHA256.json'),
            'runtime_binding': {'runtime_tree_sha256': sha(own/'FREEZE_SHA256.json'),
                                'scope': 'canonical_native_inspection_only'},
            'asset_manifest_sha256': ASSET_MAP_SHA256,
            'urdf_sha256': expected['source/source.urdf'],
            'model_sha256': expected['source/model.json'],
            'usd_sha256': expected['robot.usda'],
            'steps': STEPS, 'dt': DT, 'gravity': [0., 0., 0.],
            'physics_admitted': False, 'physical_admission': False, 'training_allowed': False}


def validate_result(directory, identity):
    directory = Path(directory)
    s = read(directory / 'state.json')
    if s.get('schema') != SCHEMA or s.get('identity') != identity:
        raise ValueError('Wrong inspection identity')
    if s.get('status') != 'completed' or s.get('inputs_unchanged') is not True:
        raise ValueError('Incomplete/failed inspection')
    if s.get('errors') != [] or s.get('native_error_events') != []:
        raise ValueError('Recorded native/runtime error')
    # This file intentionally remains live through close; inspect its post-exit bytes.
    if read(directory/'native_errors.json') != []:
        raise ValueError('Late native PhysX error event')
    if s.get('physics_admitted') is not False or s.get('physical_admission') is not False or s.get('training_allowed') is not False:
        raise ValueError('Inspection cannot admit physics or training')
    if s.get('explicit_steps_completed') != STEPS or not s.get('checks') or not all(v is True for v in s['checks'].values()):
        raise ValueError('Missing/rejected native checks')
    required = {'usd_identity', 'native_identity', 'native_frames', 'native_scene', 'native_sdf_paths', 'no_drive_gains', 'finite_samples', 'sdk_source_bound'}
    if not required.issubset(s['checks']):
        raise ValueError('Incomplete native checks')
    outputs = s.get('outputs', {})
    if not {'usd_readback.json', 'native_readback.json', 'samples.json', 'sdf_readback.json', 'resolved_stage.usda', 'runtime_api.json'}.issubset(outputs):
        raise ValueError('Missing raw inspection evidence')
    check_map(directory, outputs)
    return {'phase': PHASE, 'status': 'completed', 'scope': 'passive_import_cooking_inspection_only',
            'physics_admitted': False, 'physical_admission': False, 'training_allowed': False,
            'state_sha256': sha(directory/'state.json')}
