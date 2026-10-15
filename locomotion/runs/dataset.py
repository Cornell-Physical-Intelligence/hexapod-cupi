"""Check the retained admitted demonstration bank without a training or export path."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from locomotion.core.features import feature_contract
from locomotion.backends.isaac.config import MODEL_SHA256

SCHEMA = 'hexapod_amp_dataset_v1'
WINDOW = (100, 1000)
PROCESSING_FILES = ('amp.py', 'env_config.py', 'task.py', 'evaluation.py',
                    'force_metrics.py', 'priors/commands.py', 'priors/dataset.py')
DATASET = Path(__file__).resolve().parents[1]/'priors/datasets/amp_demonstrations_001'


def validate_processing_identity(processing):
    sources = processing.get('source_files', {})
    schema = processing.get('schema')
    if schema is None:
        expected = {'locomotion/'+name for name in PROCESSING_FILES}
    elif schema == 'hexapod_processing_sources_v2':
        path = Path(__file__).resolve().parents[1]/'priors/processing_sources_v2.json'
        expected = set(json.loads(path.read_text())['files'])
    else:
        raise ValueError('Unknown dataset processing source identity; use its historical exporter')
    if (not processing.get('numpy_version') or set(sources) != expected
            or any(not isinstance(value, str) or len(value) != 64
                   or set(value)-set('0123456789abcdef') for value in sources.values())):
        raise ValueError('Missing dataset processing source identity')


def motion_cases():
    from locomotion.tasks.task import TaskConfig, command_bank
    return [{'case_id': f'amp:command_{i:02d}', 'profile': 'omni_static',
             'command': command, 'controls': 1000}
            for i, command in enumerate(command_bank(TaskConfig()))]


def command_index(command):
    value = np.asarray(command)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('Expected one finite forward/left/yaw command')
    matches = [i for i, case in enumerate(motion_cases())
               if np.allclose(value, case['command'], atol=1e-7, rtol=0)]
    if len(matches) != 1:
        raise ValueError('Command does not match one demonstration coverage case')
    return matches[0]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_bank_identity(records):
    physics_keys = ('physics_source_files', 'physics_config', 'geometry_sha256',
                    'geometry_extrema_sha256', 'stance_sha256')
    physics = [{key: r['identity'][key] for key in physics_keys} for r in records]
    if any(value != physics[0] for value in physics[1:]):
        raise ValueError('Dataset mixes native physics identities')
    gait = [{k: v for k, v in r['identity']['optimization_config'].items()
             if k not in ('forward_mps', 'left_mps', 'yaw_rate_rad_s', 'max_iterations')} for r in records]
    if any(value != gait[0] for value in gait[1:]):
        raise ValueError('Dataset mixes gait or objective settings')
    by_command, state_hashes = {}, {}
    for record in records:
        identity = record['identity']
        ramp = identity.get('startup_ramp_controls', 0)
        if type(ramp) is not int or ramp not in (0, 50):
            raise ValueError('Unknown startup ramp')
        hashes = tuple(identity[key] for key in ('trajectory_sha256', 'optimization_input_sha256',
                                                 'optimization_result_sha256'))+(ramp,)
        previous = by_command.setdefault(command_index(record['command']), hashes)
        if previous != hashes:
            raise ValueError('Command phases use different optimized trajectories or startup ramps')
        signatures = state_hashes.setdefault(command_index(record['command']), set())
        if record['native_state_sha256'] in signatures:
            raise ValueError('Duplicate native states cannot establish a second start phase')
        signatures.add(record['native_state_sha256'])


def reviewed_clips(decision):
    if decision.get('schema') != 'hexapod_amp_motion_review_v1' or decision.get('reviewer') != 'palerdr':
        raise ValueError('The program lead (palerdr) must review this dataset')
    accepted = {r['report_sha256']: r for r in decision['clips']}
    if len(accepted) != len(decision['clips']):
        raise ValueError('Duplicate report in human review')
    return accepted


def require_review(record, accepted):
    human = accepted.get(record['report_sha256'], {})
    if (human.get('direction_accepted') is not True or human.get('tripod_accepted') is not True
            or human.get('video_sha256') != record['video_sha256']):
        raise ValueError('Missing hash-bound human direction/tripod acceptance')


def load(directory):
    """Load raw 61-value pairs; callers choose their learner device."""
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if (manifest.get('schema') != SCHEMA or manifest.get('admitted') is not True
            or manifest.get('feature_contract') != feature_contract()
            or manifest.get('model_sha256') != MODEL_SHA256
            or manifest.get('coverage') != motion_cases()
            or manifest.get('review_sha256') != sha(directory/'review.json')
            or manifest.get('review') != json.loads((directory/'review.json').read_text())
            or manifest.get('file_sha256') != sha(directory/'transitions.npz')):
        raise ValueError('Dataset schema, feature contract, coverage or bytes differ')
    with np.load(directory/'transitions.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    n = manifest['transitions']
    if (set(arrays) != {'states', 'next_states', 'commands', 'time_s', 'control_index', 'clip_id'}
            or n != len(motion_cases())*900 or arrays['states'].shape != (n, 61)
            or arrays['next_states'].shape != (n, 61) or arrays['commands'].shape != (n, 3)
            or any(arrays[key].shape != (n,) for key in ('clip_id', 'time_s', 'control_index'))
            or any(not np.isfinite(value).all() for value in arrays.values())):
        raise ValueError('Incomplete finite transition bank')
    if (manifest.get('reviewer') != 'palerdr' or manifest.get('window') != list(WINDOW)
            or not np.issubdtype(arrays['clip_id'].dtype, np.integer)
            or not np.issubdtype(arrays['control_index'].dtype, np.integer)):
        raise ValueError('Invalid review, window or clip-index contract')
    validate_processing_identity(manifest.get('processing_identity', {}))
    accepted = reviewed_clips(manifest['review'])
    seen = set()
    for clip in manifest['clips']:
        require_review(clip, accepted)
        key = (command_index(clip['command']), clip['start_phase'])
        if key in seen:
            raise ValueError('Duplicate native command/phase evidence')
        seen.add(key)
    if seen != {(i, phase) for i in range(len(motion_cases())) for phase in (0., .5)}:
        raise ValueError('Incomplete native command/phase evidence')
    require_bank_identity(manifest['clips'])
    coverage = set()
    for clip_id in np.unique(arrays['clip_id']):
        if not 0 <= clip_id < len(manifest['clips']):
            raise ValueError('Invalid clip index')
        clip = manifest['clips'][clip_id]
        selected = arrays['clip_id'] == clip_id
        if (clip['start_phase'] != 0 or selected.sum() != 900
                or not np.array_equal(arrays['control_index'][selected], np.arange(*WINDOW))
                or not np.allclose(arrays['time_s'][selected], (np.arange(*WINDOW)+1)*.02, atol=1e-9, rtol=0)
                or not np.allclose(arrays['commands'][selected], clip['command'], atol=1e-7, rtol=0)
                or not np.array_equal(arrays['states'][selected][1:], arrays['next_states'][selected][:-1])):
            raise ValueError('Dataset clip boundaries, commands or state continuity differ')
        index = command_index(clip['command'])
        if index in coverage:
            raise ValueError('Repeated command in training clips')
        coverage.add(index)
    if coverage != set(range(len(motion_cases()))):
        raise ValueError('Dataset coverage differs from declared commands')
    return arrays, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='?', default=DATASET)
    args = parser.parse_args()
    arrays, manifest = load(args.directory)
    print(json.dumps({'schema': manifest['schema'], 'transitions': len(arrays['states']),
                      'manifest_sha256': sha(args.directory/'manifest.json')}))


if __name__ == '__main__':
    main()
