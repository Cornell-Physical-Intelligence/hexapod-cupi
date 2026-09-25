"""Audit native demonstrations and export a reviewed AMP transition bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from locomotion.amp import extract_features, feature_contract
from locomotion.env_config import MODEL_SHA256, USD_SHA256
from locomotion.evaluation import score_recording
from .commands import command_index, motion_cases

SCHEMA = 'hexapod_amp_dataset_v1'
WINDOW = (100, 1000)
PROCESSING_FILES = ('amp.py', 'env_config.py', 'task.py', 'evaluation.py',
                    'force_metrics.py', 'priors/commands.py', 'priors/dataset.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def processing_identity():
    root = Path(__file__).resolve().parents[1]
    return {'numpy_version': np.__version__, 'source_files': {
        'locomotion/'+name: sha(root/name) for name in PROCESSING_FILES}}


def rotations(quaternion):
    q = np.asarray(quaternion)
    if q.shape[-1] != 4 or not np.allclose(np.linalg.norm(q, axis=-1), 1., atol=1e-5, rtol=0):
        raise ValueError('Invalid native XYZW quaternion')
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3, 3))


def reconstruct_after(trace, root_com_local):
    """Rebuild features from recorded joint, root and world-toe measurements."""
    offset = np.asarray(root_com_local)
    if offset.shape != (1, 3) or not np.isfinite(offset).all():
        raise ValueError('Expected one finite native root COM offset')
    root = trace['root_pose_xyzw']
    rotation = rotations(root[..., 3:])
    angular = trace['gyro_body_rad_s']
    com_velocity = np.einsum('...ji,...j->...i', rotation, trace['velocity_world_mps'])
    linear = com_velocity-np.cross(angular, offset)
    toes = np.einsum('...ji,...lj->...li', rotation, trace['toe_xyz_world_m']-root[..., None, :3])
    return extract_features({'q': trace['joint_position_rad'], 'dq': trace['joint_velocity_rad_s'],
        'root': root, 'linear': linear, 'angular': angular, 'toe_body': toes})


def validate_trace(trace, command, root_com_local):
    """Reject incomplete clips and reset boundaries before the fixed training crop."""
    before, after = trace['amp_state_before'], trace['amp_state_after']
    if before.shape != (1000, 1, 61) or after.shape != before.shape:
        raise ValueError('Expected a complete one-robot 1000-control native clip')
    tails = {'root_pose_xyzw': (7,), 'joint_position_rad': (18,), 'joint_velocity_rad_s': (18,),
             'gyro_body_rad_s': (3,), 'velocity_world_mps': (3,), 'toe_xyz_world_m': (6, 3)}
    if any(trace[name].shape != (1000, 1)+tail for name, tail in tails.items()):
        raise ValueError('Native telemetry shape differs from the one-robot contract')
    required = ('amp_state_before', 'amp_state_after', 'command', 'time_s',
                'root_pose_xyzw', 'joint_position_rad', 'joint_velocity_rad_s',
                'gyro_body_rad_s', 'velocity_world_mps', 'toe_xyz_world_m')
    if any(not np.isfinite(trace[name]).all() for name in required):
        raise ValueError('Nonfinite native feature or telemetry')
    times = trace['time_s']
    if times.shape != (1000,) or not np.allclose(times, (np.arange(1000)+1)*.02, atol=1e-9, rtol=0):
        raise ValueError('Native transition timestamps differ from 20 ms controls')
    for name in ('reset', 'terminated', 'truncated'):
        if trace[name].shape != (1000, 1) or np.any(trace[name] != 0):
            raise ValueError('Terminal, truncation or reset boundary in '+name)
    if trace['command'].shape != (1000, 1, 3) or not np.allclose(trace['command'], command, atol=1e-7, rtol=0):
        raise ValueError('Native commands differ from the declared motion')
    if not np.array_equal(before[1:], after[:-1]):
        raise ValueError('Native state pairs cross a discontinuity or reset')
    reconstructed = reconstruct_after(trace, root_com_local)
    parity_error = float(np.max(abs(reconstructed-after)))
    if not np.isfinite(parity_error) or parity_error > 2e-5:
        raise ValueError('Recorded AMP features differ from native measurements')
    return parity_error


def motion_diagnostics(trace, command):
    """Report achieved motion and contacts; the program lead reviews gait and direction."""
    start, stop = WINDOW
    states = trace['amp_state_after'][start:stop, 0]
    velocity = np.column_stack((-states[:, 37], states[:, 36]))
    speed = float(np.linalg.norm(command[:2]))
    contacts = trace['distal_contact']
    if contacts.shape != (1000, 1, 6) or not np.isin(contacts, (0, 1)).all():
        raise ValueError('Expected six measured distal contact states per control')
    contacts = contacts[start:stop, 0].astype(bool)
    events = (contacts[1:] != contacts[:-1]).sum(0)
    return {'mean_velocity_forward_left_mps': velocity.mean(0).tolist(),
        'mean_yaw_rad_s': float(states[:, 41].mean()),
        'signed_translation_fraction': float((velocity@np.asarray(command[:2])/speed**2).mean()) if speed else None,
        'signed_yaw_fraction': float(states[:, 41].mean()/command[2]) if command[2] else None,
        'per_foot_contact_fraction': contacts.mean(0).tolist(),
        'per_foot_contact_changes': events.tolist(),
        'tripod_a_only_fraction': float((contacts[:, [0, 2, 4]].all(1)&~contacts[:, [1, 3, 5]].any(1)).mean()),
        'tripod_b_only_fraction': float((contacts[:, [1, 3, 5]].all(1)&~contacts[:, [0, 2, 4]].any(1)).mean()),
        'review_required': 'Verify achieved direction and alternating-tripod motion against the video and contacts.',
        'new_load_acceptance_limits': False}


def audit_replay(directory):
    directory = Path(directory).resolve()
    state_path = directory/'state.json'
    state = json.loads(state_path.read_text())
    identity = state['identity']
    if (state.get('status') != 'completed' or state.get('eligible_for_motion_prior') is not True
            or identity.get('controller_kind') != 'optimized_periodic_motor_targets'
            or identity.get('learned_policy') is not False
            or identity.get('model_sha256') != MODEL_SHA256 or identity.get('usd_sha256') != USD_SHA256
            or identity.get('amp_feature_contract') != feature_contract()):
        raise ValueError('Replay lacks completed model-bound optimized motion evidence')
    if identity.get('start_phase') not in (0., .5):
        raise ValueError('Missing declared replay start phase')
    ramp = identity.get('startup_ramp_controls', 0)
    if type(ramp) is not int or ramp not in (0, 50):
        raise ValueError('Unknown startup ramp')
    for key in ('trajectory_sha256', 'optimization_input_sha256', 'optimization_result_sha256'):
        value = identity.get(key, '')
        if len(value) != 64 or set(value)-set('0123456789abcdef'):
            raise ValueError('Missing optimization identity: '+key)
    if (not identity.get('physics_source_files') or not identity.get('standing_admission')
            or not identity.get('optimization_config') or not identity.get('optimization_source_files')):
        raise ValueError('Missing native admission or optimizer identity')
    index = command_index(identity['command'])
    report_path = directory/'evaluation/report.json'
    report = json.loads(report_path.read_text())
    if (state.get('replay') != report or report.get('acquisition_complete') is not True
            or report.get('recorded_physics_steps') != 8000 or len(report.get('results', [])) != 1
            or report['results'][0].get('pass') is not True
            or report['results'][0].get('native_motor_and_joint_checks_pass') is not True
            or report['results'][0].get('native_contact_screen', {}).get('pass') is not True
            or report.get('force_metrics', {}).get('status') != 'available'):
        raise ValueError('Native motion, contact or force evidence is incomplete')
    for name, expected in report['files'].items():
        relative = Path(name)
        if relative.name != name or not (report_path.parent/relative).is_file():
            raise ValueError('Unsafe or missing native evidence file: '+name)
        if sha(report_path.parent/relative) != expected:
            raise ValueError('Changed native evidence: '+name)
    for name in ('control_trace.npz', 'force_metrics.json', 'rollout.mp4'):
        if name not in report['files']:
            raise ValueError('Missing bound native evidence: '+name)
    from locomotion.force_metrics import report_for
    saved_loads = json.loads((report_path.parent/'force_metrics.json').read_text())
    computed_loads = report_for(report_path.parent)
    for key in ('cases', 'source_files', 'capture_failure', 'model_sha256'):
        if saved_loads.get(key) != computed_loads.get(key):
            raise ValueError('Native force/torque recomputation differs: '+key)
    if (computed_loads['capture_failure'] is not None
            or len(computed_loads['cases']) != 1
            or computed_loads['cases'][0]['recorded_samples'] != 8000
            or computed_loads['cases'][0]['requested_window_complete'] is not True):
        raise ValueError('Incomplete native force/torque capture')
    trace_path = report_path.parent/'control_trace.npz'
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    parity = validate_trace(trace, identity['command'], identity['root_com_local_m'])
    rescored = score_recording(trace, {**report, 'profile': 'omni_static',
        'env_index': 0, 'command': identity['command'], 'control_dt_s': .02})
    if not rescored['pass']:
        raise ValueError('Recomputed motion screen failed')
    record = {'path': str(directory), 'command_index': index, 'command': identity['command'],
        'start_phase': identity['start_phase'], 'state_sha256': sha(state_path),
        'report_sha256': sha(report_path), 'trace_sha256': sha(trace_path),
        'native_state_sha256': hashlib.sha256(trace['amp_state_after'].tobytes()).hexdigest(),
        'video_sha256': report['files']['rollout.mp4'], 'identity': identity,
        'feature_parity_max_abs': parity, 'diagnostics': motion_diagnostics(trace, identity['command'])}
    return record, trace


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


def inspect(replays, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records, failures = [], []
    for directory in replays:
        try:
            record, _ = audit_replay(directory)
            records.append(record)
        except (ValueError, KeyError, OSError) as error:
            failures.append({'path': str(directory), 'error': str(error)})
    coverage = sorted({r['command_index'] for r in records if r['start_phase'] == 0})
    result = {'schema': SCHEMA, 'feature_contract': feature_contract(), 'clips': records,
        'processing_identity': processing_identity(),
        'failures': failures, 'missing_command_indices': sorted(set(range(len(motion_cases())))-set(coverage)),
        'missing_command_phases': [[i, phase] for i in range(len(motion_cases())) for phase in (0., .5)
            if (i, phase) not in {(r['command_index'], r['start_phase']) for r in records}],
        'admitted': False, 'human_review_pending': True}
    save(output/'audit.json', result)
    save(output/'review.json', {'schema': 'hexapod_amp_motion_review_v1', 'reviewer': None,
        'clips': [{'report_sha256': r['report_sha256'], 'video_sha256': r['video_sha256'],
                   'direction_accepted': False, 'tripod_accepted': False} for r in records]})
    return result


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


def export(replays, review, output):
    review = Path(review)
    decision = json.loads(review.read_text())
    accepted = reviewed_clips(decision)
    records, chunks, seen = [], [], set()
    for directory in replays:
        record, trace = audit_replay(directory)
        require_review(record, accepted)
        key = (record['command_index'], record['start_phase'])
        if key in seen:
            raise ValueError('Duplicate command/phase replay does not add coverage')
        seen.add(key)
        records.append(record)
        if record['start_phase'] == 0:
            start, stop = WINDOW
            chunks.append({'states': trace['amp_state_before'][start:stop, 0],
                'next_states': trace['amp_state_after'][start:stop, 0],
                'commands': trace['command'][start:stop, 0],
                'time_s': trace['time_s'][start:stop], 'control_index': np.arange(start, stop),
                'clip_id': np.full(stop-start, len(records)-1, dtype=np.int64)})
    expected = {(i, phase) for i in range(len(motion_cases())) for phase in (0., .5)}
    if seen != expected:
        raise ValueError('Missing native command/phase coverage: '+str(sorted(expected-seen)))
    require_bank_identity(records)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'review.json').write_bytes(review.read_bytes())
    arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    np.savez_compressed(output/'transitions.npz', **arrays)
    manifest = {'schema': SCHEMA, 'feature_contract': feature_contract(), 'model_sha256': MODEL_SHA256,
        'processing_identity': processing_identity(),
        'admitted': True, 'reviewer': decision['reviewer'], 'review_sha256': sha(review),
        'review': decision, 'clips': records, 'coverage': motion_cases(), 'window': list(WINDOW),
        'transitions': len(arrays['states']), 'file_sha256': sha(output/'transitions.npz'),
        'sampling': 'uniform command, then uniform transition; phase-0.5 clips are stress tests',
        'learning_benefit_tested': False, 'stage2_complete': False}
    save(output/'manifest.json', manifest)
    return manifest


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
    processing = manifest.get('processing_identity', {})
    sources = processing.get('source_files', {})
    if (not processing.get('numpy_version')
            or set(sources) != {'locomotion/'+name for name in PROCESSING_FILES}
            or any(not isinstance(value, str) or len(value) != 64
                   or set(value)-set('0123456789abcdef') for value in sources.values())):
        raise ValueError('Missing dataset processing source identity')
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


def sample(arrays, batch_size, rng):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Positive batch size required')
    indices = rng.integers(len(arrays['states']), size=batch_size)
    return np.concatenate((arrays['states'][indices], arrays['next_states'][indices]), axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('inspect', 'export', 'check'))
    parser.add_argument('--replay', type=Path, action='append', default=[])
    parser.add_argument('--review', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'check':
        arrays, manifest = load(args.output)
        print(json.dumps({'admitted': True, 'transitions': len(arrays['states'])}))
    elif args.mode == 'export':
        if args.review is None or not args.replay:
            parser.error('Export requires native replays and a human review')
        print(json.dumps(export(args.replay, args.review, args.output), indent=2))
    else:
        if not args.replay:
            parser.error('Inspection requires native replays')
        print(json.dumps(inspect(args.replay, args.output), indent=2))


if __name__ == '__main__':
    main()
