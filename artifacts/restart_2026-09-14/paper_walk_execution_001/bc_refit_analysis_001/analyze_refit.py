"""Compare three cold-start records with the preserved startup analyzer."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
A = REPO / 'artifacts/restart_2026-09-14/paper_walk_execution_001'
sys.path.insert(0, str(REPO))
from experiments.paper_walk import analyze, evaluation
from experiments.paper_walk.env import _diagnostic_servo, emitted_target
from experiments.paper_walk.env_config import KD
PINS = {}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()

def pin(path):
    path = Path(path)
    PINS[str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)] = sha(path)
    return path

def read(path): return json.loads(pin(path).read_text())

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, pin(path))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

pair = module('preserved_startup_pair', A / 'startup_pair_analysis_001/analyze_pair.py')
audit = module('preserved_native_audit', A / 'startup_pair_analysis_001/native_audit/audit_004.py')
pin(audit.load_audit_helpers())
analyze.load_arrays = pair.typed_arrays
for name in ('analyze.py', 'env.py', 'env_config.py', 'evaluation.py'):
    path = REPO / 'experiments/paper_walk' / name
    assert sha(path) == read(A / 'source_020/FREEZE_SHA256.json')[name]
    pin(path)
learner = module('refit_frozen_learner', A / 'source_020/learner.py')
geometry_path = REPO / 'artifacts/restart_2026-09-14/standing_translation_preparation_001/source/geometry/geometry.json'
geometry = {row['body']: row for row in read(geometry_path)['shapes']}
data = pair.typed_arrays(pin(REPO / 'artifacts/paper_bc_data_002/bc_dataset.npz'))
forward = np.all(data['commands'] == np.array([.05, 0, 0], np.float32), axis=1)
pool = np.flatnonzero(forward & (data['source_kind'] == 1))

ARMS = {
    'parent': ('startup_cold_001', REPO / 'artifacts/restart_2026-09-14/paper_bc_migration_002/run_001/checkpoint_update000000_migrated.pt'),
    'uniform': ('startup_cold_refit_uniform_001', REPO / 'artifacts/paper_bc_refit_001/uniform/candidate_checkpoint_update000000.pt'),
    'onset_weight20': ('startup_cold_refit_onset20_001', REPO / 'artifacts/paper_bc_refit_001/onset_weight20/candidate_checkpoint_update000000.pt'),
}

def inspect(label, attempt, checkpoint):
    directory = A / ('results_' + attempt)
    inventory = read(A / ('inventory_' + attempt + '.json'))
    files = {str(p.relative_to(directory)): sha(pin(p)) for p in directory.rglob('*') if p.is_file()}
    assert files == inventory['files']
    assert read(A / ('VERIFIED_TRANSFER_' + attempt + '.json'))['all_files_verified']
    state = read(directory / 'standing/state.json')
    assert state['status'] == 'completed'
    freeze = read(A / ('source_019' if label == 'parent' else 'source_020') / 'FREEZE_SHA256.json')
    assert state['identity']['source_files'] == freeze
    base = directory / 'standing/evaluation'
    report, capture, control, native, integrity, binding = analyze.load_recording(base)
    assert report['acquisition_complete'] and report['failure'] is None
    assert capture['failure'] is None and report['controls'] == 1000 and capture['steps'] == 8000
    assert report['initial_resets'] == 1 and report['resets_during_trial'] == 0
    assert report['policy_control_slice'] == [0, 1000] and report['startup_prefix_controls'] == 0
    assert np.array_equal(control['global_control_index'], np.arange(1000))
    assert np.array_equal(control['policy_control_index'], np.arange(1000))
    assert np.all(control['action_source'] == 'bc')
    assert np.array_equal(control['issued_action'], control['policy_action'])
    assert np.array_equal(control['issued_action'], control['actor_mean_action'])
    assert np.array_equal(control['command'], np.broadcast_to(np.array([.05, 0, 0], np.float32), (1000, 1, 3)))
    assert np.array_equal(native['command'], np.repeat(control['command'], 8, axis=0))
    assert np.array_equal(native['explicit_counter'], np.arange(1, 8001))
    assert not any(control[k].any() for k in ('terminated', 'truncated', 'reset'))
    assert np.array_equal(native['pre_joint_position_rad'][1:], native['joint_position_rad'][:-1])
    assert np.array_equal(native['pre_joint_velocity_rad_s'][1:], native['joint_velocity_rad_s'][:-1])
    initial = read(base / 'native400hz/initial_state.json')
    assert np.array_equal(native['pre_joint_position_rad'][0], np.asarray(initial['q'], np.float32))
    assert np.array_equal(native['pre_joint_velocity_rad_s'][0], np.asarray(initial['dq'], np.float32))
    for key in ('joint_position_rad', 'joint_velocity_rad_s', 'root_pose_xyzw', 'joint_target_rad', 'distal_contact', 'toe_xyz_world_m'):
        assert np.array_equal(native[key][7::8], control[key]), key
    expected = _diagnostic_servo(native['pre_joint_position_rad'][:, 0], native['pre_joint_velocity_rad_s'][:, 0],
                                native['joint_target_rad'][:, 0], np.full(18, 12., np.float32), np.asarray(KD, np.float32))
    for key, value in zip(('computed_torque_nm', 'applied_torque_nm', 'effort_ceiling_nm'), expected):
        assert np.array_equal(native[key][:, 0], value), key
    assert np.array_equal(native['applied_torque_nm'], native['native_input_pre_nm'])
    assert np.array_equal(native['joint_target_rad'], np.repeat(control['joint_target_rad'], 8, axis=0))
    audit.NATIVE = read(directory / 'standing/native/native_readback.json')
    neutral = torch.tensor(initial['q'], dtype=torch.float32)
    held = neutral.clone()
    limits = torch.tensor(audit.NATIVE['limits'], dtype=torch.float32)
    for index, action in enumerate(control['issued_action']):
        held = emitted_target(torch.from_numpy(action), held, neutral, limits[:, :, 0], limits[:, :, 1], .35, .04)
        assert np.array_equal(held.numpy(), control['joint_target_rad'][index])
    windows = {name: audit.window(native, start, end, bound) for name, start, end, bound in
               [('full', 0, 8000, .001), ('scripted_prefix', 0, 0, 0.), ('policy', 0, 8000, .001)]}
    assert windows == report['startup_physical_windows']['windows']
    packets = patches = 0
    with (base / 'native400hz/contacts.jsonl').open() as stream:
        for line in stream:
            packet = json.loads(line)
            assert packet['sequence'] == packets and packet['explicit_counter'] == int(native['explicit_counter'][packets])
            feet, other, nonfoot, _ = audit.classify(packet, native['link_pose_xyzw'][packets, 0], capture['body_names'], geometry)
            assert np.array_equal(feet, native['distal_force_world_n'][packets, 0])
            assert np.array_equal(other, native['nonfoot_force_world_n'][packets, 0])
            assert np.array_equal(np.linalg.norm(feet, axis=-1) > 1, native['distal_contact'][packets, 0])
            assert nonfoot == bool(native['nonfoot_contact'][packets, 0])
            packets += 1
            patches += len(packet['patches'])
    assert packets == 8000
    declaration = read(base / 'declaration.json')
    score = evaluation.score_recording(control, {**declaration, 'profile': 'omni_static', 'env_index': 0,
        'command': [.05, 0, 0], 'target_slew_rad': .04, 'control_dt_s': .02,
        'seed': report['cases'][0].get('seed', report['seed'])})
    original = report['results'][0]
    assert score['checks'] == original['checks']
    expected_failed = list(score['failed_bounds'])
    physical_pass = all(window['pass'] for window in windows.values())
    if not physical_pass:
        expected_failed.append('native_capture_or_original_physical_bounds')
    assert expected_failed == original['failed_bounds']
    assert (score['pass'] and physical_pass) == original['pass']
    cleanup = read(directory / 'cleanup.json')
    assert cleanup['cleanup_checked'] and cleanup['inspections'][-1]['absent']
    assert not cleanup['reservation_released'] and not any(cleanup['resources'].values())
    assert read(directory / 'standing/native/native_errors.json') == []
    assert read(directory / 'jobs/standing_contact_data_audit.json')['passed']
    saved = torch.load(pin(checkpoint), map_location='cpu', weights_only=False)
    assert state['input_checkpoint_sha256'] == sha(checkpoint) == report['checkpoint_sha256']
    assert saved['learner_sha256'] == sha(A / 'source_020/learner.py')
    model = learner.ActorCritic(learner.Config(**saved['config'])).eval()
    model.load_state_dict(saved['model'], strict=True)
    obs = control['policy_observation'][:, 0]
    with torch.no_grad():
        means, _ = model.actor(torch.from_numpy(obs))
        z = model.obs_normalizer(torch.from_numpy(obs)).numpy()
        demo = model.obs_normalizer(torch.from_numpy(data['observations'])).numpy()
    actor_error = float(np.max(np.abs(means.numpy() - control['issued_action'][:, 0])))
    assert actor_error < 2e-6
    distances = pair.distances(z, demo[pool])
    nearest = pool[distances.argmin(axis=1)]
    target_errors = np.sqrt(np.mean(((control['issued_action'][:, 0] - data['actions'][nearest]) * .35)**2, axis=1))
    summaries = {}
    for name, start, end in [('all1000', 0, 1000), ('first100', 0, 100), ('later900', 100, 1000)]:
        summary = analyze.analyze_trace({k:v[start:end] for k,v in control.items()},
            {k:v[start*8:end*8] for k,v in native.items()}, joint_names=capture['joint_names'])
        summaries[name] = summary
    return {'attempt': attempt, 'checkpoint_sha256': sha(checkpoint), 'physics_binding': binding,
        'controls': 1000, 'physics_steps': packets, 'contact_patches_reclassified': patches,
        'files_verified': len(files), 'actor_cpu_native_max_abs': actor_error,
        'original_gate_result': original, 'physical_windows': windows, 'summaries': summaries,
        'first_onset_teacher': {'dataset_row': int(nearest[0]), 'normalized_distance': float(distances[0].min()),
            'requested_target_rms_rad': float(target_errors[0])},
        'nearest_onset_target_error': {name: pair.stats(target_errors[sl]) for name, sl in
            [('first100', slice(0,100)), ('later900', slice(100,None))]},
        'servo_targets_contacts_and_gate_checks_match': True, 'own_container_absent': True,
        'video_frames': report['video_frames'], 'integrity': integrity}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    pin(Path(__file__).resolve())
    torch.set_num_threads(1)
    rows = {label: inspect(label, *values) for label, values in ARMS.items()}
    assert all(row['physics_binding'] == rows['parent']['physics_binding'] for row in rows.values())
    parent = rows['parent']
    for label in ('uniform', 'onset_weight20'):
        row = rows[label]
        row['improves_first_onset_without_worse_steady'] = bool(
            row['first_onset_teacher']['requested_target_rms_rad'] < parent['first_onset_teacher']['requested_target_rms_rad']
            and row['summaries']['later900']['tracking']['planar_error_rmse_mps'] <= parent['summaries']['later900']['tracking']['planar_error_rmse_mps'])
    assert all(sha(REPO/path if not Path(path).is_absolute() else Path(path)) == digest for path, digest in PINS.items())
    result = {'schema': 'canonical_bc_refit_native_comparison_v1', 'utc': datetime.now(timezone.utc).isoformat(),
        'arms': rows, 'same_physics': True, 'inputs_unchanged': True, 'input_sha256': PINS,
        'comparison_scope': 'One cold trial per checkpoint; first onset compares the actual action with the nearest same-command teacher label under the saved actor normalization. Later900 planar RMSE retains the prior descriptive window. Original gate decisions remain unchanged.',
        'stage2_complete': False}
    with args.output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({label: {'failed_bounds': row['original_gate_result']['failed_bounds'],
        'onset_target_rms_rad': row['first_onset_teacher']['requested_target_rms_rad'],
        'later900_planar_rmse_mps': row['summaries']['later900']['tracking']['planar_error_rmse_mps']} for label,row in rows.items()}))

if __name__ == '__main__': main()
