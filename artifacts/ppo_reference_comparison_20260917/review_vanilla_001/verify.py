"""Check the saved PPO policy, native captures and original evaluation verdicts."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.paper_walk.analyze import load_recording
from experiments.paper_walk.env import _diagnostic_servo, _diagnostic_rotation, emitted_target
from experiments.paper_walk.env_config import KD, LEGS
from experiments.paper_walk.evaluation import score_recording
from experiments.trajectory_optimization.force_metrics import report_for

A = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor(observations, weights):
    x = torch.as_tensor(observations, dtype=torch.float32)
    x = (x - weights['obs_normalizer._mean']) / (weights['obs_normalizer._std'] + .01)
    for index in (0, 2, 4, 6):
        x = torch.nn.functional.linear(x, weights[f'mlp.{index}.weight'], weights[f'mlp.{index}.bias'])
        if index != 6:
            x = torch.nn.functional.elu(x)
    return x.numpy()


def compare_checks(saved, computed):
    differences = {}
    assert set(saved) == set(computed)
    for name in saved:
        left, right = saved[name], computed[name]
        assert {k: v for k, v in left.items() if k != 'value'} == {
            k: v for k, v in right.items() if k != 'value'}, name
        if left.get('value') != right.get('value'):
            a, b = float(left['value']), float(right['value'])
            assert abs(a-b) <= 1e-5 * max(1., abs(a), abs(b)), name
            differences[name] = {'native': a, 'cpu': b, 'original_gate_status_unchanged': True}
    return differences


def main():
    train = A/'vanilla_train_001/run/standing'
    checkpoint = train/'checkpoint_update001200.pt'
    declaration = read(checkpoint.with_suffix('.json'))
    state = read(train/'state.json')
    assert state['status'] == 'completed' and not state['errors']
    assert declaration['checkpoint_sha256'] == sha(checkpoint) == state['checkpoint_sha256']
    assert declaration['updates'] == state['updates'] == 1200
    assert declaration['transitions'] == state['transitions'] == 3686400
    assert declaration['identity'] == state['identity']
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert data['infos'] == {'identity': declaration['identity'], 'updates': 1200}
    steps = sorted({int(value['step']) for value in data['optimizer_state_dict']['state'].values()})
    assert steps == [24000]
    loads = read(train/'force_metrics.json')
    assert loads['physics_steps'] == 230400
    assert loads['windows']['full_training']['samples_across_replicas'] == 29491200
    assert not declaration['identity']['motion_prior'] and not declaration['identity']['behavior_cloning']
    evaluation = A/'vanilla_evaluate_001/run/standing'
    evaluation_state = read(evaluation/'state.json')
    assert evaluation_state['status'] == 'completed' and not evaluation_state['errors']
    for name, receipt in (('vanilla_train_001', state), ('vanilla_evaluate_001', evaluation_state)):
        source = A/name/'source'
        freeze = read(source/'FREEZE_SHA256.json')
        assert receipt['identity']['source_files'] == freeze
        assert receipt['runtime_binding']['runtime_tree_sha256'] == sha(source/'FREEZE_SHA256.json')
        assert all(sha(source/file) == expected for file, expected in freeze.items())
    for key in ('model_sha256', 'usd_sha256', 'physics_source_files', 'physics_config',
                'adapter_sha256', 'entry_sha256', 'upstream_source_files', 'ppo_config'):
        assert evaluation_state['identity'][key] == declaration['identity'][key], key
    native = read(evaluation/'native/native_readback.json')
    limits = torch.tensor(native['limits'], dtype=torch.float32)
    helper = ROOT/'artifacts/restart_2026-09-14/paper_walk_execution_001/verification_evaluate_015/verify.py'
    methods = [node for node in ast.parse(helper.read_text()).body
               if isinstance(node, ast.FunctionDef) and node.name == 'classify']
    assert len(methods) == 1
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(helper), 'exec'), globals())
    geometry_path = ROOT/'artifacts/restart_2026-09-14/standing_translation_preparation_001/source/geometry/geometry.json'
    geometry = {row['body']: row for row in read(geometry_path)['shapes']}
    results = []
    for base in sorted(evaluation.glob('evaluation_*')):
        report, capture, controls, substeps, integrity, binding = load_recording(base)
        case = report['cases'][0]
        assert report['checkpoint_sha256'] == sha(checkpoint)
        assert not report['pose_forcing'] and report['resets_during_trial'] == 0
        assert capture['failure'] is None and len(substeps['sequence']) == len(controls['time_s'])*8
        predictions = actor(controls['policy_observation'], data['actor_state_dict'])
        policy_error = float(np.max(abs(predictions-controls['policy_action'])))
        assert policy_error <= 1e-5, policy_error
        assert np.array_equal(substeps['pre_joint_position_rad'][1:], substeps['joint_position_rad'][:-1])
        assert np.array_equal(substeps['pre_joint_velocity_rad_s'][1:], substeps['joint_velocity_rad_s'][:-1])
        wanted = _diagnostic_servo(substeps['pre_joint_position_rad'][:, 0],
            substeps['pre_joint_velocity_rad_s'][:, 0], substeps['joint_target_rad'][:, 0],
            np.full(18, 12., np.float32), np.asarray(KD, np.float32))
        for key, value in zip(('computed_torque_nm', 'applied_torque_nm', 'effort_ceiling_nm'), wanted):
            assert np.array_equal(substeps[key][:, 0], value), key
        assert np.array_equal(substeps['native_input_pre_nm'], substeps['applied_torque_nm'])
        initial = read(base/'native400hz/initial_state.json')
        neutral = torch.tensor(initial['q'], dtype=torch.float32)
        held = neutral.clone()
        for index, action in enumerate(controls['policy_action']):
            held = emitted_target(torch.from_numpy(action), held, neutral,
                limits[:, :, 0], limits[:, :, 1], .35, .04)
            assert np.array_equal(held.numpy(), controls['joint_target_rad'][index])
        for key in ('joint_position_rad', 'joint_velocity_rad_s', 'root_pose_xyzw', 'joint_target_rad', 'distal_contact'):
            assert np.array_equal(substeps[key][7::8], controls[key]), key
        count = patches = 0
        with (base/'native400hz/contacts.jsonl').open() as stream:
            for line in stream:
                packet = json.loads(line)
                assert packet['sequence'] == count
                assert packet['explicit_counter'] == int(substeps['explicit_counter'][count])
                feet, other, nonfoot, _ = classify(packet, substeps['link_pose_xyzw'][count, 0], capture['body_names'], geometry)
                assert np.array_equal(feet, substeps['distal_force_world_n'][count, 0])
                assert np.array_equal(other, substeps['nonfoot_force_world_n'][count, 0])
                assert nonfoot == bool(substeps['nonfoot_contact'][count, 0])
                count += 1
                patches += len(packet['patches'])
        assert count == capture['steps']
        metadata = {**read(base/'declaration.json'), 'profile': case['profile'], 'env_index': 0,
            'command': case['command'], 'target_slew_rad': .04, 'control_dt_s': .02, 'seed': report['seed']}
        scored = score_recording(controls, metadata)
        original = report['results'][0]
        differences = compare_checks(original['checks'], scored['checks'])
        assert scored['pass'] == original['passed_numeric_screen']
        assert set(scored['failed_bounds']) <= set(original['failed_bounds'])
        q, dq = substeps['joint_position_rad'], substeps['joint_velocity_rad_s']
        outside = ((q < limits[:, :, 0].numpy()-2e-6) | (q > limits[:, :, 1].numpy()+2e-6)).any(-1).sum(0)
        overspeed = (abs(dq) > np.asarray(native['native_max_velocity'])+2e-6).any(-1).sum(0)
        nonfoot_count = substeps['nonfoot_contact'].sum(0)
        assert outside.tolist() == capture['joint_bound_violation_steps']
        assert overspeed.tolist() == capture['speed_bound_violation_steps']
        assert nonfoot_count.tolist() == capture['nonfoot_contact_steps_400hz']
        contact = original['native_contact_screen']
        quiet = case['profile'] in {'quiet_stand', 'stage2_long_quiet', 'stop_to_stand'} or not np.any(case['command'])
        assert contact['maximum_nonfoot_fraction'] == (0. if quiet else .001)
        assert contact['nonfoot_fraction'] == float(nonfoot_count[0])/count
        assert contact['pass'] == (contact['nonfoot_fraction'] <= contact['maximum_nonfoot_fraction'])
        complete = (report['failure'] is None and report['controls'] == case['controls']
                    and count == case['controls']*8 and capture['failure'] is None)
        physical = {'joint_limits': bool(outside[0] == 0), 'joint_speed': bool(overspeed[0] == 0),
            'applied_torque': bool(abs(substeps['applied_torque_nm']).max() <= 1.60001),
            'nonfoot_contact': contact['pass'],
            'non_toe_clearance': bool(substeps['minimum_non_toe_floor_m'].min() >= -.001),
            'plate_height': bool(substeps['root_pose_xyzw'][:, :, 2].min() >= .055)}
        assert original['native_capture_complete'] == complete
        assert original['native_motor_and_joint_checks_pass'] == (complete and all(physical.values()))
        assert original['pass'] == (scored['pass'] and complete and all(physical.values()))
        expected_bounds = list(scored['failed_bounds'])
        if not (complete and all(physical.values())):
            expected_bounds.append('native_capture_or_original_physical_bounds')
        if not contact['pass']:
            expected_bounds.append('canonical_full_trial_400hz_nonfoot_fraction')
        assert original['failed_bounds'] == expected_bounds
        local_loads = report_for(base)
        saved_loads = read(base/'force_metrics.json')
        assert local_loads['source_files'] == saved_loads['source_files']
        assert local_loads['cases'] == saved_loads['cases']
        results.append({'case_id': case['case_id'], 'controls': report['controls'],
            'requested_controls': case['controls'], 'acquisition_complete': report['acquisition_complete'],
            'failure': report['failure'], 'physics_steps': count, 'contact_patches_reclassified': patches,
            'policy_cpu_gpu_max_abs_action_difference': policy_error, 'servo_and_targets_bitexact': True,
            'original_result': original, 'cpu_scorer_value_differences': differences,
            'full_trial_physical_checks': physical,
            'force_summary_reproduced': True, 'video_sha256': sha(base/'rollout.mp4'),
            'report_sha256': sha(base/'report.json'), 'physics_binding': binding})
    assert {row['case_id'] for row in results} == {
        'learning:translate_0.05_0deg', 'learning:quiet_20s', 'learning:forward_0.05_to_stop'}
    for name in ('vanilla_train_001', 'vanilla_evaluate_001'):
        cleanup = read(A/name/'run/cleanup.json')
        assert cleanup['cleanup_checked'] and cleanup['inspections'][-1]['absent']
        assert not cleanup['reservation_released'] and not any(cleanup['resources'].values())
        job = read(A/name/'run/jobs/standing.json')
        assert job['status'] == 'completed' and job['exit_code'] == 0
    recovery = read(A/'vanilla_train_001/RECOVERY_CLEANUP.json')
    assert recovery['returncode'] == 0 and recovery['service_failure_preserved']
    result = {'schema': 'canonical_vanilla_ppo_comparison_audit_v1',
        'utc': datetime.now(timezone.utc).isoformat(), 'checkpoint_sha256': sha(checkpoint),
        'updates': 1200, 'transitions': 3686400, 'optimizer_step_counts': steps,
        'training_load_samples': 29491200, 'cases': results, 'cleanup_verified': True,
        'native_job_exit_codes': {'train': 0, 'evaluate': 0},
        'training_wrapper_failure_preserved': True,
        'training_recovery_cleanup_sha256': sha(A/'vanilla_train_001/RECOVERY_CLEANUP.json'),
        'helper_sha256': sha(helper), 'audit_source_sha256': sha(Path(__file__)),
        'scope': 'One seed and three policy probes. CPU inference tolerance checks provenance and changes no behavior gate.',
        'stage2_complete': False}
    with (Path(__file__).parent/'RESULT.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'checkpoint_sha256': result['checkpoint_sha256'],
        'cases': [{k: row[k] for k in ('case_id', 'physics_steps', 'policy_cpu_gpu_max_abs_action_difference')} for row in results]}))


if __name__ == '__main__':
    main()
