"""Read-only, complete-recording comparison of native evaluations011 and012."""
from pathlib import Path
import json
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
A = HERE.parent
REPO = A.parents[2]
sys.path.insert(0, str(REPO))
from experiments.paper_walk.analyze import load_recording, sha, spectrum
from experiments.paper_walk.env import _diagnostic_servo, emitted_target, executed_action_feature
from experiments.paper_walk.env_config import KD, JOINT_NAMES


def read(path):
    return json.loads(Path(path).read_text())


def rms(x, axis=0):
    return np.sqrt(np.mean(np.square(np.asarray(x, float)), axis=axis))


def window(control, native, start, end):
    c = {k: v[start:end, 0] for k, v in control.items() if v.ndim > 1}
    n = {k: v[start*8:end*8, 0] for k, v in native.items() if v.ndim > 1}
    estimated, actual = c['estimated_velocity_navigation_mps'], c['critic_observation'][:, -3:]
    error = estimated.astype(float)-actual
    neutral = np.tile(np.array([0., -.3, .4], np.float32), 6)
    normalized = (c['joint_target_rad']-neutral)/.35
    delta = np.diff(c['joint_target_rad'], axis=0)
    support = n['distal_contact']
    command = c['command']
    moving = np.linalg.norm(command[:, :2], axis=-1) > 1e-8
    result = {
        'start_control_inclusive': start, 'end_control_exclusive': end,
        'endpoint_time_range_s': [(start+1)*.02, end*.02],
        'native_time_range_s': [(start*8+1)*.0025, end*.02],
        'control_samples': end-start, 'native_samples': (end-start)*8,
        'estimator_root_origin_navigation': {
            'axes': ['forward', 'left', 'up'], 'rmse_mps': rms(error).tolist(),
            'bias_mps': error.mean(0).tolist(), 'estimated_mean_mps': estimated.mean(0).tolist(),
            'native_mean_mps': actual.mean(0).tolist(), 'native_rms_mps': rms(actual).tolist(),
            'scope': 'Recorded pre-policy estimator versus same-time body-root origin navigation velocity; not COM or command.'},
        'mean_com_navigation_velocity_mps': c['velocity_navigation_mps'].mean(0).tolist(),
        'moving_mean_signed_forward_mps': float(c['velocity_navigation_mps'][moving, 0].mean()) if moving.any() else None,
        'per_joint_control_velocity_rms_rad_s': rms(c['joint_velocity_rad_s']).tolist(),
        'per_joint_native_velocity_rms_rad_s': rms(n['joint_velocity_rad_s']).tolist(),
        'maximum_control_joint_rms_rad_s': float(rms(c['joint_velocity_rad_s']).max()),
        'maximum_native_joint_rms_rad_s': float(rms(n['joint_velocity_rad_s']).max()),
        'normalized_held_target': {'formula': '(held_target-neutral)/0.35rad',
            'per_joint_mean': normalized.mean(0).tolist(), 'absolute_peak': float(abs(normalized).max()),
            'per_joint_range': np.ptp(normalized, axis=0).tolist(),
            'per_joint_joint_tracking_rms': rms((c['joint_position_rad']-c['joint_target_rad'])/.35).tolist()},
        'policy_action_abs_gt1_fraction': float((abs(c['policy_action']) > 1).mean()),
        'target_step_abs_p95_max_rad': float(np.quantile(abs(delta), .95, axis=0).max()),
        'target_step_slew_fraction': float((abs(delta) >= .04-1e-6).mean()),
        'joint_velocity_spectrum_400hz': spectrum(n['joint_velocity_rad_s'], .0025),
        'target_step_spectrum_50hz': spectrum(delta, .02),
        'support_count_histogram': np.bincount(support.sum(-1), minlength=7).tolist(),
        'missing_six_toe_native_steps': int((~support.all(-1)).sum()),
        'zero_toe_native_steps': int((~support.any(-1)).sum()),
        'toe_contact_fraction': support.mean(0).tolist(),
        'nonfoot_native_steps': int(n['nonfoot_contact'].sum()),
        'requested_saturation_max_joint_fraction': float((abs(n['computed_torque_nm']) > 1.6).mean(0).max()),
        'applied_torque_peak_nm': float(abs(n['applied_torque_nm']).max()),
        'terminated_controls': int(c['terminated'].sum()), 'truncated_controls': int(c['truncated'].sum()),
        'reset_controls': int(c['reset'].sum()),
    }
    return result


def inspect(version, batch):
    root = A/f'results_evaluate_{version}/standing/evaluation/batch_{batch:03d}'
    report, capture, c, n, integrity, binding = load_recording(root)
    initial = read(root/'native400hz/initial_state.json')
    q, dq = n['joint_position_rad'], n['joint_velocity_rad_s']
    assert np.array_equal(n['pre_joint_position_rad'][1:], q[:-1])
    assert np.array_equal(n['pre_joint_velocity_rad_s'][1:], dq[:-1])
    assert np.array_equal(n['pre_joint_position_rad'][0], np.array(initial['q'], np.float32))
    assert np.array_equal(n['pre_joint_velocity_rad_s'][0], np.array(initial['dq'], np.float32))
    assert np.array_equal(n['joint_target_rad'][7::8], c['joint_target_rad'])
    assert np.array_equal(q[7::8], c['joint_position_rad']) and np.array_equal(dq[7::8], c['joint_velocity_rad_s'])
    wanted = _diagnostic_servo(n['pre_joint_position_rad'][:,0], n['pre_joint_velocity_rad_s'][:,0],
        n['joint_target_rad'][:,0], np.full(18, 12., np.float32), np.asarray(KD, np.float32))
    for key, expected in zip(('computed_torque_nm', 'applied_torque_nm', 'effort_ceiling_nm'), wanted):
        assert np.array_equal(n[key][:,0], expected), key
    assert np.array_equal(n['applied_torque_nm'], n['native_input_pre_nm'])
    assert np.allclose(n['interval_angle_rate_rad_s'][1:], np.diff(q.astype(float), axis=0)/.0025, atol=0, rtol=0)
    neutral = torch.tensor(initial['q'], dtype=torch.float32)
    held = torch.from_numpy(np.concatenate([np.asarray(initial['q'], np.float32)[None], c['joint_target_rad'][:-1]]))
    native_meta = read(A/f'results_evaluate_{version}/standing/native/native_readback.json')
    limits = torch.tensor(native_meta['limits'], dtype=torch.float32)
    emitted = emitted_target(torch.from_numpy(c['policy_action']), held, neutral, limits[:,:,0], limits[:,:,1])
    assert np.array_equal(emitted.numpy(), c['joint_target_rad'])
    previous_feature_error = float(abs(executed_action_feature(held, neutral, .35).numpy()-c['policy_observation'][:,:,-18:]).max())
    assert previous_feature_error <= 2e-7  # CPU/CUDA float32 division, not a changed physical gate.
    speed = np.asarray(native_meta['native_max_velocity'], np.float32)
    crossings = np.argwhere(abs(dq) > speed+2e-6)
    assert len(np.unique(crossings[:,0])) == capture['speed_bound_violation_steps'][0]
    events = []
    for s, e, j in crossings:
        events.append({'sequence': int(s), 'control_index': int(n['control_index'][s]),
            'substep_index': int(n['substep_index'][s]), 'time_s': float(n['time_s'][s]),
            'joint': JOINT_NAMES[j], 'joint_index': int(j), 'reported_speed_rad_s': float(dq[s,e,j]),
            'native_speed_limit_rad_s': float(speed[e,j]),
            'interval_angle_rate_rad_s': float(n['interval_angle_rate_rad_s'][s,e,j]),
            'position_rad': float(q[s,e,j]), 'requested_torque_nm': float(n['computed_torque_nm'][s,e,j]),
            'applied_torque_nm': float(n['applied_torque_nm'][s,e,j]),
            'effort_ceiling_nm': float(n['effort_ceiling_nm'][s,e,j]),
            'control_endpoint_speed_rad_s': float(c['joint_velocity_rad_s'][s//8,e,j]),
            'control_endpoint_terminated': bool(c['terminated'][s//8,e]),
            'root_height_m': float(n['root_pose_xyzw'][s,e,2]),
            'support': n['distal_contact'][s,e].tolist(),
            'neighboring_samples': [{k: np.asarray(n[k][t,e,j]).item() for k in
                ('joint_position_rad','joint_velocity_rad_s','interval_angle_rate_rad_s','computed_torque_nm','applied_torque_nm')}
                | {'sequence':t} for t in range(max(0,s-3), min(len(q),s+4))]})
    packets, patches, categories, event_contacts = 0, 0, {}, []
    with (root/'native400hz/contacts.jsonl').open() as stream:
        for line in stream:
            packet = json.loads(line)
            assert packet['sequence'] == packets and packet['explicit_counter'] == int(n['explicit_counter'][packets])
            for patch in packet['patches']:
                categories[patch['category']] = categories.get(patch['category'], 0)+1
            if any(abs(packets-ev['sequence']) <= 1 for ev in events):
                event_contacts.append({'sequence': packets, 'patches': packet['patches']})
            packets += 1
            patches += len(packet['patches'])
    assert packets == len(q)
    start = report['results'][0]['window_start_control']
    windows = {'whole_trial': window(c,n,0,len(c['time_s'])), 'fixed_scoring_window': window(c,n,start,len(c['time_s']))}
    if batch == 2:
        windows['commanded_moving_phase'] = window(c,n,0,400)
        windows['zero_command_settling_allowance'] = window(c,n,400,500)
    return {'version':version, 'case_id':report['assigned_case_ids'][0], 'recording_report_sha256':integrity['report_sha256'],
        'hashes_verified':integrity, 'windows':windows, 'speed_events':events,
        'contact_stream':{'all_packet_order_and_counters_verified':True,'packets':packets,'patches':patches,
            'saved_category_counts':categories,'events_with_neighbor_contacts':event_contacts,
            'scope':'Full saved stream checked for ordering/counts and hashed; saved point classifications are not independently regenerated here.'},
        'native_integrity':{'servo_recurrence_and_native_input_exact':True,'pre_post_state_continuity_exact':True,
            'all_control_native_endpoints_exact':True,'post_limiter_targets_exact':True,
            'observed_previous_target_max_abs_cpu_error':previous_feature_error,
            'previous_target_feature_cpu_absolute_tolerance':2e-7},
        'preserved_failed_bounds':report['results'][0]['failed_bounds'],
        'native_motor_and_joint_checks_pass':report['results'][0]['native_motor_and_joint_checks_pass'],
        'original_acquisition_complete':report['acquisition_complete'],'original_failure':report['failure']}


if __name__ == '__main__':
    source = A/'source_017'
    assert sha(REPO/'experiments/paper_walk/learner.py') == 'b0a3cf503a0c710239d0245ef24fb961d0d05f06feaf37fffee7b2c802caf77a'
    for name in ('analyze.py','env.py','env_config.py','learner.py'):
        assert sha(source/name) == sha(REPO/'experiments/paper_walk'/name)
    assert read(A/'VERIFIED_TRANSFER_evaluate_012.json')['all_file_bytes_verified'] is True
    analyzed = read(HERE/'analysis.json')
    assert analyzed['comparison']['status'] == 'matched_identity'
    assert all(r['actor_reconstruction']['within_cpu_float_tolerance'] for r in analyzed['recordings'])
    result = {'schema':'canonical_eval011_012_comparison_v1',
        'matched_model_physics_case_duration_measurement_point':True,
        'checkpoint012_sha256':'a48fddf1f73fcaf5b93db44241390184dd36c60e031daaad87d3a63bfd2bd81a',
        'analyzer_receipt_sha256':sha(HERE/'analysis.json'),
        'source_sha256':{name:sha(REPO/'experiments/paper_walk'/name) for name in ('analyze.py','env.py','env_config.py','learner.py')},
        'cases':[{'case_id':f'batch_{i:03d}', 'previous':inspect('011',i), 'current':inspect('012',i)} for i in range(3)],
        'stage2_complete':False,'verdict_changed':False,
        'interpretation_limits':['Matched descriptive comparison, not a replicated causal isolation of the velocity BC change.',
            'BC optimizer, PPO configuration and velocity BC fields changed together; no single-factor attribution.',
            'One SDK speed spike explains the saved native rejection; its underlying simulator/contact cause is not isolated.',
            'FFT metrics summarize all retained samples within each explicitly declared window, including speed spikes.',
            'No stage2 gates or source files changed; complete raw traces and original failed verdicts preserved.']}
    with (HERE/'NUMERICAL_COMPARISON.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'cases':len(result['cases']), 'all_complete_native_integrity':True, 'stage2_complete':False}))
