"""Compare the recorded forward trials and retain the failed PPO probe verdicts."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
A = ROOT/'artifacts/ppo_reference_comparison_20260917'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


trials = {
    'original_reference': (ROOT/'artifacts/trajectory_optimizer_20260917/replay_001/standing/evaluation',
        ROOT/'artifacts/locomotion_force_metrics_20260916/forward_replay_001.json'),
    'improved_reference': (A/'replay_pack_001/replay_001/standing/evaluation',
        A/'replay_pack_001/replay_001/standing/evaluation/force_metrics.json'),
    'vanilla_ppo': (A/'vanilla_evaluate_001/run/standing/evaluation_00',
        A/'vanilla_evaluate_001/run/standing/evaluation_00/force_metrics.json'),
}
rows, inputs = {}, {}
for name, (base, load_path) in trials.items():
    report = read(base/'report.json')
    result = report['results'][0]
    load = read(load_path)['cases'][0]['windows']['commanded_locomotion_after_settle']
    assert report['seed'] == 20260914 and report['controls'] == 1000
    assert report['cases'][0]['command'] == [.05, 0., 0.]
    assert report['model_sha256'] == '7126935b35398d2a7f03271335a8a1b8f2059354af2b19d9474f22b8e73e4881'
    assert report['resets_during_trial'] == 0 and not report['pose_forcing']
    assert report['acquisition_complete'] and report['failure'] is None and load['samples'] == 7200
    for path in (base/'report.json', load_path, base/'rollout.mp4'):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    rows[name] = {'controller': 'learned standard PPO' if name == 'vanilla_ppo' else 'optimized open-loop reference',
        'ppo_updates': 1200 if name == 'vanilla_ppo' else 0,
        'checkpoint_sha256': report['checkpoint_sha256'],
        'mean_forward_speed_mps': result['metrics']['mean_velocity_mps'][0],
        'planar_error_mps': result['checks']['planar_error_mps']['value'],
        'forward_screen_pass': result['pass'], 'failed_bounds': result['failed_bounds'],
        'native_full_trial_physical_checks_pass': result['native_motor_and_joint_checks_pass'],
        'mean_vertical_support_n': load['total_vertical_support_n']['mean'],
        'support_p95_n': load['total_vertical_support_n']['p95'],
        'support_peak_n': load['total_vertical_support_n']['peak'],
        'mean_abs_applied_motor_torque_nm': load['motor_torque']['applied']['mean_abs_across_joints_nm'],
        'worst_joint_rms_torque_nm': load['motor_torque']['applied']['worst_joint_rms_nm'],
        'worst_joint': load['motor_torque']['applied']['worst_joint'],
        'per_foot_normal_load_n': {foot: value['normal_resultant_magnitude_n'] for foot, value in load['per_foot'].items()},
        'force_samples': load['samples'], 'force_window_seconds': load['duration_s']}
audit = read(A/'review_vanilla_001/RESULT.json')
training = read(A/'TRAINING_RESULT_001.json')
for path in (A/'review_vanilla_001/RESULT.json', A/'TRAINING_RESULT_001.json', A/'review_smooth_001/RESULT.json'):
    inputs[str(path.relative_to(ROOT))] = sha(path)
comparison = {'schema': 'canonical_ppo_reference_comparison_v1', 'utc': datetime.now(timezone.utc).isoformat(),
    'conditions': {'model_sha256': report['model_sha256'], 'command_mps': [.05, 0., 0.],
        'seed': 20260914, 'controls': 1000, 'control_dt_s': .02, 'limiter_rad_per_control': .04,
        'force_hz': 400, 'settling_seconds': 2., 'force_samples_per_forward_trial': 7200},
    'forward': rows,
    'reference_error_reduction_fraction': 1-rows['improved_reference']['planar_error_mps']/rows['original_reference']['planar_error_mps'],
    'baseline_training': {'updates': training['updates'], 'transitions': training['transitions'],
        'replicas': training['replicas'], 'seed': training['seed'], 'physical_terminations': training['physical_terminations']},
    'baseline_evaluation': [{'case_id': row['case_id'], 'controls': row['controls'],
        'pass': row['original_result']['pass'], 'failed_bounds': row['original_result']['failed_bounds'],
        'full_trial_physical_checks': row['full_trial_physical_checks'], 'video_sha256': row['video_sha256']} for row in audit['cases']],
    'measured_benefit': 'The improved reference reduces forward tracking error by 90.9 percent relative to the original reference. It sustains the requested forward motion under the unchanged simulator and motor limits. The 1200-update standard PPO checkpoint fails the forward, quiet and stop probes.',
    'load_interpretation': 'The improved reference has lower mean applied torque than the original reference and this PPO checkpoint. Its highest joint RMS and right-rear foot peak exceed the original reference. The PPO trial fails to achieve the requested speed, so these loads do not establish a comparison of energy efficiency at matched achieved motion.',
    'not_established': ['A benefit from using the improved reference during PPO training; that training has not run.',
        'A converged or best-tuned PPO baseline, statistical superiority across seeds, omnidirectional generalization, terrain robustness or hardware acceptance.'],
    'next_experiment_proposal': 'Compare reference-assisted PPO with this baseline under a declared common budget, then extend reference and evaluation coverage across directions and command changes.',
    'inputs': inputs, 'analysis_source_sha256': sha(Path(__file__)), 'stage2_complete': False}
with (A/'COMPARISON_001.json').open('x') as stream:
    json.dump(comparison, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({name: {k: row[k] for k in ('mean_forward_speed_mps', 'planar_error_mps', 'forward_screen_pass', 'mean_abs_applied_motor_torque_nm')} for name, row in rows.items()}, indent=2))
