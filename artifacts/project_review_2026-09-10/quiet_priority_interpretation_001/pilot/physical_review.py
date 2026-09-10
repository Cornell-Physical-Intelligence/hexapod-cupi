"""Bounded same-gate quiet result comparison, preserving reset-contaminated maxima."""
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'physical_comparison.json'
assert not OUT.exists(), 'Preserve existing evidence'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

results = {}
bindings = {}
for label, folder, analysis in (
    ('quiet_priority50', 'direct_omni_quiet_pilot_terminal_001', 'direct_omni_quiet_pilot_analysis_001'),
    ('normal_caps50', 'direct_omni_train_pilot_caps_results_001', 'direct_omni_train_pilot_caps_analysis_001'),
):
    root = REPO / 'tmp' / folder / 'run'
    ar = REPO / 'tmp' / analysis
    amap = json.loads((ar / 'INPUTS_SHA256.json').read_text())
    for rel in ('final_stop/stop_diagnostics.json', 'final_stop/stop_trace.npz', 'initial_stop/stop_diagnostics.json', 'train/training_receipt.json'):
        p = root / rel
        expected = [v for k, v in amap.items() if k.endswith('/' + rel)]
        assert len(expected) == 1 and sha(p) == expected[0], str(p)
        bindings[p.relative_to(REPO).as_posix()] = expected[0]
    for name in ('REPORT.md', 'report.json', 'INPUTS_SHA256.json'):
        bindings[(ar / name).relative_to(REPO).as_posix()] = sha(ar / name)
    stop = json.loads((root / 'final_stop/stop_diagnostics.json').read_text())
    train = json.loads((root / 'train/training_receipt.json').read_text())
    aggregate = json.loads((ar / 'report.json').read_text())
    rows = [r for s in stop['scenarios'] for r in s['replicas']]
    assert len(rows) == 48 and sorted(r['env_id'] for r in rows) == list(range(48))
    with np.load(root / 'final_stop/stop_trace.npz', allow_pickle=False) as z:
        events = []
        for index, env in zip(*np.where(z['terminated'] | z['truncated'])):
            time_s = float(z['time_s'][index])
            jump = None
            if index + 1 < len(z['time_s']):
                jump = float(np.linalg.norm(z['position_world_m'][index + 1, env, :2].astype(float) - z['position_world_m'][index, env, :2].astype(float)))
            events.append({'zero_based_trace_index': int(index), 'time_s': time_s, 'env_id': int(env),
                           'terminated': bool(z['terminated'][index, env]), 'truncated': bool(z['truncated'][index, env]),
                           'inside_scored_quiet_window': 22. < time_s <= 32.,
                           'next_row_position_jump_m': jump})
    drifts = [r['quiet']['max_planar_excursion_m'] for r in rows]
    results[label] = {
        'checkpoint_sha256': stop['checkpoint_sha256'],
        'quiet_pass_count': sum(r['quiet']['pass'] for r in rows), 'replicas': 48,
        'quiet_gates': stop['quiet_gates'], 'overrides': stop['overrides'],
        'quiet_planar_excursion_median_m': statistics.median(drifts),
        'quiet_planar_excursion_max_m': max(drifts),
        'target_step_p95_min_max': [min(r['quiet']['max_target_step_abs_p95_rad_per_20ms'] for r in rows), max(r['quiet']['max_target_step_abs_p95_rad_per_20ms'] for r in rows)],
        'sdk_joint_rms_min_max': [min(r['quiet']['max_joint_velocity_rms_rad_s'] for r in rows), max(r['quiet']['max_joint_velocity_rms_rad_s'] for r in rows)],
        'trial_events': events,
        'training_requested_peak_nm': max(train['audit']['requested_torque_max_per_row_nm']),
        'constant_saturation_regression_names_vs_original': [r['name'] for r in aggregate['constant_comparison'] if r['after']['torque_saturation_fraction'] > r['before']['torque_saturation_fraction']],
        'mean_quiet_requested_saturation_fraction_median': statistics.median(r['quiet']['mean_requested_torque_saturation_fraction'] for r in rows),
    }
assert results['normal_caps50']['quiet_gates'] == results['quiet_priority50']['quiet_gates']
assert results['normal_caps50']['overrides'] == results['quiet_priority50']['overrides']
OUT.write_text(json.dumps({'scope': 'Raw final-stop result and event-timing comparison; no new physics/source/cleanup audit',
                          'results': results, 'inputs_sha256': bindings,
                          'reset_rule': 'Preserve original failed trial and scored excursion; never relabel reset jumps as uninterrupted motion or remove a failed row',
                          'Stage2_complete': False}, indent=2) + '\n')
print(json.dumps({k: {key: v[key] for key in ('quiet_pass_count', 'quiet_planar_excursion_median_m', 'quiet_planar_excursion_max_m', 'trial_events')} for k, v in results.items()}, indent=2))
