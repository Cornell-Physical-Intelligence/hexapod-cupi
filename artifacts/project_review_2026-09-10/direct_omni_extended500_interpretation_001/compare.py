"""Descriptive actual CAPS500 versus CAPS50 comparison; no scorer changes."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import numpy as np

FILES = {
    'caps50_report': 'artifacts/omni_diagnostics_2026-09-09/direct_omni_matched_pilots_001/caps/analysis/report.json',
    'caps500_report': 'artifacts/omni_diagnostics_2026-09-09/direct_omni_extended_caps_002/cpu_analysis/remote/analysis/report.json',
    'caps50_trace': 'artifacts/omni_diagnostics_2026-09-09/direct_omni_matched_pilots_001/caps/raw/run/final_stop/stop_trace.npz',
    'caps500_trace': 'artifacts/omni_diagnostics_2026-09-09/direct_omni_extended_caps_002/curated/raw/run/final_stop/stop_trace.npz',
    'caps500_analyzer_inputs': 'artifacts/omni_diagnostics_2026-09-09/direct_omni_extended_caps_002/cpu_analysis/remote/analysis/INPUTS_SHA256.json',
    'caps50_bundle': 'artifacts/omni_diagnostics_2026-09-09/direct_omni_matched_pilots_001/FREEZE_SHA256.json',
}

def sha(path):
    with path.open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def stats(values):
    values = [float(x) for x in values]
    return {'count': len(values), 'min': min(values), 'median': statistics.median(values),
            'mean': statistics.mean(values), 'max': max(values)}

def delta(a, b):
    return {'caps50': float(a), 'caps500': float(b), 'delta_500_minus_50': float(b-a)}

def raw_quiet(path, report):
    with np.load(path, allow_pickle=False) as z:
        t = z['time_s']; q = z['joint_position_rad']; v = z['joint_velocity_rad_s']
        p = z['position_world_m']; term = z['terminated']; trunc = z['truncated']
        done = term | trunc; target = z['joint_target_rad']
        # Frozen evaluator's declared final10s window; retains reset crossings.
        mask = (t > 22.0 + 1e-8) & (t <= 32.0 + 1e-8)
        assert mask.sum() == 500 and len(t) == 1600
        cases = {r['env_id']: r for r in report['final_stop']['cases']}
        assert set(cases) == set(range(48))
        events = []
        for i, env in np.argwhere(done):
            events.append({'env_id': int(env), 'name': cases[int(env)]['name'],
                'time_s': float(t[i]), 'in_scored_quiet': bool(mask[i]),
                'terminated': bool(term[i, env]), 'truncated': bool(trunc[i, env]),
                'next_sample_planar_jump_m': float(np.linalg.norm(p[i+1, env, :2] - p[i, env, :2])) if i+1 < len(t) else None,
                'reasons': [k for k in z.files if k.startswith('reason_') and bool(z[k][i, env])]})
        angle = np.diff(q.astype(np.float64), axis=0) / np.diff(t)[:, None, None]
        rows = []
        for env in range(48):
            # Rows are pre-reset: only the interval AFTER a terminal sample
            # crosses the reset. The arrival at that terminal sample is valid.
            valid = mask[1:] & mask[:-1] & ~done[:-1, env]
            ar = np.sqrt(np.mean(angle[valid, env]**2, axis=0))
            vr = np.sqrt(np.mean(v[mask, env].astype(np.float64)**2, axis=0))
            rr = cases[env]['rate_evidence']
            np.testing.assert_allclose(ar, rr['interval_angle_rms_rad_s'], atol=1e-9, rtol=1e-9)
            # Float64 independent reduction versus analyzer float32 reduction.
            # This numerical audit tolerance is not a changed physical gate.
            np.testing.assert_allclose(vr, rr['reported_joint_rms_rad_s'], atol=2e-6, rtol=0)
            assert int(valid.sum()) == rr['valid_interval_samples']
            rows.append({'env_id':env, 'name':cases[env]['name'],
                         'SDK_joint_RMS_max_rad_s':float(vr.max()),
                         'interval_angle_joint_RMS_max_rad_s':float(ar.max()),
                         'SDK_float64_reduction_max_abs_difference_rad_s':float(np.max(np.abs(vr-np.asarray(rr['reported_joint_rms_rad_s'])))),
                         'has_trial_reset':bool(done[:,env].any()),
                         'has_quiet_reset':bool(done[mask,env].any())})
        return {'scored_window_first_last_s':[float(t[mask][0]),float(t[mask][-1])],
                'samples':int(mask.sum()), 'elapsed_endpoint_span_s':float(t[mask][-1]-t[mask][0]),
                'original_scorer_duration_s':10.0, 'reset_events':events,
                'quiet_command_abs_max':float(np.abs(z['command'][mask]).max()),
                'quiet_requested_command_abs_max':float(np.abs(z['target_command'][mask]).max()),
                'quiet_limiter_active_fraction':stats(z['target_slew_limited_fraction'][mask].mean(axis=0)),
                'quiet_joint_saturation_fraction':float(np.mean(np.abs(z['computed_torque_nm'][mask])>1.6)),
                'quiet_requested_peak_nm':float(np.abs(z['computed_torque_nm'][mask]).max()),
                'all32s_requested_peak_nm':float(np.abs(z['computed_torque_nm']).max()),
                'rate_replay_rows':rows, 'independent_rate_replay_matches_analyzer':True}

def summary(report, raw):
    rows = report['final_stop']['cases']
    reset_ids = {r['env_id'] for r in raw['reset_events']}
    quiet_reset_ids = {r['env_id'] for r in raw['reset_events'] if r['in_scored_quiet']}
    keys = ('max_planar_excursion_m','max_heading_excursion_deg','max_joint_velocity_rms_rad_s',
            'max_joint_position_range_rad','max_target_step_abs_p95_rad_per_20ms',
            'max_requested_torque_saturation_fraction','max_applied_torque_nm')
    return {'updates':report['training']['updates_completed'], 'transitions':report['training']['transitions'],
            'checkpoint_sha256':report['training']['checkpoint_sha256'],
            'quiet_passed_replicas':report['final_stop']['quiet_passed_replicas'],
            'quiet_metrics':{k:stats(r['quiet'][k] for r in rows) for k in keys},
            'no_trial_reset_descriptive_subset':stats(r['quiet']['max_planar_excursion_m'] for r in rows if r['env_id'] not in reset_ids),
            'no_quiet_reset_descriptive_subset':stats(r['quiet']['max_planar_excursion_m'] for r in rows if r['env_id'] not in quiet_reset_ids),
            'quiet_all_joint_requested_saturation':stats(j['saturation_fraction'] for r in rows for j in r['quiet_joints'].values()),
            'quiet_per_env_mean_joint_saturation':stats(statistics.mean(j['saturation_fraction'] for j in r['quiet_joints'].values()) for r in rows),
            'quiet_worst_interval_angle_joint_RMS_rad_s':max(max(r['rate_evidence']['interval_angle_rms_rad_s']) for r in rows),
            'constant_peak_requested_nm':max(r['metrics']['computed_torque_abs_max_nm'] for r in report['final_constant']['cases']),
            'constant_equal_case_mean_requested_saturation':statistics.mean(r['metrics']['torque_saturation_fraction'] for r in report['final_constant']['cases']),
            'all_stop32s_peak_requested_nm':max(r['all_requested_peak_nm'] for r in rows),
            'raw_quiet':raw}

def compare(a, b, ta, tb):
    assert not a['errors'] and not b['errors'] and b['evidence_verified']
    assert a['schema'] == 'direct315_matched_review_v1'  # Old schema has no evidence_verified field.
    old_caps = dict(a['training']['selection']['caps'])
    new_caps = dict(b['training']['selection']['caps'])
    # Native003 weights all temporal pairs by temporal_weight; native005
    # serializes the equal quiet coefficient separately for normal CAPS.
    assert 'quiet_temporal_weight' not in old_caps
    assert {**old_caps, 'quiet_temporal_weight':old_caps['temporal_weight']} == new_caps
    assert a['final_stop']['gates'] == b['final_stop']['gates']
    for k in ('options','overrides','joint_names','reward_weights'):
        assert a['final_constant'][k] == b['final_constant'][k], k
    ra, rb = raw_quiet(ta,a), raw_quiet(tb,b)
    sa, sb = summary(a,ra), summary(b,rb)
    rows = []
    keys = ('planar_error_mps','yaw_error_rad_s','torque_saturation_fraction','computed_torque_abs_max_nm',
            'reported_joint_velocity_rms_max_rad_s','finite_difference_planar_error_mps')
    for old,new in zip(a['final_constant']['cases'],b['final_constant']['cases']):
        assert old['name'] == new['name'] and old['command'] == new['command'] and old['aggregate_replicas'] == new['aggregate_replicas'] == 4
        rows.append({'name':old['name'],'command':old['command'],'aggregate_replicas':4,
                     'metrics':{k:delta(old['metrics'][k],new['metrics'][k]) for k in keys},
                     'terminations':delta(old['terminations'],new['terminations']),
                     'rate_scope':'Aggregate SDK maximum is four-replica; adjacent-angle trace covers only one replica per constant case.'})
    counts = {k:{'lower':sum(r['metrics'][k]['delta_500_minus_50']<0 for r in rows),
                 'higher':sum(r['metrics'][k]['delta_500_minus_50']>0 for r in rows)} for k in keys}
    excluded = {r['env_id'] for r in ra['reset_events']+rb['reset_events']}
    common = {label:stats(r['quiet']['max_planar_excursion_m'] for r in report['final_stop']['cases'] if r['env_id'] not in excluded)
              for label,report in [('caps50',a),('caps500',b)]}
    return {'schema':'caps500_vs_caps50_descriptive_v1',
            'scope':'Different optimization budgets and source/host versions, same declared physical/objective/evaluation contract; not a matched-budget intervention, significance test or causal attribution.',
            'caps50':sa,'caps500':sb,'directional_deltas':rows,'change_counts':counts,
            'common_no_trial_reset_subset':{'excluded_env_ids':sorted(excluded),'summaries':common,'gate_override':False},
            'quiet_gates':b['final_stop']['gates'], 'initial_constant_report_exact':a['initial_constant']==b['initial_constant'],
            'raw_caps_selections':{'caps50':old_caps,'caps500':new_caps},
            'same_declared_effective_coefficients':True,
            'quiet_failures_retained':True, 'Stage2_complete':False, 'automatic_continuation':False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    paths={k:args.repo_root/v for k,v in FILES.items()};before={k:sha(p) for k,p in paths.items()}
    inputs=json.loads(paths['caps500_analyzer_inputs'].read_text())
    assert inputs['/home/orionh/HEXAPOD_runs/mock_length_study_20260909/direct_omni_train_extended_caps_002/final_stop/stop_trace.npz']==before['caps500_trace']
    parent=json.loads(paths['caps50_bundle'].read_text())
    assert parent['caps/analysis/report.json']==before['caps50_report']
    assert parent['caps/raw/run/final_stop/stop_trace.npz']==before['caps50_trace']
    result=compare(json.loads(paths['caps50_report'].read_text()),json.loads(paths['caps500_report'].read_text()),paths['caps50_trace'],paths['caps500_trace'])
    assert before=={k:sha(p) for k,p in paths.items()}
    result['input_files']={k:{'repo_relative_path':FILES[k],'sha256':v} for k,v in before.items()}
    if args.output.exists():raise ValueError('Fresh comparison output required')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'quiet_passes':[result['caps50']['quiet_passed_replicas'],result['caps500']['quiet_passed_replicas']],
                      'change_counts':result['change_counts'],'common_no_reset':result['common_no_trial_reset_subset']},indent=2))

if __name__=='__main__':main()
