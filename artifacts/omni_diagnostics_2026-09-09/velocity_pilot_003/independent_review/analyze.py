#!/usr/bin/env python3
"""Read-only compact review of the exact candidate003 initial/final evidence."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def trace_rows(path, cases):
    data = np.load(path, allow_pickle=False)
    result = []
    names = data['joint_names'].tolist()
    for slot, env_id in enumerate(data['trace_env_ids']):
        age = data['age_s'][:, slot]
        done = data['terminated'][:, slot] | data['truncated'][:, slot]
        keep = (age >= 2.) & ~done
        continuous = keep[1:] & keep[:-1] & (np.abs(np.diff(age)-.02) < 1e-4)
        q = data['joint_position_rad'][:, slot]
        target = data['joint_target_rad'][:, slot]
        dq = data['joint_velocity_rad_s'][:, slot]
        delta = np.diff(target, axis=0)[continuous]
        fd = np.diff(q, axis=0)[continuous] / .02
        requested = data['computed_torque_nm'][:, slot][keep]
        error = (target-q)[keep]
        modeled = 30.*error-.6*dq[keep]
        action = data['raw_policy_action'][:, slot][keep]
        joint_rows = {}
        for j, name in enumerate(names):
            joint_rows[name] = {
                'endpoint_velocity_rms_rad_s': float(np.sqrt(np.mean(dq[keep,j]**2))),
                'finite_difference_velocity_rms_rad_s': float(np.sqrt(np.mean(fd[:,j]**2))),
                'target_step_abs_p95_rad': float(np.quantile(np.abs(delta[:,j]),.95)),
                'target_step_abs_max_rad': float(np.abs(delta[:,j]).max()),
                'target_range_rad': float(np.ptp(target[keep,j])),
                'position_range_rad': float(np.ptp(q[keep,j])),
                'policy_action_mean': float(action[:,j].mean()),
                'policy_action_std': float(action[:,j].std()),
                'requested_torque_rms_nm': float(np.sqrt(np.mean(requested[:,j]**2))),
            }
        result.append({
            'environment_index': int(env_id), 'scenario': cases[int(env_id)//4]['name'],
            'replica': int(env_id)%4, 'continuous_samples': int(continuous.sum()),
            'requested_pd_residual_rms_nm': float(np.sqrt(np.mean((requested-modeled)**2))),
            'executable_target_velocity_abs_max_rad_s': float(np.abs(data['executable_target_velocity_rad_s'][:,slot][keep]).max()),
            'executable_target_acceleration_abs_max_rad_s2': float(np.abs(data['executable_target_acceleration_rad_s2'][:,slot][keep]).max()),
            'joints': joint_rows,
        })
    return result


def main():
    phases = {}
    for phase in ('initial', 'final'):
        root = HERE/phase
        report = read(root/'diagnostics.json')
        quiet = read(root/'quiet_review/quiet_stand.json')
        assert report['complete'] and quiet['complete']
        assert report['stage2_complete'] is False
        directions = []
        for case in report['scenarios']:
            state = case['windows']['post_settle_nonterminal']
            command = np.asarray(case['command'])
            v = np.asarray(state['mean_velocity_mps'])
            speed = np.linalg.norm(command[:2])
            directions.append({
                'name': case['name'], 'command': case['command'],
                'mean_velocity_mps': state['mean_velocity_mps'],
                'mean_gyro_rad_s': state['mean_gyro_rad_s'],
                'velocity_in_requested_direction_mps': float(v[:2]@command[:2]/speed) if speed else None,
                **{key:state[key] for key in ('planar_error_mps','yaw_error_rad_s','torque_saturation_fraction','computed_torque_abs_max_nm','positive_mechanical_power_w','mean_raw_reward_terms')},
                'max_joint_velocity_rms_rad_s': max(j['velocity_rms_rad_s'] for j in state['joints'].values()),
                'terminations': case['terminations'], 'truncations':case['truncations'],
            })
        quiet_rows = {}
        for trial in ('quiet_stand','stop_to_stand'):
            rows = [r for r in quiet['static'] if r['name']==trial]
            keys = ('max_joint_velocity_rms_rad_s','max_joint_position_range_rad',
                    'max_target_step_abs_p95_rad_per_20ms','max_requested_torque_saturation_fraction',
                    'max_planar_excursion_m','max_heading_excursion_deg')
            quiet_rows[trial] = {'passed': sum(r['pass'] for r in rows), 'replicas':len(rows),
                'worst_metrics': {key:max(r[key] for r in rows) for key in keys},
                'failed_bounds_counts': {key:sum(key in r['failed_bounds'] for r in rows) for key in keys},
                'terminations':sum(r['terminations'] for r in rows),
                'truncations':sum(r['truncations'] for r in rows)}
        phases[phase] = {'checkpoint_sha256':report['checkpoint_sha256'],
            'observation_audit':report['observation_audit'], 'directions':directions,
            'quiet':quiet_rows, 'trace_rows':trace_rows(root/'diagnostic_trace.npz',report['scenarios'])}
    shas = {str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
            and p.name not in ('analysis.json','analysis_output.txt','README.md','FREEZE_SHA256.json')}
    output = {'stage2_complete':False,'kind':'independent_matched_50_update_pilot_review',
              'source_sha256':'fb5e472bb710daa50d1d7f16a6f3812d2ed845e2400fcc6769859b3fb0b9519e',
              'input_sha256':shas, 'phases':phases}
    (HERE/'analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    for name,phase in phases.items():
        print(name, 'checkpoint', phase['checkpoint_sha256'])
        for row in phase['directions']:
            print(row['name'], 'v',np.round(row['mean_velocity_mps'][:2],5).tolist(),
                  'yaw',round(row['mean_gyro_rad_s'][2],5), 'sat',round(100*row['torque_saturation_fraction'],3),
                  'dqmaxRMS',round(row['max_joint_velocity_rms_rad_s'],5))
        print(json.dumps(phase['quiet'],indent=2))


if __name__ == '__main__':
    main()
