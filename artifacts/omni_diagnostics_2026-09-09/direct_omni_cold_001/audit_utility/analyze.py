"""Summarize all 48 diagnostic replicas, separately inspect the 12 saved trace replicas."""
import argparse, hashlib, json, math
from pathlib import Path

CHECKPOINT = '1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'
HISTORICAL_SHA = 'f0c5c9feba99c9785bc47b69724c501f3d87b4782b3466dd98616b0382fa9ee6'
SOURCE_MAP = '4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b'
SCENARIOS = [('stand', [0.,0.,0.]), ('forward',[.1,0.,0.]), ('reverse',[-.1,0.,0.]),
             ('left',[0.,.1,0.]), ('right',[0.,-.1,0.]), ('forward_fast',[.2,0.,0.]),
             ('turn_left',[0.,0.,.2]), ('turn_right',[0.,0.,-.2]), ('arc_left',[.1,0.,.2]),
             ('arc_right',[.1,0.,-.2]), ('strafe_arc',[0.,.1,.2]), ('diagonal',[.071,.071,0.])]
FIELDS = ['samples','mean_velocity_mps','velocity_std_mps','mean_gyro_rad_s','gyro_std_rad_s',
          'planar_error_mps','yaw_error_rad_s','finite_difference_planar_error_mps',
          'reported_minus_finite_difference_velocity_rms_mps','torque_saturation_fraction',
          'computed_torque_abs_max_nm','applied_torque_abs_max_nm','positive_mechanical_power_w','tilt_rms_deg']

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())

def finite_tree(value):
    if isinstance(value,float) and not math.isfinite(value): raise ValueError('Nonfinite JSON metric')
    if isinstance(value,dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value,list):
        for v in value: finite_tree(v)

def diagnostic_summary(path, expected_slew):
    d=read(path); finite_tree(d)
    if d.get('checkpoint_sha256')!=CHECKPOINT or not d.get('complete') or d.get('kind')!='diagnostic_not_qualification':
        raise ValueError('Incomplete or wrong diagnostic')
    if d.get('overrides',{}).get('target_slew_rad_per_20ms')!=expected_slew: raise ValueError('Profile label mismatch')
    if [(x['name'],x['command']) for x in d['scenarios']]!=SCENARIOS: raise ValueError('Scenario identity/order mismatch')
    if d['options']!={'duration_s':12,'settle_s':2,'seed':7057,'trace_envs_per_scenario':1,'controller':'policy'}:
        raise ValueError('Different diagnostic allocation')
    rows=[]
    for s in d['scenarios']:
        if s['replicas']!=4 or s['windows']['all']['samples']!=2400: raise ValueError('Incomplete scenario replicas/samples')
        rows.append({**{k:s[k] for k in ('name','command','replicas','terminations','truncations','termination_reasons')},
                     'windows':{name:{k:w[k] for k in FIELDS if k in w} for name,w in s['windows'].items()}})
    return {'label': 'cold_formal004' if expected_slew==.04 else 'historical_diagnostic003',
            'diagnostics_sha256':sha(path),'checkpoint_sha256':CHECKPOINT,
            'target_slew_rad_per_20ms':expected_slew,'options':d['options'],'overrides':d['overrides'],
            'reward_weights':d['reward_weights'],'joint_names':d['joint_names'],
            'observation_audit':d['observation_audit'],'scenarios':rows,
            'terminations_all48':sum(x['terminations'] for x in rows),
            'truncations_all48':sum(x['truncations'] for x in rows)}

def trace_summary(path, diagnostic):
    import numpy as np
    with np.load(path,allow_pickle=False) as f:
        data={k:f[k] for k in f.files}
    if data['time_s'].shape!=(600,) or not np.allclose(data['time_s'],np.arange(1,601)*.02,rtol=0,atol=1e-10):
        raise ValueError('Incomplete or shifted control-time trace')
    if not np.array_equal(data['trace_env_ids'],np.arange(0,48,4)): raise ValueError('Wrong trace subset')
    if data['joint_names'].tolist()!=diagnostic['joint_names']: raise ValueError('Wrong named joint order')
    for k,v in data.items():
        if k in ('time_s','trace_env_ids','joint_names'): continue
        if v.shape[:2]!=(600,12) or not np.isfinite(v).all(): raise ValueError('Invalid trace field: '+k)
    for k in ('joint_position_rad','joint_target_rad','joint_velocity_rad_s','computed_torque_nm','applied_torque_nm'):
        if data[k].shape!=(600,12,18): raise ValueError('Wrong joint trace width')
    targets=np.array([s[1] for s in SCENARIOS])
    if not np.allclose(data['target_command'],targets[None,:,:],atol=1e-8,rtol=0): raise ValueError('Wrong trace scenario commands')
    age=data['age_s'].astype(float); term=data['terminated'].astype(bool); trunc=data['truncated'].astype(bool)
    # Interval evidence is unavailable across a reset. Endpoint-reported dq is kept separately.
    consecutive=np.isclose(age[1:]-age[:-1],.02,atol=2e-5,rtol=0)&~(term[:-1]|trunc[:-1])
    q=data['joint_position_rad'].astype(float); qt=data['joint_target_rad'].astype(float)
    dq_interval=np.diff(q,axis=0)/.02; dqt=np.diff(qt,axis=0)/.02
    rows=[]
    for i,(name,_) in enumerate(SCENARIOS):
        windows={}
        for label,mask in [('all_valid_intervals',consecutive[:,i]),
                           ('post_settle_valid_intervals',consecutive[:,i]&(age[1:,i]>=2))]:
            n=int(mask.sum()); values={'intervals':n}
            if n:
                qv=dq_interval[:,i][mask]; targetv=dqt[:,i][mask]; reported=data['joint_velocity_rad_s'][1:,i][mask]
                values.update(joint_angle_interval_velocity_rms_rad_s=float(np.sqrt(np.mean(qv*qv))),
                              raw_sdk_joint_velocity_rms_rad_s=float(np.sqrt(np.mean(reported*reported))),
                              endpoint_reported_minus_interval_velocity_rms_rad_s=float(np.sqrt(np.mean((reported-qv)**2))),
                              target_interval_velocity_rms_rad_s=float(np.sqrt(np.mean(targetv*targetv))),
                              target_max_step_rad=float(np.max(np.abs(targetv))*.02))
            windows[label]=values
        rows.append({'name':name,'actual_trace_env_id':int(data['trace_env_ids'][i]),
                     'terminations_saved_replica':int(term[:,i].sum()),'truncations_saved_replica':int(trunc[:,i].sum()),
                     'interval_windows':windows})
    return {'trace_sha256':sha(path),'control_steps':600,'saved_replicas':12,'population_replicas':48,
            'field_shapes':{k:list(v.shape) for k,v in data.items()},'scenarios':rows,
            'limits':['Only one of four replicas per scenario was saved; this cannot reproduce the all48 JSON aggregates.',
                      'Endpoint SDK joint rate and interval angle rate are distinct measurements; their difference is not a hardware or native physics diagnosis.',
                      'Legacy quaternion_world_wxyz contains raw SDK XYZW on this installed build; no quaternion transform is recomputed or silently relabeled here.',
                      '50 Hz pre-reset trace cannot prove substep torque or oscillation above 25 Hz.']}

def analyze(run,historical):
    if sha(historical)!=HISTORICAL_SHA: raise ValueError('Wrong frozen historical comparator')
    campaign=read(run/'campaign.json')
    if campaign.get('identity',{}).get('source_manifest_sha256')!=SOURCE_MAP or campaign.get('identity',{}).get('checkpoint_sha256')!=CHECKPOINT:
        raise ValueError('Wrong cold campaign identity')
    result={'campaign':campaign,'stage2_complete':False,'training_admission':False,
        'stop_transition_measured':False,
        'source_asset_cleanup_restoration_audit':'Root-owned separate proof; this analyzer does not substitute for it.',
        'files':{str(p.relative_to(run)):sha(p) for p in (run/'campaign.json',run/'standing/state.json',run/'baseline/state.json') if p.exists()}}
    result['historical_diagnostic003']=diagnostic_summary(historical,.03)
    if (run/'baseline/diagnostics.json').exists():
        new=diagnostic_summary(run/'baseline/diagnostics.json',.04); result['cold_formal004']=new
        result['trace_subset']=trace_summary(run/'baseline/diagnostic_trace.npz',new)
        old=result['historical_diagnostic003']; new_over=dict(new['overrides']); old_over=dict(old['overrides'])
        new_over.pop('target_slew_rad_per_20ms'); old_over.pop('target_slew_rad_per_20ms')
        result['comparison_checks']={'same_checkpoint':True,'same_options':new['options']==old['options'],
            'same_other_declared_overrides':new_over==old_over,'same_effective_reward_weights':new['reward_weights']==old['reward_weights'],
            'same_joint_names':new['joint_names']==old['joint_names'],
            'causal_or_training_improvement_claim':False,
            'scope':'Fresh .04 cold diagnostic versus preserved .03 historical diagnostic; both profiles and every scenario retained separately.'}
    else: result['cold_formal004']={'available':False,'reason':'Diagnostic did not finish or was not allocated; preserve campaign and phase failure evidence.'}
    return result

def markdown(report):
    lines=['# Cold direct-PPO diagnostic comparison','',
           'A completed diagnostic is not Stage 2 qualification or training admission. The original checkpoint is unchanged. The historical 0.03 and fresh 0.04 rad/20 ms profiles are labeled separately.','']
    for key in ('historical_diagnostic003','cold_formal004'):
        d=report[key];lines.extend(['## '+key,''])
        if 'scenarios' not in d: lines.extend([d['reason'],'']);continue
        lines.extend(['Each row includes all four replicas. Requested and applied torque are different quantities. Post-settle statistics exclude each episode’s first two seconds; all-window and reset statistics remain in report.json and the original diagnostic.','',
            '| Scenario | Terminations | Mean forward / left (m/s) | Mean yaw (rad/s) | Planar error (m/s) | Requested max (N m) | Saturated samples |',
            '|---|---:|---:|---:|---:|---:|---:|'])
        for s in d['scenarios']:
            w=s['windows']['post_settle'];v=w.get('mean_velocity_mps',[float('nan')]*3);g=w.get('mean_gyro_rad_s',[float('nan')]*3)
            lines.append(f"| {s['name']} | {s['terminations']} | {v[0]:.4f} / {v[1]:.4f} | {g[2]:.4f} | {w.get('planar_error_mps',float('nan')):.4f} | {w.get('computed_torque_abs_max_nm',float('nan')):.3f} | {100*w.get('torque_saturation_fraction',float('nan')):.2f}% |")
        lines.append('')
    lines.extend(['There is no moving-to-stop transition in these constant-command cases; stopping behavior is unmeasured.',
                  'The saved 50 Hz trace contains one replica per scenario. Raw reported joint rates and angle differences over one control interval remain separate. No source/input, cleanup, restoration, hardware, substep-torque, or all-direction qualification claim follows from this summary.',''])
    return '\n'.join(lines)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--historical',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);r=analyze(a.run,a.historical)
    (a.output/'report.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');(a.output/'README.md').write_text(markdown(r))
