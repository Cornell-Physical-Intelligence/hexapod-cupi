"""Descriptive comparison of frozen analyzer reports; does not introduce a gate."""
from pathlib import Path
import argparse,json,statistics,hashlib

def stats(values):
    values=list(values)
    return {'min':min(values),'median':statistics.median(values),'max':max(values),'mean':statistics.mean(values)}

def summary(d):
    cases=d['final_stop']['cases']
    stable=[r for r in cases if not r['quiet']['terminations'] and not r['quiet']['truncations']]
    return {
        'quiet_passed_replicas':d['final_stop']['quiet_passed_replicas'],
        'replicas':len(cases),
        'quiet_metrics':{k:stats(r['quiet'][k] for r in cases) for k in (
            'max_planar_excursion_m','max_heading_excursion_deg','max_joint_velocity_rms_rad_s',
            'max_joint_position_range_rad','max_target_step_abs_p95_rad_per_20ms',
            'max_requested_torque_saturation_fraction','max_applied_torque_nm')},
        'quiet_all_joint_requested_saturation_mean':statistics.mean(
            j['saturation_fraction'] for r in cases for j in r['quiet_joints'].values()),
        'quiet_max_interval_angle_joint_rms_rad_s':max(
            max(r['rate_evidence']['interval_angle_rms_rad_s']) for r in cases),
        'all_trial_terminations':sum(r['quiet']['terminations'] for r in cases),
        'all_trial_truncations':sum(r['quiet']['truncations'] for r in cases),
        'trial_reset_env_ids':[r['env_id'] for r in cases if r not in stable],
        'no_trial_reset_subset':{'count':len(stable),'drift_m':stats(r['quiet']['max_planar_excursion_m'] for r in stable),
                                 'scope':'Descriptive subset only; original all48 gate remains unchanged.'},
        'all_stop_trial_requested_peak_nm':max(r['all_requested_peak_nm'] for r in cases),
        'all_stop_trial_applied_peak_nm':max(r['all_applied_peak_nm'] for r in cases),
        'constant_requested_peak_nm':max(r['metrics']['computed_torque_abs_max_nm'] for r in d['final_constant']['cases']),
        'constant_equal_case_mean_requested_saturation':statistics.mean(r['metrics']['torque_saturation_fraction'] for r in d['final_constant']['cases']),
        'constant_terminations':sum(r['terminations'] for r in d['final_constant']['cases']),
        'updates':d['training']['updates_completed'],'transitions':d['training']['transitions'],
        'strict_reload':d['training']['strict_reload']['passed'],
        'final_checkpoint_sha256':d['training']['checkpoint_sha256'],
    }

def differences(x,y,path=''):
    if isinstance(x,dict) and isinstance(y,dict):
        out=[]
        for k in sorted(x.keys()|y.keys()):
            if k not in x or k not in y:out.append({'path':path+'/'+k,'kind':'missing'})
            else:out.extend(differences(x[k],y[k],path+'/'+k))
        return out
    if isinstance(x,list) and isinstance(y,list):
        if len(x)!=len(y):return [{'path':path,'kind':'length'}]
        return [row for i,(a,b) in enumerate(zip(x,y)) for row in differences(a,b,path+'/'+str(i))]
    return [] if x==y else [{'path':path,'caps':x,'quiet_priority':y}]

def compare(c,q):
    assert not c['errors'] and not q['errors']
    assert c['final_stop']['gates']==q['final_stop']['gates']
    assert c['final_constant']['overrides']==q['final_constant']['overrides']
    assert c['final_constant']['options']==q['final_constant']['options']
    assert c['final_constant']['joint_names']==q['final_constant']['joint_names']
    assert c['initial_constant']==q['initial_constant']
    rows=[]
    for a,b in zip(c['final_constant']['cases'],q['final_constant']['cases']):
        assert a['name']==b['name'] and a['command']==b['command'] and a['aggregate_replicas']==b['aggregate_replicas']==4
        row={'name':a['name'],'command':a['command'],'aggregate_replicas':4,'metrics':{}}
        for k in ('planar_error_mps','yaw_error_rad_s','torque_saturation_fraction',
                  'computed_torque_abs_max_nm','reported_joint_velocity_rms_max_rad_s','finite_difference_planar_error_mps'):
            row['metrics'][k]={'caps':a['metrics'][k],'quiet_priority':b['metrics'][k],
                               'difference_quiet_minus_caps':b['metrics'][k]-a['metrics'][k]}
        rows.append(row)
    count={k:{'lower':sum(r['metrics'][k]['difference_quiet_minus_caps']<0 for r in rows),
              'higher':sum(r['metrics'][k]['difference_quiet_minus_caps']>0 for r in rows)}
           for k in ('planar_error_mps','yaw_error_rad_s','torque_saturation_fraction')}
    delta=differences(c['initial_stop'],q['initial_stop'])
    return {'schema':'quiet_priority_vs_caps50_actual_v1','scope':'Matched-budget descriptive comparison; original gates preserved; no significance or causal success claim.',
            'caps':summary(c),'quiet_priority':summary(q),'constant_cases':rows,'constant_change_counts':count,
            'matched_initial_constant_exact':True,'initial_stop_report_differences':delta,
            'initial_stop_only_heading_roundoff':all(r['path'].endswith('/quiet/max_heading_excursion_deg') for r in delta),
            'initial_stop_max_heading_difference_deg':max(abs(r['quiet_priority']-r['caps']) for r in delta),
            'quiet_gates':q['final_stop']['gates'], 'Stage2_complete':False,
            'conclusion':'The stronger quiet objective did not produce quiet admission:0/48 in both. Target chatter and large requested-torque saturation remain; tracking is mixed. Do not promote this as improved omni quality.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--caps',type=Path,default=Path(__file__).parent/'caps_report.json');p.add_argument('--quiet',type=Path,default=Path(__file__).parent/'quiet_report.json');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh comparison output required')
    d=compare(json.loads(a.caps.read_text()),json.loads(a.quiet.read_text()))
    d['input_sha256']={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('caps_report.json',a.caps),('quiet_report.json',a.quiet)]}
    a.output.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'quiet_passes':[d['caps']['quiet_passed_replicas'],d['quiet_priority']['quiet_passed_replicas']],
                      'change_counts':d['constant_change_counts'],'initial_stop_max_heading_difference_deg':d['initial_stop_max_heading_difference_deg']},indent=2))

if __name__=='__main__':main()
