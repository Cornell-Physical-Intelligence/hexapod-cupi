"""CPU-only verification of all native014 samples and unchanged gate results."""
from pathlib import Path
import json
import sys
import traceback
import math
import numpy as np

HERE=Path(__file__).resolve().parent
A=HERE.parent
REPO=A.parents[2]
sys.path.insert(0,str(REPO))
from experiments.paper_walk.analyze import load_recording,sha
from experiments.paper_walk.env import _diagnostic_servo,_diagnostic_rotation
from experiments.paper_walk.env_config import KD,JOINT_NAMES,LEGS
from experiments.paper_walk.evaluation import score_recording


def read(path):return json.loads(Path(path).read_text())


def classify(packet,poses,bodies,geometry):
    feet=np.zeros((6,3));other=np.zeros((4,3));body_force={};nonfoot=False
    used=set();inactive_count=0
    for patch in packet['patches']:
        assert patch['env']==0 and 0<=patch['buffer_index']<1024 and patch['buffer_index'] not in used
        used.add(patch['buffer_index']);body=patch['body'];assert body in bodies
        f=patch['normal_force_n'];normal=np.asarray(patch['normal_world'],np.float32)
        point=np.asarray(patch['point_world_m'],np.float32);sep=patch['separation_m']
        assert np.isfinite(np.r_[f,normal,point,sep]).all()
        inactive=f==0 and np.all(normal==0) and sep==0
        assert inactive==patch['inactive_zero_normal']
        assert abs(np.linalg.norm(normal)-1)<=1e-3 or inactive
        inactive_count+=int(inactive)
        if body.endswith('_tibia'):
            s=geometry[body];T=np.asarray(s['shape_to_link']);pose=poses[bodies.index(body)]
            local=(point-pose[:3])@_diagnostic_rotation(pose[3:])
            local=(local-T[:3,3])@T[:3,:3]
            assert np.allclose(local,patch['shape_point_m'],atol=2e-12,rtol=0)
            bounds=np.asarray(s['cap_bounds_m']);skin=s['contact_offset_m']
            cap=local[0]>=s['cap_lower_x_m']-1e-12 and local[0]<=bounds[1,0]+skin
            cap=cap and np.all(local[1:]>=bounds[0,1:]-skin) and np.all(local[1:]<=bounds[1,1:]+skin)
            category='toe' if cap else 'shaft'
        else:category='coxa' if body.endswith('_coxa') else 'femur' if body.endswith('_femur') else 'body'
        assert category==patch['category']
        vector=np.float32(f)*normal
        if category=='toe':feet[LEGS.index(body[:2])]+=vector
        else:
            other[('body','coxa','femur','shaft').index(category)]+=vector
            body_force[body]=body_force.get(body,np.zeros(3))+f*np.asarray(patch['normal_world'])
            nonfoot=nonfoot or abs(f)>1
    assert len(used)<1024
    nonfoot=nonfoot or any(np.linalg.norm(v)>1 for v in body_force.values())
    return feet,other,nonfoot,inactive_count


def support_summary(n,start,end):
    contact=n['distal_contact'][start:end,0];forces=n['distal_force_world_n'][start:end,0]
    missing=~contact
    runs=[]
    for toe in range(6):
        mask=missing[:,toe];boundaries=np.flatnonzero(np.diff(np.r_[False,mask,False]))
        durations=boundaries[1::2]-boundaries[::2]
        runs.append({'toe':LEGS[toe],'missing_steps':int(mask.sum()),'episodes':len(durations),
            'longest_missing_s':float(durations.max()*.0025) if len(durations) else 0.})
    return {'start_sequence':start,'end_sequence_exclusive':end,'samples':end-start,
        'missing_six_toe_samples':int(missing.any(-1).sum()),'zero_toe_samples':int((~contact.any(-1)).sum()),
        'support_count_histogram':np.bincount(contact.sum(-1),minlength=7).tolist(),
        'per_toe':runs,'minimum_toe_force_norm_n':np.linalg.norm(forces,axis=-1).min(0).tolist()}


def inspect(index,geometry):
    base=A/f'results_evaluate_014/standing/evaluation/batch_{index:03d}'
    saved,capture,c,n,integrity,binding=load_recording(base)
    declaration=read(base/'declaration.json');case=saved['cases'][0]
    metadata={**declaration,'profile':case['profile'],'env_index':0,'command':case['command'],
              'target_slew_rad':.04,'control_dt_s':.02,'seed':case.get('seed',saved['seed'])}
    recomputed=score_recording(c,metadata)
    original=saved['results'][0]
    quantile_compatibility=None
    if case['profile']!='omni_static':
        key='max_target_step_abs_p95_rad_per_20ms'
        local_value=recomputed['metrics'][key]
        start_control=original['window_start_control']
        samples=np.sort(abs(np.diff(c['joint_target_rad'][start_control:,0],axis=0)),axis=0)
        rank=np.float32(len(samples)-1)*np.float32(.95)
        lower=int(rank);fraction=rank-np.float32(lower)
        difference=samples[lower+1]-samples[lower]
        interpolated=np.where(fraction>=.5,samples[lower+1]-difference*(1-fraction),samples[lower]+difference*fraction)
        archived_value=float(interpolated.max())
        assert archived_value==original['metrics'][key]
        assert recomputed['checks'][key]['status']==original['checks'][key]['status']
        quantile_compatibility={'local_numpy':np.__version__,'local_default_value':local_value,
            'archived_float32_rank_value':archived_value,'absolute_difference':abs(local_value-archived_value),
            'rank_float32':float(rank),'gate_status_unchanged':True,
            'scope':'Artifact-only explicit float32 rank/interpolation reproduces the archived native NumPy percentile arithmetic; no source or gate changed.'}
        recomputed['metrics'][key]=archived_value
        recomputed['checks'][key]['value']=archived_value
    vertical_compatibility=None
    metric='vertical_velocity_abs_p95_mps'
    if recomputed['metrics'][metric]!=original['metrics'][metric]:
        samples=np.sort(abs(c['velocity_world_mps'][original['window_start_control']:,0,2]))
        rank=np.float32(len(samples)-1)*np.float32(.95)
        lower=int(rank);fraction=rank-np.float32(lower)
        difference=samples[lower+1]-samples[lower]
        value=float(samples[lower+1]-difference*(1-fraction) if fraction>=.5 else samples[lower]+difference*fraction)
        assert value==original['metrics'][metric]
        vertical_compatibility={'local_default_value':recomputed['metrics'][metric],
            'archived_float32_rank_value':value,'rank_float32':float(rank),
            'scope':'Artifact-only native float32 percentile interpolation; metric is descriptive in these profiles.'}
        recomputed['metrics'][metric]=value
    heading_compatibility=None
    metric='max_heading_excursion_deg'
    if metric in recomputed['metrics'] and recomputed['metrics'][metric]!=original['metrics'][metric]:
        local_value=recomputed['metrics'][metric];native_value=original['metrics'][metric]
        quat=c['root_pose_xyzw'][original['window_start_control']:,0,3:].astype(np.float64)
        x,y,z,w=quat.T
        heading=np.unwrap(np.array([math.atan2(float(a),float(b)) for a,b in zip(-1+2*(x*x+z*z),2*(w*z-x*y))]))
        double_value=float(np.degrees(abs(heading-heading[0]).max()))
        local_check=recomputed['checks'][metric];native_check=original['checks'][metric]
        assert local_check['bound']==native_check['bound']==2. and local_check['operator']==native_check['operator']=='<='
        assert local_value<=2. and native_value<=2. and double_value<=2.
        assert local_check['status']==native_check['status']=='pass'
        heading_compatibility={'local_numpy_float32_value':local_value,'recorded_native_value':native_value,
            'independent_float64_math_atan2_value':double_value,'absolute_native_local_difference_deg':abs(local_value-native_value),
            'unchanged_gate_deg':2.,'all_three_values_pass_original_gate':True,
            'scope':'Cross-runtime float32 heading metric is not bit-exact. Both original/local scorers and a float64 recomputation pass the unchanged gate. No original value is overwritten and no compatibility tolerance substitutes for a gate.'}
    for key,value in recomputed.items():
        if key in ('pass','failed_bounds'):continue
        if heading_compatibility and key in ('metrics','checks'):
            for name,actual in value.items():
                if name=='max_heading_excursion_deg':continue
                assert actual==original[key][name],(key,name)
        else:assert value==original[key],key
    assert recomputed['pass']==original['passed_numeric_screen']
    assert original['failed_bounds']==recomputed['failed_bounds']
    initial=read(base/'native400hz/initial_state.json')
    native=read(A/'results_evaluate_014/standing/native/native_readback.json')
    limits=np.asarray(native['limits'],np.float32);speed=np.asarray(native['native_max_velocity'],np.float32)
    q,dq=n['joint_position_rad'],n['joint_velocity_rad_s']
    assert np.array_equal(n['pre_joint_position_rad'][1:],q[:-1])
    assert np.array_equal(n['pre_joint_velocity_rad_s'][1:],dq[:-1])
    assert np.array_equal(n['pre_joint_position_rad'][0],np.asarray(initial['q'],np.float32))
    assert np.array_equal(q[7::8],c['joint_position_rad']) and np.array_equal(dq[7::8],c['joint_velocity_rad_s'])
    wanted=_diagnostic_servo(n['pre_joint_position_rad'][:,0],n['pre_joint_velocity_rad_s'][:,0],
        n['joint_target_rad'][:,0],np.full(18,12.,np.float32),np.asarray(KD,np.float32))
    for key,value in zip(('computed_torque_nm','applied_torque_nm','effort_ceiling_nm'),wanted):
        assert np.array_equal(n[key][:,0],value),key
    assert np.array_equal(n['applied_torque_nm'],n['native_input_pre_nm'])
    violations=((q<limits[:,:,0]-2e-6)|(q>limits[:,:,1]+2e-6)).any(-1).sum(0)
    speed_violations=(abs(dq)>speed+2e-6).any(-1).sum(0)
    assert violations.tolist()==capture['joint_bound_violation_steps']
    assert speed_violations.tolist()==capture['speed_bound_violation_steps']
    assert violations[0]==speed_violations[0]==0
    assert np.max(abs(n['applied_torque_nm']))<=1.60001
    assert not c['terminated'].any() and not c['truncated'].any() and not c['reset'].any()
    assert not n['nonfoot_contact'].any()
    assert n['minimum_non_toe_floor_m'].min()>=-.001 and n['root_pose_xyzw'][:,:,2].min()>=.055
    assert n['minimum_non_toe_floor_m'].min()==capture['minimum_non_toe_floor_m'][0]
    assert n['root_pose_xyzw'][:,:,2].min()==capture['minimum_plate_height_m'][0]
    assert saved['controls']==case['controls'] and len(q)==case['controls']*8
    assert saved['acquisition_complete'] and saved['failure'] is None and capture['failure'] is None
    assert original['native_capture_complete'] and original['native_motor_and_joint_checks_pass']
    start=original['window_start_control']*8
    missing=(~n['distal_contact'][start:,0].all(-1)).sum()
    if case['profile']!='omni_static':assert missing==original['checks']['missing_six_toe_count_400hz']['value']
    packets=patches=inactive=0;events=[]
    with (base/'native400hz/contacts.jsonl').open() as stream:
        for line in stream:
            packet=json.loads(line)
            assert packet['sequence']==packets and packet['explicit_counter']==int(n['explicit_counter'][packets])
            feet,other,nonfoot,zeros=classify(packet,n['link_pose_xyzw'][packets,0],capture['body_names'],geometry)
            assert np.array_equal(feet,n['distal_force_world_n'][packets,0])
            assert np.array_equal(other,n['nonfoot_force_world_n'][packets,0])
            assert np.array_equal(np.linalg.norm(feet,axis=-1)>1,n['distal_contact'][packets,0])
            assert nonfoot==bool(n['nonfoot_contact'][packets,0])
            if index==2 and packets>=start and not n['distal_contact'][packets,0].all():
                absent=np.flatnonzero(~n['distal_contact'][packets,0]);absent_legs=[LEGS[x] for x in absent]
                events.append({'sequence':packets,'time_s':float(n['time_s'][packets]),
                    'missing_toes':absent_legs,'toe_force_norm_n':np.linalg.norm(feet,axis=-1).tolist(),
                    'missing_toe_packets':[x for x in packet['patches'] if x['body'][:2] in absent_legs],
                    'root_height_m':float(n['root_pose_xyzw'][packets,0,2])})
            packets+=1;patches+=len(packet['patches']);inactive+=zeros
    assert packets==len(q)
    previous=read(A/f'results_evaluate_013/standing/evaluation/batch_{index:03d}/report.json')['results'][0]
    return {'case_id':case['case_id'],'controls':len(c['time_s']),'physics_steps':packets,
        'original_gate_verdicts_reproduced':True,'original_metrics_exact_except_declared_heading':True,
        'percentile_compatibility':quantile_compatibility,'vertical_percentile_compatibility':vertical_compatibility,'heading_compatibility':heading_compatibility,'original_pass':original['pass'],
        'failed_bounds':original['failed_bounds'],'original_checks':original['checks'],
        'native_joint_limit_violation_steps':int(violations[0]),'native_speed_violation_steps':int(speed_violations[0]),
        'maximum_sdk_speed_rad_s':float(abs(dq).max()),'maximum_interval_angle_rate_rad_s':float(abs(n['interval_angle_rate_rad_s']).max()),
        'servo_and_actual_native_input_exact':True,'pre_post_state_and_control_endpoint_continuity_exact':True,
        'native_contact_packets_reclassified':packets,'native_patches_reclassified':patches,'inactive_zero_tuples':inactive,
        'nonfoot_steps':int(n['nonfoot_contact'].sum()),'terminations':int(c['terminated'].sum()),
        'truncations':int(c['truncated'].sum()),'resets':int(c['reset'].sum()),
        'maximum_applied_torque_nm':float(abs(n['applied_torque_nm']).max()),
        'maximum_requested_torque_nm':float(abs(n['computed_torque_nm']).max()),
        'whole_trial_support':support_summary(n,0,len(q)),
        'fixed_score_window_support':support_summary(n,start,len(q)),
        'stop_quiet_missing_events':events,
        'fixed_window_mean_root_height_m':float(n['root_pose_xyzw'][start:,0,2].mean()),
        'fixed_window_mean_q':n['joint_position_rad'][start:,0].mean(0).tolist(),
        'fixed_window_mean_target':n['joint_target_rad'][start:,0].mean(0).tolist(),
        'fixed_window_action_clipped_fraction':float((abs(c['policy_action'][start//8:,0])>1).mean()),
        'metrics013':previous['metrics'],'metrics014':original['metrics'],
        'failed_bounds013':previous['failed_bounds'],'input_integrity':integrity}


def main():
    inputs=[p for p in (A/'results_evaluate_014').rglob('*') if p.is_file()]
    source=A/'source_018';freeze=read(source/'FREEZE_SHA256.json')
    inputs.extend([source/'FREEZE_SHA256.json',A/'VERIFIED_TRANSFER_evaluate_014.json',Path(__file__)])
    inputs.extend(A/f'results_evaluate_013/standing/evaluation/batch_{i:03d}/report.json' for i in range(3))
    for name in ('env.py','env_config.py','evaluation.py','evaluate.py','analyze.py'):
        path=REPO/'experiments/paper_walk'/name;assert sha(path)==freeze[name];inputs.append(path)
    geometry_path=REPO/'artifacts/restart_2026-09-14/standing_translation_preparation_001/source/geometry/geometry.json'
    geometry={r['body']:r for r in read(geometry_path)['shapes']};inputs.append(geometry_path)
    identity=read(A/'results_evaluate_014/standing/identity.json')
    assert sha(geometry_path)==identity['geometry_sha256']
    assert read(A/'VERIFIED_TRANSFER_evaluate_014.json')['all_file_bytes_verified']
    before={str(p):sha(p) for p in inputs}
    result={'schema':'independent_evaluate014_native_verification_v1','status':'running','stage2_complete':False}
    try:
        result.update(cases=[inspect(i,geometry) for i in range(3)],status='completed',all_verification_checks_passed=True)
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc(),all_verification_checks_passed=False)
    result['input_sha256']=before;result['inputs_unchanged']=before=={str(p):sha(p) for p in inputs}
    result['all_verification_checks_passed']=result['all_verification_checks_passed'] and result['inputs_unchanged']
    result['limitations']=['Scoring reuses exact original maintained scorer; independent code separately reconstructs every saved contact point and force.',
        'Recorded clearance channels are gated; all exact mesh vertices are not independently transformed again.',
        'Heading is not bit-exact on the local runtime; declared local/native/float64 values independently pass the unchanged 2-degree gate.',
        'Verification passing means the original numerical verdicts are reproducible, not full-suite or policy qualification.']
    with (HERE/'RESULT_002.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','input_sha256')},indent=2))
    if not result['all_verification_checks_passed']:raise SystemExit(1)


if __name__=='__main__':main()
