"""Bounded CPU audit of complete source019 cold/settled startup evidence."""
from pathlib import Path
import ast, datetime, hashlib, json, sys, traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
A=HERE.parents[1]
REPO=A.parents[2]
sys.path.insert(0,str(REPO))
from experiments.paper_walk.analyze import verified_files
from experiments.paper_walk.env import _diagnostic_servo,_diagnostic_rotation,emitted_target
from experiments.paper_walk.env_config import KD,JOINT_NAMES,LEGS
from experiments.paper_walk.evaluation import score_recording

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def load_recording(base):
    report=read(base/'report.json')
    verified=verified_files(base,report['files'])
    def arrays(path,labels=False):
        with np.load(path,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
        for k,v in data.items():
            if k=='action_source' and labels:
                assert v.ndim==1 and v.dtype.kind=='U'
                assert set(v.tolist())<={'bc','scripted_neutral'}
            else:assert v.dtype.kind in 'biuf' and np.isfinite(v).all(),k
        return data
    c=arrays(base/'control_trace.npz',labels=True)
    capture=read(base/'native400hz/capture.json')
    native_verified=verified_files(base/'native400hz',capture['files'])
    chunks=[arrays(base/'native400hz'/name) for name in capture['substep_files']]
    assert all(set(chunk)==set(chunks[0]) for chunk in chunks)
    native={k:np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
    assert all(len(v)==report['controls'] for v in c.values())
    assert all(len(v)==capture['steps'] for v in native.values())
    assert np.allclose(np.diff(c['time_s']),.02,rtol=0,atol=1e-9)
    verified.update({'native400hz/'+k:v for k,v in native_verified.items()})
    integrity={'report_sha256':sha(base/'report.json'),'verified_files':verified,
        'scope':'Artifact-only loader accepts only the explicit bc/scripted_neutral string label channel; all numeric channels finite and lengths/hash checks retained.'}
    return report,capture,c,native,integrity,None


def load_audit_helpers():
    # Reuse only these two pure artifact audit functions, not its old trial entry.
    prior=A/'verification_evaluate_015/verify.py'
    tree=ast.parse(prior.read_text())
    names=('classify','support_summary')
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert len(nodes)==2
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(prior),'exec'),globals())
    return prior

def window(raw,start,end,nonfoot_bound):
    q=raw['joint_position_rad'][start:end]
    if not len(q):return {'applicable':False,'expected_substeps':0,'recorded_substeps':0,'pass':True}
    take=lambda key:raw[key][start:end]
    limits=np.asarray(NATIVE['limits']);speed=np.asarray(NATIVE['native_max_velocity'])
    r=dict(applicable=True,expected_substeps=end-start,recorded_substeps=len(q),global_substep_slice=[start,end],
        joint_bound_violation_steps=int(((q<limits[:,:,0]-2e-6)|(q>limits[:,:,1]+2e-6)).any(-1).sum()),
        speed_bound_violation_steps=int((abs(take('joint_velocity_rad_s'))>speed+2e-6).any(-1).sum()),
        maximum_applied_nm=float(abs(take('applied_torque_nm')).max()),
        maximum_requested_nm=float(abs(take('computed_torque_nm')).max()),
        requested_saturation_fraction_400hz=float((abs(take('computed_torque_nm'))>1.6).mean()),
        nonfoot_steps=int(take('nonfoot_contact').sum()),maximum_nonfoot_fraction=nonfoot_bound,
        missing_six_toe_steps=int((~take('distal_contact').all(-1)).sum()),
        minimum_non_toe_floor_m=float(take('minimum_non_toe_floor_m').min()),
        minimum_plate_height_m=float(take('root_pose_xyzw')[:,:,2].min()))
    r['nonfoot_fraction']=r['nonfoot_steps']/len(q)
    r['pass']=bool(len(q)==end-start and not r['joint_bound_violation_steps'] and not r['speed_bound_violation_steps']
        and r['maximum_applied_nm']<=1.60001 and r['minimum_non_toe_floor_m']>=-.001
        and r['minimum_plate_height_m']>=.055 and r['nonfoot_fraction']<=nonfoot_bound)
    return r

def inspect(arm,geometry):
    global NATIVE
    parent=A/f'results_startup_{arm}_001';base=parent/'standing/evaluation'
    report,capture,c,n,integrity,binding=load_recording(base)
    startup=read(base/'startup_result.json');declaration=read(base/'declaration.json')
    NATIVE=read(parent/'standing/native/native_readback.json');initial=read(base/'native400hz/initial_state.json')
    handoff=read(base/'policy_handoff.json');identity=read(parent/'standing/identity.json')
    prefix={'cold':0,'neutral4':200}[arm];count=prefix+1000;seq=np.arange(count*8)
    assert identity['source_files']==FREEZE
    assert read(parent/'launch_binding.json')['source_freeze_sha256']==sha(A/'source_019/FREEZE_SHA256.json')
    assert report['checkpoint_sha256']==sha(CHECKPOINT)==startup['checkpoint_sha256']
    assert report['source_sha256']==FREEZE['evaluate.py']
    assert report['model_sha256']=='7126935b35398d2a7f03271335a8a1b8f2059354af2b19d9474f22b8e73e4881'
    assert report['urdf_sha256']=='9492fde54c50170e940b49ccb3037d7b413743a5b77bc1d8707a539d44349e78'
    assert report['controls']==count and capture['steps']==len(seq)
    assert report['acquisition_complete'] and report['failure'] is None and capture['failure'] is None
    assert not report['allocation_limit_reached'] and report['pose_forcing'] is False
    assert report['initial_resets']==1 and report['resets_during_trial']==0
    assert not startup['diagnostic_policy_screen_passed'] and not startup['stage2_complete']
    assert startup['policy_state_unchanged'] and startup['integrity_failure'] is None
    assert report['startup_prefix_controls']==prefix and report['policy_control_slice']==[prefix,count]
    assert report['recorded_physics_steps']==len(seq) and report['video_frames']==count//2
    assert np.array_equal(n['sequence'],seq) and np.array_equal(n['control_index'],seq//8)
    assert np.array_equal(n['substep_index'],seq%8)
    assert np.array_equal(n['explicit_counter'],seq+capture['initial_counter']+1)
    assert capture['final_counter']==capture['initial_counter']+len(seq)
    assert np.allclose(n['time_s'],(seq+1)*.0025,rtol=0,atol=1e-9)
    assert np.array_equal(c['global_control_index'],np.arange(count))
    assert np.array_equal(c['policy_control_index'],np.r_[np.full(prefix,-1),np.arange(1000)])
    assert (c['action_source'][:prefix]=='scripted_neutral').all() and (c['action_source'][prefix:]=='bc').all()
    assert np.array_equal(c['issued_action'],c['policy_action'])
    assert np.array_equal(c['actor_mean_action'][prefix:],c['issued_action'][prefix:])
    assert not c['issued_action'][:prefix].any() and not c['command'][:prefix].any()
    assert np.array_equal(c['command'],c['requested_command'])
    assert np.array_equal(c['command'][prefix:],np.broadcast_to(np.array([.05,0,0],np.float32),(1000,1,3)))
    assert np.array_equal(n['command'],np.repeat(c['command'],8,axis=0))
    assert not any(c[k].any() for k in ['terminated','truncated','reset'])
    for dataset in [c,n]:
        assert all(np.isfinite(v).all() for v in dataset.values() if v.dtype.kind not in 'US')
    q,dq=n['joint_position_rad'],n['joint_velocity_rad_s']
    assert np.array_equal(n['pre_joint_position_rad'][1:],q[:-1])
    assert np.array_equal(n['pre_joint_velocity_rad_s'][1:],dq[:-1])
    assert np.array_equal(n['pre_joint_position_rad'][0],np.asarray(initial['q'],np.float32))
    assert np.array_equal(n['pre_joint_velocity_rad_s'][0],np.asarray(initial['dq'],np.float32))
    for key in ['joint_position_rad','joint_velocity_rad_s','root_pose_xyzw','joint_target_rad','distal_contact','toe_xyz_world_m']:
        assert np.array_equal(n[key][7::8],c[key]),key
    expected=_diagnostic_servo(n['pre_joint_position_rad'][:,0],n['pre_joint_velocity_rad_s'][:,0],
        n['joint_target_rad'][:,0],np.full(18,12.,np.float32),np.asarray(KD,np.float32))
    for key,value in zip(('computed_torque_nm','applied_torque_nm','effort_ceiling_nm'),expected):
        assert np.array_equal(n[key][:,0],value),key
    assert np.array_equal(n['applied_torque_nm'],n['native_input_pre_nm'])
    assert np.array_equal(n['joint_target_rad'],np.repeat(c['joint_target_rad'],8,axis=0))
    assert np.array_equal((abs(n['computed_torque_nm'])>1.6).reshape(count,8,1,18).sum(1),c['saturation_count_400hz'])
    assert np.array_equal(n['nonfoot_contact'].reshape(count,8,1).sum(1),c['nonfoot_contact_count_400hz'])
    assert np.array_equal((~n['distal_contact'].all(-1)).reshape(count,8,1).sum(1),c['missing_six_toe_count_400hz'])
    neutral=torch.tensor(initial['q'],dtype=torch.float32);held=neutral.clone()
    limits=torch.tensor(NATIVE['limits'],dtype=torch.float32)
    for i,action in enumerate(c['issued_action']):
        held=emitted_target(torch.from_numpy(action),held,neutral,limits[:,:,0],limits[:,:,1],.35,.04)
        assert np.array_equal(held.numpy(),c['joint_target_rad'][i]),('emitted',i)
    assert np.array_equal(c['joint_target_rad'][:prefix],np.broadcast_to(neutral.numpy(),(prefix,1,18)))
    assert handoff['global_control_index']==prefix and handoff['physics_counter']==prefix*8
    assert handoff['extra_reset_or_physics_step'] is False
    assert np.array_equal(np.asarray(handoff['native_state_before_policy']['q'],np.float32),n['pre_joint_position_rad'][prefix*8])
    assert np.array_equal(np.asarray(handoff['native_state_before_policy']['dq'],np.float32),n['pre_joint_velocity_rad_s'][prefix*8])
    windows={name:window(n,start,end,bound) for name,start,end,bound in [
        ('full',0,len(seq),.001),('scripted_prefix',0,prefix*8,0.),('policy',prefix*8,len(seq),.001)]}
    assert windows==report['startup_physical_windows']['windows']
    assert all(w['pass'] for w in windows.values())
    packets=patches=inactive=0
    with (base/'native400hz/contacts.jsonl').open() as stream:
        for line in stream:
            packet=json.loads(line)
            assert packet['sequence']==packets and packet['explicit_counter']==int(n['explicit_counter'][packets])
            feet,other,nonfoot,zeros=classify(packet,n['link_pose_xyzw'][packets,0],capture['body_names'],geometry)
            assert np.array_equal(feet,n['distal_force_world_n'][packets,0])
            assert np.array_equal(other,n['nonfoot_force_world_n'][packets,0])
            assert np.array_equal(np.linalg.norm(feet,axis=-1)>1,n['distal_contact'][packets,0])
            assert nonfoot==bool(n['nonfoot_contact'][packets,0])
            packets+=1;patches+=len(packet['patches']);inactive+=zeros
    assert packets==len(seq)
    # Score the unchanged 1,000-row policy view with global timestamps intact.
    tail={k:v[prefix:] for k,v in c.items()};case=report['cases'][0]
    score=score_recording(tail,{**declaration,'profile':'omni_static','env_index':0,'command':[.05,0,0],
        'target_slew_rad':.04,'control_dt_s':.02,'seed':case.get('seed',report['seed'])})
    saved=report['results'][0]
    assert score['failed_bounds']==saved['failed_bounds']==['planar_error_mps']
    assert score['pass'] is False and score['checks']==saved['checks']
    differences={k:{'native':v,'local':score['metrics'][k]} for k,v in saved['metrics'].items() if score['metrics'][k]!=v}
    compatibility=None
    if differences:
        assert set(differences)=={'vertical_velocity_abs_p95_mps'},differences
        values=np.sort(abs(tail['velocity_world_mps'][100:,0,2]))
        rank=np.float32(len(values)-1)*np.float32(.95)
        lo=int(rank);fraction=float(rank-lo);delta=values[lo+1]-values[lo]
        archived=float(values[lo+1]-delta*(1-fraction) if fraction>=.5 else values[lo]+delta*fraction)
        assert archived==saved['metrics']['vertical_velocity_abs_p95_mps']
        compatibility={'metric':'vertical_velocity_abs_p95_mps','values':differences['vertical_velocity_abs_p95_mps'],
            'native_float32_rank_reconstruction':archived,'scope':'Descriptive percentile only; no gate or native value changed. All gate checks and verdicts reproduce exactly.'}
    assert read(parent/'standing/native/native_errors.json')==[]
    assert read(parent/'jobs/standing_contact_data_audit.json')['passed']
    cleanup=read(parent/'cleanup.json')
    assert cleanup['cleanup_checked'] and cleanup['inspections'][-1]['absent'] and not cleanup['reservation_released']
    support={name:support_summary(n,start,end) for name,start,end in [
        ('full',0,len(seq)),('policy',prefix*8,len(seq)),('fixed_policy_score',(prefix+100)*8,len(seq))]}
    if prefix:support['scripted_prefix']=support_summary(n,0,prefix*8)
    return dict(arm=arm,controls=count,physics_steps=packets,patches_reclassified=patches,inactive_zero_tuples=inactive,
        scripted_controls=prefix,bc_controls=1000,policy_score_global_control_slice=[prefix+100,count],
        policy_screen_pass=False,failed_bounds=score['failed_bounds'],unchanged_gate_checks_exact=True,descriptive_percentile_compatibility=compatibility,
        original_metrics=score['metrics'],windows=windows,support=support,
        servo_demands_ceiling_applied_native_input_bitexact=True,pre_post_continuity_bitexact=True,
        all_control_targets_bitexact=True,bc_actor_issued_equal=True,scripted_zero_and_neutral_equal=True,
        native_contact_forces_categories_support_bitexact=True,controls_terminated_truncated_reset=[0,0,0],
        maximum_sdk_joint_speed_rad_s=float(abs(dq).max()),
        maximum_interval_angle_rate_rad_s=float(abs(n['interval_angle_rate_rad_s']).max()),
        own_container_absent=True,video_frame_count=report['video_frames'],integrity=integrity,
        native_readback_sha256=sha(parent/'standing/native/native_readback.json'),initial_state_sha256=sha(base/'native400hz/initial_state.json'))

if __name__=='__main__':
    torch.set_num_threads(2)
    prior=load_audit_helpers();FREEZE=read(A/'source_019/FREEZE_SHA256.json')
    CHECKPOINT=REPO/'artifacts/restart_2026-09-14/paper_bc_migration_002/run_001/checkpoint_update000000_migrated.pt'
    geometry_path=REPO/'artifacts/restart_2026-09-14/standing_translation_preparation_001/source/geometry/geometry.json'
    geometry={r['body']:r for r in read(geometry_path)['shapes']}
    inputs=[Path(__file__),prior,CHECKPOINT,geometry_path,A/'source_019/FREEZE_SHA256.json']
    for name,digest in FREEZE.items():
        path=A/'source_019'/name;assert sha(path)==digest;inputs.append(path)
    for name in ['analyze.py','env.py','env_config.py','evaluation.py']:
        path=REPO/'experiments/paper_walk'/name;assert sha(path)==FREEZE[name];inputs.append(path)
    for arm in ['cold','neutral4']:
        base=A/f'results_startup_{arm}_001'
        transfer=A/f'VERIFIED_TRANSFER_startup_{arm}_001.json';assert read(transfer)['all_files_verified'];inputs.append(transfer)
        inputs.extend(p for p in base.rglob('*') if p.is_file())
        assert read(base/'standing/identity.json')['geometry_sha256']==sha(geometry_path)
    before={str(p.relative_to(REPO)):sha(p) for p in inputs}
    result={'schema':'startup_pair_native_audit_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'native_actions':0,'maintained_source_edits':False,'stage2_complete':False}
    try:
        result['arms']=[inspect(arm,geometry) for arm in ['cold','neutral4']]
        assert result['arms'][0]['native_readback_sha256']==result['arms'][1]['native_readback_sha256']
        assert result['arms'][0]['initial_state_sha256']==result['arms'][1]['initial_state_sha256']
        result['all_checks_passed']=True
    except BaseException as e:
        result.update(all_checks_passed=False,error=repr(e),traceback=traceback.format_exc())
    result['input_sha256']=before
    result['inputs_unchanged']=all(sha(REPO/k)==v for k,v in before.items())
    result['limitations']=['CPU artifact audit reuses hash-matching maintained servo/scorer and two pure prior audit helpers; it does not rerun physics.',
        'Every saved contact point is transformed/reclassified with its recorded native link pose and exact frozen cap geometry; full mesh vertices are not transformed again for clearance.',
        'Recorded controls prove actor/issued equality on BC rows, not independent checkpoint network reconstruction; the parent analysis owns reconstruction and history diagnostics.',
        'Initial state and actual counters are verified; separate app/process identity relies on root launch and cleanup records.',
        'Scripted prefix support and motor metrics are diagnostic only, not learned quiet or stop qualification. All 96 formal Stage2 cases remain unrun.',
        'Video byte hash and reported frame count are checked; decoding/visual inspection is outside this native audit.']
    dump(HERE/'RESULT_003.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['arms','input_sha256']},indent=2))
    if not result['all_checks_passed'] or not result['inputs_unchanged']:raise SystemExit(1)
