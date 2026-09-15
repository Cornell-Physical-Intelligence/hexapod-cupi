"""Typed startup evidence analysis, separate prefix/policy scopes, no native work."""
from pathlib import Path
import collections
import datetime
import hashlib
import importlib.util
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
A=HERE.parent
ROOT=A.parents[2]
PINS={}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def pin(path):
    PINS[str(path.relative_to(ROOT))]=sha(path)
    return path


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,pin(path))
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


def typed_arrays(path):
    with np.load(path,allow_pickle=False) as f:result={k:f[k].copy() for k in f.files}
    for key,value in result.items():
        if key=='action_source':
            assert value.dtype.kind=='U' and set(value.tolist())<={'bc','scripted_neutral'}
        else:
            assert value.dtype.kind in 'biuf' and np.isfinite(value).all(),key
    return result


def load_arm(analyzer,arm):
    directory=A/f'results_startup_{arm}_001'
    inventory=json.loads(pin(A/f'inventory_startup_{arm}_001.json').read_text())
    actual={str(x.relative_to(directory)) for x in directory.rglob('*') if x.is_file()}
    assert actual==set(inventory['files']), (actual^set(inventory['files']))
    for name,digest in inventory['files'].items():assert sha(pin(directory/name))==digest,name
    transfer=json.loads(pin(A/f'VERIFIED_TRANSFER_startup_{arm}_001.json').read_text())
    assert transfer['all_files_verified']
    analyzer.load_arrays=typed_arrays
    report,capture,control,native,integrity,binding=analyzer.load_recording(directory/'standing/evaluation')
    prefix=0 if arm=='cold' else 200
    count=prefix+1000
    assert report['acquisition_complete'] and report['failure'] is None
    assert report['controls']==count and capture['steps']==count*8
    assert report['policy_control_slice']==[prefix,count] and report['initial_resets']==1 and report['resets_during_trial']==0
    assert np.array_equal(control['global_control_index'],np.arange(count))
    assert np.array_equal(control['policy_control_index'],np.r_[np.full(prefix,-1),np.arange(1000)])
    assert np.all(control['action_source'][:prefix]=='scripted_neutral') and np.all(control['action_source'][prefix:]=='bc')
    assert np.array_equal(control['policy_action'],control['issued_action'])
    assert not control['issued_action'][:prefix].any()
    assert np.array_equal(control['issued_action'][prefix:],control['actor_mean_action'][prefix:])
    assert np.all(control['command'][:prefix]==0)
    assert np.all(control['command'][prefix:]==np.array([.05,0,0],np.float32))
    assert np.array_equal(native['explicit_counter'],np.arange(1,count*8+1))
    assert capture['initial_counter']==0 and capture['final_counter']==count*8
    handoff=json.loads((directory/'standing/evaluation/policy_handoff.json').read_text())
    assert handoff['global_control_index']==prefix and handoff['physics_counter']==prefix*8
    assert handoff['extra_reset_or_physics_step'] is False
    for key,channel in [('obs','policy_observation'),('critic','critic_observation'),('amp','amp_state_before')]:
        assert np.array_equal(np.asarray(handoff['observations_before_policy'][key],dtype=np.float32),control[channel][prefix])
    state=json.loads((directory/'standing/state.json').read_text())
    neutral=np.asarray(state['identity']['physics_config']['neutral_joint_position_rad'],dtype=np.float32)
    expected_held=neutral[None] if not prefix else control['joint_target_rad'][prefix-1]
    assert np.array_equal(np.asarray(handoff['actual_held_target_rad'],np.float32),expected_held)
    assert np.all(control['joint_target_rad'][:prefix]==neutral)
    if prefix:
        assert np.array_equal(control['amp_state_before'][prefix],control['amp_state_after'][prefix-1])
    result={'arm':arm,'inventory_files_verified':len(actual),'inventory_sha256':sha(A/f'inventory_startup_{arm}_001.json'),
            'complete_controls':count,'scripted_prefix_controls':prefix,'bc_controls':1000,
            'full_native_steps':count*8,'handoff_counter':prefix*8,'action_labels_and_issued_actions_exact':True,
            'handoff_obs_critic_amp_and_previous_held_exact':True,'prefix_is_not_bc_quiet_qualification':True,
            'original_results':report['results'],'native_capture_integrity':integrity,
            'original_startup_windows':report['startup_physical_windows'],
            'checkpoint_sha256':state['input_checkpoint_sha256'],'physics_binding':binding}
    for label,start,end in [('full',0,count),('policy',prefix,count),('policy_first100',prefix,prefix+100),('policy_later900',prefix+100,count)]+([('scripted_prefix',0,prefix)] if prefix else []):
        result[label]=analyzer.analyze_trace({k:v[start:end] for k,v in control.items()},
                         {k:v[start*8:end*8] for k,v in native.items()},joint_names=capture['joint_names'])
        result[label]['scope']=f'{label}: global controls[{start},{end}); absolute recorded timestamps retained. Relative diagnostic event times start at this slice; no gate or raw record changed.'
    return result,control,native,handoff


def stats(x):
    x=np.asarray(x,dtype=float)
    return {'mean':float(x.mean()),'median':float(np.median(x)),'p95':float(np.quantile(x,.95)),'min':float(x.min()),'max':float(x.max())}


def distances(x,y):
    x=x.astype(float);y=y.astype(float)
    return np.sqrt(np.maximum((x*x).sum(1)[:,None]+(y*y).sum(1)[None]-2*x@y.T,0)/x.shape[1])


def main():
    import torch
    torch.set_num_threads(1)
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    pin(Path(__file__).resolve())
    analyzer=module('startup_numeric_analyzer',ROOT/'experiments/paper_walk/analyze.py')
    learner=module('startup_frozen_learner019',A/'source_019/learner.py')
    pin(A/'source_019/FREEZE_SHA256.json');pin(A/'source_019/evaluate.py')
    checkpoint=pin(ROOT/'artifacts/restart_2026-09-14/paper_bc_migration_002/run_001/checkpoint_update000000_migrated.pt')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['learner_sha256']==sha(A/'source_019/learner.py') and saved['schema']==learner.SCHEMA
    model=learner.ActorCritic(learner.Config(**saved['config'])).eval();model.load_state_dict(saved['model'],strict=True)
    original=torch.load(pin(ROOT/'artifacts/paper_bc_fit_004/candidate_checkpoint_update000000.pt'),map_location='cpu',weights_only=False)
    assert all(torch.equal(v,original['model'][k]) for k,v in saved['model'].items())
    with np.load(pin(ROOT/'artifacts/paper_bc_data_002/bc_dataset.npz'),allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    pin(A/'analysis_startup_cold_001/FAILED_GENERIC_ANALYZER_001.json')
    with torch.no_grad():norm_demo=model.obs_normalizer(torch.from_numpy(data['observations'])).numpy()
    forward=np.all(data['commands']==np.array([.05,0,0],np.float32),axis=1)
    pools={'forward_all':np.flatnonzero(forward),'forward_onset':np.flatnonzero(forward&(data['source_kind']==1)),
           'forward_steady':np.flatnonzero(forward&(data['source_kind']==0))}
    groups={'all':np.arange(231),'q':np.array([42*f+i for f in range(5) for i in range(6,24)]),
            'dq':np.array([42*f+i for f in range(5) for i in range(24,42)]),
            'gyro':np.array([42*f+i for f in range(5) for i in range(3)]),
            'gravity':np.array([42*f+i for f in range(5) for i in range(3,6)]),
            'held_target':np.arange(213,231),'command':np.arange(210,213)}
    rows={};per_row={};controls={}
    for arm in ('cold','neutral4'):
        row,c,n,h=load_arm(analyzer,arm);controls[arm]=c
        assert row['checkpoint_sha256']==sha(checkpoint)
        prefix=row['scripted_prefix_controls'];o=c['policy_observation'][prefix:,0]
        with torch.no_grad():
            mu,velocity=model.actor(torch.from_numpy(o));z=model.obs_normalizer(torch.from_numpy(o)).numpy()
        row['bc_actor_reconstruction']={'rows':len(o),'cpu_native_action_max_absolute_difference':float(np.abs(mu.numpy()-c['issued_action'][prefix:,0]).max()),
            'cpu_native_estimator_max_absolute_difference':float(np.abs(velocity.numpy()-c['estimated_velocity_navigation_mps'][prefix:,0]).max()),
            'native_actor_mean_equals_issued_action_exact':True,'scope':'Only actual BC suffix rows reconstructed; scripted prefix issued actions are zero, and unused actor means are not mislabeled as BC control.'}
        row['handoff_root_height_m']=float(c['amp_state_before'][prefix,0,42])
        row['nearest']={};row['outside_demo_ranges']={}
        windows={'all1000':slice(None),'first100':slice(0,100),'later900':slice(100,None)}
        for pool,ids in pools.items():
            row['nearest'][pool]={}
            for group,cols in groups.items():
                dist=distances(z[:,cols],norm_demo[ids][:,cols]);idx=dist.argmin(1);near=ids[idx];mind=dist.min(1)
                row['nearest'][pool][group]={w:stats(mind[sl]) for w,sl in windows.items()}
                if group=='all':
                    per_row[arm+'__'+pool+'__nearest_row']=near;per_row[arm+'__'+pool+'__distance']=mind
                    actor=c['issued_action'][prefix:,0]
                    err=np.sqrt(np.mean(((actor-data['actions'][near])*.35)**2,axis=1))
                    held=np.sqrt(np.mean((c['joint_target_rad'][prefix:,0]-data['applied_joint_target_rad'][near])**2,axis=1))
                    row['nearest'][pool]['actor_to_nearest_teacher_requested_target_rms_rad']={w:stats(err[sl]) for w,sl in windows.items()}
                    row['nearest'][pool]['held_to_nearest_teacher_applied_target_rms_rad']={w:stats(held[sl]) for w,sl in windows.items()}
                    row['nearest'][pool]['handoff']={'dataset_row':int(near[0]),'raw_control':int(data['control_index'][near[0]]),
                        'replica':int(data['env_index'][near[0]]),'normalized_distance':float(mind[0]),
                        'q_current_difference_rms_rad':float(np.sqrt(np.mean((o[0,174:192]-data['observations'][near[0],174:192])**2))),
                        'teacher_prehold_root_height_m':float(data['states'][near[0],42]),
                        'requested_target_error_rms_rad':float(err[0]),'actual_held_target_error_rms_rad':float(held[0])}
            ids_onset=pools['forward_onset']
        for group,cols in groups.items():
            lo=data['observations'][:,cols].min(0);hi=data['observations'][:,cols].max(0)
            mask=(o[:,cols]<lo)|(o[:,cols]>hi)
            row['outside_demo_ranges'][group]={w:{'scalar_fraction':float(mask[sl].mean()),'rows_with_any':int(mask[sl].any(1).sum())} for w,sl in windows.items()}
        rows[arm]=row
    assert rows['cold']['physics_binding']==rows['neutral4']['physics_binding']
    result={'schema':'independent_bc_startup_pair_analysis_v1','started_utc':started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'same_checkpoint_model_normalizers_and_physics':True,'bc_model_exact_against_original_fit004':True,'arms':rows,
        'generic_analyzer_failure_preserved':str((A/'analysis_startup_cold_001/FAILED_GENERIC_ANALYZER_001.json').relative_to(ROOT)),
        'typed_loader_scope':'Only action_source may be a Unicode channel; exact labels/prefix/indices verified. All other channels must be finite numeric. Maintained source and raw files unchanged.',
        'nearest_scope':'Saved actor normalization, including its clamp; per-group RMS nearest distances, actual same-command teacher labels only. Nearest targets are not expert labels for the policy state. Empirical ranges are descriptive, not new gates.',
        'limitations':['Warm and cold are one run each, not a statistical replication.','Scripted neutral prefix is not BC quiet qualification.','All original forward windows and failed planar gates are retained.','Paired intervention changes physical and observed context together; it does not isolate root-height causality.','Independent full raw audit is delivered separately under native_audit.','No image/video appearance is inferred from numerical records; root performs RGB review.'],
        'input_sha256':PINS,'stage2_complete':False}
    assert all(sha(ROOT/p)==v for p,v in PINS.items())
    result['inputs_unchanged']=True
    with (HERE/'RESULT.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    with (HERE/'nearest_rows.npz').open('xb') as f:np.savez_compressed(f,**per_row)
    print(json.dumps({'complete':True,'input_files':len(PINS),'arms':{k:{'controls':v['complete_controls'],'handoff':v['nearest']['forward_onset']['handoff'],'actor':v['bc_actor_reconstruction']} for k,v in rows.items()}}))


if __name__=='__main__':main()
