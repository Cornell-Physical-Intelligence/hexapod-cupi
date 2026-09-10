"""Independent actual consumer003 recovery ledger audit; no admission writer."""
from pathlib import Path
import argparse
import json
import numpy as np
from audit_base001 import load, sha, require, moving

CONSUMER='fd9bef87dda976f9b229e7541408d24e674245ca2c51231dfce9075b81d38334'
HISTORY=('history','history_valid','last_fd','last_fd_valid','ready','episodes','steps')

def equal(a,b,message):
    require(np.array_equal(a,b),message)

def replay_ledger(directory,trace):
    directory=Path(directory);summary=json.loads((directory/'summary.json').read_text())
    files=summary['files_sha256']
    actual={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='summary.json'}
    require(actual==files,'Ledger files differ from immutable map')
    records=[load(directory/name) for name in sorted(files)]
    controls,n=trace['learning_active'].shape
    require(summary['controls']==controls and summary['replicas']==n,'Ledger dimensions differ')
    terminal=trace['training_terminated'];truncated=trace['training_truncated'];ended=terminal|truncated
    by_control={};completed=np.zeros(n,dtype=int);details=[];bootstrap_count=0
    for index,r in enumerate(records):
        kind=str(r['kind']);c=int(r['control']);selected=r['selected'];ids=np.flatnonzero(selected)
        require(selected.shape==(n,) and selected.dtype==bool and len(ids)>0,'Malformed selected rows')
        require(1<=c<=controls and bool(r['verified']),'Unverified or out-of-range ledger record')
        for key,value in r.items():
            if value.dtype.kind in 'fc':require(np.isfinite(value).all(),'Nonfinite ledger '+key)
        if kind=='reset':
            require(c not in by_control,'Duplicate reset at one control')
            by_control[c]=r
            equal(selected,ended[c-1],'Reset mask differs from actual outcomes')
            equal(r['terminated'],terminal[c-1],'Reset terminal mask differs')
            equal(r['truncated'],truncated[c-1],'Reset timeout mask differs')
            equal(r['episode_before'],trace['episode_id'][c-1],'Wrong final episode')
            equal(r['episode_after'],r['episode_before']+selected,'Wrong next episode')
            equal(r['final_episode'][:,0],r['episode_before'][selected],'Final packet is not pre-reset')
            equal(r['next_episode'][:,0],r['episode_after'][selected],'Reset packet wrong episode')
            for key in HISTORY:equal(r['before__'+key][~selected],r['after__'+key][~selected],'Unselected history changed '+key)
            require(not r['after__ready'][selected].any(),'Reset actor history remains ready')
            for key,width in [('policy',846),('critic',849)]:
                require(r['final_'+key].shape==(len(ids),width),'Wrong final packet width')
                require(r['next_'+key].shape==(len(ids),width) and not r['next_'+key].any(),'Reset actor packet leaked history')
            require(not r['next_learning_valid'].any(),'Recovery packet allowed learning')
            details.append({'control':c,'rows':ids.tolist(),'terminated':terminal[c-1,ids].tolist(),
                'reference_failure':trace['next_reference_failure'][c-1,ids].tolist()})
        elif kind=='history_initialization':
            for key in HISTORY:equal(r['before__'+key][~selected],r['reset__'+key][~selected],'Selective history reset contaminated another row '+key)
            require(not r['reset__history'][selected].any() and not r['reset__history_valid'][selected].any(),'Old history was not cleared')
            require(not r['reset__last_fd_valid'][selected].any(),'Reset interval-rate validity leaked')
            require((r['encoded__history_valid'][selected].sum(-1)==1).all(),'First recovered history has stale frames')
            require(not r['encoded__last_fd_valid'][selected].any() and (r['encoded__steps'][selected]==0).all(),'First recovered interval/step state invalid')
            require(r['learning_valid'].all(),'Recovered packet not enabled')
            equal(r['packet_episode'][:,0],r['episode'][selected],'Recovered episode differs')
            equal(r['policy'],r['critic'][:,:846],'Actor/critic shared packet differs')
            history=r['encoded__history'][selected].reshape(len(ids),315).astype(np.float32)
            equal(r['policy'][:,:315],history,'Actual actor packet did not use the recovered history')
            equal(r['policy'][:,315:320],r['encoded__history_valid'][selected].astype(np.float32),'Actual actor history-valid flags differ')
            for row,origin in zip(ids,r['origin_record']):
                if origin<0:
                    require(c==200 and r['episode'][row]==0,'Unexpected initial history event')
                    continue
                require(origin<index,'Recovery origin is not a prior record')
                reset=records[int(origin)];start=int(reset['control'])
                require(str(reset['kind'])=='reset' and reset['selected'][row] and c-start==200,'Recovery has wrong origin/duration')
                require(not trace['learning_active'][start:c,row].any(),'Recovery controls entered learning')
                equal(trace['episode_id'][start:c,row],np.full(200,reset['episode_after'][row]),'Recovery episode changed')
                if reset['terminated'][row]:completed[row]+=1
        elif kind=='bootstrap':
            bootstrap_count+=1;require(c in by_control,'Bootstrap lacks prior actual reset')
            reset=by_control[c]
            for key in ('selected','final_policy','final_critic','next_policy','next_critic','next_learning_valid','next_episode'):
                equal(r[key],reset[key],'Actual critic/reset packet differs '+key)
            equal(r['terminated'],terminal[c-1],'Bootstrap terminal mask differs')
            equal(r['truncated'],truncated[c-1],'Bootstrap timeout mask differs')
            require(not r['used_next_value'][r['terminated']].any(),'True terminal bootstrapped nonzero')
            require(r['normalization_rows']==0 and str(r['source'])=='unoptimized_recovery_probe_no_PPO_storage','Admission collected or optimized a rollout')
            require(not r['truncated'].any(),'A 512-control admission unexpectedly timed out')
        else:raise ValueError('Unknown actual ledger kind '+kind)
    require(set(by_control)==set(np.flatnonzero(ended.any(-1))+1),'Actual terminal ledger incomplete')
    require(bootstrap_count==len(by_control),'Actual bootstrap event count differs')
    require(int(completed.sum())==summary['completed_finite_recovery_rows'],'Completed recovery summary differs')
    equal(np.asarray(summary['per_environment_episodes']),ended.sum(0),'Final episode count differs')
    return {'record_count':len(records),'event_count':len(by_control),'bootstrap_records':bootstrap_count,
        'completed_finite_recoveries_per_row':completed.tolist(),'completed_finite_recovery_rows':int(completed.sum()),
        'events':details,'actual_final_packet_and_reset_history_replayed':True,
        'terminal_bootstrap_zero_replayed':True,'GPU_timeout_bootstrap_observed':False,
        'normalization_storage_and_optimizer_updates_claimed':0}

def progress(trace,state):
    per=[];start=200;controls,n=trace['learning_active'].shape
    # Independently derive forward heading from XYZW quaternion; never use
    # the source's stored rotation matrix to reproduce its own result.
    q=trace['quaternion_world_xyzw'][:-1].astype(np.float64)
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    x,y,z,w=np.moveaxis(q,-1,0)
    fx=2*(w*z-x*y);fy=-(1-2*(x*x+z*z));norm=np.hypot(fx,fy)
    require((norm>0).all(),'Undefined planar body heading')
    d=trace['position_world_m'][1:]-trace['position_world_m'][:-1]
    projected=(d[:,:,0]*fx+d[:,:,1]*fy)/norm
    for row in range(n):
        active=trace['learning_active'][start:,row]
        same=trace['episode_id'][start:,row]==trace['episode_id'][start-1:-1,row]
        mask=active & (trace['requested_command'][start:,row,0]!=0) & same
        displacement=float(projected[start-1:,row][mask].sum())
        velocity=float(-trace['velocity_body_mps'][start:,row,1][mask].mean()) if mask.any() else 0.
        r={'row':row,'active_controls':int(active.sum()),'excluded_recovery_controls':int((~active).sum()),
            'finite_terminal_transitions':int(trace['training_terminated'][start:,row].sum()),
            'time_limit_transitions':int(trace['training_truncated'][start:,row].sum()),
            'moving_intervals_without_reset_crossing':int(mask.sum()),
            'measured_forward_increment_sum_m':displacement,'raw_SDK_mean_forward_mps':velocity}
        recorded=state['recovery_per_environment'][row]
        for key,value in r.items():
            if isinstance(value,float):require(abs(value-recorded[key])<=1e-8,'Independent position/heading projection disagrees '+key)
            else:require(recorded[key]==value,'Actual per-row count disagrees '+key)
        per.append(r)
    timings=np.asarray(state['recovery_control_seconds']);require(timings.shape==(512,) and np.isfinite(timings).all() and (timings>0).all(),'Invalid measured loop timing')
    return {'per_environment':per,'timing_s':{'wall':state['recovery_wall_s'],'sum_controls':float(timings.sum()),
        'mean':float(timings.mean()),'p50':float(np.quantile(timings,.5)),'p95':float(np.quantile(timings,.95)),
        'max':float(timings.max()),'actual_env_controls_per_wall_s':512*n/state['recovery_wall_s']},
        'memory':{k:v for k,v in state.items() if 'memory' in k.lower() or 'rss' in k.lower()},
        'rate_fidelity_qualified':False,'physical_admission':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    require(not a.out.exists(),'Audit output immutable')
    state=json.loads((a.phase/'state.json').read_text());raw=a.phase/'raw';result={'integrity_passed':False,'physical_admission':False,'training_allocation_approved':False}
    try:
        require(state['identity']['consumer_freeze_sha256']==CONSUMER and state['mode']=='learning_recovery_32','Wrong frozen consumer or mode')
        trace=load(raw/'trace.npz');physics=load(raw/'physics_substeps.npz');reference=load(raw/'reference_states.npz')
        clocks=load(raw/'sensor_clocks.npz');resets=load(raw/'reset_clocks.npz') if (raw/'reset_clocks.npz').exists() else None
        events=json.loads((raw/'episodes.json').read_text())
        base=moving(trace,physics,clocks,reference,events,[],resets)
        for key in ('actor_packet_history_directly_exported','optimizer_final_bootstrap_directly_exported','unexported_state_scope'):base.pop(key)
        for key in ('raw_residual_action','residual_goal_rad','residual_position_rad','residual_velocity_rad_s'):
            require(np.isfinite(trace[key]).all() and not trace[key].any(),'Nonzero or invalid residual during zero-action admission: '+key)
        requested=np.abs(physics['computed_torque_nm'][1:]).reshape(712,8,32,18).transpose(0,2,1,3)
        post=np.zeros((712,32),bool);post[200:]=True;active=trace['learning_active']
        base['requested_torque_by_learning_mask']={name:{'row_controls':int(mask.sum()),
            'max_nm':float(requested[mask].max()),'samples_above_1p6':int((requested[mask]>1.6).sum())}
            for name,mask in [('initial_startup',~post),('active',post&active),('excluded_recovery',post&~active)]}
        base['exact_zero_residual_action_and_state_replayed']=True
        result.update(raw=base,ledger=replay_ledger(raw/'learning_ledger',trace),progress=progress(trace,state))
        require(state['status']=='completed' and state['learning_recovery_passed'] and state['PPO_updates_completed']==0 and not state['policy_training_started'],'Incomplete or unapproved optimization')
        require(result['ledger']['completed_finite_recovery_rows']>=1,'No actual completed finite recovery')
        result['integrity_passed']=True
    except Exception as exc:result['error']=repr(exc)
    result['input_sha256']={str(p.relative_to(a.phase)):sha(p) for p in a.phase.rglob('*') if p.is_file()}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','raw','ledger','progress')}))
    return 0 if result['integrity_passed'] else 1

if __name__=='__main__':raise SystemExit(main())
