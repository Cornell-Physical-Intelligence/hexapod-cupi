"""Independent CPU replay of frozen moving-consumer raw evidence.

This is an audit, never a launcher, reward replacement or acceptance gate.
History contents and optimizer bootstrap tensors are not exported by consumer001;
their GPU correctness cannot be inferred from the episode ledger alone.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

CONSUMER = 'dd49fd3e3639e453b067b9d4d67e5465db20865200c6a17616dd6936b527b842'
ENDPOINTS = {'root_link_position_world_m':'position_world_m',
    'root_link_quaternion_world_xyzw':'quaternion_world_xyzw',
    'root_link_velocity_world_mps':'velocity_world_mps',
    'computed_torque_nm':'computed_torque_nm','applied_torque_nm':'applied_torque_nm',
    'joint_position_rad':'joint_position_rad','joint_velocity_rad_s':'joint_velocity_rad_s'}

def require(ok, message):
    if not bool(ok): raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def load(path):
    with np.load(path,allow_pickle=False) as archive:
        return {k:archive[k] for k in archive.files}

def ticks(value):
    value=np.asarray(value,dtype=np.float32).copy()
    for _ in range(8): value=np.add(value,np.float32(.0025),dtype=np.float32)
    return value

def substeps(trace, raw):
    count,n=trace['joint_position_rad'].shape[:2]
    size=count*8+1
    require(raw['joint_position_rad'].shape==(size,n,18),'Incomplete named 400 Hz joint data')
    require(np.array_equal(raw['relative_physics_index'],np.arange(size)),'Missing/reordered physics sample')
    require(np.array_equal(raw['control_index'],np.r_[-1,np.repeat(np.arange(count),8)]),'Wrong control indices')
    require(np.array_equal(raw['substep_index'],np.r_[0,np.tile(np.arange(1,9),count)]),'Wrong substep indices')
    require(np.all(np.diff(raw['sim_step_counter'])==1),'Physics counter discontinuity')
    require(np.all(np.abs(np.diff(raw['sdk_sim_timestamp_s'])-.0025)<=1e-7),'SDK timestamp cadence differs')
    require(np.allclose(raw['time_s'],np.arange(size)*.0025,rtol=0,atol=1e-12),'Synthetic relative substep time differs')
    for key,ordinary in ENDPOINTS.items():
        require(np.isfinite(raw[key]).all(),'Nonfinite raw physics '+key)
        require(np.array_equal(raw[key][8::8],trace[ordinary]),'Endpoint mismatch '+key)
    require(np.max(np.abs(raw['applied_torque_nm']))<=float(np.float32(1.6)),'Applied motor cap exceeded')
    requested=np.abs(raw['computed_torque_nm'][1:]).reshape(count,8,n,18)
    return requested.max((1,3))>1.6,{
        'completed_controls':count,'physical_samples':size,
        'all_substep_max_requested_torque_nm':float(requested.max()),
        'all_substep_max_applied_torque_nm':float(np.abs(raw['applied_torque_nm']).max()),
        'requested_excess_samples':int(np.count_nonzero(requested>1.6)),
        'post_startup_max_requested_torque_nm':float(requested[200:].max()) if count>200 else None,
        'joint_rate_semantics':'Raw SDK joint rate retained; no physical rate-fidelity claim'}

def clocks(clock, ended, resets=None):
    current=clock['sensor_timestamp_s'];last=clock['sensor_last_update_s']
    count,n=ended.shape
    require(current.shape==(count,n,14) and current.dtype==np.float32,'Wrong actual sensor clock shape/dtype')
    require(np.array_equal(current,last) and np.array_equal(current,clock['expected_timestamp_s']),
            'Lazy contact cache/expected clock mismatch')
    require(np.isfinite(current).all() and np.all(current>=0),'Invalid sensor clock')
    require(not clock['sensor_outdated'].any() and clock['all_sensors_valid'].all(),'Invalid/stale sensor')
    require(not clock['sensor_age_s'].any() and clock['contact_valid'].all(),'Invalid measured sensor age')
    reset_rows=[]
    for i in range(count-1):
        previous=current[i].copy();mask=ended[i]
        if mask.any():
            require(resets is not None,'Missing actual reset-clock readback')
            index=len(reset_rows);reset_rows.append(i)
            require(np.array_equal(resets['reset_mask'][index],mask),'Reset mask differs from terminal ledger')
            for key in ('timestamp','last_update'):
                require(np.all(resets[key][index,mask]==0),'Selected reset clocks were not zero')
                require(np.array_equal(resets[key][index,~mask],current[i,~mask]),'Unselected reset clocks changed')
            require(resets['outdated'][index,mask].all(),'Reset rows were not marked outdated')
            previous[mask]=0
        require(np.array_equal(current[i+1],ticks(previous)),'Nonconsecutive actual eight-tick sensor advancement')
    # A final-row reset has no next physical sample but still needs its receipt.
    if ended[-1].any():
        index=len(reset_rows);reset_rows.append(count-1);mask=ended[-1]
        require(resets is not None and np.array_equal(resets['reset_mask'][index],mask),'Missing final reset receipt')
        require(np.all(resets['timestamp'][index,mask]==0) and np.all(resets['last_update'][index,mask]==0)
                and resets['outdated'][index,mask].all(),'Invalid final selected reset clocks')
        require(np.array_equal(resets['timestamp'][index,~mask],current[-1,~mask]),'Final unselected clocks changed')
    require((0 if resets is None else len(resets['reset_mask']))==len(reset_rows),'Extra reset-clock receipts')
    return {'all_14_sensor_clocks_replayed':True,'clock_reset_events':len(reset_rows),
            'first_clock_epoch':'Initial absolute sensor epoch is not independently exported'}

def moving(trace, raw, clock, reference, events, updates, resets=None):
    count,n=trace['learning_active'].shape
    require(count>=200 and n in (32,128),'Explicit startup and admitted replica count required')
    for key in ('learning_active','training_terminated','training_truncated','next_reference_failure','reward_scorable'):
        require(trace[key].shape==(count,n) and trace[key].dtype==np.bool_,'Wrong boolean row field '+key)
    require(not any(k.startswith('field_present__') and not v.all() for k,v in trace.items()),
            'Partial trace fields: preserve failed prefix; no complete replay claim')
    active=trace['learning_active'];term=trace['training_terminated'];timeout=trace['training_truncated'];ended=term|timeout
    require(not (term&timeout).any() and not (ended&~active).any(),'Invalid terminal/timeout learning classification')
    episode=trace['episode_id'];expected=np.zeros(n,dtype=np.int64)
    age=np.zeros(n,dtype=np.int64);warm=np.full(n,200,dtype=np.int64)
    expected_events=[]
    for i in range(count):
        enabled=warm==0
        require(np.array_equal(episode[i],expected),'Episode state crossed/reset early at control '+str(i))
        require(np.array_equal(active[i],enabled),'Recovery learning mask differs at control '+str(i))
        command=np.zeros((n,3));command[:,0]=.005*(enabled & (np.arange(n)%4!=0) & (age<1200))
        require(np.array_equal(command,trace['requested_command'][i]),'Observed command schedule differs at control '+str(i))
        require(not (timeout[i] & (age!=2199)).any(),'Timeout did not use final active control 2200')
        age[enabled]+=1;warm=np.maximum(warm-1,0)
        if ended[i].any():
            expected_events.append((i,np.flatnonzero(ended[i]),expected.copy()))
            expected[ended[i]]+=1;age[ended[i]]=0;warm[ended[i]]=200
    require(len(events)==len(expected_events),'Reset event ledger count differs')
    for event,(i,ids,old) in zip(events,expected_events):
        require(event['control']==i+1 and event['ended_rows']==ids.tolist() and event['episodes']==old.tolist(),
                'Reset event identity/time differs')
        require(event['reset_kind']=='original_inherited_reset_only_before_next_action','Unknown physical reset mechanism')
        q=np.asarray(event['reset_reference_joint_target_rad'])
        require(q.shape==(len(ids),18) and np.isfinite(q).all(),'Incomplete reset target evidence')
    require(np.array_equal(raw['episode_id'][1:].reshape(count,8,n),np.repeat(episode[:,None,:],8,axis=1)),
            'Substeps pooled across wrong episode')
    crossing=np.r_[np.zeros((1,n),dtype=bool),np.diff(raw['episode_id'],axis=0)!=0]
    require(np.array_equal(raw['crosses_episode_reset'],crossing),'Reset-crossing interval mask differs')
    saturation,physical=substeps(trace,raw)
    nonfoot=trace['base_contact']|trace['shaft_contact'].any(-1)|trace['coxa_contact'].any(-1)|trace['femur_contact'].any(-1)
    required=np.where((trace['requested_command']!=0).any(-1)|(reference['current_leg']>=0),5,6)
    physical_fail=trace['terminated']|(active&(saturation|nonfoot|(trace['distal_contact'].sum(-1)<required)))
    require(np.array_equal(term,active&(physical_fail|trace['next_reference_failure'])),'Recorded physical/reference terminal cause differs')
    require(np.array_equal(trace['reward_scorable'],active&~physical_fail),'Ordinary reward scoring mask differs')
    ordinary=trace['reward_before_event'];expected_reward=np.where(active,ordinary,0)-3*term.astype(ordinary.dtype)
    require(np.array_equal(trace['reward'],expected_reward),'Event reward absent/double charged or recovery rewarded')
    require(not ordinary[~trace['reward_scorable']].any(),'Non-scorable ordinary reward is nonzero')
    for key,value in trace.items():
        if key.startswith('reward_component__'):
            require(np.isfinite(value).all() and not value[~trace['reward_scorable']].any(),'Effective reward component mask differs: '+key)
    for i,report in enumerate(updates):
        lo=200+i*256;hi=lo+256
        require(hi<=count,'Update claims unrecorded physical transitions')
        expected_counts={'valid_learning_transitions':int(active[lo:hi].sum()),
            'finite_terminal_transitions':int(term[lo:hi].sum()),'time_limit_transitions':int(timeout[lo:hi].sum()),
            'recovery_transitions_excluded':int((~active[lo:hi]).sum())}
        require(all(report[k]==v for k,v in expected_counts.items()),'Actual PPO learning-count report disagrees with raw window')
        require(report['controls_per_replica']==256,'Wrong PPO rollout length')
    physical.update(clocks(clock,ended,resets))
    physical.update(active_transitions=int(active[200:].sum()),recovery_transitions_excluded=int((~active[200:]).sum()),
        finite_terminal_transitions=int(term.sum()),time_limit_transitions=int(timeout.sum()),
        requested_spikes_terminating_active_rows=int((active&saturation).sum()),
        PPO_updates_replayed=len(updates),ended_episodes_per_environment=ended.sum(0).tolist(),
        actor_packet_history_directly_exported=False,optimizer_final_bootstrap_directly_exported=False,
        unexported_state_scope='History/reset and final-pre-reset bootstrap are CPU-tested; no direct GPU tensor replay claim',
        physical_admission=False,Stage2_complete=False)
    return physical

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();require(not args.out.exists(),'Audit output must be new')
    state=json.loads((args.phase/'state.json').read_text());base=args.phase/'raw'
    result={'scope':'independent moving raw schema audit; not physical admission','integrity_passed':False,
            'phase_status':state.get('status'),'consumer_sha256':CONSUMER,'Stage2_complete':False}
    files=['state.json','raw/trace.npz','raw/physics_substeps.npz','raw/sensor_clocks.npz','raw/reference_states.npz','raw/episodes.json']
    try:
        require(state['identity']['consumer_freeze_sha256']==CONSUMER,'Wrong frozen consumer')
        require(state['mode'] in ('profile_32','profile_128','train_10','train_25'),'Wrong moving mode')
        updates=[json.loads(line) for line in (args.phase/'updates.jsonl').read_text().splitlines()] if (args.phase/'updates.jsonl').exists() else []
        reset=load(base/'reset_clocks.npz') if (base/'reset_clocks.npz').exists() else None
        result['replay']=moving(load(base/'trace.npz'),load(base/'physics_substeps.npz'),load(base/'sensor_clocks.npz'),
            load(base/'reference_states.npz'),json.loads((base/'episodes.json').read_text()),updates,reset)
        result['integrity_passed']=True
    except Exception as exc: result['error']=repr(exc)
    result['input_sha256']={name:sha(args.phase/name) for name in files if (args.phase/name).is_file()}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','replay')}))
    return 0 if result['integrity_passed'] else 1

if __name__=='__main__':raise SystemExit(main())
