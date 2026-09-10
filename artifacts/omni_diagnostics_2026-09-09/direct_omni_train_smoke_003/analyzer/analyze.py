"""Matched direct315 evidence review. No dispatch, actor, simulator or gate mutation."""
import argparse, hashlib, json, math, re
from pathlib import Path
import numpy as np
from optimizer_summary import summarize as summarize_optimizer, human as optimizer_human

NATIVE_FREEZE='1ab678beeba6bf74978c210c4e38f13655e25728fe66e3007e3b1db5e19b6b20'
ORIGINAL='1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'
SCHEMA='direct315_quiet_priority_native_v3'
SOURCE='ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62'
PLAN='eee25089a65ec112f5eeed9c70f6310ef019ca5f077618c736fcfaa859739fc2'
QUIET_GATES={'max_planar_excursion_m':.01,'max_heading_excursion_deg':2.,'max_joint_velocity_rms_rad_s':.03,
             'max_joint_position_range_rad':.02,'max_target_step_abs_p95_rad_per_20ms':.002,
             'max_requested_torque_saturation_fraction':.005,'max_applied_torque_nm':1.60001}
CONSTANT_METRICS=('planar_error_mps','yaw_error_rad_s','finite_difference_planar_error_mps',
                  'torque_saturation_fraction','computed_torque_abs_max_nm','applied_torque_abs_max_nm','tilt_rms_deg')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

class Inputs:
    def __init__(self):self.hashes={}
    def pin(self,path):
        path=Path(path).resolve();value=sha(path)
        if str(path) in self.hashes and self.hashes[str(path)]!=value:raise ValueError('Input changed during review: '+str(path))
        self.hashes[str(path)]=value;return path
    def json(self,path):return json.loads(self.pin(path).read_text())
    def text(self,path):return self.pin(path).read_text()
    def npz(self,path):
        with np.load(self.pin(path),allow_pickle=False) as z:
            data={k:z[k] for k in z.files}
        for key,value in data.items():
            if value.dtype.kind not in 'US' and not np.isfinite(value).all():raise ValueError('Nonfinite raw '+key)
        return data
    def unchanged(self):
        for path,bound in self.hashes.items():
            if sha(path)!=bound:raise ValueError('Reviewed input changed: '+path)

def require(condition,message):
    if not condition:raise ValueError(message)

def close(actual,expected,label):
    require(np.allclose(actual,expected,rtol=2e-5,atol=2e-6,equal_nan=False),'Raw/report mismatch: '+label)

def cadence(data,count):
    t=data['time_s'];require(t.shape==(count,),'Wrong trace control count')
    close(t,(np.arange(count)+1)*.02,'50Hz time axis')

def named_rows(report):
    rows=report['scenarios'];names=[r['name'] for r in rows]
    require(len(names)==12 and len(set(names))==12,'Twelve distinct directions required')
    return {r['name']:r for r in rows}

def trace_rates(data,column,start=100):
    q=data['joint_position_rad'][start:,column];v=data['joint_velocity_rad_s'][start:,column]
    target=data['joint_target_rad'][start:,column];terminal=data['terminated'][:,column]|data['truncated'][:,column]
    age=data['age_s'][:,column]
    # Adjacent samples crossing a reset are excluded only from derivative evidence.
    # All event counts and acceptance failures remain in the report.
    valid=~terminal[start:-1] & (age[start+1:]>age[start:-1])
    delta=np.diff(q.astype(np.float64),axis=0)/.02
    td=np.diff(target.astype(np.float64),axis=0)
    result={'traced_env_id':int(data.get('trace_env_ids',np.arange(data['command'].shape[1]))[column]),
            'samples':len(q),'valid_interval_samples':int(valid.sum()),
            'reported_joint_rms_rad_s':np.sqrt(np.mean(v*v,axis=0)).tolist(),
            'joint_position_range_rad':np.ptp(q,axis=0).tolist()}
    if valid.any():
        result.update(interval_angle_rms_rad_s=np.sqrt(np.mean(delta[valid]**2,axis=0)).tolist(),
                      target_step_rms_rad=np.sqrt(np.mean(td[valid]**2,axis=0)).tolist(),
                      target_step_p95_abs_rad=np.quantile(np.abs(td[valid]),.95,axis=0).tolist(),
                      raw_endpoint_minus_interval_rms_rad_s=np.sqrt(np.mean((v[1:][valid]-delta[valid])**2,axis=0)).tolist())
    return result

def constant(inputs,directory):
    directory=Path(directory);d=inputs.json(directory/'diagnostics.json');raw=inputs.npz(directory/'diagnostic_trace.npz')
    require(d['complete'] and d['kind']=='diagnostic_not_qualification','Incomplete constant diagnostic')
    require(d['control_sample_dt_s']==.02 and d['physics_dt_s']==.0025,'Changed constant timing')
    require(d['options']=={'duration_s':12,'settle_s':2,'seed':7057,'trace_envs_per_scenario':1,'controller':'policy'},'Constant protocol differs')
    require(d['overrides']['target_slew_rad_per_20ms']==.04,'Historical .03 must remain a separate comparison')
    cadence(raw,600);require(raw['command'].shape==(600,12,3),'Constant raw trace is one of four replicas per case, not all48')
    require(raw['joint_names'].tolist()==d['joint_names'],'Runtime joint order differs')
    cases=[]
    for index,(name,row) in enumerate(named_rows(d).items()):
        require(row['replicas']==4,'Constant summary must contain four replicas')
        window=row['windows']['post_settle'];all_window=row['windows']['all']
        # Four-replica reported metrics cannot be reconstructed from the one-replica trace.
        metrics={key:window[key] for key in CONSTANT_METRICS}
        require(all(math.isfinite(x) for x in metrics.values()),'Nonfinite constant aggregate')
        metrics.update(mean_velocity_mps=window['mean_velocity_mps'],mean_gyro_rad_s=window['mean_gyro_rad_s'],
                       mean_nonfoot_body_count=window.get('mean_raw_reward_terms',{}).get('nonfoot'),
                       reported_joint_velocity_rms_max_rad_s=max(j['velocity_rms_rad_s'] for j in window['joints'].values()))
        cases.append({'name':name,'command':row['command'],'aggregate_replicas':4,'metrics':metrics,
                      'joints':window['joints'],'terminations':row['terminations'],'truncations':row['truncations'],
                      'termination_reasons':row['termination_reasons'],'all_window_applied_peak_nm':all_window['applied_torque_abs_max_nm'],
                      'sampled_replica_rates':trace_rates(raw,index),
                      'requested_saturation_above_existing_0_005_bound':metrics['torque_saturation_fraction']>.005})
    return {'checkpoint_sha256':d['checkpoint_sha256'],'options':d['options'],'overrides':d['overrides'],
            'reward_weights':d['reward_weights'],'joint_names':d['joint_names'],'cases':cases,
            'scope':'Four-replica direction summaries plus one traced replica; never fabricate unrecorded replica data or full formal admission'}

def compare_constant(before,after):
    for key in ('options','overrides','reward_weights','joint_names'):
        require(before[key]==after[key],'Constant comparison confound: '+key)
    b={x['name']:x for x in before['cases']};a={x['name']:x for x in after['cases']}
    require(set(a)==set(b),'Direction sets differ')
    result=[]
    for name in b:
        require(b[name]['command']==a[name]['command'],'Command differs: '+name)
        delta={k:a[name]['metrics'][k]-b[name]['metrics'][k] for k in CONSTANT_METRICS}
        result.append({'name':name,'command':b[name]['command'],'before':b[name]['metrics'],'after':a[name]['metrics'],
                       'delta_after_minus_before':delta,
                       'termination_delta':a[name]['terminations']-b[name]['terminations'],
                       'truncation_delta':a[name]['truncations']-b[name]['truncations'],
                       'directional_observations':{'planar_error_lower':delta['planar_error_mps']<0,'yaw_error_lower':delta['yaw_error_rad_s']<0,
                           'requested_saturation_lower':delta['torque_saturation_fraction']<0},
                       'note':'Sign of measured change only, not a statistical or acceptance threshold'})
    return result

def recompute_quiet(raw,index):
    start=1100;q=raw['joint_position_rad'][start:,index];v=raw['joint_velocity_rad_s'][start:,index]
    target=raw['joint_target_rad'][start:,index];p=raw['position_world_m'][start:,index]
    xyzw=raw['quaternion_world_xyzw'][start:,index]
    close(raw['quaternion_world_wxyz'][start:,index],xyzw[:,[3,0,1,2]],'explicit quaternion conversion')
    x,y,z,w=xyzw.T;heading=np.unwrap(np.arctan2(-1+2*(x*x+z*z),2*(w*z-x*y)))
    requested=raw['computed_torque_nm'][start:,index];applied=raw['applied_torque_nm'][start:,index]
    rms=np.sqrt(np.mean(v*v,axis=0));target95=np.quantile(np.abs(np.diff(target,axis=0)),.95,axis=0)
    metrics={'window_samples':len(q),'window_duration_s':len(q)*.02,
             'max_planar_excursion_m':float(np.linalg.norm(p[:,:2]-p[0,:2],axis=-1).max()),
             'max_heading_excursion_deg':float(np.degrees(np.abs(heading-heading[0]).max())),
             'max_joint_velocity_rms_rad_s':float(rms.max()),'max_joint_position_range_rad':float(np.ptp(q,axis=0).max()),
             'max_target_step_abs_p95_rad_per_20ms':float(target95.max()),
             'max_requested_torque_saturation_fraction':float((np.abs(requested)>1.6).mean(0).max()),
             'max_applied_torque_nm':float(np.abs(applied).max()),
             'terminations':int(raw['terminated'][:,index].sum()),'truncations':int(raw['truncated'][:,index].sum())}
    metrics['failed_bounds']=[k for k,bound in QUIET_GATES.items() if metrics[k]>bound]
    metrics['pass']=not metrics['failed_bounds'] and metrics['terminations']==0 and metrics['truncations']==0
    return metrics

def stop(inputs,directory):
    directory=Path(directory);d=inputs.json(directory/'stop_diagnostics.json');raw=inputs.npz(directory/'stop_trace.npz')
    require(d.get('complete') is True and d['kind']=='direct315_move_to_zero_diagnostic_v1','Incomplete stop diagnostic')
    require(d['quiet_gates']==QUIET_GATES,'Quiet gates changed')
    require(d['controls']==1600 and d['replicas']==48 and raw['command'].shape==(1600,48,3),'Wrong stop allocation')
    cadence(raw,1600);names=raw['joint_names'].tolist();cases=[]
    for i,(name,row) in enumerate(named_rows(d).items()):
        require(len(row['replicas'])==4,'Missing stop replica')
        for j,replica in enumerate(row['replicas']):
            index=i*4+j;require(replica['env_id']==index,'Stop replica mapping changed')
            target=np.zeros((1600,3));target[200:800]=row['command'];close(raw['target_command'][:,index],target,'explicit move-to-stop target')
            quiet=recompute_quiet(raw,index)
            for key in QUIET_GATES:close(quiet[key],replica['quiet'][key],'quiet '+key)
            for key in ('pass','failed_bounds','terminations','truncations'):
                require(quiet[key]==replica['quiet'][key],'Quiet verdict differs: '+key)
            require(np.all(raw['command'][1100:,index]==0),'Scored quiet window still commanded to move')
            rates=trace_rates(raw,index,1100)
            nonfoot=raw.get('reward_term_nonfoot')
            cases.append({'name':name,'replica':j,'env_id':index,'command':row['command'],'quiet':quiet,
                          'motion':replica['motion'],'quiet_joints':replica['quiet']['joints'],'rate_evidence':rates,
                          'nonfoot_env_steps_all':int((nonfoot[:,index]>0).sum()) if nonfoot is not None else None,
                          'all_applied_peak_nm':float(np.abs(raw['applied_torque_nm'][:,index]).max()),
                          'all_requested_peak_nm':float(np.abs(raw['computed_torque_nm'][:,index]).max())})
    return {'checkpoint_sha256':d['checkpoint_sha256'],'overrides':d['overrides'],'gates':QUIET_GATES,'cases':cases,
            'quiet_passed_replicas':sum(r['quiet']['pass'] for r in cases),'total_replicas':48,
            'scope':'Unchanged10 s quiet bounds independently replayed; any trial reset invalidates its replica'}

def compare_stop(before,after):
    require(before['gates']==after['gates'] and before['overrides']==after['overrides'],'Stop comparison profile differs')
    require([(r['name'],r['replica'],r['command']) for r in before['cases']]==[(r['name'],r['replica'],r['command']) for r in after['cases']],'Stop case alignment differs')
    return [{'name':b['name'],'replica':b['replica'],'env_id':b['env_id'],
             'before_pass':b['quiet']['pass'],'after_pass':a['quiet']['pass'],
             'before_failed_bounds':b['quiet']['failed_bounds'],'after_failed_bounds':a['quiet']['failed_bounds'],
             'delta_after_minus_before':{k:a['quiet'][k]-b['quiet'][k] for k in QUIET_GATES},
             'termination_delta':a['quiet']['terminations']-b['quiet']['terminations']} for b,a in zip(before['cases'],after['cases'])]

def parse_timings(text):
    clean=re.sub(r'\x1b\[[0-9;]*m','',text)
    return {'collection_seconds':[float(x) for x in re.findall(r'Collection time:\s*([\d.]+)s',clean)],
            'learning_seconds':[float(x) for x in re.findall(r'Learning time:\s*([\d.]+)s',clean)],
            'reported_steps_per_second':[float(x) for x in re.findall(r'Steps per second:\s*([\d.]+)',clean)]}

def training(inputs,directory,log=None):
    directory=Path(directory);receipt=inputs.json(directory/'training_receipt.json');state=inputs.json(directory/'state.json')
    raw=inputs.npz(directory/'training_trace.npz');audit=receipt['audit'];n=audit['replicas'];t=audit['controls'];cadence(raw,t)
    require(raw['command'].shape==(t,n,3),'Training raw allocation mismatch')
    for filename,key in [('training_trace.npz','trace_sha256'),('training_joint_trace.npz','joint_trace_sha256'),('training_events.json','event_ledger_sha256')]:
        require(sha(inputs.pin(directory/filename))==audit[key],'Training raw receipt hash differs')
    close(raw['terminated'].sum(0),audit['terminations_per_row'],'termination counts')
    close(raw['truncated'].sum(0),audit['truncations_per_row'],'timeout counts')
    close(raw['requested_torque_abs_max_nm'].max(0),audit['requested_torque_max_per_row_nm'],'per-row requested torque')
    close(raw['applied_torque_abs_max_nm'].max(0),audit['applied_torque_max_per_row_nm'],'per-row applied torque')
    close(raw['requested_saturation_fraction'].mean(0),audit['requested_saturation_fraction_per_row'],'per-row requested saturation')
    events=inputs.json(directory/'training_events.json');expected=np.argwhere(raw['terminated']|raw['truncated'])
    actual=[]
    for event in events:
        control=event['control']-1
        for local,index in enumerate(event['ids']):
            actual.append((control,index))
            for key in ('command','terminated','truncated','requested_torque_abs_max_nm','nonfoot'):
                close(event['fields'][key][local],raw[key][control,index],'event '+key)
    require(sorted(actual)==sorted(map(tuple,expected)),'Event ledger does not cover exact pre-reset rows')
    updates=receipt.get('updates_completed',0);optimizer=receipt.get('optimizer_updates',[])
    require(len(optimizer)==updates and [x['completed_update'] for x in optimizer]==list(range(1,updates+1)),'Actual optimizer update counter differs')
    complete=receipt.get('complete') is True
    if complete:
        require(receipt==state.get('training_receipt'),'State and final training receipt differ')
        require(t==updates*24,'Completed training control budget differs')
        cp=inputs.pin(directory/'policy/final.pt');require(sha(cp)==receipt['final_checkpoint_sha256']==receipt['reload']['checkpoint_sha256']==state['checkpoint_sha256'],'Final checkpoint identity differs')
        require(all(receipt['reload'].get(k) is True for k in ('passed','exact_actor_critic_normalizer_optimizer','exact_deterministic_action')),'Strict reload not passed')
        for row in receipt['decision_checkpoints'].values():require(sha(inputs.pin(directory/'policy'/row['file']))==row['sha256'],'Decision checkpoint changed')
    rows=[]
    for i in range(n):
        zero=np.all(raw['command'][:,i]==0,axis=-1)
        rows.append({'env_id':i,'controls':t,'zero_command_controls':int(zero.sum()),'moving_command_controls':int((~zero).sum()),
                     'terminations':int(raw['terminated'][:,i].sum()),'truncations':int(raw['truncated'][:,i].sum()),
                     'nonfoot_env_steps':int(raw['nonfoot'][:,i].sum()),'requested_peak_nm':float(raw['requested_torque_abs_max_nm'][:,i].max()),
                     'applied_peak_nm':float(raw['applied_torque_abs_max_nm'][:,i].max()),'requested_saturation_fraction':float(raw['requested_saturation_fraction'][:,i].mean()),
                     'zero_command_reported_joint_rms_rad_s':float(np.sqrt(np.mean(raw['reported_joint_velocity_rms_rad_s'][zero,i]**2))) if zero.any() else None,
                     'zero_command_target_delta_rms_rad':float(np.sqrt(np.mean(raw['target_delta_rms_rad'][zero,i]**2))) if zero.any() else None})
    wall=receipt['wall_seconds'];require(math.isfinite(wall) and wall>0,'Invalid measured wall time')
    timings=parse_timings(inputs.text(log)) if log is not None and Path(log).is_file() else None
    return {'complete':complete,'selection':receipt['selection'],'updates_completed':updates,'controls':t,'replicas':n,
            'transitions':t*n,'training_wrapper_wall_seconds':wall,'wrapper_transitions_per_second':t*n/wall,
            'wall_scope':'learn plus final reload/decision copy; actual RSL collection and learning timings listed separately',
            'native_rsl_timings':timings,'optimizer_updates':optimizer,'strict_reload':receipt.get('reload'),
            'checkpoint_sha256':receipt.get('final_checkpoint_sha256'),'decision_checkpoints':receipt.get('decision_checkpoints'),
            'initialization':inputs.json(directory/'repair_initialization.json'),'learned_std':receipt.get('learned_std'),
            'cuda_peak_allocated_bytes':receipt.get('cuda_peak_allocated_bytes'),'cuda_peak_reserved_bytes':receipt.get('cuda_peak_reserved_bytes'),
            'per_row':rows,'scope':'Raw finite episode failures remain failures; 50Hz endpoint telemetry is not substep torque proof'}

def human(report):
    lines=['# Direct PPO comparison','',report['conclusion'],'',
           'Acquisition completion is not physical acceptance. Every result retains the original 1.6 Nm cap and quiet bounds; no threshold or statistical significance is invented.','']
    if report.get('training'):
        t=report['training'];lines += [f"Training completed {t['updates_completed']} updates, {t['controls']} controls × {t['replicas']} replicas ({t['transitions']:,} transitions). Wrapper time {t['training_wrapper_wall_seconds']:.2f} s; {t['wrapper_transitions_per_second']:,.0f} transitions/s. Strict reload reported {bool((t.get('strict_reload') or {}).get('passed'))}.",'']
        rows=t['per_row'];lines += [f"Training events: {sum(x['terminations'] for x in rows)} terminations, {sum(x['truncations'] for x in rows)} timeouts, {sum(x['nonfoot_env_steps'] for x in rows)} nonfoot environment-steps. Requested peak {max(x['requested_peak_nm'] for x in rows):.3f} Nm; applied peak {max(x['applied_peak_nm'] for x in rows):.6f} Nm.",'']
    if report.get('training_receipt_forensics'):
        f=report['training_receipt_forensics'];lines += [f"Receipt-only evidence: producer reports {f.get('updates_completed')} updates, complete={f.get('complete')}, strict reload={f.get('reload_passed')}. Raw/checkpoint verification remains separate.",'']
    if report.get('optimizer_diagnostics') is not None:lines += optimizer_human(report['optimizer_diagnostics'])+['']
    comparison=report.get('constant_comparison')
    if comparison:
        lines += ['Constant directions use four-replica summaries. Joint-rate traces cover one replica per direction; those trace values are never represented as all four.','',
                  '| Direction | Planar error before → after (m/s) | Yaw error before → after (rad/s) | Requested saturation before → after | Requested peak after (Nm) | Δterm |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in comparison:
            b,a=r['before'],r['after'];lines.append(f"| {r['name']} | {b['planar_error_mps']:.4f} → {a['planar_error_mps']:.4f} | {b['yaw_error_rad_s']:.4f} → {a['yaw_error_rad_s']:.4f} | {100*b['torque_saturation_fraction']:.2f}% → {100*a['torque_saturation_fraction']:.2f}% | {a['computed_torque_abs_max_nm']:.3f} | {r['termination_delta']:+d} |")
        worse={k:[r['name'] for r in comparison if r['delta_after_minus_before'][k]>0] for k in ['planar_error_mps','yaw_error_rad_s','torque_saturation_fraction']}
        lines += ['', 'Observed regressions (any positive difference, not a significance test):']
        for key,names in worse.items():lines.append('- '+key+': '+(', '.join(names) if names else 'none')+'.')
    final=report.get('final_stop')
    if final:
        lines += ['',f"Final stop/quiet: {final['quiet_passed_replicas']}/48 replicas passed the unchanged ten-second quiet checks.",'',
                  '| Direction | Passing replicas | Failed bounds across replicas | Worst raw joint RMS (rad/s) | Worst drift (mm) |',
                  '|---|---:|---|---:|---:|']
        for name in dict.fromkeys(x['name'] for x in final['cases']):
            rows=[x for x in final['cases'] if x['name']==name];bounds=sorted({b for r in rows for b in r['quiet']['failed_bounds']})
            if any(r['quiet']['terminations'] or r['quiet']['truncations'] for r in rows):bounds.append('trial reset')
            lines.append(f"| {name} | {sum(r['quiet']['pass'] for r in rows)}/4 | {', '.join(bounds) or 'none'} | {max(r['quiet']['max_joint_velocity_rms_rad_s'] for r in rows):.4f} | {1000*max(r['quiet']['max_planar_excursion_m'] for r in rows):.2f} |")
        lines += ['', 'All 48 stop replicas, named-joint values and before/after deltas are in report.json. Raw SDK rates and adjacent-angle interval averages remain separate; neither is substituted into the other’s score.']
    if report.get('stop_baseline_available') is False:lines += ['', 'No historical moving-to-stop baseline exists for the cold constant run. The final stop result is an absolute screen, not an improvement claim.']
    if report['errors']:lines += ['', 'Evidence problems:']+['- '+x for x in report['errors']]
    lines += ['', 'Next decision: '+report['next_decision'],'','Input paths and SHA256 values are recorded in INPUTS_SHA256.json and rechecked after analysis. No source, checkpoint, input or GPU job is modified.']
    return '\n'.join(lines)+'\n'

def analyze(campaign,cold,output):
    campaign,cold,output=map(lambda p:Path(p).resolve(),(campaign,cold,output))
    require(not output.exists() and campaign not in output.parents and cold not in output.parents,'Fresh output outside immutable inputs required')
    inputs=Inputs()
    report={'schema':'direct315_quiet_priority_matched_review_v2','native_contract_freeze_sha256':NATIVE_FREEZE,
            'source_manifest_sha256':SOURCE,'plan_sha256':PLAN,
            'errors':[],'Stage2_complete':False,'automatic_continuation':False,'stop_baseline_available':False,
            'phase_receipts':{},'scope':'Read-only numerical and forensic review, not host admission or physical qualification'}
    def attempt(name,fn):
        try:
            value=fn();report[name]=value;return value
        except Exception as exc:
            report['errors'].append(name+': '+repr(exc));return None
    def check(label,fn):
        try:fn();return True
        except Exception as exc:report['errors'].append(label+': '+repr(exc));return False
    def object_input(path):
        value=inputs.json(path);require(isinstance(value,dict),'Expected JSON object');return value
    c=attempt('campaign',lambda:object_input(campaign/'campaign.json')) or {}
    old=attempt('cold_campaign',lambda:object_input(cold/'campaign.json')) or {}
    identity=c.get('identity') if isinstance(c.get('identity'),dict) else {}
    selected=identity.get('selection') if isinstance(identity.get('selection'),dict) else {}
    allocation=c.get('allocation');branch=c.get('branch')
    def campaign_identity():
        require(identity.get('schema')==SCHEMA,'Wrong/missing native campaign schema')
        require(identity.get('source_manifest_sha256')==SOURCE and identity.get('plan_sha256')==PLAN,'Unbound native source/plan')
        require(len(SOURCE)==64 and len(PLAN)==64,'Analyzer source binding is not finalized')
        require(identity.get('checkpoint_sha256')==ORIGINAL and identity.get('actor_width')==315 and identity.get('critic_width')==318,'Original checkpoint or observation contract differs')
        require(allocation in ('smoke','pilot') and branch in ('quiet_priority','caps','curriculum'),'Undeclared branch/allocation')
        require(allocation!='smoke' or branch=='quiet_priority','This source requires its own quiet-priority smoke')
        expected={'schema':SCHEMA,'allocation':allocation,'branch':branch,
            'replicas':32 if allocation=='smoke' else 1024,'controls_per_update':24,'updates':2 if allocation=='smoke' else 50,
            'caps':{'temporal_weight':.1 if branch!='curriculum' else 0.,'spatial_weight':.1 if branch!='curriculum' else 0.,
                    'noise_scale':1.,'noise_seed':1157,'quiet_temporal_weight':1.0 if branch=='quiet_priority' else (.1 if branch=='caps' else 0.)}}
        require(selected==expected,'Native training selection differs')
        if allocation=='pilot':
            require(all(isinstance(identity.get(k),str) and re.fullmatch('[0-9a-f]{64}',identity[k]) for k in ('smoke_campaign_sha256','smoke_state_sha256','smoke_receipt_sha256')),'Missing same-source smoke provenance hashes')
    check('campaign identity',campaign_identity)
    def cold_identity():
        require(old.get('identity',{}).get('source_manifest_sha256')=='4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b','Unbound cold004 source')
        require(old.get('status')=='completed' and old.get('terminal_inputs_unchanged') is True,'Cold baseline integrity incomplete')
    check('cold baseline identity',cold_identity)
    expected_phases=(['standing','train','final_constant','final_stop'] if allocation=='smoke' else
                     ['standing','initial_constant','initial_stop','train','final_constant','final_stop'] if allocation=='pilot' else [])
    check('phase allocation',lambda:require(bool(expected_phases) and c.get('planned_phases')==expected_phases,'Campaign phase allocation differs'))
    for phase in expected_phases:
        def phase_identity():
            state=object_input(campaign/phase/'state.json');job=object_input(campaign/'jobs'/(phase+'.json'))
            report['phase_receipts'][phase]={'state_status':state.get('status'),'job_status':job.get('status'),'cleanup_checked':job.get('cleanup_checked'),'exit_code':job.get('exit_code')}
            require(state.get('plan_sha256')==PLAN and state.get('urdf_sha256')=='e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c','Phase asset/plan identity differs')
            binding=state.get('runtime_binding') or {}
            require(binding.get('source_manifest_sha256')==SOURCE and binding.get('runtime_tree_sha256')=='abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280','Phase runtime identity differs')
            if c.get('status')=='completed':
                require(state.get('status')=='completed' and job.get('status')=='completed' and job.get('exit_code')==0 and job.get('cleanup_checked') is True,'Completed campaign contains an incomplete/unclean phase')
                accepted=c.get('accepted_phases',{}).get(phase,{})
                require(accepted.get('state_sha256')==sha(campaign/phase/'state.json'),'Accepted phase state changed')
        check(phase+' identity/ownership receipt',phase_identity)
    receipt=attempt('training_receipt_raw',lambda:object_input(campaign/'train/training_receipt.json'))
    if receipt is not None:
        # Retain reported counts even if missing raw files prevent independent replay.
        reload=receipt.get('reload') or {}
        report['training_receipt_forensics']={
            'complete':receipt.get('complete'),'updates_completed':receipt.get('updates_completed'),
            'reload_passed':reload.get('passed') if isinstance(reload,dict) else None,
            'final_checkpoint_sha256':receipt.get('final_checkpoint_sha256'),'error':receipt.get('error'),'audit_error':receipt.get('audit_error'),
            'scope':'Producer-reported values only; no independent raw/checkpoint verification implied'}
        check('receipt selection',lambda:require(receipt.get('selection')==selected,'Receipt and campaign selections differ'))
        diag=attempt('optimizer_diagnostics',lambda:summarize_optimizer(receipt))
        if diag is not None and diag['errors']:
            report['errors'].append('Optimizer diagnostic inventory has '+str(len(diag['errors']))+' evidence gaps; see optimizer_diagnostics.errors')
        attempt('training',lambda:training(inputs,campaign/'train',campaign/'logs/train.log'))
        if c.get('status')=='completed':
            check('completed training receipt',lambda:require(receipt.get('complete') is True and report.get('training',{}).get('complete') is True,'Completed campaign lacks fully verified training'))
    else:report['errors'].append('No usable training receipt; no completed updates inferred')
    # The raw receipt remains an immutable input, not duplicated into a large report.
    report.pop('training_receipt_raw',None)
    if (campaign/'train/failure.json').is_file():attempt('runtime_failure',lambda:inputs.json(campaign/'train/failure.json'))
    baseline=campaign/'initial_constant' if (campaign/'initial_constant/diagnostics.json').is_file() else cold/'baseline'
    report['constant_baseline_kind']='same_pilot_initial' if baseline.parent==campaign else 'historical_cold_formal004'
    attempt('initial_constant',lambda:constant(inputs,baseline))
    if (campaign/'final_constant/diagnostics.json').is_file():attempt('final_constant',lambda:constant(inputs,campaign/'final_constant'))
    if report.get('initial_constant') and report.get('final_constant'):attempt('constant_comparison',lambda:compare_constant(report['initial_constant'],report['final_constant']))
    if (campaign/'initial_stop/stop_diagnostics.json').is_file():
        attempt('initial_stop',lambda:stop(inputs,campaign/'initial_stop'));report['stop_baseline_available']=True
    if (campaign/'final_stop/stop_diagnostics.json').is_file():attempt('final_stop',lambda:stop(inputs,campaign/'final_stop'))
    if report.get('initial_stop') and report.get('final_stop'):attempt('stop_comparison',lambda:compare_stop(report['initial_stop'],report['final_stop']))
    if c.get('status')!='completed' or c.get('terminal_inputs_unchanged') is not True:report['errors'].append('Campaign did not complete terminal input integrity; no continuation admission')
    if c.get('status')=='completed':check('accepted phase inventory',lambda:require(set(c.get('accepted_phases',{}))==set(expected_phases),'Accepted phase inventory differs'))
    if not report.get('final_constant') or not report.get('final_stop'):report['errors'].append('Matched final constant and stop diagnostics are incomplete or unverified')
    train=report.get('training')
    if train and train.get('checkpoint_sha256'):
        for key in ['final_constant','final_stop']:
            if key in report and report[key]['checkpoint_sha256']!=train['checkpoint_sha256']:report['errors'].append(key+' checkpoint does not match trained final')
    if 'initial_constant' in report and report['initial_constant']['checkpoint_sha256']!=ORIGINAL:report['errors'].append('Initial diagnostic did not use original checkpoint')
    # Never serialize NaN/Infinity as apparently valid evidence. Preserve the raw
    # input hash, replace only the report copy, and record every affected field.
    nonfinite_fields=[]
    def clean(value,path='report'):
        if isinstance(value,float) and not math.isfinite(value):
            nonfinite_fields.append('Nonfinite report field replaced with null: '+path);return None
        if isinstance(value,dict):return {k:clean(v,path+'.'+str(k)) for k,v in value.items()}
        if isinstance(value,list):return [clean(v,path+'['+str(i)+']') for i,v in enumerate(value)]
        return value
    report=clean(report)
    report['errors'].extend(nonfinite_fields)
    check_result=True
    try:inputs.unchanged()
    except Exception as exc:
        check_result=False;report['errors'].append('Input immutability: '+repr(exc))
    report['reviewed_inputs_unchanged']=check_result
    report['evidence_verified']=not report['errors']
    if report['errors']:
        report['conclusion']='This campaign has incomplete or inconsistent evidence; preserved partial results are diagnostic only.'
        report['next_decision']='Resolve the listed infrastructure/evidence failure before allocating a successor; no automatic training or promotion.'
    elif allocation=='smoke':
        report['conclusion']='The two-update integration smoke completed. It establishes execution/reload evidence, not convergence or smooth walking.'
        report['next_decision']='Review every directional and quiet failure before separately dispatching the bounded quiet-priority pilot. Existing failures remain unqualified.'
    else:
        report['conclusion']='The bounded pilot completed. Physical changes and regressions remain separate from optimizer loss and sparse gradients.'
        report['next_decision']='Compare the matched constant and stop evidence with the preserved pilots. Losses, gradient alignment and learning-rate trends alone cannot promote or extend training.'
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (output/'REPORT.md').write_text(human(report))
    (output/'INPUTS_SHA256.json').write_text(json.dumps(inputs.hashes,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False)
    for key in ('campaign','cold','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=analyze(a.campaign,a.cold,a.output)
    print(json.dumps({'conclusion':r['conclusion'],'updates':(r.get('training') or {}).get('updates_completed'),'errors':r['errors'],'report':str(a.output/'REPORT.md')},indent=2))
