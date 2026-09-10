"""Host-safe standard-library admission/result contract; no simulator or NumPy imports."""
import hashlib, json, math
from pathlib import Path
from direct_config import ALLOCATIONS, CHECKPOINT, SCHEMA, SMOKE_BRANCH, protocol, selection

PLAN='robot/hexapod_mkii_length_study/training_plan.json'
PARENT='4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b'
URDF='e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c'
PHASES=('standing','initial_constant','initial_stop','train','final_constant','final_stop')
EXPECTED_OVERRIDES={'target_slew_rad_per_20ms':.04,'reward_weights':{'stand_joint_velocity':-.5,'stand_target_velocity':-.15,'stand_posture':-2.,'action_rate':-.075,'saturation':-.75,'torque_excess':-.6,'worst_torque_excess':-.2,'stand_raw_action':0.},'observation_noise_scale':1.,'target_filter_time_constant_s':0.}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def finite(x):
    if type(x) not in (int,float) or not math.isfinite(x): raise ValueError('Expected finite numeric evidence')
    return x
def check_hash(p,bound):
    if not isinstance(bound,str) or len(bound)!=64 or sha(p)!=bound: raise ValueError('Missing/changed evidence '+str(p))

def verify_smoke_campaign(smoke, identity):
    smoke=Path(smoke);campaign=read(smoke/'campaign.json')
    required=('standing','train','final_constant','final_stop')
    if campaign.get('status')!='completed' or campaign.get('terminal_inputs_unchanged') is not True or campaign.get('allocation')!='smoke' or campaign.get('branch')!=SMOKE_BRANCH or campaign.get('identity')!=identity:
        raise ValueError('Smoke campaign did not complete same-source terminal integrity')
    if set(campaign.get('accepted_phases',{}))!=set(required) or campaign.get('planned_phases')!=list(required):
        raise ValueError('Smoke phase inventory differs')
    train=validate_result(smoke/'train','train',identity)
    for phase in required:
        current=validate_result(smoke/phase,phase,identity,train['checkpoint_sha256'])
        if campaign['accepted_phases'][phase]!=current: raise ValueError('Smoke accepted receipt changed')
    return sha(smoke/'campaign.json')

def verify_inputs(args):
    source=Path(args.source).resolve(); manifest=read(source/'campaign_source_hashes.json')
    actual={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*') if p.is_file() and p.name!='campaign_source_hashes.json'}
    if actual!=manifest or any(p.is_symlink() for p in source.rglob('*')): raise ValueError('Changed/unlisted source input')
    origin=read(source/'source_origin.json');plan=read(source/PLAN)
    if origin.get('schema')!=SCHEMA or origin.get('cold_parent_source_sha256')!=PARENT: raise ValueError('Wrong shared direct training lineage')
    if plan['omni']['direct_recovery_training']!=protocol() or plan['omni']['overrides']!=EXPECTED_OVERRIDES: raise ValueError('Wrong behavior/physical/reward profile')
    if (plan['validation_num_envs'],plan['validation_control_steps'],plan['evaluation_num_envs'],plan['training_num_envs'],plan['training_iterations'],plan['physics_dt_s'],plan['decimation'])!=(32,1000,48,1024,50,.0025,8): raise ValueError('Wrong declared allocation or timestep')
    if plan['omni']['diagnostics']!={'duration_s':12,'settle_s':2,'seed':7057,'trace_envs_per_scenario':1,'controller':'policy'}: raise ValueError('Historical constant diagnostic changed')
    repair=plan['omni']['repair_training']
    if repair!={'checkpoint_sha256':CHECKPOINT,'exploration_std':.1,'entropy_coef':0.,'optimizer':'reset','learning_rate':5e-5}: raise ValueError('Checkpoint/optimizer/exploration initialization changed')
    check_hash(args.checkpoint,CHECKPOINT)
    selected=selection('train',args.allocation,args.branch,None,None)
    if args.allocation=='smoke' and args.branch!=SMOKE_BRANCH: raise ValueError('Integration smoke exercises positive quiet-priority CAPS only')
    identity={'schema':SCHEMA,'source_manifest_sha256':sha(source/'campaign_source_hashes.json'),'plan_sha256':sha(source/PLAN),
              'checkpoint_sha256':CHECKPOINT,'actor_width':315,'critic_width':318,'selection':selected,
              'overrides':EXPECTED_OVERRIDES,'diagnostic_options':plan['omni']['diagnostics'],'Stage2_complete':False}
    if args.allocation=='pilot':
        smoke=getattr(args,'smoke',None)
        if smoke is None: raise ValueError('Pilot requires completed same-source actual quiet-priority smoke')
        smoke_identity={**identity,'selection':selection('train','smoke',SMOKE_BRANCH,None,None)}
        identity['smoke_campaign_sha256']=verify_smoke_campaign(smoke,smoke_identity)
        identity['smoke_state_sha256']=sha(Path(smoke)/'train/state.json')
        identity['smoke_receipt_sha256']=sha(Path(smoke)/'train/training_receipt.json')
    return identity

validate_inputs=verify_inputs

def validate_optimizer_diagnostics(receipt, selected):
    if receipt.get('optimizer_diagnostics_schema')!='direct315_actor_gradients_v1':
        raise ValueError('Optimizer diagnostic schema missing')
    updates=receipt.get('optimizer_updates')
    if not isinstance(updates,list) or len(updates)!=selected['updates']:
        raise ValueError('Optimizer update inventory differs')
    batch_size=selected['replicas']*24//4
    for number,update in enumerate(updates,1):
        if update.get('completed_update')!=number:raise ValueError('Optimizer update label differs')
        rows=update.get('minibatches')
        if not isinstance(rows,list) or len(rows)!=20:raise ValueError('Minibatch inventory differs')
        for mb,row in enumerate(rows,1):
            if row.get('update')!=number or row.get('minibatch')!=mb:raise ValueError('Minibatch label differs')
            finite(row.get('kl_mean'))
            for key in ('learning_rate_before','learning_rate_after'):
                rate=finite(row.get(key))
                if not 1e-5<=rate<=1e-2:raise ValueError('Adaptive rate outside existing bounds')
            counts=row.get('pair_counts')
            if selected['branch']!='curriculum':
                if not isinstance(counts,dict) or set(counts)!={'valid_pairs','quiet_pairs','moving_pairs'}:raise ValueError('Missing pair counts')
                if any(type(v) is not int or not 0<=v<=batch_size for v in counts.values()):raise ValueError('Invalid pair count')
                if counts['quiet_pairs']+counts['moving_pairs']!=counts['valid_pairs']:raise ValueError('Pair count sum differs')
            sparse=number in (1,10,25,50) and mb in (1,20)
            g=row.get('gradient')
            if sparse:
                if not isinstance(g,dict) or set(g.get('norms',{}))!={'ppo_actor','quiet_temporal','moving_temporal','spatial'}:raise ValueError('Missing sparse actor gradients')
                for value in [*g['norms'].values(),g.get('component_sum_norm'),g.get('combined_actor_before_clip'),g.get('combined_actor_after_clip')]:
                    if finite(value)<0:raise ValueError('Negative gradient norm')
                cosine=g.get('ppo_quiet_cosine')
                if cosine is not None and not -1.00001<=finite(cosine)<=1.00001:raise ValueError('Invalid gradient cosine')
            elif g is not None:raise ValueError('Unexpected sparse gradient row')
        if finite(update.get('learning_rate'))!=rows[-1]['learning_rate_after']:raise ValueError('Final minibatch rate differs')
    return {'schema':'direct315_actor_gradients_v1','updates':len(updates),'minibatches':20*len(updates),
            'sparse_actor_gradient_rows':2*sum(n in (1,10,25,50) for n in range(1,len(updates)+1))}

def validate_result(directory,phase,identity,expected_checkpoint_sha256=None):
    directory=Path(directory)
    if phase not in PHASES: raise ValueError('Undeclared phase')
    state=read(directory/'state.json')
    if state.get('status')!='completed' or any(state.get(k)!=v for k,v in {'variant':'f050_t060','urdf_sha256':URDF,'plan_sha256':identity['plan_sha256'],'stance_index':0}.items()): raise ValueError('Incomplete/wrong-source phase')
    expected_mode='validate' if phase=='standing' else ('train' if phase=='train' else 'evaluate')
    expected_selection=identity['selection'] if phase=='train' else selection(expected_mode,None,None,None if phase=='standing' else phase.split('_')[1],None)
    if state.get('mode')!=expected_mode or state.get('direct_selection')!=expected_selection: raise ValueError('Wrong phase selection')
    result={'phase':phase,'state_sha256':sha(directory/'state.json'),'source_manifest_sha256':identity['source_manifest_sha256'],'Stage2_complete':False}
    if phase=='standing':
        a=read(directory/'admission.json')
        if not a.get('gate',{}).get('passed') or any(a.get(k)!=state.get(k) for k in ('variant','urdf_sha256','plan_sha256','stance_index')): raise ValueError('Matching standing admission absent')
        result.update(passed=True,admission_sha256=sha(directory/'admission.json'));return result
    if phase=='train':
        receipt=read(directory/'training_receipt.json');sel=identity['selection']
        if receipt!=state.get('training_receipt') or receipt.get('selection')!=sel or receipt.get('complete') is not True or receipt.get('updates_completed')!=sel['updates'] or state.get('iterations')!=sel['updates']: raise ValueError('Incomplete/mismatched bounded learning receipt')
        optimizer_diagnostics=validate_optimizer_diagnostics(receipt,sel)
        reload=receipt.get('reload',{})
        if not all(reload.get(k) is True for k in ('passed','exact_actor_critic_normalizer_optimizer','exact_deterministic_action')) or reload.get('optimizer_entries',0)<=0: raise ValueError('Actual strict checkpoint reload unproved')
        bound=receipt['final_checkpoint_sha256'];check_hash(directory/'policy/final.pt',bound)
        if bound!=state.get('checkpoint_sha256') or bound!=reload.get('checkpoint_sha256'): raise ValueError('Checkpoint receipt mismatch')
        init=read(directory/'repair_initialization.json')
        if init.get('checkpoint_sha256')!=CHECKPOINT or not init.get('actor_and_critic_preserved_except_std') or not init.get('observation_normalizers_preserved') or init.get('optimizer_state_entries')!=0: raise ValueError('Wrong warm start')
        audit=receipt['audit']
        if audit.get('controls')!=sel['updates']*24 or audit.get('replicas')!=sel['replicas']: raise ValueError('Wrong actual control count')
        for key in ('terminations_per_row','truncations_per_row','requested_torque_max_per_row_nm','applied_torque_max_per_row_nm','requested_saturation_fraction_per_row'):
            if len(audit.get(key,[]))!=sel['replicas']: raise ValueError('Missing per-replica audit')
            for value in audit[key]:
                if finite(value)<0: raise ValueError('Negative raw metric')
        if max(audit['applied_torque_max_per_row_nm'])>1.60001: raise ValueError('Applied cap exceeded')
        for filename,key in [('training_trace.npz','trace_sha256'),('training_joint_trace.npz','joint_trace_sha256'),('training_events.json','event_ledger_sha256')]: check_hash(directory/filename,audit[key])
        expected_updates=[2] if sel['allocation']=='smoke' else [10,25,50]
        if set(receipt['decision_checkpoints'])!={str(x) for x in expected_updates}: raise ValueError('Decision checkpoint inventory differs')
        for update in expected_updates:
            row=receipt['decision_checkpoints'][str(update)]
            if row.get('file')!='decision_'+str(update).zfill(3)+'.pt' or row.get('completed_updates')!=update: raise ValueError('Wrong decision checkpoint label')
            check_hash(directory/'policy'/row['file'],row['sha256'])
        finite(receipt['wall_seconds'])
        result.update(complete=True,optimizer_diagnostics=optimizer_diagnostics,updates_completed=sel['updates'],checkpoint_sha256=bound,receipt_sha256=sha(directory/'training_receipt.json'),scope='Bounded optimization/integrity, not physical acceptance');return result
    expected=CHECKPOINT if phase.startswith('initial_') else expected_checkpoint_sha256
    if expected is None or state.get('checkpoint_sha256')!=expected: raise ValueError('Evaluation checkpoint unbound')
    filename='diagnostics.json' if phase.endswith('_constant') else 'stop_diagnostics.json'
    report=read(directory/filename)
    if not report.get('complete') or report.get('checkpoint_sha256')!=expected or report.get('overrides')!=EXPECTED_OVERRIDES or len(report.get('scenarios',[]))!=12: raise ValueError('Missing/mismatched directional diagnostic')
    obs=report['observation_audit']
    if obs.get('actor_width')!=315 or obs.get('critic_width')!=318 or any(obs.get(k)!=0 for k in ('max_command_slice_difference','max_same_step_repeat_difference','max_history_shift_difference')): raise ValueError('Observation history/command audit failed')
    if phase.endswith('_constant'):
        if report.get('kind')!='diagnostic_not_qualification' or report.get('options')!=identity['diagnostic_options']: raise ValueError('Wrong unchanged constant diagnostic')
        for row in report['scenarios']:
            if finite(row['windows']['all']['applied_torque_abs_max_nm'])>1.60001: raise ValueError('Applied cap exceeded')
        trace='diagnostic_trace.npz'
    else:
        if report.get('kind')!='direct315_move_to_zero_diagnostic_v1' or report.get('controls')!=1600 or report.get('replicas')!=48: raise ValueError('Wrong stop protocol')
        if finite(report['all_control_applied_torque_abs_max_nm'])>1.60001: raise ValueError('Applied cap exceeded outside scored windows')
        failed=[]
        for row in report['scenarios']:
            if len(row.get('replicas',[]))!=4: raise ValueError('Missing stop replicas')
            for replica in row['replicas']:
                quiet=replica['quiet']
                if quiet.get('window_samples')!=500 or quiet.get('window_duration_s')!=10.: raise ValueError('Incomplete quiet window')
                if finite(quiet['max_applied_torque_nm'])>1.60001 or finite(replica['motion']['applied_torque_abs_max_nm'])>1.60001: raise ValueError('Applied cap exceeded')
                if not quiet['pass']: failed.append(replica['env_id'])
        result['quiet_failed_env_ids']=failed;trace='stop_trace.npz'
    result.update(complete=True,checkpoint_sha256=expected,report_sha256=sha(directory/filename),trace_sha256=sha(directory/trace),scope='Acquisition completion only; all acceptance failures retained')
    return result

def runtime_arguments(phase,allocation,branch):
    if phase not in PHASES: raise ValueError('Undeclared phase')
    selected=selection('train',allocation,branch,None,None)
    if allocation=='smoke' and (branch!=SMOKE_BRANCH or phase not in ('standing','train','final_constant','final_stop')): raise ValueError('Smoke scope is standing, CAPS two updates and matched final diagnostics only')
    mode='validate' if phase=='standing' else ('train' if phase=='train' else 'evaluate')
    argv=['/source/tools/train_length_study.py','--package','/source/robot/hexapod_mkii_length_study','--output','/output/'+phase,'--variant','f050_t060','--stance-index','0','--mode',mode,
          '--headless','--device','cuda:0','--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry']
    if phase!='standing': argv+=['--admission','/admission/admission.json','--checkpoint','/checkpoint/original.pt' if phase=='train' else '/checkpoint/evaluated.pt']
    if phase=='train': argv+=['--direct-allocation',allocation,'--direct-branch',branch,'--iterations',str(selected['updates'])]
    elif phase!='standing': argv+=['--direct-evaluation',phase.split('_')[1]]
    return argv
