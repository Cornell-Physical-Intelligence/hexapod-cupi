"""Exact physical dependencies for the new, bounded residual PPO consumer."""
from pathlib import Path
import hashlib,importlib.util,json,sys
from types import SimpleNamespace
SOURCE='04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
OBSERVATION='22402b0079b0b5ac0acdd4e1d7e819fe206b20ee1883077f7de6eaca55808a63'
DEVICE_ADAPTER='be4870a8f7ca0857ba50ffd3a0c92ac5925b9a816767d66d826af7b59957aa0e'
DEVICE_CAMPAIGN='1e22a3b292be40501d97c6fd4a35e5a015c873fb773c2a9dca1ae5d28d8833b1'
RUNTIME='abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280'
RUNTIME_SCHEMA='c2ab56b61c8ca56412090cd42cf109c90576bd5e78602a17eb29ada6d0559845'
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);p=path.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');p.replace(path)
def verify_tree(root,manifest,expected):
    if sha(root/manifest)!=expected:raise ValueError('Wrong manifest '+str(root))
    if any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Symbolic dependency substitution')
    actual={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()};actual.pop(manifest)
    if actual!=read(root/manifest):raise ValueError('Changed dependency '+str(root))

def verify_standing(path,campaign_directory):
    path=path.resolve()
    if path!=(campaign_directory/'standing/admission.json').resolve():raise ValueError('Fresh same-campaign standing receipt required')
    admission=read(path)
    if sha(path)!=sha(path.with_name('state.json')):raise ValueError('Standing state/admission differ')
    gate=admission.get('gate',{})
    if (admission.get('status')!='completed' or admission.get('control_steps')!=1000
        or admission.get('identity',{}).get('source_manifest_sha256')!=SOURCE
        or gate.get('passed') is not True or gate.get('num_envs')!=32
        or gate.get('all_replica_quiet',{}).get('passed') is not True
        or len(gate.get('all_replica_quiet',{}).get('per_environment',[]))!=32
        or any(r.get('pass') is not True or r.get('window_duration_s',0)<10. for r in gate.get('all_replica_quiet',{}).get('per_environment',[]))):
        raise ValueError('Exact fresh32x1000 physical+allquiet standing admission required')
    return sha(path)

def verify_inputs(args,*,require_standing=True):
    own=sha(HERE/'FREEZE_SHA256.json');verify_tree(HERE,'FREEZE_SHA256.json',own)
    verify_tree(args.source_root,'campaign_source_hashes.json',SOURCE)
    verify_tree(args.observation_bundle,'FREEZE_SHA256.json',OBSERVATION)
    verify_tree(args.bridge,'FREEZE_SHA256.json',DEVICE_ADAPTER)
    spec=importlib.util.spec_from_file_location('_bound_smoke_input_contract',args.bridge/'smoke_contract.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify(SimpleNamespace(source_root=args.source_root,run=args.run,observation_bundle=args.observation_bundle,
        package=args.package,num_envs=32,controls=264))
    if sha(args.device_run/'campaign.json')!=DEVICE_CAMPAIGN:raise ValueError('Exact completed physical device001 proof required')
    campaign=read(args.device_run/'campaign.json')
    if campaign.get('both_bridge_phases_passed') is not True or campaign.get('source_and_inputs_unchanged') is not True:raise ValueError('Device bridge incomplete')
    for phase,n in [('replicas_1',1),('replicas_32',32)]:
        receipt=read(args.device_run/(phase+'_accepted.json'))
        if receipt!=campaign['accepted_phases'][phase] or receipt['passed'] is not True or receipt['replicas']!=n or receipt['schema_sha256']!=RUNTIME_SCHEMA:
            raise ValueError('Device exact result/width/schema mismatch')
        for name,bound in receipt['files_sha256'].items():
            if sha(args.device_run/phase/name)!=bound:raise ValueError('Device raw evidence changed '+phase+'/'+name)
    standing=verify_standing(args.standing,args.output.parent) if require_standing else None
    return dict(schema='c_reference_masked_moving_training_consumer_v2',consumer_freeze_sha256=own,
        physical_source_sha256=SOURCE,observation_freeze_sha256=OBSERVATION,bridge_freeze_sha256=DEVICE_ADAPTER,
        device_admission_sha256=DEVICE_CAMPAIGN,standing_admission_sha256=standing,
        directional_admission_sha256=sha(args.run/'wave/state.json'),runtime_schema_sha256=RUNTIME_SCHEMA,
        requested_envelope={'forward_mps':[0,.005],'left_mps':[0,0],'yaw_rad_s':[0,0]},
        maximum_PPO_updates=25,initial_allocation_updates=10,controls_per_update=256,training_commands=[[0.,0.,0.],[.005,0.,0.]],native_velocity_fidelity_qualified=False,
        original_prototype_training_flags_unchanged=True,Stage2_complete=False)

def _quiet_pass(report,replicas):
    import math
    bounds={'max_planar_excursion_m':.01,'max_heading_excursion_deg':2.,'max_joint_velocity_rms_rad_s':.03,
        'max_joint_position_range_rad':.02,'max_target_step_abs_p95_rad_per_20ms':.002,
        'max_requested_torque_saturation_fraction':.005,'max_applied_torque_nm':1.60001}
    rows=report.get('per_environment',[])
    if report.get('passed') is not True or report.get('original_quiet_bounds')!=bounds or len(rows)!=replicas:
        raise ValueError('Incomplete unchanged per-replica quiet report')
    for row in rows:
        if row.get('pass') is not True or row.get('window_duration_s',0)<10 or row.get('terminations')!=0 or row.get('truncations')!=0:
            raise ValueError('Quiet failure or incomplete window')
        for key,limit in bounds.items():
            value=row.get(key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=limit:
                raise ValueError('Quiet bound exceeded or nonfinite: '+key)


def validate_result(output,identity):
    """Stdlib-only host receipt check; raw numeric replay is a separate audit."""
    import math
    output=Path(output);state=read(output/'state.json');mode=state.get('mode')
    modes=('calibrate','profile_32','profile_128','train_10','train_25',
           'evaluate_initial','evaluate_010','evaluate_025','quiet_010','quiet_025')
    if mode not in modes or state.get('status')!='completed' or state.get('identity')!=identity:
        raise ValueError('Phase did not complete with its exact source/admission identity')
    if state.get('source_inputs_unchanged') is not True or state.get('runtime_binding',{}).get('runtime_tree_sha256')!=RUNTIME:
        raise ValueError('Phase final integrity/runtime check failed')
    if state.get('Stage2_complete') is not False:raise ValueError('This consumer cannot complete Stage2')
    policy=state.get('policy_identity',{});bindings=policy.get('bindings',{})
    required={'physical_source_sha256','observation_freeze_sha256','device_admission_sha256','directional_admission_sha256',
              'standing_admission_sha256','calibration_sha256','consumer_source_sha256','observation_schema_sha256'}
    if (policy.get('lineage')!='c_wave005_masked_moving_residual_PPO_v1' or policy.get('scope')!='admitted_bounded_moving_PPO'
        or policy.get('actor_width')!=846 or policy.get('critic_width')!=849 or policy.get('actions')!=18
        or policy.get('maximum_updates')!=25 or policy.get('old_checkpoint_transfer_allowed') is not False
        or policy.get('plan_sha256')!=sha(HERE/'plan.json')
        or policy.get('recovery_controls_excluded_from_learning') is not True or set(bindings)!=required
        or bindings.get('consumer_source_sha256')!=identity['consumer_freeze_sha256']
        or bindings.get('observation_schema_sha256')!=RUNTIME_SCHEMA):raise ValueError('Wrong moving policy/checkpoint contract')
    for key in required-{'calibration_sha256','consumer_source_sha256','observation_schema_sha256'}:
        if bindings[key]!=identity[key]:raise ValueError('Policy admission differs: '+key)
    calibration=output if mode=='calibrate' else output.parent/'calibration'
    if sha(calibration/'calibration.json')!=bindings['calibration_sha256']:
        raise ValueError('Bound calibration changed')
    checked_calibration=read(calibration/'calibration.json')
    if checked_calibration.get('passed') is not True or checked_calibration.get('initial_std')!=.02 or set(checked_calibration.get('trials',{}))!={'zero_mean','sampled'}:
        raise ValueError('Required calibration was not passed')
    for report in checked_calibration['trials'].values():_quiet_pass(report,32)
    if mode!='calibrate':
        which={'evaluate_010':('train_10','decision_010.pt'),'quiet_010':('train_10','decision_010.pt'),
            'train_25':('train_10','decision_010.pt'),'evaluate_025':('train_25','decision_025.pt'),
            'quiet_025':('train_25','decision_025.pt')}.get(mode,('calibration','initial.pt'))
        prior=output.parent/which[0]/which[1]
        if state.get('input_checkpoint_sha256')!=sha(prior) or read(prior.with_suffix('.pt.json'))!={**policy,'checkpoint_sha256':sha(prior)}:
            raise ValueError('Phase used a different checkpoint or policy identity')
    raw_mode=mode.startswith(('profile','train'))
    if mode=='calibrate':controls=2200
    elif mode.startswith('profile'):controls=712
    elif mode=='train_10':controls=2760
    elif mode=='train_25':controls=4040
    elif mode.startswith('evaluate'):controls=2400
    else:controls=1200
    session=state.get('session',{})
    if session.get('controls')!=controls or session.get('failure') is not None:
        raise ValueError('Missing complete bounded physical session')
    if raw_mode:
        sub=read(output/'raw/physics_substep_review.json')
        if session!=read(output/'raw/session.json') or session.get('physical_admission') is not False or session.get('evaluation_gates_changed') is not False:
            raise ValueError('Training/profile session receipt differs or claims admission')
        required_paths=['raw/trace.npz','raw/sensor_clocks.npz','raw/reference_states.npz','raw/physics_substeps.npz','raw/episodes.json']
    else:
        sub=session.get('substeps',{})
        required_paths=['trace.npz','sensor_clocks.npz','physics_substeps.npz','physics_control_integrals.npz',
                        'observation_schema.json','reference_state.npz','final_observation.npz']
        if read(output/'observation_schema.json').get('schema_sha256')!=RUNTIME_SCHEMA:
            raise ValueError('Written observation schema differs')
    if (sub.get('completed_controls')!=controls or sub.get('samples_including_initial')!=1+controls*8
        or sub.get('error') is not None or sub.get('method_restored') is not True):
        raise ValueError('Raw substep capture/restoration incomplete')
    for name in required_paths:
        if not (output/name).is_file() or (output/name).stat().st_size==0:raise ValueError('Missing raw phase evidence: '+name)
    if mode.startswith('profile'):
        n=int(mode.split('_')[1]);timings=state.get('profile_control_seconds',[])
        if (state.get('replicas')!=n or state.get('profile_controls')!=512 or state.get('profile_passed') is not True
            or len(timings)!=512 or state.get('active_transitions')!=512*n or read(output/'raw/episodes.json')!=[]):
            raise ValueError('Incomplete or failed device timing phase')
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in timings):
            raise ValueError('Nonfinite/invalid device timing')
        wall=state.get('profile_wall_s')
        if isinstance(wall,bool) or not isinstance(wall,(int,float)) or not math.isfinite(wall) or wall<=0 or wall+1e-6<sum(timings):
            raise ValueError('Invalid/inconsistent profile wall time')
    expected_updates=int(mode.split('_')[1]) if mode.startswith('train') else 0
    if state.get('PPO_updates_completed')!=expected_updates:raise ValueError('Wrong number of actual PPO updates')
    if state.get('policy_training_started') is not mode.startswith('train'):
        raise ValueError('Unexpected or missing optimization in this phase')
    if mode.startswith('train'):
        std=state.get('learned_std_per_joint',[])
        if (state.get('optimizer_entries')!=17 or state.get('deterministic_reload_max_difference')!=0
            or state.get('reload',{}).get('actor_critic_normalizer_and_optimizer_exact') is not True
            or state.get('evaluation_still_required') is not True or len(std)!=18
            or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in std)):
            raise ValueError('Actual PPO/reload checkpoint proof incomplete')
        updates=state.get('updates',[]);start=1 if mode=='train_10' else 11
        if [r.get('updates_completed') for r in updates]!=list(range(start,expected_updates+1)):
            raise ValueError('Missing/reordered optimization update reports')
        for report in updates:
            if report.get('controls_per_replica')!=256 or report.get('valid_learning_transitions',0)<8:
                raise ValueError('Invalid masked rollout update')
        name='decision_'+str(expected_updates).zfill(3)+'.pt'
        if read(output/(name+'.json'))!={**policy,'checkpoint_sha256':sha(output/name)} or state.get('decision_checkpoint')!=read(output/(name+'.json')):
            raise ValueError('Decision checkpoint bytes/sidecar/state differ')
    if mode=='calibrate':
        meta={**policy,'checkpoint_sha256':sha(output/'initial.pt')}
        if read(output/'initial.pt.json')!=meta or state.get('initial_checkpoint')!=meta:
            raise ValueError('Immutable initial checkpoint differs')
    if mode.startswith('evaluate'):
        report=read(output/'retention.json')
        if state.get('retention_passed') is not True or report.get('passed') is not True or report.get('completed_legs')!=list(range(6)):
            raise ValueError('Original full forward-stop retention gate did not pass')
        _quiet_pass(report.get('final_quiet',{}),1)
    if mode.startswith('quiet'):
        if state.get('quiet_passed') is not True:raise ValueError('Quiet phase rejected')
        _quiet_pass(read(output/'quiet.json'),32)
    if any(path.is_symlink() for path in output.rglob('*')):raise ValueError('Phase evidence must not contain symlinks')
    return {'passed':True,'mode':mode,'phase':'calibration' if mode=='calibrate' else mode,'controls':controls,'source_identity':identity,'Stage2_complete':False,
            'automatic_continuation_allowed':False,
            'files_sha256':{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}}
