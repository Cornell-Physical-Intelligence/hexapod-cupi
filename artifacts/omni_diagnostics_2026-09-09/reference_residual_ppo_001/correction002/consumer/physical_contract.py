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
    return dict(schema='c_reference_residual_external_training_consumer_v1',consumer_freeze_sha256=own,
        physical_source_sha256=SOURCE,observation_freeze_sha256=OBSERVATION,bridge_freeze_sha256=DEVICE_ADAPTER,
        device_admission_sha256=DEVICE_CAMPAIGN,standing_admission_sha256=standing,
        directional_admission_sha256=sha(args.run/'wave/state.json'),runtime_schema_sha256=RUNTIME_SCHEMA,
        requested_envelope={'forward_mps':[0,.005],'left_mps':[0,0],'yaw_rad_s':[0,0]},
        maximum_PPO_updates=2,optimization_command=[0.,0.,0.],native_velocity_fidelity_qualified=False,
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
    """Stdlib host guard over exact output/state contracts; raw source retained.

    The physical consumer independently evaluates every pre-reset row and every
    substep. This host layer does not pretend to re-run those numeric analyses.
    """
    output=Path(output);state=read(output/'state.json');mode=state.get('mode')
    if mode not in ('smoke','evaluate_initial','evaluate_final'):raise ValueError('Unknown phase')
    if state.get('status')!='completed' or state.get('identity')!=identity or state.get('source_inputs_unchanged') is not True:
        raise ValueError('Consumer phase did not complete with exact inputs')
    if state.get('runtime_binding',{}).get('runtime_tree_sha256')!=RUNTIME or state.get('Stage2_complete') is not False:
        raise ValueError('Wrong runtime or inflated completion')
    controls=3248 if mode=='smoke' else 2400;session=state.get('session',{});sub=session.get('substeps',{})
    if session.get('controls')!=controls or session.get('failure') is not None or session.get('long_training_admitted') is not False:
        raise ValueError('Incomplete or failed bounded session')
    if (sub.get('completed_controls')!=controls or sub.get('samples_including_initial')!=1+8*controls
        or sub.get('error') is not None or sub.get('method_restored') is not True):raise ValueError('Substep capture/cleanup incomplete')
    policy=state.get('policy_identity',{});bindings=policy.get('bindings',{})
    if (policy.get('lineage')!='c_wave005_finite_position_residual_PPO_smoke_v1' or policy.get('scope')!='admitted_two_update_physical_smoke'
        or policy.get('actor_width')!=846 or policy.get('critic_width')!=849 or policy.get('maximum_updates')!=2
        or bindings.get('consumer_source_sha256')!=identity['consumer_freeze_sha256']
        or bindings.get('observation_schema_sha256')!=RUNTIME_SCHEMA):raise ValueError('Wrong policy lineage/schema/source')
    for key in ('physical_source_sha256','observation_freeze_sha256','device_admission_sha256','directional_admission_sha256','standing_admission_sha256'):
        if bindings.get(key)!=identity[key]:raise ValueError('Policy admission binding differs: '+key)
    if state.get('reload',{}).get('actor_critic_normalizer_and_optimizer_exact') is not True:raise ValueError('Strict readback missing')
    paths=['state.json','trace.npz','sensor_clocks.npz','physics_substeps.npz','physics_control_integrals.npz',
        'physics_substep_review.json','observation_schema.json','reference_state.npz','final_observation.npz']
    if mode=='smoke':
        if (state.get('PPO_updates_completed')!=2 or state.get('policy_training_started') is not True
            or session.get('policy_steps')!=3048 or state.get('optimizer_entries')!=17
            or state.get('deterministic_reload_max_difference')!=0 or state.get('post_update_quiet_passed') is not True):
            raise ValueError('Exact two-update/reload/quiet smoke incomplete')
        calibration=read(output/'calibration.json')
        if calibration.get('passed') is not True or calibration.get('initial_std')!=.02 or set(calibration.get('trials',{}))!={'zero_mean','sampled'}:
            raise ValueError('Calibration was not passed')
        for trial in calibration['trials'].values():_quiet_pass(trial,32)
        if sha(output/'calibration.json')!=bindings['calibration_sha256']:raise ValueError('Calibration binding changed')
        _quiet_pass(read(output/'post_update_quiet.json'),32)
        for phase in ('initial','final'):
            name=phase+'.pt';meta=read(output/(name+'.json'))
            if meta!={**policy,'checkpoint_sha256':sha(output/name)} or state.get(phase+'_checkpoint')!=meta:
                raise ValueError('Immutable checkpoint identity differs: '+name)
            paths += [name,name+'.json']
        paths+=['calibration.json','post_update_quiet.json']
    else:
        if state.get('policy_training_started') is not False or state.get('PPO_updates_completed')!=0 or state.get('retention_passed') is not True:
            raise ValueError('Unexpected training or incomplete retention')
        retention=read(output/'retention.json')
        if retention.get('passed') is not True or retention.get('completed_legs')!=list(range(6)):raise ValueError('Physical motion/contact retention incomplete')
        _quiet_pass(retention.get('final_quiet',{}),1);paths+=['retention.json']
    if read(output/'observation_schema.json').get('schema_sha256')!=RUNTIME_SCHEMA:raise ValueError('Written observation schema differs')
    return dict(passed=True,mode=mode,controls=controls,source_identity=identity,
        files_sha256={name:sha(output/name) for name in paths},longer_training_admitted=False,Stage2_complete=False)
