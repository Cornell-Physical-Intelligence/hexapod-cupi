"""Explicit new-lineage initialization, calibrated exploration and immutable checkpoints."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from velocity_action import LINEAGE,verify_new_lineage_checkpoint

INITIAL_STD=.005
SAMPLED_CALIBRATION_STEPS=1000
LEARNING_RATE=1.e-4
IDENTITY_KEYS={'variant','urdf_sha256','plan_sha256','stance_index','source_sha256'}

def save_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def checkpoint_metadata(config,identity):
    if set(identity)!=IDENTITY_KEYS:raise ValueError('Exact source/asset/plan identity keys required; reserved fields cannot be overridden')
    return {**identity,**config.contract(),'action_semantics':'normalized_joint_target_velocity',
            'max_acceleration_rad_s2':config.max_acceleration_rad_s2}

def initialize_scratch(runner,config):
    actor=runner.alg.actor;critic=runner.alg.critic
    linear=[m for m in actor.mlp.modules() if isinstance(m,torch.nn.Linear)]
    critic_linear=[m for m in critic.mlp.modules() if isinstance(m,torch.nn.Linear)]
    if linear[0].in_features!=495 or linear[-1].out_features!=18 or critic_linear[0].in_features!=498:
        raise ValueError('Expected new 495/498 actor/critic architecture')
    if runner.alg.optimizer.state or runner.current_learning_iteration!=0:raise ValueError('Candidate initialization requires scratch runner')
    if actor.distribution.std_type!='scalar' or actor.distribution.std_param.shape!=(18,):raise ValueError('Expected 18 scalar Gaussian standard deviations')
    with torch.no_grad():
        linear[-1].weight.zero_();linear[-1].bias.zero_();actor.distribution.std_param.fill_(INITIAL_STD)
    return {'initialization':'scratch_zero_actor_mean_head','initial_action_std':INITIAL_STD,
            'initial_action_std_per_joint':actor.distribution.std_param.detach().cpu().tolist(),
            'optimizer':'fresh_adam','optimizer_state_entries':len(runner.alg.optimizer.state),
            'learning_rate':runner.alg.learning_rate,'learning_rate_schedule':runner.alg.schedule,
            'entropy_coef':runner.alg.entropy_coef,'action_semantics':'normalized_joint_target_velocity',
            'actor_width':495,'critic_width':498,'controller':config.contract(),'stage2_complete':False}

def prepare_algorithm_buffers_for_load(algorithm):
    """Exact reviewed inference-buffer repair from isaaclab/train_mkii_fourbar.py.

    Copied locally to avoid importing a different physical-task entrypoint, which
    also changes import paths. Only inference buffers are replaced, preserving values.
    """
    replaced=[]
    for model_name in ('actor','critic','rnd'):
        model=getattr(algorithm,model_name,None)
        if model is None:continue
        for module_name,module in model.named_modules():
            for name,tensor in tuple(module.named_buffers(recurse=False)):
                if tensor.is_inference():
                    with torch.inference_mode(False):setattr(module,name,tensor.clone())
                    replaced.append('.'.join(filter(None,(model_name,module_name,name))))
    return replaced

def prepare_local_logger_for_early_checkpoint(runner):
    """RSL 5.0.1 defers writer creation until learn, but save_model reads it.

    Initialize only the absent local-writer sentinel. Calling the full writer
    initializer here would create it twice when learn starts. No logging config,
    learned tensor, normalizer or optimizer state changes.
    """
    logger=runner.logger
    if not hasattr(logger,'writer'):
        if logger.cfg.get('logger')!='tensorboard' or logger.log_dir is None:
            raise ValueError('Early checkpoint requires the configured local TensorBoard logger')
        logger.writer=None


def save_candidate(runner,path,config,identity):
    path=Path(path);sidecar=path.with_suffix(path.suffix+'.json')
    if path.exists() or sidecar.exists():raise FileExistsError('Candidate checkpoint and sidecar are immutable')
    meta=checkpoint_metadata(config,identity)
    path.parent.mkdir(parents=True,exist_ok=True)
    prepare_local_logger_for_early_checkpoint(runner)
    runner.save(str(path),infos={'candidate_contract':meta})
    meta['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(sidecar,meta)
    return meta

def verified_candidate_metadata(path,config,identity):
    expected=checkpoint_metadata(config,identity)
    path=Path(path);meta=json.loads(path.with_suffix(path.suffix+'.json').read_text())
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    verify_new_lineage_checkpoint(meta,config,meta.get('checkpoint_sha256'),sha)
    if {k:v for k,v in meta.items() if k!='checkpoint_sha256'}!=expected:
        raise ValueError('Candidate source/plan/asset/controller identity mismatch')
    return meta

def load_candidate(runner,path,config,identity):
    meta=verified_candidate_metadata(path,config,identity)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    embedded=saved.get('infos',{}).get('candidate_contract')
    if embedded!={k:v for k,v in meta.items() if k!='checkpoint_sha256'}:raise ValueError('Checkpoint sidecar differs from embedded contract')
    for name,width in [('actor',495),('critic',498)]:
        state=saved.get(name+'_state_dict',{})
        if state.get('mlp.0.weight',torch.empty(0,0)).shape[1:]!=(width,):raise ValueError('Checkpoint tensor schema mismatch')
        if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite checkpoint state')
    repaired=prepare_algorithm_buffers_for_load(runner.alg)
    runner.load(str(path),strict=True,map_location=runner.device)
    for name in ('actor','critic'):
        actual=getattr(runner.alg,name).state_dict();expected=saved[name+'_state_dict']
        if set(actual)!=set(expected) or any(not torch.equal(actual[k].detach().cpu(),expected[k]) for k in actual):
            raise RuntimeError('Strict checkpoint readback differs from saved tensors')
    return {**meta,'inference_buffers_made_writable':repaired}

def summarize_calibration(data,joint_names,dt,config,*,quiet_required):
    """Each replica/joint is retained; a noisy replica cannot hide in a mean."""
    from omni_quiet_review import quiet_metrics
    required={'applied_torque_nm','computed_torque_nm','joint_position_rad','joint_velocity_rad_s','joint_target_rad',
              'position_world_m','quaternion_world_wxyz','terminated','truncated','reward_term_nonfoot',
              'executable_target_velocity_rad_s','executable_target_acceleration_rad_s2'}
    if not required.issubset(data):raise ValueError(f'Missing pre-reset calibration fields: {required-set(data)}')
    if not all(np.isfinite(value).all() for value in data.values()):raise ValueError('Nonfinite calibration trace')
    start=round(2/dt);rows=[]
    for i in range(data['joint_position_rad'].shape[1]):
        quiet=quiet_metrics(data,i,start,joint_names,dt)
        requested=data['computed_torque_nm'][start:,i]
        q=data['joint_position_rad'][start:,i]
        target=data['joint_target_rad'][:,i]
        delta=np.diff(target,axis=0)
        row={'environment_index':i,'quiet_window':quiet,
             'joint_fd_velocity_rms_rad_s':np.sqrt(np.mean((np.diff(q,axis=0)/dt)**2,axis=0)).tolist(),
             'max_target_step_rad':float(np.abs(delta).max()),
             'max_target_velocity_rad_s':float(np.abs(data['executable_target_velocity_rad_s'][:,i]).max()),
             'max_target_acceleration_rad_s2':float(np.abs(data['executable_target_acceleration_rad_s2'][:,i]).max()),
             'max_applied_torque_nm':float(np.abs(data['applied_torque_nm'][:,i]).max()),
             'post_settle_max_joint_saturation_fraction':float((np.abs(requested)>1.6).mean(0).max()),
             'post_settle_nonfoot_steps':int((data['reward_term_nonfoot'][start:,i]>0).sum()),
             'terminations':int(data['terminated'][:,i].sum()),'truncations':int(data['truncated'][:,i].sum())}
        row['passed']=(row['terminations']==0 and row['truncations']==0 and row['post_settle_nonfoot_steps']==0
            and row['post_settle_max_joint_saturation_fraction']<=.005 and row['max_applied_torque_nm']<=1.60001
            and row['max_target_step_rad']<=config.max_velocity_rad_s*dt+1e-6
            and row['max_target_velocity_rad_s']<=config.max_velocity_rad_s+1e-6
            and row['max_target_acceleration_rad_s2']<=config.max_acceleration_rad_s2+1e-4
            and (quiet['pass'] or not quiet_required))
        rows.append(row)
    return {'per_environment':rows,'passed':all(row['passed'] for row in rows),
            'quiet_required':quiet_required,'full_30s_quiet_review_completed':False}

@torch.inference_mode()
def calibrate_exploration(env,runner,output,identity):
    """Actual zero-mean actor distribution; still commands, no walking training."""
    from tensordict import TensorDict
    reports=[];output=Path(output)
    env.omni_diagnostic_enabled=True
    runner.alg.eval_mode()
    try:
        for label,stochastic,steps in [('zero_mean',False,250),('sampled_std_0p005',True,SAMPLED_CALIBRATION_STEPS)]:
            env.reset(seed=7057);env.episode_length_buf.zero_();env.set_evaluation_targets(torch.zeros(env.num_envs,3,device=env.device))
            samples=[];raw=[]
            for i in range(steps):
                obs=env._get_observations()
                if not all(torch.isfinite(v).all() for v in obs.values()):raise RuntimeError('Nonfinite calibration observation')
                action=runner.alg.actor(TensorDict(obs,batch_size=[env.num_envs]),stochastic_output=stochastic)
                raw.append(action.detach().cpu().numpy().copy())
                _,_,term,trunc,_=env.step(action)
                sample=env.omni_diagnostic_sample
                if not np.array_equal(sample['terminated'],term.cpu().numpy()) or not np.array_equal(sample['truncated'],trunc.cpu().numpy()):
                    raise RuntimeError('Calibration pre-reset event mismatch')
                samples.append(sample)
            data={key:np.stack([v[key] for v in samples]) for key in samples[0]}
            summary=summarize_calibration(data,list(env._robot.joint_names),env.step_dt,env.target_velocity_controller.config,quiet_required=not stochastic)
            action=np.stack(raw)
            np.savez_compressed(output/(label+'_trace.npz'),**data,joint_names=np.array(env._robot.joint_names))
            reports.append({'name':label,'steps':steps,'measured_raw_action_mean_per_joint':action.mean((0,1)).tolist(),
                'measured_raw_action_std_per_joint':action.std((0,1)).tolist(),**summary})
    finally:env.omni_diagnostic_enabled=False
    ids=torch.arange(0,env.num_envs,2,device=env.device);other=torch.arange(1,env.num_envs,2,device=env.device)
    before=env.target_velocity_controller.position[other].clone();env._reset_idx(ids)
    obs=env._get_observations();repeat=env._get_observations()
    reset_error=float((env.target_velocity_controller.position[ids]-env._previous_processed_joint_target[ids]).abs().max())
    reset_velocity=float(env.target_velocity_controller.velocity[ids].abs().max())
    unaffected_error=float((env.target_velocity_controller.position[other]-before).abs().max()) if len(other) else 0.
    repeat_error=float((obs['policy']-repeat['policy']).abs().max())
    state_error=float((obs['policy'].reshape(env.num_envs,5,99)[ids,:,81:]).abs().max())
    result={'complete':True,'kind':'new_action_standing_calibration_not_walking','stage2_complete':False,
            **identity,'initial_std':INITIAL_STD,'controller':env.candidate_contract,'trials':reports,
            'reset_target_error_rad':reset_error,'reset_target_velocity_rad_s':reset_velocity,
            'unaffected_reset_target_error_rad':unaffected_error,'same_step_observation_difference':repeat_error,
            'reset_history_velocity_error':state_error}
    result['passed']=all(r['passed'] for r in reports) and max(reset_error,reset_velocity,unaffected_error,repeat_error,state_error)<=1e-6
    save_json(output/'calibration.json',result)
    return result


def source_digest(root):
    """Bind executable source, including reward code, pinned runtime and shims."""
    root=Path(root);paths=set()
    for folder in ('tools','isaaclab','packages','experiments/c_length_study/runtime'):
        paths.update(p for p in (root/folder).rglob('*.py') if '__pycache__' not in p.parts)
    paths.update((root/'experiments/c_length_study/runtime').glob('*.json'))
    if not paths:raise ValueError('No frozen C-study source found')
    encoded={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    return hashlib.sha256(json.dumps(encoded,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def preflight_candidate(args):
    """Validate new-lineage inputs before AppLauncher; scratch training only."""
    from velocity_action import VelocityActionConfig
    if args.mode not in ('validate','probe','train','evaluate'):raise ValueError('Candidate supports validate/probe/train/evaluate only; video requires later reviewed adapter')
    if args.mode!='evaluate' and args.checkpoint is not None:raise ValueError('This candidate never resumes or transfers a checkpoint into training')
    if args.mode=='evaluate' and args.checkpoint is None:raise ValueError('Evaluate requires an explicit candidate checkpoint')
    if args.mode in ('validate','probe') and args.iterations is not None:raise ValueError('Standing probe has exactly two schema-smoke updates; no iteration override')
    plan_path=args.package/'training_plan.json';plan=json.loads(plan_path.read_text())
    manifest=json.loads((args.package/'manifest.json').read_text())
    omni=plan.get('omni',{})
    if omni.get('architecture')!=LINEAGE or 'reference_controller' in plan or 'repair_training' in omni:
        raise ValueError('Explicit new target-velocity architecture required; no reference or old repair lineage')
    if set(omni.get('velocity_candidate',{}))!={'profile','max_acceleration_rad_s2'}:raise ValueError('Explicit controller profile and acceleration required')
    config=VelocityActionConfig(**omni['velocity_candidate'])
    if (omni.get('initial_exploration_std')!=INITIAL_STD
        or omni.get('sampled_calibration_control_steps')!=SAMPLED_CALIBRATION_STEPS):
        raise ValueError('Candidate 002 requires explicit std 0.005 and 20 s sampled calibration in the matched plan')
    override=omni.get('overrides',{})
    if override.get('observation_noise_scale')!=1. or override.get('target_filter_time_constant_s')!=0.:
        raise ValueError('Initial comparison requires observation noise 1 and position filter off')
    if 'target_slew_rad_per_20ms' in override and override['target_slew_rad_per_20ms']!=config.max_velocity_rad_s*.02:
        raise ValueError('Legacy slew field disagrees with explicit candidate profile')
    required={'physics_dt_s':.0025,'decimation':8,'validation_num_envs':32,'validation_control_steps':1000,
              'training_num_envs':1024,'evaluation_num_envs':48}
    if any(plan.get(k)!=v for k,v in required.items()):raise ValueError('Candidate requires explicit admitted dt, 32 standing / 1024 training / 48 evaluation layout')
    iterations=args.iterations if args.iterations is not None else plan.get('training_iterations')
    if type(iterations) is not int or not 1<=iterations<=100:raise ValueError('Initial candidate PPO pilot must be explicitly bounded to 1–100 updates')
    if not 0<=args.stance_index<len(plan['variants'][args.variant]['stances']):raise ValueError('Invalid stance index')
    records=[r for r in manifest['variants'] if r['variant']==args.variant]
    if len(records)!=1:raise ValueError('Unique morphology record required')
    stance=plan['variants'][args.variant]['stances'][args.stance_index]
    expected_pose={manifest['link_joint_mapping'][leg]['joints'][part]:float(np.deg2rad(degrees))
                   for leg in ('lf','lm','lr','rf','rm','rr') for part,degrees in (('coxa',0),('femur',40),('tibia',120))}
    if (set(stance['joint_positions_rad'])!=set(expected_pose)
        or any(not np.isclose(stance['joint_positions_rad'][name],value,atol=1e-10,rtol=0) for name,value in expected_pose.items())
        or not np.isclose(stance['root_height_at_contact_m'],.13053251856352807,atol=1e-10,rtol=0)
        or not np.isclose(stance['suggested_reset_root_height_m'],.13653251856352808,atol=1e-10,rtol=0)):
        raise ValueError('Candidate requires the admitted C 40/120 degree stance and root heights by runtime joint names')
    record=records[0];sha=hashlib.sha256((args.package/record['urdf']).read_bytes()).hexdigest()
    if sha!=record['sha256']:raise ValueError('URDF hash mismatch')
    if sha!='e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c':raise ValueError('This candidate is bound to the previously admitted serial C URDF')
    motor=manifest['actuator_config_snapshot']
    if any(motor.get(k)!=v for k,v in {'effort_limit':1.6,'stiffness':30.,'damping':.6}.items()):
        raise ValueError('Candidate must preserve RS05 cap and existing PD gains')
    identity={'variant':args.variant,'urdf_sha256':sha,'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              'stance_index':args.stance_index,'source_sha256':source_digest(Path(__file__).resolve().parents[1])}
    if args.mode!='validate':
        admission=json.loads(args.admission.read_text()) if args.admission else {}
        if admission.get('gate',{}).get('passed') is not True or any(admission.get(k)!=v for k,v in identity.items()):
            raise ValueError('Candidate requires a passed standing admission matching source, plan, asset and stance')
    if args.mode=='evaluate':verified_candidate_metadata(args.checkpoint,config,identity)
    if args.mode=='train':
        calibration=json.loads(args.calibration.read_text()) if args.calibration else {}
        smoke=json.loads(args.runner_smoke.read_text()) if args.runner_smoke else {}
        if (calibration.get('passed') is not True or smoke.get('passed') is not True
            or calibration.get('initial_std')!=INITIAL_STD
            or any(calibration.get(k)!=v or smoke.get(k)!=v for k,v in identity.items())
            or smoke.get('calibration_sha256')!=hashlib.sha256(args.calibration.read_bytes()).hexdigest()):
            raise ValueError('Scratch walking requires matching passed noise calibration and actual runner learn/save/reload smoke')
    for name in ('state.json','admission.json','calibration.json','evaluation.json'):
        if (args.output/name).exists():raise FileExistsError('Candidate outputs are immutable; use a fresh run directory')
    return identity,config


def run_schema_smoke(env,runner,output,config,identity):
    """Actual RSL learns two stand-only updates, then strict save/reload equality."""
    from tensordict import TensorDict
    output=Path(output)
    with torch.inference_mode():
        env.reset(seed=8057);env.episode_length_buf.zero_()
        env.set_evaluation_targets(torch.zeros(env.num_envs,3,device=env.device))
    if not env.omni_evaluation:raise ValueError('Schema smoke must be stand-only evaluation environment')
    # The new initial candidate has no old weights or normalizers to inherit.
    save_candidate(runner,output/'policy'/'initial.pt',config,identity)
    # Start optimization from a settled physical hold; this is still calibration.
    with torch.inference_mode():
        for _ in range(100):
            _,_,term,trunc,_=env.step(torch.zeros(env.num_envs,18,device=env.device))
            if term.any() or trunc.any():raise RuntimeError('Standing warmup failed before schema-smoke optimization')
    env.candidate_schema_snapshots=[];env.omni_diagnostic_enabled=True
    try:runner.learn(num_learning_iterations=2,init_at_random_ep_len=False)
    finally:env.omni_diagnostic_enabled=False
    samples=env.candidate_schema_snapshots;env.candidate_schema_snapshots=None
    if not samples:raise RuntimeError('Runner produced no captured optimization steps')
    data={k:np.stack([v[k] for v in samples]) for k in samples[0]}
    if not all(np.isfinite(v).all() for v in data.values()):raise RuntimeError('Nonfinite runner-smoke motor/state trace')
    motors=[]
    for i in range(env.num_envs):
        row={'environment_index':i,'terminations':int(data['terminated'][:,i].sum()),
             'truncations':int(data['truncated'][:,i].sum()),
             'max_joint_saturation_fraction':float((np.abs(data['computed_torque_nm'][:,i])>1.6).mean(0).max()),
             'nonfoot_contact_steps':int((data['reward_term_nonfoot'][:,i]>0).sum()),
             'max_applied_torque_nm':float(np.abs(data['applied_torque_nm'][:,i]).max())}
        row['passed']=row['terminations']==0 and row['truncations']==0 and row['nonfoot_contact_steps']==0 and row['max_joint_saturation_fraction']<=.005 and row['max_applied_torque_nm']<=1.60001
        motors.append(row)
    np.savez_compressed(output/'runner_smoke_trace.npz',**data,joint_names=np.array(env._robot.joint_names))
    saved=save_candidate(runner,output/'policy'/'schema_smoke.pt',config,identity)
    runner.alg.eval_mode()
    with torch.inference_mode():
        fixed=TensorDict({k:v.clone() for k,v in env._get_observations().items()},batch_size=[env.num_envs])
        before=runner.alg.actor(fixed,stochastic_output=False).detach().cpu().clone()
    loaded=load_candidate(runner,output/'policy'/'schema_smoke.pt',config,identity)
    with torch.inference_mode():after=runner.alg.actor(fixed,stochastic_output=False).detach().cpu()
    difference=float((before-after).abs().max())
    finite=bool(torch.isfinite(before).all() and torch.isfinite(after).all())
    report={'complete':True,'kind':'actual_rsl_two_update_standing_schema_smoke_not_walking','stage2_complete':False,
            **identity,'calibration_sha256':hashlib.sha256((output/'calibration.json').read_bytes()).hexdigest(),
            'learning_updates':2,'iteration_recorded_by_rsl':runner.current_learning_iteration,
            'checkpoint_sha256':saved['checkpoint_sha256'],'readback_action_max_difference':difference,
            'optimizer_state_entries':len(runner.alg.optimizer.state),'per_environment_motor_screen':motors,
            'learned_std_per_joint':runner.alg.actor.distribution.std_param.detach().cpu().tolist(),
            'inference_buffers_made_writable':loaded['inference_buffers_made_writable']}
    report['passed']=finite and difference==0 and bool(runner.alg.optimizer.state) and all(row['passed'] for row in motors)
    save_json(output/'runner_smoke.json',report)
    return report
