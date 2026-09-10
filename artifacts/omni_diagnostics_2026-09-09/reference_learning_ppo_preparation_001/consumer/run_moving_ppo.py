#!/usr/bin/env python3
"""External moving consumer; bounded phases, new identity, immutable checkpoints."""
import argparse
import faulthandler
import json
from pathlib import Path
import sys
import time
import traceback
sys.dont_write_bytecode=True
from campaign_contract import verify_inputs,sha,read,save,_quiet_pass,RUNTIME,RUNTIME_SCHEMA,validate_learning_decision

MODES=('calibrate','learning_recovery_32','profile_32','profile_128','train_10','train_25',
       'evaluate_initial','evaluate_010','evaluate_025','quiet_010','quiet_025')


def accepted_calibration(args,identity):
    folder=args.output.parent/'calibration';state=read(folder/'state.json')
    if state.get('status')!='completed' or state.get('identity')!=identity or state.get('mode')!='calibrate':
        raise ValueError('Exact completed same-campaign calibration required')
    calibration=read(folder/'calibration.json')
    if calibration.get('passed') is not True or calibration.get('initial_std')!=.02 or set(calibration.get('trials',{}))!={'zero_mean','sampled'}:
        raise ValueError('Both explicit exploration trials required')
    for trial in calibration['trials'].values():_quiet_pass(trial,32)
    policy=state['policy_identity']
    if policy['bindings']['calibration_sha256']!=sha(folder/'calibration.json'):
        raise ValueError('Calibration identity changed')
    meta=read(folder/'initial.pt.json')
    if meta!={**policy,'checkpoint_sha256':sha(folder/'initial.pt')} or state.get('initial_checkpoint')!=meta:
        raise ValueError('Scratch initial checkpoint changed')
    return folder,policy


def require_profile(args,identity,replicas):
    path=args.output.parent/('profile_'+str(replicas))/'state.json'
    state=read(path)
    if state.get('status')!='completed' or state.get('identity')!=identity or state.get('profile_passed') is not True:
        raise ValueError('Completed same-source device profile required')
    if state.get('profile_controls')!=512 or state.get('replicas')!=replicas:
        raise ValueError('Wrong profile size or duration')
    return sha(path)


def require_continuation(args,identity):
    if args.decision_receipt is None:raise ValueError('No automatic10-to25 update allocation')
    receipt=read(args.decision_receipt)
    if receipt.get('schema')!='moving_PPO_10_to25_review_v1' or receipt.get('identity')!=identity or receipt.get('accepted') is not True:
        raise ValueError('Explicit same-source measured continuation review required')
    expected={name:sha(args.output.parent/name/'state.json') for name in ('train_10','evaluate_initial','evaluate_010','quiet_010')}
    if receipt.get('phase_state_sha256')!=expected:raise ValueError('Continuation review does not bind exact completed phases')
    for name in expected:
        state=read(args.output.parent/name/'state.json')
        if state.get('status')!='completed' or state.get('identity')!=identity:
            raise ValueError('Continuation requires completed matched phases')
        if name.startswith('evaluate') and state.get('retention_passed') is not True:raise ValueError('Retention gate did not pass')
        if name=='quiet_010' and state.get('quiet_passed') is not True:raise ValueError('Quiet gate did not pass')
    if receipt.get('measured_improvement_review') is not True or receipt.get('no_direction_or_quiet_regression_review') is not True:
        raise ValueError('Quantitative improvement and no-regression review are required, beyond gate passage')
    return sha(args.decision_receipt)


def recovery_metrics(session):
    import numpy as np
    rows=session.rows;start=200
    active=np.stack([r['learning_active'] for r in rows])[start:]
    command=np.stack([r['requested_command'] for r in rows])[start:]
    episode=np.stack([r['episode_id'] for r in rows])
    position=np.stack([r['position_world_m'] for r in rows])
    rotation=np.stack([r['rotation_world_from_body'] for r in rows])
    delta=position[start:]-position[start-1:-1]
    axis=-rotation[start-1:-1,:,:,1];axis[:,:,2]=0
    axis/=np.maximum(np.linalg.norm(axis,axis=-1,keepdims=True),1e-20)
    same=episode[start:]==episode[start-1:-1]
    moving=active&(command[:,:,0]!=0)&same
    forward=(delta*axis).sum(-1)
    term=np.stack([r['training_terminated'] for r in rows])[start:]
    timeout=np.stack([r['training_truncated'] for r in rows])[start:]
    raw_velocity=np.stack([r['velocity_body_mps'] for r in rows])[start:]
    return [dict(row=i,active_controls=int(active[:,i].sum()),excluded_recovery_controls=int((~active[:,i]).sum()),
        finite_terminal_transitions=int(term[:,i].sum()),time_limit_transitions=int(timeout[:,i].sum()),
        moving_intervals_without_reset_crossing=int(moving[:,i].sum()),
        measured_forward_increment_sum_m=float(forward[moving[:,i],i].sum()),
        raw_SDK_mean_forward_mps=float(-raw_velocity[moving[:,i],i,1].mean()) if moving[:,i].any() else 0.,
        physical_admission=False) for i in range(session.num_envs)]


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    for key in ('source-root','run','device-run','bridge','observation-bundle','package','standing','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--mode',choices=MODES,required=True)
    parser.add_argument('--decision-receipt',type=Path)
    parser.add_argument('--preflight-only',action='store_true')
    early,_=parser.parse_known_args()
    for key,value in vars(early).items():
        if isinstance(value,Path):setattr(early,key,value.resolve())
    if early.output.name!=('calibration' if early.mode=='calibrate' else early.mode):
        raise ValueError('Exact phase output directory name required')
    roots=[early.source_root,early.run,early.device_run,early.bridge,early.observation_bundle,early.package,Path(__file__).parent.resolve()]
    if early.output.exists() or any(p==early.output or p in early.output.parents for p in roots):
        raise ValueError('Fresh output outside all immutable inputs required')
    identity=verify_inputs(early,require_standing=not early.preflight_only)
    if early.preflight_only:print(json.dumps(identity,indent=2));return 0
    calibration_folder=policy_identity=None
    if early.mode!='calibrate':calibration_folder,policy_identity=accepted_calibration(early,identity)
    profile_bindings={}
    learning_decision=validate_learning_decision(early,identity) if early.mode=='train_10' else None
    continuation=require_continuation(early,identity) if early.mode=='train_25' else None
    sys.path.insert(0,str(early.source_root/'tools'))
    from c_study_runtime import bootstrap_c_study_runtime
    runtime=bootstrap_c_study_runtime()
    if runtime['runtime_tree_sha256']!=RUNTIME:raise ValueError('Wrong pinned C runtime')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser);parser.add_argument('-h','--help',action='help');args=parser.parse_args()
    for key,value in vars(early).items():setattr(args,key,value)
    if args.device!='cuda:0':raise ValueError('Explicit CUDA0 physical run required')
    args.output.mkdir(parents=True)
    state={'status':'initializing','mode':args.mode,'identity':identity,'runtime_binding':runtime,
        'PPO_updates_completed':0,'Stage2_complete':False,'started_unix':time.time(),
        'policy_training_started':False,
        'profile_bindings':profile_bindings,'continuation_receipt_sha256':continuation,'learning_decision':learning_decision}
    save(args.output/'state.json',state);app=env=session=runner=None
    try:
        faulthandler.enable();faulthandler.dump_traceback_later(45,repeat=True)
        print('REFERENCE_SCREEN_APP_START',flush=True);app=AppLauncher(args).app
        faulthandler.cancel_dump_traceback_later();print('REFERENCE_SCREEN_APP_READY',flush=True)
        import numpy as np
        import torch
        import warp as wp
        import importlib.metadata
        import rsl_rl
        sys.path.insert(0,str(args.bridge));sys.path.insert(0,str(args.observation_bundle))
        from sensor_freshness import verify_installed_sources
        from reference_physics_env import build_reference_environment
        from screen_contract import OPTIONS
        from moving_session import MovingSession
        from rollout_session import ReferenceResidualSession
        from physical_scores import quiet_trial,forward_stop_trial
        from moving_runner import initialize_scratch,contract,runner_config,save_checkpoint,reload_checkpoint
        from masked_runner import MaskedRunner
        if importlib.metadata.version('rsl-rl-lib')!='5.0.1':raise ValueError('Exact RSL5.0.1 required')
        for row in read(Path(__file__).parent/'RSL_SOURCE_PARITY.json')['sources']:
            if sha(Path(rsl_rl.__file__).parent.parent/row['relative_file'])!=row['installed_Spark_sha256']:
                raise ValueError('Installed RSL implementation differs')
        state['sensor_sources']=verify_installed_sources()
        import importlib
        state['reset_API_sources']={}
        for relative,record in read(Path(__file__).parent/'SDK_RESET_API.json').items():
            module=importlib.import_module('isaaclab.'+relative[:-3].replace('/','.'))
            actual=sha(Path(module.__file__))
            if actual!=record['sha256']:raise ValueError('Installed reset/step API changed: '+relative)
            state['reset_API_sources'][relative]=actual
        moving_mode=args.mode.startswith(('profile','train')) or args.mode=='learning_recovery_32'
        args.num_envs=128 if args.mode=='profile_128' else (1 if args.mode.startswith('evaluate') else 32)
        args.variant='f050_t060';args.stance_index=0;state['replicas']=args.num_envs
        env,manifest,oldplan,record,stance,layout,audit,controller=build_reference_environment(args,OPTIONS)
        env.reset(seed=0);env.episode_length_buf.zero_()
        points=np.asarray(read(args.source_root/'robot/hexapod_mkii_length_study/candidate_c_reference.json')['stance']['toes'],dtype=float)
        if moving_mode:
            session=MovingSession(env,layout,points,stance,warp_to_torch=wp.to_torch,output=args.output/'raw')
        else:
            session=ReferenceResidualSession(env,layout,points,stance,warp_to_torch=wp.to_torch,
                initial_commands=[[.005,0.,0.]] if args.mode.startswith('evaluate') else None)
        state.update(asset_audit=audit,layout=layout,controller=controller)
        with session:
            if session.encoder.spec()['schema_sha256']!=RUNTIME_SCHEMA:raise ValueError('Named846/849 schema changed')
            torch.manual_seed(157)
            runner=MaskedRunner(session,runner_config(env.device),log_dir=str(args.output/'runner_logs'),device=env.device)
            state['initialization']=initialize_scratch(runner);runner.alg.eval_mode()
            if moving_mode:runner.alg.evidence_callback=session.integrity.bootstrap
            def act(stochastic=False):
                with torch.inference_mode():return runner.alg.actor(session.get_observations(),stochastic_output=stochastic)
            if args.mode=='calibrate':
                calibration={'passed':False,'initial_std':.02,'trials':{}}
                save(args.output/'calibration.json',calibration)
                for label,stochastic in [('zero_mean',False),('sampled',True)]:
                    start=len(session.rows)
                    for _ in range(1000):session.step(act(stochastic))
                    trial=quiet_trial(session.rows,session.names,start,len(session.rows))
                    calibration['trials'][label]=trial;save(args.output/'calibration.json',calibration)
                    if not trial['passed']:raise RuntimeError('Unchanged calibration rejected: '+label)
                calibration['passed']=True;save(args.output/'calibration.json',calibration)
                bindings={key:identity[key] for key in ('physical_source_sha256','observation_freeze_sha256','device_admission_sha256','directional_admission_sha256','standing_admission_sha256')}
                bindings.update(calibration_sha256=sha(args.output/'calibration.json'),consumer_source_sha256=identity['consumer_freeze_sha256'],observation_schema_sha256=RUNTIME_SCHEMA)
                policy_identity=contract('admitted_bounded_moving_PPO',bindings)
                state['initial_checkpoint']=save_checkpoint(runner,args.output/'initial.pt',policy_identity)
            else:
                which={'evaluate_010':('train_10','decision_010.pt'),'quiet_010':('train_10','decision_010.pt'),
                    'train_25':('train_10','decision_010.pt'),'evaluate_025':('train_25','decision_025.pt'),
                    'quiet_025':('train_25','decision_025.pt')}.get(args.mode,('calibration','initial.pt'))
                checkpoint=args.output.parent/which[0]/which[1]
                state['input_checkpoint_sha256']=sha(checkpoint)
                state['reload']=reload_checkpoint(runner,checkpoint,policy_identity);runner.alg.eval_mode()
                if args.mode=='learning_recovery_32':
                    timings=[];valid=0;start=time.perf_counter()
                    for _ in range(512):
                        beginning=time.perf_counter();action=act(False)
                        if torch.count_nonzero(action):raise RuntimeError('Recovery admission requires exact zero actor residual')
                        observation,_,_,extra=session.step(action)
                        runner.alg.audit_unoptimized_transition(observation,extra)
                        valid+=int(extra['masked_transition']['learnable'].sum());timings.append(time.perf_counter()-beginning)
                    state.update(recovery_controls=512,recovery_wall_s=time.perf_counter()-start,
                        recovery_control_seconds=timings,active_transitions=valid,
                        recovery_per_environment=recovery_metrics(session),learning_recovery_passed=False)
                    proof=session.integrity.summary()
                    if (session.failure is not None or proof['completed_finite_recovery_rows']<1
                        or not proof['all_records_verified'] or proof['bootstrap_event_records']!=proof['finite_reset_events']):
                        raise RuntimeError('Actual finite-event/recovery integrity proof incomplete; no PPO allocation')
                    if runner.alg.storage.step or runner.alg.normalizer_rows or runner.alg.optimizer.state:
                        raise RuntimeError('Recovery admission silently collected or optimized PPO')
                    state['learning_recovery_passed']=True
                elif args.mode.startswith('profile'):
                    measurements=[];start=time.perf_counter();valid=0
                    for _ in range(512):
                        beginning=time.perf_counter();_,_,done,extra=session.step(act(False))
                        valid+=int(extra['masked_transition']['learnable'].sum());measurements.append(time.perf_counter()-beginning)
                    state.update(profile_controls=512,profile_wall_s=time.perf_counter()-start,
                        active_transitions=valid,profile_control_seconds=measurements,
                        profile_passed=not session.events and session.failure is None)
                    if not state['profile_passed']:raise RuntimeError('Zero-residual device profile had a row outcome; review before allocation')
                elif args.mode.startswith('train'):
                    maximum=10 if args.mode=='train_10' else 25
                    if runner.current_learning_iteration!=(0 if maximum==10 else 10):raise ValueError('Wrong moving checkpoint update count')
                    state['policy_training_started']=True;save(args.output/'state.json',state)
                    def progress(report):
                        with (args.output/'updates.jsonl').open('a') as f:f.write(json.dumps(report,allow_nan=False)+'\n')
                        state['PPO_updates_completed']=report['updates_completed'];save(args.output/'state.json',state)
                    state['updates']=runner.collect_updates(maximum-runner.current_learning_iteration,maximum_updates=maximum,after_update=progress)
                    name='decision_'+str(maximum).zfill(3)+'.pt'
                    state['decision_checkpoint']=save_checkpoint(runner,args.output/name,policy_identity)
                    runner.alg.eval_mode()
                    with torch.inference_mode():fixed=session.get_observations();before=runner.alg.actor(fixed).clone()
                    state['reload']=reload_checkpoint(runner,args.output/name,policy_identity)
                    with torch.inference_mode():after=runner.alg.actor(fixed)
                    if not torch.equal(before,after):raise RuntimeError('Decision checkpoint deterministic reload differed')
                    state['deterministic_reload_max_difference']=0.
                    state['learned_std_per_joint']=runner.alg.actor.distribution.std_param.detach().cpu().tolist()
                    state['optimizer_entries']=len(runner.alg.optimizer.state)
                    state['evaluation_still_required']=True
                elif args.mode.startswith('evaluate'):
                    for k in range(2200):
                        if k==1199:session.queue_commands([[0.,0.,0.]])
                        session.step(act(False))
                    result=forward_stop_trial(session);save(args.output/'retention.json',result)
                    state['retention_passed']=result['passed']
                    if not result['passed']:raise RuntimeError('Original deterministic forward/stop admission rejected')
                else:
                    start=len(session.rows)
                    for _ in range(1000):session.step(act(False))
                    result=quiet_trial(session.rows,session.names,start,len(session.rows));save(args.output/'quiet.json',result)
                    state['quiet_passed']=result['passed']
                    if not result['passed']:raise RuntimeError('Original32-replica quiet admission rejected')
            state['policy_identity']=policy_identity
            state['status']='completed'
    except BaseException as error:
        state.update(status='rejected',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        try:
            if session is not None:
                try:
                    if hasattr(session,'recovery'):
                        session.export();state['session']=read(args.output/'raw/session.json')
                        if args.mode=='learning_recovery_32':
                            ledger_path=args.output/'raw/learning_ledger/summary.json'
                            record={'schema':'moving_learning_recovery_admission_v1','identity':identity,
                                'infrastructure_passed':state.get('learning_recovery_passed',False),
                                'ledger':read(ledger_path),'ledger_sha256':sha(ledger_path),
                                'physical_admission':False,'automatic_training_allowed':False}
                            save(args.output/'learning_integrity.json',record)
                            state['learning_integrity_sha256']=sha(args.output/'learning_integrity.json')
                    else:state['session']=session.export(args.output)
                except BaseException as error:state.update(status='failed',export_error=repr(error))
            try:
                if runner is not None and getattr(runner.logger,'writer',None) is not None:
                    runner.logger.writer.flush();runner.logger.writer.close()
            except BaseException as error:state.update(status='failed',logger_finalize_error=repr(error))
            try:
                state['source_inputs_unchanged']=verify_inputs(args)==identity
                if not state['source_inputs_unchanged']:raise ValueError('Inputs changed')
            except BaseException as error:state.update(status='failed',integrity_error=repr(error))
            state['finished_unix']=time.time();save(args.output/'state.json',state)
        finally:
            try:
                if env is not None:env.close()
            except BaseException as error:
                state.update(status='failed',environment_close_error=repr(error));save(args.output/'state.json',state)
            finally:
                try:
                    if app is not None:app.close()
                except BaseException as error:
                    state.update(status='failed',application_close_error=repr(error));save(args.output/'state.json',state)
    if state['status']!='completed':raise RuntimeError('Bounded phase failed final checks')
    return 0

if __name__=='__main__':raise SystemExit(main())
