#!/usr/bin/env python3
"""Two scratch PPO updates with actual calibration and separate matched retention."""
import argparse,faulthandler,json,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
from physical_contract import verify_inputs,save,sha,read,RUNTIME,RUNTIME_SCHEMA

def main():
    parser=argparse.ArgumentParser(add_help=False)
    for key in ('source-root','run','device-run','bridge','observation-bundle','package','standing','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--mode',choices=('smoke','evaluate_initial','evaluate_final'),required=True)
    parser.add_argument('--smoke-output',type=Path);parser.add_argument('--preflight-only',action='store_true')
    early,_=parser.parse_known_args()
    for key,value in vars(early).items():
        if isinstance(value,Path):setattr(early,key,value.resolve())
    inputs=(early.source_root,early.run,early.device_run,early.bridge,early.observation_bundle,early.package,Path(__file__).parent.resolve())
    if early.output.exists() or any(p==early.output or p in early.output.parents for p in inputs):raise ValueError('Fresh output outside immutable inputs required')
    identity=verify_inputs(early,require_standing=not early.preflight_only)
    if early.mode=='smoke' and early.smoke_output is not None:raise ValueError('Scratch smoke cannot consume a prior policy')
    if early.mode!='smoke' and early.smoke_output!=(early.output.parent/'smoke'):raise ValueError('Matched same-campaign smoke required')
    if early.preflight_only:print(json.dumps(identity,indent=2));return 0
    sys.path.insert(0,str(early.source_root/'tools'))
    from c_study_runtime import bootstrap_c_study_runtime
    runtime=bootstrap_c_study_runtime()
    if runtime['runtime_tree_sha256']!=RUNTIME:raise ValueError('Wrong pinned C runtime')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser);parser.add_argument('-h','--help',action='help');args=parser.parse_args()
    for key,value in vars(early).items():setattr(args,key,value)
    if args.device!='cuda:0':raise ValueError('Explicit CUDA0 physical smoke required')
    args.output.mkdir(parents=True);state=dict(status='initializing',mode=args.mode,identity=identity,runtime_binding=runtime,
        policy_training_started=False,PPO_updates_completed=0,Stage2_complete=False,started_unix=time.time())
    save(args.output/'state.json',state);app=env=session=runner=None
    try:
        faulthandler.enable();faulthandler.dump_traceback_later(45,repeat=True)
        print('REFERENCE_SCREEN_APP_START',flush=True);app=AppLauncher(args).app
        faulthandler.cancel_dump_traceback_later();print('REFERENCE_SCREEN_APP_READY',flush=True)
        import numpy as np
        import torch
        import warp as wp
        sys.path.insert(0,str(args.bridge));sys.path.insert(0,str(args.observation_bundle))
        from sensor_freshness import verify_installed_sources
        from reference_physics_env import build_reference_environment
        from screen_contract import OPTIONS
        from rollout_session import ReferenceResidualSession
        from physical_scores import quiet_trial,forward_stop_trial
        from residual_runner import initialize_scratch,contract,runner_config,save_checkpoint,reload_checkpoint,plan
        import rsl_rl,importlib.metadata
        from rsl_rl.runners import OnPolicyRunner
        if importlib.metadata.version('rsl-rl-lib')!='5.0.1':raise ValueError('Exact reviewed RSL5.0.1 required')
        for row in read(Path(__file__).parent/'RSL_SOURCE_PARITY.json')['sources']:
            path=Path(rsl_rl.__file__).parent.parent/row['relative_file']
            if sha(path)!=row['installed_Spark_sha256']:raise ValueError('RSL implementation changed: '+row['relative_file'])
        state['sensor_source_readback']=verify_installed_sources()
        args.num_envs=32 if args.mode=='smoke' else 1;args.variant='f050_t060';args.stance_index=0
        env,manifest,oldplan,record,stance,layout,audit,controller=build_reference_environment(args,OPTIONS)
        env.reset(seed=0);env.episode_length_buf.zero_()
        points=np.asarray(read(args.source_root/'robot/hexapod_mkii_length_study/candidate_c_reference.json')['stance']['toes'],dtype=float)
        initial_commands=None if args.mode=='smoke' else [[.005,0.,0.]]
        session=ReferenceResidualSession(env,layout,points,stance,warp_to_torch=wp.to_torch,initial_commands=initial_commands)
        state.update(asset_audit=audit,layout=layout,controller=controller)
        with session:
            if session.encoder.spec()['schema_sha256']!=RUNTIME_SCHEMA:raise ValueError('Runtime named846/849 schema differs from actual device admission')
            torch.manual_seed(157)
            runner=OnPolicyRunner(session,runner_config(env.device),log_dir=str(args.output/'runner_logs'),device=env.device)
            state['initialization']=initialize_scratch(runner);runner.alg.eval_mode()
            def actor_step(stochastic=False):
                with torch.inference_mode():action=runner.alg.actor(session.get_observations(),stochastic_output=stochastic)
                return session.step(action)
            if args.mode=='smoke':
                calibration=dict(scope='actual32replica_standing_position_residual_exploration',initial_std=.02,passed=False,trials={})
                save(args.output/'calibration.json',calibration)
                for label,stochastic in [('zero_mean',False),('sampled',True)]:
                    start=len(session.rows)
                    for _ in range(1000):actor_step(stochastic)
                    result=quiet_trial(session.rows,session.names,start,len(session.rows))
                    calibration['trials'][label]=result;save(args.output/'calibration.json',calibration)
                    if not result['passed']:raise RuntimeError('Per-replica exploration calibration rejected: '+label)
                calibration['passed']=True;save(args.output/'calibration.json',calibration)
                bindings={key:identity[key] for key in ('physical_source_sha256','observation_freeze_sha256','device_admission_sha256','directional_admission_sha256','standing_admission_sha256')}
                bindings.update(calibration_sha256=sha(args.output/'calibration.json'),consumer_source_sha256=identity['consumer_freeze_sha256'],observation_schema_sha256=RUNTIME_SCHEMA)
                policy_identity=contract('admitted_two_update_physical_smoke',bindings)
                state['policy_identity']=policy_identity
                state['initial_checkpoint']=save_checkpoint(runner,args.output/'initial.pt',policy_identity)
                state['policy_training_started']=True;save(args.output/'state.json',state)
                session.optimization_steps=0
                runner.learn(num_learning_iterations=2,init_at_random_ep_len=False)
                if session.optimization_steps!=48:raise RuntimeError('Runner did not perform exactly48controls')
                session.optimization_steps=None;state['PPO_updates_completed']=2
                state['final_checkpoint']=save_checkpoint(runner,args.output/'final.pt',policy_identity)
                runner.alg.eval_mode()
                with torch.inference_mode():fixed=session.get_observations();before=runner.alg.actor(fixed,stochastic_output=False).clone()
                # Exercise actual writable-buffer restoration and Adam/model load;
                # no more optimization is permitted after this readback.
                state['reload']=reload_checkpoint(runner,args.output/'final.pt',policy_identity)
                runner.alg.eval_mode()
                with torch.inference_mode():after=runner.alg.actor(fixed,stochastic_output=False)
                if not torch.equal(before,after):raise RuntimeError('Deterministic action changed on strict reload')
                state['deterministic_reload_max_difference']=float((before-after).abs().max())
                start=len(session.rows)
                for _ in range(1000):actor_step(False)
                quiet=quiet_trial(session.rows,session.names,start,len(session.rows));save(args.output/'post_update_quiet.json',quiet)
                state['post_update_quiet_passed']=quiet['passed']
                if not quiet['passed']:raise RuntimeError('PPO lost required per-replica quiet standing')
                state['learned_std_per_joint']=runner.alg.actor.distribution.std_param.detach().cpu().tolist()
                state['optimizer_entries']=len(runner.alg.optimizer.state)
                state['matched_forward_stop_evaluations_still_required']=True
            else:
                smoke=read(args.smoke_output/'state.json')
                if smoke.get('status')!='completed' or smoke.get('identity')!=identity or smoke.get('PPO_updates_completed')!=2:
                    raise ValueError('Exact accepted same-source two-update smoke required')
                if sha(args.smoke_output/'calibration.json')!=smoke['policy_identity']['bindings']['calibration_sha256']:
                    raise ValueError('Calibration bytes differ from checkpoint identity')
                which='initial' if args.mode=='evaluate_initial' else 'final'
                checkpoint=args.smoke_output/(which+'.pt')
                state['reload']=reload_checkpoint(runner,checkpoint,smoke['policy_identity']);runner.alg.eval_mode()
                state['checkpoint_sha256']=sha(checkpoint);state['policy_identity']=smoke['policy_identity']
                for k in range(2200):
                    if k==1199:session.queue_commands([[0.,0.,0.]])
                    actor_step(False)
                result=forward_stop_trial(session);save(args.output/'retention.json',result)
                if not result['passed']:raise RuntimeError('Full-C deterministic forward-stop retention rejected')
                state['retention_passed']=True
            state['status']='completed'
    except Exception as error:
        state.update(status='rejected',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        try:
            if session is not None:
                try:state['session']=session.export(args.output)
                except Exception as error:state.update(status='failed',export_error=repr(error))
            try:
                if runner is not None and getattr(runner.logger,'writer',None) is not None:
                    runner.logger.writer.flush();runner.logger.writer.close()
            except Exception as error:state.update(status='failed',logger_finalize_error=repr(error))
            try:
                state['source_inputs_unchanged']=verify_inputs(args)==identity
                if not state['source_inputs_unchanged']:raise ValueError('Consumer/input identity changed')
            except Exception as error:state.update(status='failed',integrity_error=repr(error),source_inputs_unchanged=False)
            state['finished_unix']=time.time();save(args.output/'state.json',state)
        finally:
            try:
                if env is not None:env.close()
            except Exception as error:
                state.update(status='failed',environment_close_error=repr(error));save(args.output/'state.json',state)
            finally:
                try:
                    if app is not None:app.close()
                except Exception as error:
                    state.update(status='failed',application_close_error=repr(error));save(args.output/'state.json',state)
    if state['status']!='completed':raise RuntimeError('Bounded PPO phase failed final integrity/export checks')
    return 0
if __name__=='__main__':raise SystemExit(main())
