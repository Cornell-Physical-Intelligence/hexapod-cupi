#!/usr/bin/env python3
"""Short source009 hold with source-bound CUDA packing/reference/observation checks."""
import argparse,faulthandler,json,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
from smoke_contract import verify,save,RUNTIME

def main():
    parser=argparse.ArgumentParser(add_help=False)
    for key in ('source-root','run','observation-bundle','package','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--num-envs',type=int,choices=(1,32),required=True);parser.add_argument('--controls',type=int,default=264)
    parser.add_argument('--preflight-only',action='store_true')
    early,_=parser.parse_known_args()
    for key in ('source_root','run','observation_bundle','package','output'):setattr(early,key,getattr(early,key).resolve())
    if early.output.exists() or any(p==early.output or p in early.output.parents for p in (early.source_root,early.run,early.observation_bundle,early.package,Path(__file__).parent.resolve())):
        raise ValueError('Fresh output outside all immutable inputs required')
    identity=verify(early)
    if early.preflight_only:print(json.dumps(identity,indent=2));return 0
    sys.path.insert(0,str(early.source_root/'tools'))
    from c_study_runtime import bootstrap_c_study_runtime
    runtime=bootstrap_c_study_runtime()
    if runtime['runtime_tree_sha256']!=RUNTIME:raise ValueError('Wrong pinned C runtime')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser);parser.add_argument('-h','--help',action='help');args=parser.parse_args()
    for key,value in vars(early).items():setattr(args,key,value)
    if args.device!='cuda:0':raise ValueError('This is an explicit CUDA0 measurement smoke')
    args.output.mkdir(parents=True)
    state={'status':'initializing','identity':identity,'runtime_binding':runtime,'policy_training_started':False,'stage2_complete':False}
    save(args.output/'state.json',state);faulthandler.enable();faulthandler.dump_traceback_later(45,repeat=True)
    app=None;env=None;last=None;timings=[];sample_count=0;fresh_log=[];samples=[];physics_recorder=None;wave=None;encoder=None
    try:
        print('REFERENCE_SCREEN_APP_START',flush=True);app=AppLauncher(args).app
        faulthandler.cancel_dump_traceback_later();print('REFERENCE_SCREEN_APP_READY',flush=True)
        import numpy as np
        import torch
        import warp as wp
        sys.path.insert(0,str(args.observation_bundle))
        from device_pack import DeviceTelemetry
        from observation import ObservationBuilder
        from batch_wave import BatchWave005
        from sensor_freshness import ContactFreshness,verify_installed_sources
        from pre_reset_device import capture_before_reset,require_one
        from reference_physics_env import build_reference_environment
        from screen_contract import OPTIONS
        from canonical_stance_startup import CanonicalStanceStartup
        from physics_telemetry import array,capture_measured_state
        from physics_substeps import PhysicsSubstepRecorder
        state['sensor_source_readback']=verify_installed_sources()
        args.variant='f050_t060';args.stance_index=0
        env,manifest,plan,record,stance,layout,audit,controller_contract=build_reference_environment(args,OPTIONS)
        env.reset(seed=0);env.episode_length_buf.zero_()
        points=np.asarray(json.loads((args.source_root/'robot/hexapod_mkii_length_study/candidate_c_reference.json').read_text())['stance']['toes'],dtype=float)
        reader=DeviceTelemetry(env,layout,points);freshness=ContactFreshness(env,wp.to_torch)
        state.update(asset_audit=audit,layout=layout,controller=controller_contract)
        zero=torch.zeros((args.num_envs,18),device=env.device);commands=torch.zeros((args.num_envs,3),device=env.device,dtype=torch.float64)
        target=array(env.reference_residual_controller.reference_position)
        nominal=np.asarray([stance['joint_positions_rad'][name] for name in layout['joint_names_runtime']],dtype=np.float32).astype(np.float64)
        nominal=np.broadcast_to(nominal,target.shape).copy();limits=array(env._robot.data.soft_joint_pos_limits)
        if not np.array_equal(nominal,array(env._robot.data.default_joint_pos).astype(np.float64)):
            raise RuntimeError('Named source009 stance differs from actual nominal articulation target')
        startup=CanonicalStanceStartup(target,nominal,limits[...,0],limits[...,1],duration_s=2.,dt=.02)
        wave=None;encoder=None;clock={'step':0};capture_times=[];fresh_log=[];max_requested=0.;max_applied=0.
        def capture(terminated,truncated):
            torch.cuda.synchronize();start=time.perf_counter();fresh=freshness.read_after_normal_updates(8)
            value=reader.capture(env,time_s=torch.full((args.num_envs,),(clock['step']+1)*.02,device=env.device,dtype=torch.float64),terminated=terminated,truncated=truncated,
                contact_valid=fresh['contact_valid'],contact_age_s=fresh['contact_age_s'])
            torch.cuda.synchronize();capture_times.append((time.perf_counter()-start)*1000)
            fresh_log.append({k:v.detach().cpu().numpy().copy() for k,v in fresh.items() if isinstance(v,torch.Tensor)})
            return value
        physics_recorder=PhysicsSubstepRecorder(env)
        with physics_recorder,capture_before_reset(env,capture) as samples:
            for step in range(264):
                clock['step']=step
                if step==200:
                    wave=BatchWave005(layout['joint_names_runtime'],args.num_envs,device=env.device)
                    initial=wave.reset(last['measurement'],torch.ones(args.num_envs,device=env.device,dtype=torch.bool),torch.zeros(args.num_envs,device=env.device,dtype=torch.int64))
                    if not initial['valid'].all():raise RuntimeError('Measured settled reference reset rejected')
                    encoder=ObservationBuilder(layout['joint_names_runtime'],args.num_envs,device=env.device)
                    encoder.reset(torch.ones(args.num_envs,device=env.device,dtype=torch.bool),torch.zeros(args.num_envs,device=env.device,dtype=torch.int64),last['measurement']['time_s'])
                    # Seed the initial valid history from the actual settled sample.
                    seed=dict(last,reference=initial,controller=env.reference_residual_target,requested_twist=commands,
                        world_up=commands.new_tensor([0.,0.,1.]).expand(args.num_envs,-1),episode_ids=torch.zeros(args.num_envs,device=env.device,dtype=torch.int64),
                        step_indices=torch.zeros(args.num_envs,device=env.device,dtype=torch.int64),critic_simulator_reported_twist=last['measurement']['velocity_body_mps'])
                    seeded=encoder.build(seed)
                    if not seeded['valid'].all():raise RuntimeError('Settled initial observation rejected')
                torch.cuda.synchronize();start=time.perf_counter()
                if wave is None:
                    ref=startup.sample(step+1)
                    env.set_reference_targets(ref['q_ref'],ref['analytic_velocity_rad_s'],ref['analytic_acceleration_rad_s2'],ref['valid'])
                else:
                    ref=wave.step(last['measurement'],commands)
                    if not ref['valid'].all():raise RuntimeError('Reference tensor rejected a measured row')
                    env.set_reference_targets(ref['q_ref'],ref['v_ref'],ref['a_ref'],ref['valid'])
                torch.cuda.synchronize();ref_ms=(time.perf_counter()-start)*1000
                env.set_evaluation_targets(commands.float());before=len(samples);physics_recorder.begin_control(step)
                start=time.perf_counter()
                with torch.inference_mode():obs,reward,terminated,truncated,_=env.step(zero)
                torch.cuda.synchronize();step_ms=(time.perf_counter()-start)*1000
                last=require_one(samples,before,terminated,truncated);sample_count=step+1;m=last['measurement']
                physics_recorder.end_control({key:value.detach().cpu().numpy() for key,value in m.items()})
                if not m['measurement_valid'].all() or terminated.any() or truncated.any():raise RuntimeError('Measured bridge row invalid or physical reset occurred')
                if not all(torch.isfinite(value).all() for value in obs.values()) or not torch.isfinite(reward).all():raise RuntimeError('Legacy physics observations/reward nonfinite')
                if step>=200:
                    request=float(np.abs(np.stack([row['computed_torque_nm'] for row in physics_recorder.rows[-8:]])).max())
                    applied=float(np.abs(np.stack([row['applied_torque_nm'] for row in physics_recorder.rows[-8:]])).max())
                    max_requested=max(max_requested,request);max_applied=max(max_applied,applied)
                    if request>1.6 or applied>1.6 or m['distal_contact'].sum(-1).min()<6 or any(m[key].any() for key in ('shaft_contact','coxa_contact','femur_contact','base_contact')):
                        raise RuntimeError('Short hold exceeded unchanged torque/contact bounds')
                    packet=dict(last,reference=ref,controller=env.reference_residual_target,requested_twist=commands,
                        world_up=commands.new_tensor([0.,0.,1.]).expand(args.num_envs,-1),episode_ids=torch.zeros(args.num_envs,device=env.device,dtype=torch.int64),
                        step_indices=torch.full((args.num_envs,),step-199,device=env.device,dtype=torch.int64),critic_simulator_reported_twist=m['velocity_body_mps'])
                    torch.cuda.synchronize();start=time.perf_counter();encoded=encoder.build(packet);torch.cuda.synchronize();encode_ms=(time.perf_counter()-start)*1000
                    if not encoded['valid'].all() or encoded['policy'].shape!=(args.num_envs,846) or encoded['critic'].shape!=(args.num_envs,849):raise RuntimeError('New typed observation contract rejected')
                    timings.append({'control':step+1,'reference_and_target_set_ms':ref_ms,'env_step_including_capture_ms':step_ms,'nested_device_capture_ms':capture_times[-1],'observation_history_ms':encode_ms})
                if (step+1)%100==0:print('REFERENCE_DEVICE_SMOKE',step+1,flush=True)
        old=capture_measured_state(env,layout,points,time_s=264*.02);m=last['measurement'];compared=[]
        for key in sorted(set(old)&set(m)):
            actual=m['contact_point_world_m_raw'] if key=='contact_point_world_m' else m[key]
            np.testing.assert_allclose(actual.detach().cpu().numpy(),old[key],rtol=2e-6,atol=2e-7,equal_nan=True,err_msg=key)
            compared.append(key)
        np.savez_compressed(args.output/'sensor_clocks.npz',**{key:np.stack([r[key] for r in fresh_log]) for key in fresh_log[0]})
        save(args.output/'timings.json',{'samples':timings,'scope':'ActualCUDA hold only; env.step retains legacy CPU copies, source009400Hz observer and nested capture; noPPOthroughput projection'})
        save(args.output/'observation_schema.json',encoder.spec())
        state.update(status='completed',controls=sample_count,bridge_passed=True,comparison_keys=compared,postsettle_max_requested_nm=max_requested,postsettle_max_applied_nm=max_applied,
            short_hold_is_not_quiet_admission=True,actor_width=846,critic_width=849,substep_torque_checked=True,
            legacy_env_step_includes_original_CPU_telemetry_copies=True)
        np.savez_compressed(args.output/'final_observation.npz',policy=encoded['policy'].cpu().numpy(),critic=encoded['critic'].cpu().numpy(),
            raw_sdk_joint_velocity_rad_s=encoded['raw_sdk_joint_velocity_rad_s'].cpu().numpy(),interval_joint_rate_rad_s=encoded['interval_joint_rate_rad_s'].cpu().numpy(),interval_rate_valid=encoded['interval_rate_valid'].cpu().numpy())
    except Exception as error:
        if samples:last=samples[-1]
        state.update(status='failed',error=repr(error),traceback=traceback.format_exc(),controls=sample_count)
        raise
    finally:
        try:
            if physics_recorder is not None:
                try:state['substep_review']=physics_recorder.export(args.output)
                except Exception as export_error:state.update(status='failed',substep_export_error=repr(export_error))
            if wave is not None:
                import numpy as np
                np.savez_compressed(args.output/'last_reference_state.npz',**{key:value.detach().cpu().numpy() for key,value in wave.s.items()})
            if last is not None:
                import numpy as np
                np.savez_compressed(args.output/'last_device_sample.npz',**{key:value.detach().cpu().numpy() for key,value in last['measurement'].items()})
            if fresh_log:
                import numpy as np
                np.savez_compressed(args.output/'sensor_clocks.npz',**{key:np.stack([r[key] for r in fresh_log]) for key in fresh_log[0]})
            save(args.output/'timings.json',{'samples':timings,'scope':'ActualCUDA hold only; env.step retains legacy CPU copies, source009400Hz observer and nested capture; noPPOthroughput projection'})
            try:
                state['source_inputs_unchanged']=verify(args)==identity
                if not state['source_inputs_unchanged']:raise RuntimeError('Source identity changed')
            except Exception as integrity_error:
                state.update(status='failed',source_inputs_unchanged=False,integrity_error=repr(integrity_error))
            save(args.output/'state.json',state)
        finally:
            if env is not None:env.close()
            if app is not None:app.close()
    if state['status']!='completed':raise RuntimeError('Device smoke did not complete with unchanged sources')
    return 0
if __name__=='__main__':raise SystemExit(main())
