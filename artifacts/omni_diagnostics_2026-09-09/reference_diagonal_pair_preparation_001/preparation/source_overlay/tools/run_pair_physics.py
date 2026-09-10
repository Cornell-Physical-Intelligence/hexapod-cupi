#!/usr/bin/env python3
"""Bounded full-C named diagonal-pair static load transfer; no wave/PPO or body-pose prescription."""
from __future__ import annotations
import argparse,faulthandler,json,time,traceback
from pathlib import Path
from c_study_runtime import bootstrap_c_study_runtime
RUNTIME=bootstrap_c_study_runtime()
from isaaclab.app import AppLauncher
from screen_contract import OPTIONS,STARTUP,save
from pair_screen_contract import PAIR_PROTOCOL,PAIR_CASES,preflight
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--geometry-reference',type=Path,required=True);p.add_argument('--admission',type=Path,required=True)
p.add_argument('--variant',default='f050_t060');p.add_argument('--stance-index',type=int,default=0)
p.add_argument('--pair-case',choices=list(PAIR_CASES),required=True)
p.add_argument('--mode',choices=['pair'],required=True);p.add_argument('--num-envs',type=int,required=True);p.add_argument('--steps',type=int,required=True)
AppLauncher.add_app_launcher_args(p);args=p.parse_args();SOURCE=Path(__file__).resolve().parents[1]
identity,geometry=preflight(args,SOURCE);args.output.mkdir(parents=True)
faulthandler.enable();faulthandler.dump_traceback_later(45,repeat=True)
print('REFERENCE_SCREEN_APP_START',flush=True);app=AppLauncher(args).app
faulthandler.cancel_dump_traceback_later();print('REFERENCE_SCREEN_APP_READY',flush=True)
import numpy as np
import torch
from reference_physics_env import build_reference_environment
from canonical_stance_startup import CanonicalStanceStartup
from physics_telemetry import array,capture_measured_state,pre_reset_capture,require_single_pre_reset_sample
from physics_substeps import PhysicsSubstepRecorder
from load_transfer import PairLoadTransfer
from score_transfer import check_substep_batch,score_transfer
from pair_rollout import emit_control,check_measured_pair_state


def serializable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serializable(v) for v in value]
    return value


def stack(rows,names):
    return {**{k:np.stack([r[k] for r in rows]) for k in rows[0]},'joint_names':np.asarray(names)} if rows else {}


def main():
    env=None;startup_recorder=None;pair_recorder=None;startup_rows=[];pair_rows=[];references=[];post_measurements=[]
    state={'status':'initializing','mode':'pair','identity':identity,'runtime_binding':RUNTIME,
        'pair_case':args.pair_case,'pair_partition':PAIR_CASES[args.pair_case],'started_unix':time.time(),'stage2_complete':False,'terrain_qualified':False,
        'policy_training_started':False,'physics_qualification':False,'proposed_diagnostic_criteria_met':False}
    clock={'step':0};failure=None
    try:
        env,manifest,plan,record,stance,layout,asset_audit,controller_contract=build_reference_environment(args,OPTIONS)
        state.update(asset_audit=asset_audit,controller=controller_contract,layout=layout)
        points=np.asarray(geometry['stance']['toes'],dtype=float)
        if tuple(geometry['joint_names'])!=layout['joint_names_leg_major']:raise ValueError('Named geometry order differs')
        env.reset(seed=0);env.episode_length_buf.zero_()
        zero=torch.zeros((1,18),device=env.device)
        initial=array(env.reference_residual_controller.reference_position)
        nominal=np.asarray([stance['joint_positions_rad'][name] for name in layout['joint_names_runtime']],dtype=np.float32).astype(np.float64)[None,:]
        if not np.array_equal(nominal,array(env._robot.data.default_joint_pos).astype(np.float64)):
            raise RuntimeError('Named canonical target differs from actual articulation default')
        limits=array(env._robot.data.soft_joint_pos_limits)
        startup=CanonicalStanceStartup(initial,nominal,limits[...,0],limits[...,1],duration_s=STARTUP['duration_s'],dt=env.step_dt)
        save(args.output/'startup_reference.json',{**startup.contract(),'joint_names_runtime':layout['joint_names_runtime']})
        def capture():
            row=capture_measured_state(env,layout,points,time_s=(clock['step']+1)*env.step_dt)
            target=env.reference_residual_target
            if target is None:raise RuntimeError('Missing emitted target at pre-reset capture')
            row.update({k:array(v) for k,v in target.items()})
            row['requested_command']=np.zeros((1,3))
            return row
        state['status']='running_startup';save(args.output/'state.json',state)
        startup_recorder=PhysicsSubstepRecorder(env)
        with startup_recorder,pre_reset_capture(env,capture) as captured:
            for step in range(200):
                clock['step']=step;ref=startup.sample(step+1)
                row,term,trunc=emit_control(env,ref,captured,startup_recorder,step,startup_rows,zero)
                if term.any() or trunc.any():raise ValueError('Startup terminal event; preserve pre-reset evidence')
        startup_dir=args.output/'startup';startup_dir.mkdir()
        np.savez_compressed(startup_dir/'trace.npz',**stack(startup_rows,layout['joint_names_runtime']))
        state['startup_observer']=startup_recorder.export(startup_dir)
        snapshot=startup_rows[-1]  # Actual pre-reset flags and exact4s measurement.
        generator=PairLoadTransfer(layout['joint_names_runtime'],pair_leg_names=PAIR_CASES[args.pair_case]['pair_legs']);reset=generator.reset(snapshot)
        if not np.array_equal(reset['q_ref'],array(env.reference_residual_controller.reference_position)):
            raise RuntimeError('Pair reset changed canonical emitted target/preload')
        save(args.output/'pair_reset.json',serializable(reset))
        state['status']='running_pair';state['control_steps']=200;save(args.output/'state.json',state)
        pair_recorder=PhysicsSubstepRecorder(env)
        with pair_recorder,pre_reset_capture(env,capture) as captured:
            for local_step in range(1100):
                clock['step']=200+local_step
                snapshot=pair_rows[-1] if pair_rows else startup_rows[-1]
                ref=generator.step(snapshot,stop=False,dt=env.step_dt)
                if np.asarray(ref['valid']).shape!=(1,) or np.asarray(ref['valid']).dtype!=np.bool_ or not ref['valid'][0]:
                    failure='Pair reference rejected before target emission: '+str(ref.get('failure_reason'))
                    save(args.output/'rejected_reference.json',serializable(ref));break
                references.append(serializable(ref))
                row,term,trunc=emit_control(env,ref,captured,pair_recorder,local_step,pair_rows,zero)
                check_substep_batch(pair_recorder.rows[-8:],local_step,initial_counter=pair_recorder.base_counter,control_row=row)
                if term.any() or trunc.any():failure='Physical terminal event; preserved pre-reset sample';break
                post_measurements.append(serializable(check_measured_pair_state(generator,row)))
                if (local_step+1)%100==0:
                    state['control_steps']=201+local_step;save(args.output/'state.json',state)
                    print(f'PAIR_SCREEN {local_step+1}/1100',flush=True)
        data=stack(pair_rows,layout['joint_names_runtime'])
        if data:np.savez_compressed(args.output/'trace.npz',**data)
        save(args.output/'reference_states.json',references)
        save(args.output/'post_step_measurements.json',post_measurements)
        state['pair_observer']=pair_recorder.export(args.output)
        state['pair_observer_time_scope']={'relative_time_origin_physical_s':4.,'all1100diagnostic_controls_scored_without_settling_exclusion':True,
            'generic_observer_postsettle_prefix_is_only_a_relative_index_summary_not_the_pair_admission_window':True}
        result=None
        if failure is None and len(pair_rows)==1100:
            result=score_transfer(data,references,substeps=pair_recorder.data(),dt=env.step_dt,pair_leg_names=PAIR_CASES[args.pair_case]['pair_legs'])
        passed=bool(result and result['proposed_criteria_met'])
        state.update(status='completed' if passed else 'rejected',failure=failure,diagnostic_result=result,
            proposed_diagnostic_criteria_met=passed,control_steps=200+len(pair_rows),diagnostic_controls=len(pair_rows),finished_unix=time.time())
        save(args.output/'state.json',state);return 0 if passed else 1
    except Exception as exc:
        state.update(status='failed',failure=repr(exc),traceback=traceback.format_exc(),control_steps=len(startup_rows)+len(pair_rows),finished_unix=time.time())
        for recorder,directory,rows in [(startup_recorder,args.output/'startup',startup_rows),(pair_recorder,args.output,pair_rows)]:
            if recorder is not None and recorder.rows:
                try:
                    directory.mkdir(exist_ok=True);recorder.export(directory)
                    if rows:np.savez_compressed(directory/'partial_trace.npz',**stack(rows,layout['joint_names_runtime']))
                except Exception as error:state.setdefault('evidence_export_errors',[]).append(repr(error))
        save(args.output/'reference_states.json',references)
        save(args.output/'post_step_measurements.json',post_measurements);save(args.output/'state.json',state);return 1
    finally:
        if env is not None:env.close()
        app.close()
if __name__=='__main__':raise SystemExit(main())
