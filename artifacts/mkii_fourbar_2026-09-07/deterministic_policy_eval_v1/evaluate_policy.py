#!/usr/bin/env python3
"""External deterministic flat-policy evaluation; no learner updates or rendering."""
from __future__ import annotations
import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
import numpy as np
from evaluation_common import (SCHEMA,DT,CASES,ScheduledCommands,EndStateCapture,assert_command_observation,
    helpers,verify_inputs,tool_identity,require_request,schedule_descriptor,summarize,body_to_navigation)


def parser(add_launcher_args=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('source-dir','capture-tools-dir','checkpoint','admission','training-report','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    if add_launcher_args:
        add_launcher_args(p);p.set_defaults(visualizer=[],enable_cameras=False)
    return p


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv);early,_=parser().parse_known_args(argv)
    out=early.output_dir.resolve(strict=True);common=helpers(early.capture_tools_dir)
    if any((out/name).exists() for name in ('report.json','states.npz','metadata.json','metrics.json')):
        raise ValueError('Refusing to overwrite evaluation evidence')
    report={'schema':SCHEMA,'pass':False,'errors':[],'controls_completed':0,'policy_skill_pass':None,
        'hardware_admission':False,'navigation_or_terrain_qualification':False}
    persisted=False
    def finish(error=None):
        nonlocal persisted
        if persisted:return
        if error is not None:report['errors'].append(f'{type(error).__name__}: {error}')
        report['pass']=(not report['errors'] and report['controls_completed']==450
            and all(report.get(k) is True for k in ('checkpoint_unchanged','policy_state_unchanged','physics_coverage_pass')))
        common.write_json(out/'report.json',report);persisted=True
        print('DETERMINISTIC_POLICY_EVAL_RESULT',report['pass'],report['controls_completed'],flush=True)
    try:
        verified=verify_inputs(common,early.source_dir,early.checkpoint,early.admission,early.training_report)
        layout=common.environment_layout(early.source_dir,verified['runtime_manifest'])
        if os.environ.get('HEXAPOD_MKII_ENVIRONMENT_LAYOUT','grid_2m_v1')!=layout:raise ValueError('Environment layout mismatch')
        tools_before=tool_identity(Path(__file__).parent)
        if not (out/'admitted').is_file():raise ValueError('Evaluation has no guarded admission barrier')
        require_request(common.read_json(out/'request.json'),verified,tools_before)
        report.update(contract=verified['contract'],input_sha256=verified['input_sha256'],evaluation_tools_sha256=tools_before,
            schedule=schedule_descriptor(),num_envs=len(CASES),policy_dt_s=DT)
        source=early.source_dir.resolve()
        training=common.load_module(source/'isaaclab/train_mkii_fourbar.py','train_mkii_fourbar')
        from hexapod_env.tasks.mkii_fourbar_v1.register import register_mkii_fourbar_v1
        register_mkii_fourbar_v1()
        from isaaclab_tasks.utils import add_launcher_args,launch_simulation,resolve_task_config,setup_preset_cli
        saved=sys.argv
        try:
            sys.argv=[__file__,*argv];args,overrides=setup_preset_cli(parser(add_launcher_args))
            if overrides or args.device!='cuda:0' or args.visualizer not in (None,[]) or args.enable_cameras:
                raise ValueError('Only reviewed headless CUDA evaluation without task/physics overrides is allowed')
            sys.argv=[__file__];cfg,agent_cfg=resolve_task_config(common.TASK_ID,'rsl_rl_cfg_entry_point')
        finally:sys.argv=saved
        cfg.scene.num_envs=len(CASES);cfg.seed=agent_cfg.seed=verified['training']['seed']
        if cfg.sim.dt*cfg.decimation!=DT or cfg.episode_length_s<=9.:raise ValueError('Schedule exceeds reviewed task time contract')
        if importlib.metadata.version('rsl-rl-lib')!='5.0.1':raise ValueError('Unreviewed RSL release')
        if 'pxr' in sys.modules:raise ValueError('Standalone USD imported before Kit')
        with launch_simulation(cfg,args):
            env=guard=commands=end_capture=None;rows=[]
            try:
                import torch
                import gymnasium as gym
                from rsl_rl.runners import OnPolicyRunner
                from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
                from validate_mkii_fourbar import PhysicalMetrics
                env=gym.make(common.TASK_ID,cfg=cfg);raw=env.unwrapped
                comparison=training.compare_admitted_runtime(verified['runtime_manifest'],raw.runtime_manifest)
                report['runtime_manifest']=raw.runtime_manifest;report['admitted_runtime_comparison']=comparison
                if not comparison['pass'] or raw.num_envs!=len(CASES) or raw._robot.num_bodies!=31:
                    raise ValueError('Evaluation differs from admitted physical runtime')
                wrapped=RslRlVecEnvWrapper(env,clip_actions=agent_cfg.clip_actions)
                agent_cfg.device=str(raw.device);runner=OnPolicyRunner(wrapped,training.runner_config_dict(agent_cfg,'5.0.1'),log_dir=None,device=agent_cfg.device)
                infos=runner.load(str(args.checkpoint),strict=True,map_location=agent_cfg.device)
                if infos!={'contract':verified['contract'],'next_iteration':verified['sidecar']['next_iteration']}:raise ValueError('Embedded checkpoint identity mismatch')
                training.restore_adaptive_learning_rate(runner.alg)
                ph=training.parameters_digest(runner);ah=training.algorithm_state_digest(runner)
                if ph!=verified['training']['policy_after_sha256'] or ah!=verified['training']['algorithm_after_sha256']:
                    raise ValueError('Loaded actor/critic/optimizer/normalizer/scheduler differs from training')
                policy=runner.get_inference_policy(device=agent_cfg.device)
                def arr(value):
                    return (value.torch if hasattr(value,'torch') else value).detach().cpu().numpy().copy()
                def nav(value):
                    return body_to_navigation(arr(value))
                def state():
                    data=raw._robot.data;pads,shafts,slip,bad=raw._contact_state()
                    lin,ang=nav(data.root_lin_vel_b),nav(data.root_ang_vel_b)
                    return {'com_velocity_navigation':np.column_stack((lin[:,:2],ang[:,2])),
                        'com_linear_velocity_navigation':lin,'angular_velocity_navigation':ang,
                        'plate_pos_w_m':arr(data.root_pos_w),'plate_quat_w_xyzw':arr(data.root_quat_w),
                        'joint_pos_rad':arr(raw.motor_state('joint_pos')),'joint_vel_rad_s':arr(raw.motor_state('joint_vel')),
                        'joint_acc_rad_s2':arr(raw.motor_state('joint_acc')),
                        'applied_motor_torque_nm':arr(raw.motor_state('applied_torque')),
                        'raw_motor_demand_nm':arr(raw.motor_telemetry('raw_demand_nm')),
                        'motor_clipping_nm':arr(raw.motor_telemetry('clipping_nm')),
                        'burst_headroom':arr(raw.motor_telemetry('burst_headroom')),
                        'foot_pad':arr(pads),'foot_shaft':arr(shafts),'foot_slip_speed_mps':arr(slip),'invalid_contact':arr(bad)}
                commands=ScheduledCommands(raw);commands.__enter__()
                end_capture=EndStateCapture(raw,state);end_capture.__enter__()
                metrics=PhysicalMetrics(raw,raw.kinematics);metrics.window='settle'
                guard=training.PhysicalTrainingGuard(raw,metrics);guard.__enter__()
                with torch.inference_mode():
                    for step in range(450):
                        if (out/'stop_requested').exists():raise RuntimeError('Supervisor requested evaluation stop')
                        phase,obs=commands.prepare(step,wrapped);metrics.window=phase
                        row={'pre_'+k:v for k,v in state().items()}
                        row['observation_policy']=arr(obs['policy']);row['command_navigation']=arr(raw._commands)
                        actions=policy(obs,stochastic_output=False)
                        if not bool(torch.isfinite(actions).all()):raise ValueError('Nonfinite policy action')
                        row['policy_action']=arr(actions)
                        obs,rewards,dones,_=wrapped.step(actions)
                        if end_capture.calls!=step+1:raise ValueError('Missing/extra pre-reset final-state capture')
                        row.update({'end_'+k:v for k,v in end_capture.latest.items()})
                        row['post_plate_pos_w_m']=arr(raw._robot.data.root_pos_w)
                        row['post_plate_quat_w_xyzw']=arr(raw._robot.data.root_quat_w)
                        row['post_processed_target_rad']=arr(raw._processed_actions)
                        row['reward']=arr(rewards);row['done']=arr(dones)
                        row['terminated']=arr(raw.reset_terminated);row['truncated']=arr(raw.reset_time_outs)
                        for reason,value in raw.last_termination_reasons.items():row['termination_reason_'+reason]=arr(value)
                        row['post_command_navigation']=arr(raw._commands)
                        assert_command_observation(row['command_navigation'],arr(obs['policy']))
                        if not np.array_equal(row['command_navigation'],row['post_command_navigation']):raise ValueError('Sampler replaced scheduled command')
                        row['simulation_time_s']=(step+1)*DT
                        if any(not np.isfinite(value).all() for value in row.values()):raise ValueError('Nonfinite evaluation trace')
                        rows.append(row);report['controls_completed']=step+1;policy.reset(dones)
                        if (step+1)%50==0:common.write_json(out/'progress.json',{'controls_completed':step+1,'phase':phase,'physical_metrics':metrics.windows})
                guard.require_coverage(450);report['physics_coverage_pass']=True
                report['physics_substeps']=guard.total;report['physical_metrics']=metrics.windows
                report['policy_state_unchanged']=(training.parameters_digest(runner)==ph and training.algorithm_state_digest(runner)==ah)
                if not report['policy_state_unchanged']:raise ValueError('Evaluation altered policy/normalization/optimizer state')
                states={k:np.asarray([row[k] for row in rows]) for k in rows[0]}
                result=summarize(states);np.savez_compressed(out/'states.npz',**states)
                common.write_json(out/'metrics.json',result)
                metadata={'schema':SCHEMA,'schedule':schedule_descriptor(),'source_sha256':verified['contract']['sha256'],
                    'input_sha256':verified['input_sha256'],'evaluation_tools_sha256':tools_before,'root_quaternion_order':'XYZW',
                    'active_motor_names':list(raw.active_joint_names),'body_names':list(raw._robot.body_names),
                    'environment_origins_w_m':arr(raw.scene.env_origins).tolist(),
                    'state_alignment':'pre precedes action; end is final physics state before done reset; post follows automatic reset if any',
                    'command_timing':'schedule injected before fresh consumed observation; original sampler replaced temporarily, including reset callback',
                    'random_sampler_calls_during_evaluation':0,'scheduled_reset_callback_calls':commands.calls,
                    'policy_parameters_sha256':ph,'algorithm_state_sha256':ah,'checkpoint_written':False,
                    'termination_reason_fields':[k for k in states if k.startswith('termination_reason_')]}
                common.write_json(out/'metadata.json',metadata)
                report['artifacts']={name:{'sha256':common.digest(out/name),'bytes':(out/name).stat().st_size} for name in ('states.npz','metadata.json','metrics.json')}
                common.require_same_inputs(verified,verify_inputs(common,args.source_dir,args.checkpoint,args.admission,args.training_report))
                helpers(args.capture_tools_dir)
                if tools_before!=tool_identity(Path(__file__).parent):raise ValueError('Evaluation tools changed during execution')
                report['checkpoint_unchanged']=True;finish()
            except BaseException as error:
                if guard is not None:report.update(physical_metrics=guard.metrics.windows,physics_substeps=guard.total)
                if rows and not (out/'states.npz').exists():
                    partial=out/'states_partial.npz'
                    np.savez_compressed(partial,**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
                    report['partial_trace']={'path':partial.name,'sha256':common.digest(partial),'completed_controls':len(rows)}
                finish(error);raise
            finally:
                if guard is not None:guard.__exit__(None,None,None)
                if end_capture is not None:end_capture.__exit__()
                if commands is not None:commands.__exit__()
                # Kit can exit the process during close; all evidence above is durable first.
                if env is not None:env.close()
    except BaseException as error:finish(error);raise
    return 0 if report['pass'] else 1

if __name__=='__main__':raise SystemExit(main())
