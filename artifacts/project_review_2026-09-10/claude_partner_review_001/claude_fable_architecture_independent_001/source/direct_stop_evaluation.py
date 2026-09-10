"""Real movement-to-zero commands, full per-replica traces and unchanged quiet bounds."""
import json
from pathlib import Path
import numpy as np
import torch
from direct_quiet_metrics import QUIET_GATES, quiet_metrics

def target_at(control, moving_targets):
    if type(control) is not int or not 0 <= control < 1600:
        raise ValueError('Exact 32 s stop diagnostic control range required')
    return moving_targets if 200 <= control < 800 else torch.zeros_like(moving_targets)

def summarize(data, scenarios, names, dt):
    from omni_diagnostics import _summary
    if dt != .02 or data['command'].shape != (1600,48,3):
        raise ValueError('Complete32 s /48-replica stop diagnostic required')
    rows=[]
    for i, scenario in enumerate(scenarios):
        replicas=[]
        for replica in range(4):
            index=i*4+replica
            quiet=quiet_metrics(data,index,1100,names,dt)
            q=data['joint_position_rad'][1100:,index]
            quiet['interval_angle_velocity_rms_rad_s']=np.sqrt(np.mean((np.diff(q,axis=0)/dt)**2,axis=0)).tolist()
            quiet['interval_angle_note']='50Hz adjacent-angle interval average; separate from raw SDK joint velocity and unchanged gates'
            subset={k:v[:,index:index+1] for k,v in data.items()}
            mask=np.zeros((1600,1),dtype=bool);mask[300:800]=True
            replicas.append({'env_id':index,'motion':_summary(subset,mask,names),'quiet':quiet,
                             'requested_target_nonzero_controls':int(np.any(data['target_command'][:,index]!=0,axis=-1).sum())})
        rows.append({**scenario,'replicas':replicas})
    return rows

@torch.inference_mode()
def evaluate_stop(env, runner, plan, output, checkpoint_sha):
    from tensordict import TensorDict
    from omni_diagnostics import diagnostic_scenarios
    from omni_flat_evaluation import save
    if env.num_envs!=48 or env.step_dt!=.02: raise ValueError('Stop diagnostic requires48 environments at50Hz')
    output=Path(output);scenarios=diagnostic_scenarios()
    targets=torch.tensor([s['command'] for s in scenarios],device=env.device).repeat_interleave(4,0)
    policy=runner.get_inference_policy(device=env.device)
    env.omni_diagnostic_enabled=True; snapshots=[]
    observation_audit={'actor_width':315,'critic_width':318,'max_command_slice_difference':0.,'max_same_step_repeat_difference':0.,'max_history_shift_difference':0.}
    previous=None; previous_done=None; complete=False
    try:
        env.reset(seed=27057);env.episode_length_buf.zero_()
        for control in range(1600):
            env.set_evaluation_targets(target_at(control,targets))
            obs=env._get_observations();actor=obs['policy'];repeat=env._get_observations()
            observation_audit['max_same_step_repeat_difference']=max(observation_audit['max_same_step_repeat_difference'],float((actor-repeat['policy']).abs().max()))
            command=env._commands*torch.tensor([5.,5.,2.5],device=env.device)
            observation_audit['max_command_slice_difference']=max(observation_audit['max_command_slice_difference'],float((actor[:,-57:-54]-command).abs().max()))
            if previous is not None and (~previous_done).any():
                valid=~previous_done
                observation_audit['max_history_shift_difference']=max(observation_audit['max_history_shift_difference'],float((actor[valid,:-63]-previous[valid,63:]).abs().max()))
            previous=actor.clone()
            _,_,term,trunc,_=env.step(policy(TensorDict(obs,batch_size=[48])))
            previous_done=term|trunc
            sample={k:v.copy() for k,v in env.omni_diagnostic_sample.items()}
            # Exact bound legacy capture stores Isaac XYZW under its old WXYZ label.
            # Preserve raw pre-reset values, then explicitly convert this NEW trace.
            sample['quaternion_world_xyzw']=sample['quaternion_world_wxyz'].copy()
            sample['quaternion_world_wxyz']=sample['quaternion_world_xyzw'][...,[3,0,1,2]].copy()
            snapshots.append(sample)
            if not all(np.isfinite(v).all() for v in sample.values()): raise RuntimeError('Nonfinite stop diagnostic')
            if np.abs(sample['applied_torque_nm']).max()>1.60001: raise RuntimeError('Applied cap exceeded in preserved stop trace')
            if not np.array_equal(sample['terminated'],term.cpu().numpy()) or not np.array_equal(sample['truncated'],trunc.cpu().numpy()): raise RuntimeError('Pre-reset stop evidence mismatch')
            if (control+1)%100==0: print('DIRECT_STOP '+str(control+1)+'/1600',flush=True)
        complete=True
    finally:
        env.omni_diagnostic_enabled=False
        if snapshots:
            data={k:np.stack([s[k] for s in snapshots]) for k in snapshots[0]}
            names=list(env._robot.joint_names)
            np.savez_compressed(output/'stop_trace.npz',**data,joint_names=np.array(names),time_s=(np.arange(len(snapshots))+1)*.02)
            report={'complete':complete,'kind':'direct315_move_to_zero_diagnostic_v1','checkpoint_sha256':checkpoint_sha,
                    'controls':len(snapshots),'replicas':48,'scenarios':summarize(data,scenarios,names,.02) if complete else [],
                    'all_control_applied_torque_abs_max_nm':float(np.abs(data['applied_torque_nm']).max()),
                    'quiet_gates':QUIET_GATES,'observation_audit':observation_audit,
                    'schedule_s':{'initial_zero':[0,4],'motion':[4,16],'stop':[16,32],'scored_quiet':[22,32]},
                    'overrides':plan['omni']['overrides'],'Stage2_complete':False,
                    'scope':'Complete is acquisition only, failed quiet/torque/termination bounds remain failures; 50Hz not substep qualification'}
            save(output/'stop_diagnostics.json',report)
    return report
