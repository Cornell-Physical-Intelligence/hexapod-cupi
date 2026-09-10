"""Bounded training evidence and real save/reload check; legacy physics is unchanged."""
import copy, hashlib, json, shutil, time
from pathlib import Path
import numpy as np
import torch
from direct_config import decision_updates, gradient_updates

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, d): Path(p).write_text(json.dumps(d, indent=2, allow_nan=False)+'\n')

def audited_environment(parent):
    class AuditedDirectEnv(parent):
        def __init__(self, *a, **kw):
            self.direct_audit_rows=[]; self.direct_audit_events=[]; self.direct_audit_samples=[]
            super().__init__(*a, **kw)
        def _get_rewards(self):
            command=self._commands.clone()
            reward=super()._get_rewards()
            d=self._robot.data
            distal, shaft, _=self._get_foot_contact_state()
            coxa=self._coxa_contact_sensor.data.net_forces_w_history.torch.norm(dim=-1).amax(1)>1
            femur=torch.cat([s.data.net_forces_w_history.torch.norm(dim=-1).amax(1)>1 for s in self._femur_contact_sensors],1)
            nonfoot=torch.cat((coxa,femur,shaft),1).any(-1)
            dq=d.joint_vel.torch; target_delta=self._processed_actions-self.omni_previous_target
            v=self._vector_in_command_frame(d.root_lin_vel_b.torch)
            w=self._vector_in_command_frame(d.root_ang_vel_b.torch)
            row={'command':command,'reward':reward, 'age_s':self._episode_elapsed_s,
                 'requested_torque_abs_max_nm':d.computed_torque.torch.abs().amax(-1),
                 'applied_torque_abs_max_nm':d.applied_torque.torch.abs().amax(-1),
                 'requested_saturation_fraction':(d.computed_torque.torch.abs()>1.6).float().mean(-1),
                 'reported_joint_velocity_rms_rad_s':dq.square().mean(-1).sqrt(),
                 'target_delta_rms_rad':target_delta.square().mean(-1).sqrt(),
                 'velocity_navigation_mps':v,'gyro_navigation_rad_s':w,
                 'position_world_m':d.root_pos_w.torch,'quaternion_world_xyzw':d.root_quat_w.torch,
                 'terminated':self.reset_terminated,'truncated':self.reset_time_outs,
                 'nonfoot':nonfoot,'support_count':distal.sum(-1)}
            self.direct_audit_rows.append({k:x.detach().clone() for k,x in row.items()})
            detail={'joint_position_rad':d.joint_pos.torch,'joint_velocity_rad_s':dq,
                    'joint_target_rad':self._processed_actions,'computed_torque_nm':d.computed_torque.torch,
                    'applied_torque_nm':d.applied_torque.torch,'raw_policy_action':self.omni_raw_policy_action}
            self.direct_audit_samples.append({k:x[:8].detach().clone() for k,x in detail.items()})
            ids=torch.nonzero(self.reset_terminated|self.reset_time_outs).flatten()
            if len(ids):
                self.direct_audit_events.append({'control':len(self.direct_audit_rows),'ids':ids.detach().clone(),
                                                'fields':{k:x[ids].detach().clone() for k,x in {**row,**detail}.items()}})
            finite=torch.stack([torch.isfinite(x).all() for x in [*row.values(),*detail.values()]]).all()
            if not bool(finite) or bool((row['applied_torque_abs_max_nm']>1.60001).any()):
                raise RuntimeError('Nonfinite direct training evidence or applied cap exceeded; pre-reset row preserved')
            return reward
    return AuditedDirectEnv

def export_audit(env, output):
    output=Path(output)
    if not env.direct_audit_rows:
        return {'controls':0,'scope':'No physical learning samples collected'}
    raw={k:torch.stack([r[k] for r in env.direct_audit_rows]).cpu().numpy() for k in env.direct_audit_rows[0]}
    raw['time_s']=(np.arange(len(env.direct_audit_rows))+1)*env.step_dt
    raw['joint_names']=np.array(env._robot.joint_names)
    np.savez_compressed(output/'training_trace.npz',**raw)
    sample={k:torch.stack([r[k] for r in env.direct_audit_samples]).cpu().numpy() for k in env.direct_audit_samples[0]}
    np.savez_compressed(output/'training_joint_trace.npz',**sample,env_ids=np.arange(min(8,env.num_envs)),time_s=raw['time_s'],joint_names=raw['joint_names'])
    events=[{'control':e['control'],'ids':e['ids'].cpu().tolist(),'fields':{k:v.cpu().tolist() for k,v in e['fields'].items()}} for e in env.direct_audit_events]
    save(output/'training_events.json',events)
    return {'controls':len(env.direct_audit_rows),'replicas':env.num_envs,
            'terminations_per_row':raw['terminated'].sum(0).astype(int).tolist(),
            'truncations_per_row':raw['truncated'].sum(0).astype(int).tolist(),
            'requested_torque_max_per_row_nm':raw['requested_torque_abs_max_nm'].max(0).tolist(),
            'applied_torque_max_per_row_nm':raw['applied_torque_abs_max_nm'].max(0).tolist(),
            'requested_saturation_fraction_per_row':raw['requested_saturation_fraction'].mean(0).tolist(),
            'trace_sha256':sha(output/'training_trace.npz'),'joint_trace_sha256':sha(output/'training_joint_trace.npz'),
            'event_ledger_sha256':sha(output/'training_events.json'),
            'scope':'50 Hz pre-reset endpoint evidence; no 400 Hz motor qualification or SDK-rate fidelity claim'}

def equal_tree(a,b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,dict): return isinstance(b,dict) and set(a)==set(b) and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b

def verify_reload(runner, wrapped, checkpoint):
    saved=torch.load(checkpoint,map_location=runner.device,weights_only=False)
    before=copy.deepcopy(runner.alg.save())
    obs=wrapped.get_observations()
    with torch.inference_mode(): action=runner.alg.actor(obs).clone()
    # RSL updates normalization during inference-mode rollout; update() replaces
    # _std with an inference tensor. Strict load_state_dict copies into buffers
    # outside that context. Normalize only such buffers after learning, preserving
    # all values and parameter/optimizer objects; never alter the checkpoint.
    normalized_buffers=[]
    with torch.inference_mode(False), torch.no_grad():
        for model_name in ('actor','critic'):
            for module_name,module in getattr(runner.alg,model_name).named_modules():
                for name,value in tuple(module.named_buffers(recurse=False)):
                    if torch.is_inference(value):
                        ordinary=value.detach().clone()
                        if torch.is_inference(ordinary) or not torch.equal(value,ordinary):
                            raise RuntimeError('Inference buffer value-preserving clone failed')
                        setattr(module,name,ordinary)
                        normalized_buffers.append('.'.join(x for x in (model_name,module_name,name) if x))
        runner.load(str(checkpoint),strict=True,map_location=runner.device)
    after=runner.alg.save()
    if not equal_tree(before,after) or not equal_tree(saved['actor_state_dict'],after['actor_state_dict']) or not equal_tree(saved['critic_state_dict'],after['critic_state_dict']):
        raise RuntimeError('Strict actor/critic/normalizer/optimizer reload differs')
    with torch.inference_mode(): exact=torch.equal(action,runner.alg.actor(obs))
    if not exact: raise RuntimeError('Deterministic action changed after strict reload')
    return {'passed':True,'exact_actor_critic_normalizer_optimizer':True,'exact_deterministic_action':True,
            'optimizer_entries':len(runner.alg.optimizer.state),'checkpoint_sha256':sha(checkpoint),
            'inference_buffers_cloned_after_learning':normalized_buffers,
            'reload_method':'Strict same-runner load after value-preserving inference-buffer normalization'}

def learn_and_verify(env, wrapped, runner, selected, output):
    output=Path(output); start=time.perf_counter(); start_iteration=runner.current_learning_iteration
    receipt={'selection':selected,'initial_runner_iteration':start_iteration,'complete':False,'Stage2_complete':False,
             'updates_completed':0,'optimizer_updates':[],
             'optimizer_diagnostics_schema':'direct315_actor_gradients_v1'}
    original_update=runner.alg.update
    def observed_update():
        tick=time.perf_counter();losses=original_update()
        receipt['updates_completed']+=1
        receipt['optimizer_updates'].append({'completed_update':receipt['updates_completed'],
            'optimizer_wall_seconds':time.perf_counter()-tick,'losses':{k:float(v) for k,v in losses.items()},
            'learning_rate':float(runner.alg.learning_rate),
            'minibatches':copy.deepcopy(runner.alg.last_update_diagnostics)})
        return losses
    runner.alg.update=observed_update
    try:
        if selected['allocation']=='extended':
            if runner.alg.storage.num_transitions_per_env!=24 or runner.alg.storage.num_envs!=1024 or runner.cfg.get('save_interval')!=1 or runner.cfg.get('num_steps_per_env')!=24 or runner.cfg.get('max_iterations')!=500 or tuple(runner.alg.gradient_update_milestones)!=gradient_updates(selected):
                raise RuntimeError('Extended runner allocation/save/diagnostic cadence differs')
        # Same native20 s episodes and randomized initial episode ages as old direct PPO.
        runner.learn(num_learning_iterations=selected['updates'],init_at_random_ep_len=True)
        checkpoint=output/'policy/final.pt'
        if checkpoint.exists(): raise FileExistsError(checkpoint)
        runner.save(str(checkpoint))
        receipt['reload']=verify_reload(runner,wrapped,checkpoint)
        expected=selected['updates']*selected['controls_per_update']
        if len(env.direct_audit_rows)!=expected or receipt['updates_completed']!=selected['updates']: raise RuntimeError('Actual collected controls or optimizer updates differ from bounded allocation')
        receipts={}
        for update in decision_updates(selected):
            native=output/'policy'/('model_'+str(start_iteration+update-1)+'.pt')
            target=output/'policy'/('decision_'+str(update).zfill(3)+'.pt')
            if not native.is_file() or target.exists(): raise RuntimeError('Missing or existing immutable decision checkpoint')
            shutil.copyfile(native,target)
            receipts[str(update)]={'file':target.name,'sha256':sha(target),'native_iteration':start_iteration+update-1,'completed_updates':update}
        receipt.update(complete=True,updates_completed=selected['updates'],decision_checkpoints=receipts,
                       final_checkpoint_sha256=sha(checkpoint),final_runner_iteration=runner.current_learning_iteration,
                       learned_std=runner.alg.actor.distribution.std_param.detach().cpu().tolist())
        if selected['allocation']=='extended':
            from direct_contract import validate_extended_checkpoints
            receipt['extended_readback']=validate_extended_checkpoints(output,receipt)
        return receipt
    except BaseException as exc:
        receipt['error']=repr(exc)
        if selected['allocation']=='extended': receipt['complete']=False
        raise
    finally:
        runner.alg.update=original_update
        receipt['wall_seconds']=time.perf_counter()-start
        try: receipt['audit']=export_audit(env,output)
        except BaseException as exc:
            receipt.update(complete=False,audit_error=repr(exc));save(output/'training_receipt.json',receipt);raise
        if torch.cuda.is_available():
            receipt['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated(env.device)
            receipt['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved(env.device)
        save(output/'training_receipt.json',receipt)
