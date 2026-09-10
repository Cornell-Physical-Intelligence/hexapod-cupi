"""Bounded actual-physics RSL vector interface over the immutable source009 env.

No body writes, old checkpoint, frozen-bundle edits or automatic reset recovery.
Any failed reference, sensor row, physical terminal or motor bound aborts the
entire short experiment with its pre-reset evidence. This is not a long-run env.
"""
import time
import numpy as np
import torch
from tensordict import TensorDict
from observation import ObservationBuilder
from batch_wave import BatchWave005
from device_pack import DeviceTelemetry
from sensor_freshness import ContactFreshness
from pre_reset_device import capture_before_reset,require_one
from canonical_stance_startup import CanonicalStanceStartup
from physics_substeps import PhysicsSubstepRecorder
from physics_telemetry import array


class ReferenceResidualSession:
    """The external entrypoint must prove exact admissions before construction.

    Inputs are already-built source009 physics and its exact named layout. All
    measured packets retain simulator provenance and raw-rate uncertainty.
    """
    def __init__(self,env,layout,points,stance,*,warp_to_torch,initial_commands=None):
        if env.num_envs not in (1,32) or env.step_dt!=.02:raise ValueError('Only bounded1/32replica50Hz sessions')
        self.env=env;self.layout=layout;self.names=tuple(layout['joint_names_runtime'])
        self.num_envs=env.num_envs;self.num_actions=18;self.device=env.device
        self.max_episode_length=env.max_episode_length;self.episode_length_buf=env.episode_length_buf
        self.cfg={'scope':'new_finite_position_residual_PPO_smoke_consumer','autoreset_supported':False,
            'prototype_training_flags_unchanged':True,'native_velocity_fidelity_qualified':False}
        self.reader=DeviceTelemetry(env,layout,points);self.freshness=ContactFreshness(env,warp_to_torch)
        self.zero=torch.zeros((self.num_envs,18),device=self.device)
        self.commands=torch.zeros((self.num_envs,3),device=self.device,dtype=torch.float64)
        self.initial_commands=self.commands.clone() if initial_commands is None else self._validate_commands(initial_commands).clone()
        self.episodes=torch.zeros(self.num_envs,device=self.device,dtype=torch.int64)
        self.control=0;self.policy_steps=0;self.last=None;self.reference=None;self.encoder=None;self.wave=None
        self.rows=[];self.clock_rows=[];self.encoded=None;self.failure=None;self.pending_commands=None
        self.captured=[];self.capture_context=None;self.optimization_steps=None
        self.recorder=PhysicsSubstepRecorder(env)
        initial=array(env.reference_residual_controller.reference_position)
        nominal=np.asarray([stance['joint_positions_rad'][name] for name in self.names],dtype=np.float32).astype(np.float64)
        nominal=np.broadcast_to(nominal,initial.shape).copy()
        if not np.array_equal(nominal,array(env._robot.data.default_joint_pos).astype(np.float64)):
            raise ValueError('Canonical stance differs from source009 articulation default')
        limits=array(env._robot.data.soft_joint_pos_limits)
        self.startup=CanonicalStanceStartup(initial,nominal,limits[...,0],limits[...,1],duration_s=2.,dt=.02)

    def _capture(self,terminated,truncated):
        fresh=self.freshness.read_after_normal_updates(8)
        sample=self.reader.capture(self.env,time_s=torch.full((self.num_envs,),(self.control+1)*.02,device=self.device,dtype=torch.float64),
            terminated=terminated,truncated=truncated,contact_valid=fresh['contact_valid'],contact_age_s=fresh['contact_age_s'])
        self.clock_rows.append({k:v.detach().cpu().numpy().copy() for k,v in fresh.items() if isinstance(v,torch.Tensor)})
        return sample

    def _physical_step(self,ref,actions,*,post_startup):
        if self.failure is not None:raise RuntimeError('Failed session cannot resume: '+self.failure)
        try:
            if actions.shape!=(self.num_envs,18) or not torch.isfinite(actions).all():raise ValueError('Finite18-joint residual actions required')
            if not ref['valid'].all():raise ValueError('Reference rejected before target emission')
            before=len(self.captured);self.recorder.begin_control(self.control)
            v=ref['analytic_velocity_rad_s'] if 'analytic_velocity_rad_s' in ref else ref['v_ref']
            a=ref['analytic_acceleration_rad_s2'] if 'analytic_acceleration_rad_s2' in ref else ref['a_ref']
            self.env.set_reference_targets(ref['q_ref'],v,a,ref['valid']);self.env.set_evaluation_targets(self.commands.float())
            with torch.inference_mode():old_obs,reward,term,trunc,extras=self.env.step(actions.float())
            sample=require_one(self.captured,before,term,trunc);self.last=sample;m=sample['measurement']
            row={k:value.detach().cpu().numpy().copy() for k,value in m.items()}
            row.update({k:value.detach().cpu().numpy().copy() for k,value in self.env.reference_residual_target.items()})
            row['requested_command']=self.commands.detach().cpu().numpy().copy();row['reward']=reward.detach().cpu().numpy().copy()
            self.rows.append(row);self.recorder.end_control(row);self.control+=1
            if not m['measurement_valid'].all() or term.any() or trunc.any():raise ValueError('Invalid/stale measurement or physical reset; pre-reset row retained')
            if not all(torch.isfinite(v).all() for v in old_obs.values()) or not torch.isfinite(reward).all():raise ValueError('Original physics reward/observation nonfinite')
            if post_startup:
                for key in ('computed_torque_nm','applied_torque_nm'):
                    value=np.stack([r[key] for r in self.recorder.rows[-8:]])
                    if not np.isfinite(value).all() or np.abs(value).max()>1.6:raise ValueError('Post-startup400Hz motor bound failed: '+key)
                if any(m[k].any() for k in ('shaft_contact','coxa_contact','femur_contact','base_contact')):raise ValueError('Nonfoot contact')
                required=torch.where(self.commands.norm(dim=-1)==0,6,5)
                # During a finite stop the reference may still have one swing.
                if self.wave is not None:required=torch.where(self.wave.s['current_leg']>=0,5,required)
                if (m['distal_contact'].sum(-1)<required).any():raise ValueError('Required measured supports lost')
            if self.control%100==0:print('REFERENCE_RESIDUAL_PPO_CONTROL',self.control,flush=True)
            return reward,term|trunc,extras
        except Exception as error:
            self.failure=repr(error)
            if self.captured:self.last=self.captured[-1]
            raise

    def __enter__(self):
        self.recorder.__enter__();self.capture_context=capture_before_reset(self.env,self._capture)
        self.captured=self.capture_context.__enter__()
        try:
            for _ in range(200):self._physical_step(self.startup.sample(self.control+1),self.zero,post_startup=False)
            m=self.last['measurement'];self.wave=BatchWave005(self.names,self.num_envs,device=self.device)
            self.reference=self.wave.reset(m,torch.ones(self.num_envs,device=self.device,dtype=torch.bool),self.episodes)
            if not self.reference['valid'].all():raise ValueError('Settled measured wave reset failed')
            self.encoder=ObservationBuilder(self.names,self.num_envs,device=self.device)
            self.encoder.reset(torch.ones(self.num_envs,device=self.device,dtype=torch.bool),self.episodes,m['time_s'])
            self.commands.copy_(self.initial_commands)
            self._encode()
            return self
        except Exception:
            self.__exit__(*__import__('sys').exc_info());raise

    def __exit__(self,kind,value,traceback):
        try:
            if self.capture_context is not None:self.capture_context.__exit__(kind,value,traceback)
        finally:self.recorder.__exit__(kind,value,traceback)
        return False

    def _encode(self):
        packet=dict(self.last,reference=self.reference,controller=self.env.reference_residual_target,
            requested_twist=self.commands,world_up=self.commands.new_tensor([0.,0.,1.]).expand(self.num_envs,-1),
            episode_ids=self.episodes,step_indices=torch.full_like(self.episodes,self.policy_steps),
            critic_simulator_reported_twist=self.last['measurement']['velocity_body_mps'])
        encoded=self.encoder.build(packet)
        if not encoded['valid'].all():raise ValueError('New846/849 history/state packet rejected')
        if encoded['policy_training_allowed'] is not False:raise ValueError('Frozen prototype flags unexpectedly changed')
        self.encoded=encoded

    def _validate_commands(self,commands):
        command=torch.as_tensor(commands,device=self.device,dtype=torch.float64)
        if command.shape!=(self.num_envs,3) or not torch.isfinite(command).all():raise ValueError('Complete finite forward/left/yaw command batch required')
        # Initial smoke and retention are restricted to the already physically
        # demonstrated forward/stop envelope. All-bearing training is separate.
        if (command[:,1:]!=0).any() or (command[:,0]<0).any() or (command[:,0]>.005).any():
            raise ValueError('Initial smoke supports demonstrated0..0.005forward and stop only; new bearings require new admission')
        return command

    def queue_commands(self,commands):
        command=self._validate_commands(commands)
        if self.failure is not None:raise RuntimeError('Cannot queue into a failed session')
        # The current action still uses its already observed command. After that
        # control, the queued command enters the NEXT observation and target step.
        # This explicit20ms latency preserves the frozen single-build history rule.
        self.pending_commands=command.clone()

    def get_observations(self):
        if self.failure is not None or self.encoded is None:raise RuntimeError('No valid current observed state')
        return TensorDict({k:self.encoded[k].clone() for k in ('policy','critic')},batch_size=[self.num_envs])

    def step(self,actions):
        if self.failure is not None:raise RuntimeError('Failed session cannot advance reference or physics')
        if self.optimization_steps is not None:
            if self.optimization_steps>=48 or self.commands.any() or self.pending_commands is not None:
                raise RuntimeError('Exactly48 stand-only optimization controls maximum')
            self.optimization_steps+=1
        try:
            self.reference=self.wave.step(self.last['measurement'],self.commands)
            reward,done,extras=self._physical_step(self.reference,actions,post_startup=True)
            if self.pending_commands is not None:
                self.commands.copy_(self.pending_commands);self.pending_commands=None
            self.policy_steps+=1;self._encode()
            return self.get_observations(),reward,done.long(),extras
        except Exception as error:
            self.failure=repr(error);raise

    def export(self,output):
        from pathlib import Path
        import json
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        if self.encoded is not None:np.savez_compressed(output/'final_observation.npz',**{k:self.encoded[k].detach().cpu().numpy() for k in ('policy','critic','raw_sdk_joint_velocity_rad_s','interval_joint_rate_rad_s','interval_rate_valid')})
        if self.rows:np.savez_compressed(output/'trace.npz',**{k:np.stack([r[k] for r in self.rows]) for k in self.rows[0]},joint_names=np.asarray(self.names))
        if self.clock_rows:np.savez_compressed(output/'sensor_clocks.npz',**{k:np.stack([r[k] for r in self.clock_rows]) for k in self.clock_rows[0]})
        report=self.recorder.export(output)
        if self.encoder is not None:(output/'observation_schema.json').write_text(json.dumps(self.encoder.spec(),indent=2)+'\n')
        if self.wave is not None:np.savez_compressed(output/'reference_state.npz',**{k:v.detach().cpu().numpy() for k,v in self.wave.s.items()})
        return {'controls':self.control,'policy_steps':self.policy_steps,'failure':self.failure,'substeps':report,
            'autoreset_supported':False,'long_training_admitted':False,'prototype_flags_unchanged':True}
