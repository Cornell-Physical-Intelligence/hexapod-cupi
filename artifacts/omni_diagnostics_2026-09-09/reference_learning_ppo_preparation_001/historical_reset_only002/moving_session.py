"""Training-only row recovery over exact source009 physical stepping.

Evaluation uses the immutable non-recovering evaluator. This consumer keeps
every raw pre-reset outcome and never promotes a recovered episode to admission.
"""
from contextlib import ExitStack
from pathlib import Path
import json
import numpy as np
import torch
from tensordict import TensorDict
from observation import ObservationBuilder
from batch_wave import BatchWave005
from device_pack import DeviceTelemetry
from sensor_freshness import ContactFreshness
from physics_substeps import PhysicsSubstepRecorder
from recovery import RecoveryTargets, defer_training_resets, rebase_reset_clocks
from reward_adapter import install_reward


def tensor(value):
    return value.torch if hasattr(value,'torch') else value


class MovingSession:
    def __init__(self, env, layout, points, stance, *, warp_to_torch, output):
        if env.num_envs not in (32,128) or env.step_dt!=.02:
            raise ValueError('New moving consumer supports explicit32/128 replicas at50Hz')
        self.env=env;self.num_envs=env.num_envs;self.num_actions=18;self.device=env.device
        self.names=tuple(layout['joint_names_runtime']);self.output=Path(output)
        if self.output.exists():raise FileExistsError('Moving raw output is immutable')
        self.output.mkdir(parents=True)
        self.cfg={'scope':'new_masked_moving_training_consumer','evaluation_gates_unchanged':True,
            'reset_recovery_controls_learned':False,'native_velocity_fidelity_qualified':False}
        self.max_episode_length=2200;self.episode_length_buf=env.episode_length_buf
        self.reader=DeviceTelemetry(env,layout,points);self.freshness=ContactFreshness(env,warp_to_torch)
        self.recorder=PhysicsSubstepRecorder(env)
        self.wave=BatchWave005(self.names,self.num_envs,device=self.device)
        self.encoder=ObservationBuilder(self.names,self.num_envs,device=self.device)
        initial=env.reference_residual_controller.reference_position.clone()
        nominal=torch.tensor([stance['joint_positions_rad'][name] for name in self.names],device=self.device,dtype=torch.float32).double().expand_as(initial)
        if not torch.equal(nominal,tensor(env._robot.data.default_joint_pos).double()):
            raise ValueError('Named canonical target differs from exact imported default')
        self.recovery=RecoveryTargets(initial,nominal,tensor(env._robot.data.soft_joint_pos_limits))
        self.episodes=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.age=torch.zeros_like(self.episodes);self.control=0
        self.commands=torch.zeros(self.num_envs,3,device=self.device,dtype=torch.float64)
        self.quiet_only=torch.arange(self.num_envs,device=self.device)%4==0
        self.zero=torch.zeros(self.num_envs,18,device=self.device)
        self.active=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.scorable=self.active.clone();self.failure=None;self.last=None;self.reference=None;self.pending=None
        self.rows=[];self.clocks=[];self.events=[];self.reference_rows=[];self.reset_clocks=[]
        self.control_epochs=[]
        self.capture_count=0;self.encoded=None;self.stack=None
        self.exported=False
        env.episode_length_buf.zero_()

    def _fatal(self, message):
        self.failure=message
        raise RuntimeError(message)

    def _capture(self, terminated, truncated):
        fresh=self.freshness.read_after_normal_updates(8)
        self.clocks.append({k:v.detach().cpu().numpy().copy() for k,v in fresh.items() if isinstance(v,torch.Tensor)})
        self.last=self.reader.capture(self.env,time_s=torch.full((self.num_envs,),(self.control+1)*.02,device=self.device,dtype=torch.float64),
            terminated=terminated,truncated=truncated,contact_valid=fresh['contact_valid'],contact_age_s=fresh['contact_age_s'])
        self.capture_count+=1;m=self.last['measurement']
        # Preserve the raw sample before any classification can raise.
        row={k:v.detach().cpu().numpy().copy() for k,v in m.items()}
        row.update({k:v.detach().cpu().numpy().copy() for k,v in self.env.reference_residual_target.items()})
        row.update(episode_id=self.episodes.cpu().numpy().copy(),learning_active=self.active.cpu().numpy().copy(),
                   requested_command=self.commands.cpu().numpy().copy())
        self.rows.append(row)
        if not m['measurement_valid'].all():self._fatal('Nonfinite/stale physical measurement; reject current rollout')
        if truncated.any():self._fatal('Unexpected native90s truncation; training horizon is44s')
        requested=np.stack([r['computed_torque_nm'] for r in self.recorder.rows[-8:]])
        applied=np.stack([r['applied_torque_nm'] for r in self.recorder.rows[-8:]])
        if len(requested)!=8 or not np.isfinite(requested).all() or not np.isfinite(applied).all():
            self._fatal('Missing/nonfinite physical substep torque')
        # The actuator clips to float32(1.6); comparisons retain that exact
        # representable value. Evaluation thresholds and scoring are untouched.
        if np.abs(applied).max()>float(np.float32(1.6)):
            self._fatal('Applied torque exceeded the actual1.6 actuator cap')
        saturation=torch.as_tensor(np.abs(requested).max(axis=(0,2))>1.6,device=self.device)
        nonfoot=m['base_contact']|m['shaft_contact'].any(-1)|m['coxa_contact'].any(-1)|m['femur_contact'].any(-1)
        required=torch.where((self.commands!=0).any(-1)|(self.wave.s['current_leg']>=0),5,6)
        support=m['distal_contact'].sum(-1)<required
        self.physical_terminal=terminated | (self.active & (saturation|nonfoot|support))
        self.terminal_reasons=torch.stack([terminated,saturation,nonfoot,support],-1)
        if (terminated & ~self.active).any():self._fatal('Physical reset recovery terminated')
        self.scorable=self.active & ~self.physical_terminal

    def __enter__(self):
        self.stack=ExitStack()
        try:
            self.stack.enter_context(self.recorder)
            self.stack.enter_context(defer_training_resets(self.env,self._capture))
            self.stack.enter_context(install_reward(self.env,self))
            for _ in range(200):self._advance(self.zero,initializing=True)
            if not self.active.all():self._fatal('Initial canonical recovery did not complete')
            return self
        except BaseException:
            self.__exit__(*__import__('sys').exc_info());raise

    def __exit__(self,kind,value,tb):
        if value is not None:self.failure=repr(value)
        try:
            if self.stack is not None:self.stack.__exit__(kind,value,tb)
        finally:self.export()
        return False

    def _schedule(self):
        command=torch.zeros_like(self.commands)
        command[:,0]=.005*(self.active & ~self.quiet_only & (self.age<1200)).to(torch.float64)
        return command

    def _encode(self):
        packet=dict(self.last,reference=self.wave.output(),controller=self.env.reference_residual_target,
            requested_twist=self.commands,world_up=self.commands.new_tensor([0.,0.,1.]).expand(self.num_envs,-1),
            episode_ids=self.episodes,step_indices=self.age,
            critic_simulator_reported_twist=self.last['measurement']['velocity_body_mps'])
        result=self.encoder.build(packet)
        required=self.active & ~self.physical_terminal
        if (required & ~result['valid']).any():self._fatal('Active846/849 packet failed history/state/freshness contract')
        self.encoded=result
        return self._packet(required)

    def _packet(self, valid):
        if self.encoded is None:
            policy=torch.zeros(self.num_envs,846,device=self.device);critic=torch.zeros(self.num_envs,849,device=self.device)
        else:
            policy=torch.where(valid[:,None],self.encoded['policy'],0.)
            critic=torch.where(valid[:,None],self.encoded['critic'],0.)
        return TensorDict({'policy':policy.clone(),'critic':critic.clone(),
            'learning_valid':valid[:,None].clone(),'episode_id':self.episodes[:,None].clone()},
            batch_size=[self.num_envs],device=self.device)

    def get_observations(self):
        if self.failure is not None:raise RuntimeError('Failed session cannot provide actor state')
        return self._packet(self.active)

    def _preview(self):
        # Only .s is mutable in the bound BatchWave005. Predict the next knot
        # before publishing an actor packet so a reference failure ends the
        # transition which caused it, rather than charging an unexecuted action.
        current=self.wave.s
        with torch.inference_mode(False):
            self.wave.s={k:v.clone() for k,v in current.items()}
        try:
            result=self.wave.step(self.last['measurement'],self.commands)
            future=self.wave.s
        finally:self.wave.s=current
        self.pending={'state':future,'reference':result,'episode':self.episodes.clone(),
                      'age':self.age.clone(),'command':self.commands.clone()}
        return self.active & ~result['valid']

    def _reset_rows(self, ended, terminated, truncated):
        ids=torch.nonzero(ended).flatten()
        if not len(ids):return
        self.events.append({'control':self.control,'episodes':self.episodes.cpu().tolist(),
            'ended_rows':ids.cpu().tolist(),'reason_flags':self.terminal_reasons.cpu().tolist(),
            'reference_failure':self.pending['reference']['failure_code'].cpu().tolist()})
        # Sole physical state write: the original inherited reset, after raw
        # final observations and bootstrap inputs have already been copied.
        # Reward has already been evaluated with deferred native flags false.
        # Publish the real training outcome now so inherited reset summaries
        # do not silently report zero terminations for an ended training row.
        # DirectRLEnv allocates reset_buf during env.step under inference mode.
        # Its ordinary automatic reset executes in that same mode; preserve
        # those SDK tensor semantics for this deferred original reset as well.
        with torch.inference_mode():
            self.env.reset_terminated.copy_(terminated)
            self.env.reset_time_outs.copy_(truncated)
            self.env.reset_buf.copy_(ended)
            self.env._reset_idx(ids)
            self.env.episode_length_buf[ids]=0
        self.events[-1]['reset_reference_joint_target_rad']=self.env.reference_residual_controller.reference_position[ids].cpu().tolist()
        self.events[-1]['reset_kind']='original_inherited_reset_only_before_next_action'
        self.reset_clocks.append(rebase_reset_clocks(self.freshness,ended))
        self.episodes[ended]+=1;self.age[ended]=0;self.active[ended]=False
        self.recovery.reset(ended,self.env.reference_residual_controller.reference_position)
        self.wave.s['ready'][ended]=False;self.wave.s['failure'][ended]=1
        self.encoder.ready[ended]=False
        self.commands[ended]=0.
        # Pending rows are disabled until a separately verified200-control
        # recovery and fresh per-row wave/history initialization complete.
        self.pending['state']['ready'][ended]=False
        self.pending['state']['failure'][ended]=1
        self.pending['episode']=self.episodes.clone();self.pending['age']=self.age.clone()
        self.pending['command']=self.commands.clone()

    def _advance(self,actions,*,initializing=False):
        if self.failure is not None:raise RuntimeError('Failed campaign cannot advance')
        if self.control>=200+25*256:self._fatal('Bounded physical control allocation exhausted')
        if actions.shape!=(self.num_envs,18) or not torch.isfinite(actions).all():self._fatal('Malformed residual action')
        before_active=self.active.clone();before_episode=self.episodes.clone()
        recovery=self.recovery.sample_next()
        if self.active.any():
            if self.pending is None or not torch.equal(self.pending['episode'],self.episodes) or not torch.equal(self.pending['age'],self.age) or not torch.equal(self.pending['command'],self.commands):
                self._fatal('Next reference cache does not match observed episode/command')
            self.wave.s=self.pending['state'];self.reference=self.pending['reference']
        else:self.reference=self.wave.output()
        q=torch.where(self.active[:,None],self.reference['q_ref'],recovery['q_ref'])
        v=torch.where(self.active[:,None],self.reference['v_ref'],recovery['v_ref'])
        a=torch.where(self.active[:,None],self.reference['a_ref'],recovery['a_ref'])
        action=torch.where(self.active[:,None],actions,0.)
        self.reference_rows.append({k:value.detach().cpu().numpy().copy() for k,value in self.wave.s.items()})
        self.control_epochs.append(self.episodes.cpu().numpy().copy())
        captures=self.capture_count;self.recorder.begin_control(self.control)
        self.env.set_reference_targets(q,v,a,torch.ones_like(self.active))
        self.env.set_evaluation_targets(self.commands.float())
        with torch.inference_mode():old_obs,reward,term,trunc,extras=self.env.step(action.float())
        if self.capture_count!=captures+1 or term.any() or trunc.any():self._fatal('Training reset deferral/capture contract failed')
        self.recorder.end_control(self.rows[-1]);self.control+=1
        if not all(torch.isfinite(x).all() for x in old_obs.values()) or not torch.isfinite(reward).all():self._fatal('Nonfinite inherited physics output')
        self.rows[-1]['reward_before_event']=reward.detach().cpu().numpy().copy()
        self.rows[-1]['reward_scorable']=self.scorable.cpu().numpy().copy()
        if hasattr(self,'reward_components'):
            for key,value in self.reward_components.items():
                self.rows[-1]['reward_component__'+key]=value.cpu().numpy().copy()
        self.age[before_active]+=1
        completed=self.recovery.advanced()
        if completed.any():
            m=self.last['measurement']
            if (m['distal_contact'][completed].sum(-1)!=6).any() or self.terminal_reasons[completed].any():
                self._fatal('Recovered canonical stance failed unchanged contact/motor checks')
            result=self.wave.reset(m,completed,self.episodes)
            if not result['valid'][completed].all():self._fatal('Recovered measured reference initialization failed')
            self.encoder.reset(completed,self.episodes,m['time_s'])
            self.active[completed]=True;self.recovery.warming[completed]=False
        self.commands.copy_(self._schedule())
        final=self._encode() if self.active.any() else self._packet(self.active)
        failed_reference=self._preview() if self.active.any() else torch.zeros_like(self.active)
        if (failed_reference & ~before_active).any():self._fatal('First recovered reference knot failed')
        terminated=before_active & (self.physical_terminal|failed_reference)
        truncated=before_active & ~terminated & (self.age>=2200)
        ended=terminated|truncated
        reward=torch.where(before_active,reward,0.)-3.*terminated.float()
        final_valid=before_active & ~self.physical_terminal
        transition={'learnable':before_active,'terminated':terminated,'truncated':truncated,
            'fatal':torch.zeros_like(self.active),'episode_id':before_episode,
            'final_observation':final.clone(),'final_observation_valid':final_valid,
            'final_episode_id':before_episode.clone()}
        self.rows[-1].update(training_terminated=terminated.cpu().numpy().copy(),
            training_truncated=truncated.cpu().numpy().copy(),reward=reward.cpu().numpy().copy(),
            next_reference_failure=failed_reference.cpu().numpy().copy())
        self._reset_rows(ended,terminated,truncated)
        if self.control%100==0:print('MOVING_PPO_CONTROL',self.control,'active',int(self.active.sum()),flush=True)
        return self.get_observations(),reward,ended,{'masked_transition':transition}

    def step(self,actions):
        try:return self._advance(actions)
        except BaseException as error:
            self.failure=repr(error)
            raise

    def export(self):
        if self.exported:return
        # Bounded32-replica pilot stores all raw samples.128 is restricted to
        # timing smoke by its outer allocation, pending measured memory cost.
        for name,rows in [('trace',self.rows),('sensor_clocks',self.clocks),('reference_states',self.reference_rows)]:
            if rows:
                keys=set.union(*(set(row) for row in rows));saved={}
                for key in sorted(keys):
                    template=next(row[key] for row in rows if key in row)
                    present=np.asarray([key in row for row in rows],dtype=bool)
                    saved[key]=np.stack([row[key] if key in row else np.zeros_like(template) for row in rows])
                    if not present.all():saved['field_present__'+key]=present
                np.savez_compressed(self.output/(name+'.npz'),**saved)
        data=self.recorder.data()
        if data:
            # Resets are discontinuities between controls. Never integrate
            # their position jump as physical velocity or pool episode motion.
            row_episodes=np.asarray(self.control_epochs)
            sub_episode=np.zeros((len(self.recorder.rows),self.num_envs),dtype=np.int64)
            for index,row in enumerate(self.recorder.rows[1:],1):
                sub_episode[index]=row_episodes[int(row['control_index'])]
            data['episode_id']=sub_episode
            data['crosses_episode_reset']=np.concatenate([np.zeros((1,self.num_envs),dtype=bool),np.diff(sub_episode,axis=0)!=0])
            np.savez_compressed(self.output/'physics_substeps.npz',**data,joint_names=np.asarray(self.names))
        (self.output/'physics_substep_review.json').write_text(json.dumps({**self.recorder.identity,
            'samples_including_initial':len(self.recorder.rows),'completed_controls':self.recorder.completed_controls,
            'error':self.recorder.error,'method_restored':self.env.scene.update==self.recorder.original,
            'reset_crossing_intervals_not_kinematic_measurements':True,'physical_admission':False},indent=2)+'\n')
        (self.output/'episodes.json').write_text(json.dumps(self.events,indent=2,allow_nan=False)+'\n')
        if self.reset_clocks:
            np.savez_compressed(self.output/'reset_clocks.npz',**{k:np.stack([r[k].cpu().numpy() for r in self.reset_clocks]) for k in ('reset_mask','timestamp','last_update','outdated')})
        (self.output/'session.json').write_text(json.dumps({'controls':self.control,'failure':self.failure,
            'episodes':self.episodes.cpu().tolist(),'actor_width':846,'critic_width':849,
            'physical_admission':False,'evaluation_gates_changed':False},indent=2)+'\n')
        self.exported=True
