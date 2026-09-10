"""Executable new C-study environment adapter; import only after AppLauncher.

Requires the pinned C-study bootstrap. Does not register or replace any existing
production/mock task, and never loads an old absolute-position checkpoint.
"""
from c_study_runtime import bootstrap_c_study_runtime
RUNTIME_BINDING=bootstrap_c_study_runtime()
import torch
from omni_flat_env import OmniFlatEnv, configure_omni
from omni_flat_math import ObservationHistory
from velocity_action import JointTargetVelocity,VelocityActionConfig,append_executable_state,LINEAGE


def candidate_options(value):
    if not isinstance(value,dict) or set(value)!={'profile','max_acceleration_rad_s2'}:
        raise ValueError('velocity_candidate requires explicit profile and max_acceleration_rad_s2')
    return VelocityActionConfig(**value)


def configure_velocity_candidate(cfg, options, reward_overrides=None):
    config=candidate_options(options)
    configure_omni(cfg,reward_overrides)
    if cfg.omni_target_filter_time_constant_s!=0:
        raise ValueError('New velocity action cannot inherit the old target filter')
    cfg.observation_space=495;cfg.state_space=498
    cfg.candidate_action_config=options
    cfg.processed_joint_target_slew_limit_rad_per_20ms=config.contract()['max_position_step_rad_per_20ms']
    # Kp/Kd, motor caps, failure gates, physics and command distribution are not changed.
    return config.contract()


class VelocityCandidateEnv(OmniFlatEnv):
    """New 495/498 schema and named target-velocity actions at 50 Hz.

    The initial candidate retains zero joint-velocity feedforward, as in the
    direct-PPO baseline. Its executable velocity is the discrete position-target
    derivative, made observable; it is not implicitly a new motor velocity drive.
    """
    def __init__(self,cfg,render_mode=None,*,evaluation=False,**kwargs):
        super().__init__(cfg,render_mode,evaluation=evaluation,**kwargs)
        config=candidate_options(cfg.candidate_action_config)
        names=tuple(self._robot.joint_names);limits=self._robot.data.soft_joint_pos_limits.torch
        if not torch.allclose(limits,limits[:1].expand_as(limits),atol=0,rtol=0):
            raise ValueError('This candidate requires identical declared limits across environments')
        self.target_velocity_controller=JointTargetVelocity(names,
            dict(zip(names,limits[0,:,0].tolist())),dict(zip(names,limits[0,:,1].tolist())),
            self.num_envs,config,device=self.device,dtype=self._actions.dtype)
        self.omni_history=ObservationHistory(self.num_envs,5,99,self.device)
        self.candidate_previous_target_velocity=torch.zeros_like(self._actions)
        self.candidate_target_acceleration=torch.zeros_like(self._actions)
        self.candidate_acceleration_limited=torch.zeros_like(self._actions,dtype=torch.bool)
        self.candidate_joint_braking=torch.zeros_like(self._actions,dtype=torch.bool)
        self.candidate_action_clipped=torch.zeros_like(self._actions,dtype=torch.bool)
        self.candidate_contract={**config.contract(),'joint_names':list(names),
            'joint_velocity_feedforward':'zero_preserve_existing_PD_semantics','runtime_binding':RUNTIME_BINDING,
            'action_semantics':'normalized_joint_target_velocity'}
        # Imported zero coordinates can precede the first reset and lie outside
        # soft limits. Perform the canonical physical reset, then anchor to the
        # exact joint positions that the pinned parent writes into physics.
        self._reset_idx(None)

    def _pre_physics_step(self,actions):
        # Deliberately do not call the absolute-position parent's action mapper.
        c=self.target_velocity_controller
        if getattr(self,'omni_diagnostic_enabled',False):
            self.omni_diagnostic_start_position=self._robot.data.root_pos_w.torch.clone()
            self.omni_diagnostic_start_quaternion=self._robot.data.root_quat_w.torch.clone()
        self.omni_raw_policy_action=actions.clone()
        self.omni_step+=1
        self.omni_previous_action.copy_(self._actions)
        self.omni_previous_velocity.copy_(self._robot.data.joint_vel.torch)
        self.omni_previous_target.copy_(c.position)
        self.candidate_previous_target_velocity.copy_(c.velocity)
        result=c.step(actions,self.step_dt)
        self._actions=result.clipped_action
        self._processed_actions=result.target_position_rad
        self.candidate_target_acceleration=result.target_acceleration_rad_s2
        self.candidate_acceleration_limited=result.acceleration_limited
        self.candidate_joint_braking=result.joint_braking_active
        self.candidate_action_clipped=result.action_clipped
        self._previous_processed_joint_target.copy_(self._processed_actions)
        self._has_previous_processed_joint_target.fill_(True)
        # No after-the-fact position slew clipping occurs. Log acceleration and
        # braking interventions separately rather than relabeling them as slew.
        self._joint_target_slew_limited_fraction=torch.zeros(self.num_envs,device=self.device)

    def _get_rewards(self):
        d=self._robot.data
        if not torch.isfinite(d.applied_torque.torch).all() or not torch.isfinite(d.computed_torque.torch).all():
            raise RuntimeError('Nonfinite applied or requested motor torque')
        reward=super()._get_rewards()
        if getattr(self,'omni_measurements_enabled',False):
            self.omni_measurement_sample['max_applied_torque']=d.applied_torque.torch.abs().amax(-1).clone()
            self.omni_measurement_sample['joint_velocity_squared_mean']=d.joint_vel.torch.square().mean(-1).clone()
        if getattr(self,'omni_diagnostic_enabled',False):
            sample=self.omni_diagnostic_sample
            # The old capture helper describes an absolute-position mapper.
            # Those hypothetical targets have no meaning in this new lineage.
            sample.pop('filtered_target_rad',None);sample.pop('unfiltered_target_rad',None)
            # Installed Isaac Lab uses XYZW; the shared historical trace key
            # incorrectly says WXYZ. Preserve raw data and convert explicitly
            # for the existing quiet-review heading helper.
            raw_quaternion=d.root_quat_w.torch.detach().cpu().numpy().copy()
            sample['quaternion_world_xyzw']=raw_quaternion
            sample['quaternion_world_wxyz']=raw_quaternion[..., [3,0,1,2]].copy()
            sample['executable_target_velocity_rad_s']=self.target_velocity_controller.velocity.detach().cpu().numpy().copy()
            sample['executable_target_acceleration_rad_s2']=self.candidate_target_acceleration.detach().cpu().numpy().copy()
            sample['candidate_acceleration_limited']=self.candidate_acceleration_limited.detach().cpu().numpy().copy()
            sample['candidate_joint_braking']=self.candidate_joint_braking.detach().cpu().numpy().copy()
            if getattr(self,'candidate_schema_snapshots',None) is not None:
                self.candidate_schema_snapshots.append(sample)
        return reward

    def _apply_action(self):
        self._robot.set_joint_position_target_index(target=self._processed_actions)
        self._robot.set_joint_velocity_target_index(target=torch.zeros_like(self._processed_actions))

    def _get_observations(self):
        d=self._robot.data
        current=torch.cat((self._vector_in_command_frame(d.root_ang_vel_b.torch)*.25,
            self._vector_in_command_frame(d.projected_gravity_b.torch),
            self._commands*torch.tensor([5.,5.,2.5],device=self.device),
            d.joint_pos.torch-d.default_joint_pos.torch,d.joint_vel.torch*.05,self._actions),-1)
        if self.omni_noise_step!=self.omni_step:
            scales=torch.zeros(63,device=self.device)
            scales[:3]=.00375;scales[3:6]=.01;scales[9:27]=.005;scales[27:45]=.0025
            self.omni_noise=torch.randn_like(current)*scales*self.cfg.omni_observation_noise_scale
            self.omni_noise_step=self.omni_step
        frame=append_executable_state(current+self.omni_noise,self.target_velocity_controller,d.default_joint_pos.torch)
        actor=self.omni_history.observe(frame,self.omni_step)
        critic=torch.cat((actor,self._vector_in_command_frame(d.root_lin_vel_b.torch)*5.),-1)
        return {'policy':actor,'critic':critic}

    def _reset_idx(self,env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self,'target_velocity_controller'):
            ids=(torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids)
            # The pinned parent just wrote these exact positions to physics.
            q=self._previous_processed_joint_target[ids].clone()
            self.target_velocity_controller.reset(q,ids)
            self._processed_actions[ids]=q
            self.candidate_previous_target_velocity[ids]=0
            self.candidate_target_acceleration[ids]=0
            self.candidate_acceleration_limited[ids]=False
            self.candidate_joint_braking[ids]=False
            self.candidate_action_clipped[ids]=False
