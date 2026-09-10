"""Physics-only full-C reference/residual adapter; import after AppLauncher.

No PPO runner or checkpoint load is provided. The 675/678 observation includes
sensor/history and complete executable target state, but a future learned actor
must additionally observe the external gait's phase/contact/reference state.
"""
from c_study_runtime import bootstrap_c_study_runtime
RUNTIME_BINDING = bootstrap_c_study_runtime()
import torch
from omni_flat_env import OmniFlatEnv, configure_omni
from omni_flat_math import ObservationHistory
from reference_residual import ResidualConfig, ReferenceResidualTarget


def configure_reference_residual_physics(cfg, options, reward_overrides=None):
    if not isinstance(options, dict) or set(options) != set(ResidualConfig.__dataclass_fields__):
        raise ValueError('Explicit complete reference/residual controller configuration required')
    config = ResidualConfig(**options)
    configure_omni(cfg, reward_overrides)
    if cfg.omni_target_filter_time_constant_s != 0:
        raise ValueError('Physics prototype cannot inherit an extra position filter')
    cfg.reference_residual_options = dict(options)
    cfg.observation_space = 675
    cfg.state_space = 678
    cfg.processed_joint_target_slew_limit_rad_per_20ms = config.contract()['total_position_step_rad_per_20ms']
    return config.contract()


class ReferenceResidualPhysicsEnv(OmniFlatEnv):
    """Only named joint targets enter physics; body motion is measured outcome."""
    def __init__(self, cfg, render_mode=None, *, evaluation=True, **kwargs):
        if evaluation is not True:
            raise ValueError('Reference physics prototype does not admit PPO training yet')
        super().__init__(cfg, render_mode, evaluation=True, **kwargs)
        config = ResidualConfig(**cfg.reference_residual_options)
        names = tuple(self._robot.joint_names)
        limits = self._robot.data.soft_joint_pos_limits.torch
        if not torch.equal(limits, limits[:1].expand_as(limits)):
            raise ValueError('Reference proof requires identical named joint limits for all replicas')
        # Float64 makes reference knot differences explicit before the actual
        # float32 articulation target cast, whose error is retained in telemetry.
        self.reference_residual_controller = ReferenceResidualTarget(
            names, dict(zip(names,limits[0,:,0].tolist())), dict(zip(names,limits[0,:,1].tolist())),
            self.num_envs, config, device=self.device, dtype=torch.float64)
        self.omni_history = ObservationHistory(self.num_envs,5,135,self.device)
        self._reference_fresh = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.reference_residual_contract = {**config.contract(), 'joint_names':list(names),
            'runtime_binding':RUNTIME_BINDING, 'mode':'physics_prototype_no_PPO',
            'observation_width':675, 'critic_width':678,
            'full_reference_generator_state_in_policy_observation':False,
            'joint_velocity_feedforward':'zero_preserve_current_PD_semantics'}
        self.reference_residual_target = None
        self.reference_residual_sample = None
        self._reset_idx(None)

    def set_reference_targets(self, q_ref, v_ref, a_ref, valid):
        """One fresh batch of reference knots, already in runtime joint order.

        Validity must include geometry and measured contact-transition checks in
        the external generator. The controller independently checks named joint
        bounds and executable knot rates; it never silently repairs the gait.
        v_ref/a_ref are retained as separately supplied derivative metadata.
        """
        c = self.reference_residual_controller
        self._reference_fresh.fill_(False)
        shape = c.reference_position.shape
        self._pending_reference = {
            'q':c._tensor(q_ref,shape,'q_ref').clone(),
            'v':c._tensor(v_ref,shape,'v_ref').clone(),
            'a':c._tensor(a_ref,shape,'a_ref').clone(),
            'valid':torch.as_tensor(valid,device=self.device).clone()}
        valid_tensor = self._pending_reference['valid']
        if valid_tensor.dtype != torch.bool or valid_tensor.shape != (self.num_envs,) or not valid_tensor.all():
            raise ValueError('Every environment needs an explicitly valid reference')
        self._reference_fresh.fill_(True)

    def _pre_physics_step(self, actions):
        if not self._reference_fresh.all():
            raise RuntimeError('Fresh contact-aware reference required for every control step')
        c = self.reference_residual_controller
        pending = self._pending_reference
        result = c.step(pending['q'],actions,reference_valid=pending['valid'],dt=self.step_dt,
                        analytic_reference_velocity=pending['v'],analytic_reference_acceleration=pending['a'])
        # Do not call the inherited absolute-position action mapper.
        self.omni_diagnostic_start_position = self._robot.data.root_pos_w.torch.clone()
        self.omni_diagnostic_start_quaternion = self._robot.data.root_quat_w.torch.clone()
        self.omni_raw_policy_action = actions.clone()
        self.omni_step += 1
        self.omni_previous_action.copy_(self._actions)
        self.omni_previous_velocity.copy_(self._robot.data.joint_vel.torch)
        self.omni_previous_target.copy_(self._processed_actions)
        # Parent resets mutate these persisted tensors, including outside a
        # rollout's inference_mode. Allocate normal writable storage explicitly.
        with torch.inference_mode(False):
            self._actions = torch.tanh(actions).detach().clone()
            self._processed_actions = result['target_position_rad'].to(dtype=self._actions.dtype).detach().clone()
            self._joint_target_slew_limited_fraction = torch.zeros(self.num_envs,device=self.device)
        result['applied_position_target_rad'] = self._processed_actions.to(dtype=torch.float64).clone()
        result['position_target_cast_error_rad'] = result['applied_position_target_rad']-result['target_position_rad']
        result['reference_to_executable_lag_rad'] = result['target_position_rad']-result['reference_position_rad']
        self.reference_residual_target = result
        self._previous_processed_joint_target.copy_(self._processed_actions)
        self._has_previous_processed_joint_target.fill_(True)
        self._reference_fresh.fill_(False)

    def _apply_action(self):
        self._robot.set_joint_position_target_index(target=self._processed_actions)
        self._robot.set_joint_velocity_target_index(target=torch.zeros_like(self._processed_actions))

    def _get_rewards(self):
        d = self._robot.data
        if not torch.isfinite(d.applied_torque.torch).all() or not torch.isfinite(d.computed_torque.torch).all():
            raise RuntimeError('Nonfinite physical requested/applied torque')
        reward = super()._get_rewards()
        if not torch.isfinite(reward).all():
            raise RuntimeError('Nonfinite physical reward')
        self.reference_residual_sample = {k:v.detach().cpu().numpy().copy()
                                         for k,v in self.reference_residual_target.items()}
        if getattr(self,'omni_diagnostic_enabled',False):
            sample = self.omni_diagnostic_sample
            sample.pop('filtered_target_rad',None)
            sample.pop('unfiltered_target_rad',None)
            raw = d.root_quat_w.torch.detach().cpu().numpy().copy()
            sample['quaternion_world_xyzw'] = raw
            sample['quaternion_world_wxyz'] = raw[..., [3,0,1,2]].copy()
            sample.update(self.reference_residual_sample)
        return reward

    def _get_observations(self):
        d = self._robot.data
        current = torch.cat((self._vector_in_command_frame(d.root_ang_vel_b.torch)*.25,
            self._vector_in_command_frame(d.projected_gravity_b.torch),
            self._commands*torch.tensor([5.,5.,2.5],device=self.device),
            d.joint_pos.torch-d.default_joint_pos.torch,d.joint_vel.torch*.05,self._actions),-1)
        if self.omni_noise_step != self.omni_step:
            scales=torch.zeros(63,device=self.device)
            scales[:3]=.00375;scales[3:6]=.01;scales[9:27]=.005;scales[27:45]=.0025
            self.omni_noise=torch.randn_like(current)*scales*self.cfg.omni_observation_noise_scale
            self.omni_noise_step=self.omni_step
        known = self.reference_residual_controller.observable_state(d.default_joint_pos.torch).to(current.dtype)
        frame = torch.cat((current+self.omni_noise,known),-1)
        actor = self.omni_history.observe(frame,self.omni_step)
        critic = torch.cat((actor,self._vector_in_command_frame(d.root_lin_vel_b.torch)*5.),-1)
        return {'policy':actor,'critic':critic}

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self,'reference_residual_controller'):
            ids = torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
            q = self._previous_processed_joint_target[ids].clone()
            self.reference_residual_controller.reset(q,q,env_ids=ids)
            self._processed_actions[ids] = q
            self._reference_fresh[ids] = False
