"""CPU execution of actual wrapper hooks, with explicit minimal SDK doubles."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_reference_residual import NAMES


class History:
    def __init__(self,n,h,w,device):
        self.data=torch.zeros(n,h*w);self.width=w;self.step=-1
    def observe(self,current,step):
        if step!=self.step:
            self.data=torch.cat((self.data[:,self.width:],current),-1);self.step=step
        return self.data.clone()
    def reset(self,ids):self.data[ids]=0


def box(t):return SimpleNamespace(torch=t)


class Parent:
    def __init__(self,cfg,*args,**kwargs):
        self.cfg=cfg;self.num_envs=2;self.device='cpu';self.step_dt=.02
        self.omni_step=0;self.omni_noise_step=-1
        self._actions=torch.zeros(2,18)
        self._processed_actions=torch.zeros_like(self._actions)
        self._previous_processed_joint_target=self._processed_actions.clone()
        self._has_previous_processed_joint_target=torch.zeros(2,dtype=torch.bool)
        self.omni_previous_action=self._actions.clone();self.omni_previous_velocity=self._actions.clone()
        self.omni_previous_target=self._actions.clone();self._commands=torch.zeros(2,3)
        self.omni_diagnostic_enabled=True
        limits=torch.stack((torch.ones(2,18)*-1,torch.ones(2,18)),dim=-1)
        quat=torch.tensor([[0.,0.,np.sin(np.deg2rad(5.)),np.cos(np.deg2rad(5.))]]).repeat(2,1)
        data=SimpleNamespace(soft_joint_pos_limits=box(limits),default_joint_pos=box(torch.zeros(2,18)),
            joint_pos=box(torch.zeros(2,18)),joint_vel=box(torch.zeros(2,18)),
            applied_torque=box(torch.ones(2,18)*.5),computed_torque=box(torch.ones(2,18)*.5),
            root_pos_w=box(torch.zeros(2,3)),root_quat_w=box(quat),
            root_ang_vel_b=box(torch.zeros(2,3)),root_lin_vel_b=box(torch.zeros(2,3)),
            projected_gravity_b=box(torch.tensor([[0.,0.,-1.]]).repeat(2,1)))
        self.sent_position=None;self.sent_velocity=None
        self._robot=SimpleNamespace(joint_names=NAMES,data=data,
            set_joint_position_target_index=lambda target:setattr(self,'sent_position',target.clone()),
            set_joint_velocity_target_index=lambda target:setattr(self,'sent_velocity',target.clone()))
    def _reset_idx(self,ids):
        ids=torch.arange(2) if ids is None else ids
        self._previous_processed_joint_target[ids]=0
        if hasattr(self,'omni_history'):self.omni_history.reset(ids)
    def _get_rewards(self):
        self.omni_diagnostic_sample={'quaternion_world_wxyz':self._robot.data.root_quat_w.torch.numpy().copy(),
                                    'filtered_target_rad':np.zeros((2,18)), 'unfiltered_target_rad':np.zeros((2,18))}
        return torch.zeros(2)
    def _vector_in_command_frame(self,value):return value


def load_wrapper():
    mods={name:ModuleType(name) for name in ('c_study_runtime','omni_flat_env','omni_flat_math')}
    mods['c_study_runtime'].bootstrap_c_study_runtime=lambda:{'CPU_fixture':True}
    mods['omni_flat_env'].OmniFlatEnv=Parent
    def configure(cfg,overrides):
        cfg.omni_target_filter_time_constant_s=0.;cfg.omni_observation_noise_scale=1.
    mods['omni_flat_env'].configure_omni=configure
    mods['omni_flat_math'].ObservationHistory=History
    spec=importlib.util.spec_from_file_location('reference_env_fixture',Path(__file__).with_name('reference_residual_env.py'))
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,mods):spec.loader.exec_module(module)
    return module


class WrapperTests(unittest.TestCase):
    def make(self):
        mod=load_wrapper();cfg=SimpleNamespace()
        mod.configure_reference_residual_physics(cfg,dict(profile='formal_004',residual_radius_rad=.02,
            residual_velocity_rad_s=.25,residual_acceleration_rad_s2=2.,total_acceleration_rad_s2=8.))
        return mod,mod.ReferenceResidualPhysicsEnv(cfg)

    def test_actual_hooks_zero_reference_and_observable_state(self):
        _,env=self.make();q=torch.zeros(2,18,dtype=torch.float64);raw=q.float()
        for _ in range(5):
            env.set_reference_targets(q,q,q,torch.ones(2,dtype=torch.bool))
            env._pre_physics_step(raw);env._apply_action();reward=env._get_rewards()
            self.assertTrue(torch.equal(env.sent_position,raw));self.assertTrue(torch.equal(env.sent_velocity,raw))
            self.assertTrue(torch.isfinite(reward).all())
            obs=env._get_observations();again=env._get_observations()
            self.assertEqual(obs['policy'].shape,(2,675));self.assertEqual(obs['critic'].shape,(2,678))
            self.assertTrue(torch.equal(obs['policy'],again['policy']))
            self.assertEqual(float(np.abs(env.reference_residual_sample['reference_to_executable_lag_rad']).max()),0.)
        sample=env.omni_diagnostic_sample
        self.assertNotIn('filtered_target_rad',sample)
        np.testing.assert_array_equal(sample['quaternion_world_wxyz'],sample['quaternion_world_xyzw'][...,[3,0,1,2]])
        self.assertFalse(env.reference_residual_contract['full_reference_generator_state_in_policy_observation'])

    def test_stale_reference_nonfinite_torque_and_training_rejected(self):
        mod,env=self.make();q=torch.zeros(2,18)
        with self.assertRaises(RuntimeError):env._pre_physics_step(q)
        env.set_reference_targets(q,q,q,torch.ones(2,dtype=torch.bool))
        with self.assertRaises(ValueError):env.set_reference_targets(q,q,q,torch.zeros(2,dtype=torch.bool))
        with self.assertRaises(RuntimeError):env._pre_physics_step(q)
        with self.assertRaises(ValueError):mod.ReferenceResidualPhysicsEnv(env.cfg,evaluation=False)
        env.set_reference_targets(q,q,q,torch.ones(2,dtype=torch.bool));env._pre_physics_step(q)
        env._robot.data.applied_torque.torch[0,0]=float('nan')
        with self.assertRaises(RuntimeError):env._get_rewards()

    def test_actual_wrapper_persistent_actions_reset_outside_inference(self):
        _,env=self.make();q=torch.zeros(2,18)
        env.set_reference_targets(q,q,q,torch.ones(2,dtype=torch.bool))
        with torch.inference_mode():
            env._pre_physics_step(torch.ones_like(q))
            env._apply_action();env._get_rewards()
        # The real pinned parent writes _actions before our own reset hook.
        env._actions[0]=0
        env._joint_target_slew_limited_fraction[0]=0
        env._reset_idx(torch.tensor([0]))
        self.assertFalse(torch.is_inference(env._actions))
        self.assertFalse(torch.is_inference(env._processed_actions))
        self.assertFalse(torch.is_inference(env._joint_target_slew_limited_fraction))
        self.assertTrue(torch.equal(env._processed_actions[0],q[0]))


if __name__=='__main__':unittest.main()
