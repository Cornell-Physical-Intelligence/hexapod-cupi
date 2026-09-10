import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import tempfile
import numpy as np
import torch

TMP=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(TMP/'reference_policy_observation_005_001'),str(TMP/'reference_device_smoke_001'),
              str(TMP/'reference_physics_adapter_009/source_009/tools')]
from fixtures import Rig
from sensor_freshness import ContactFreshness
from recovery import RecoveryTargets,rebase_reset_clocks,defer_training_resets
from canonical_stance_startup import CanonicalStanceStartup
import moving_session


class Sensor:
    def __init__(self,n):
        self.cfg=types.SimpleNamespace(update_period=.0025);self._is_initialized=True
        self._timestamp=torch.zeros(n,dtype=torch.float32);self._timestamp_last_update=self._timestamp.clone()
        self._is_outdated=torch.zeros(n,dtype=torch.bool)
    @property
    def data(self):
        self._timestamp_last_update[self._is_outdated]=self._timestamp[self._is_outdated]
        self._is_outdated.zero_()
        return self


class Reader:
    def __init__(self,env,*args):self.env=env
    def capture(self,env,**kwargs):
        packet=env.rig.packet();m=packet['measurement']
        m.update(time_s=kwargs['time_s'],terminated=kwargs['terminated'],truncated=kwargs['truncated'],
            contact_valid=kwargs['contact_valid'],contact_age_s=kwargs['contact_age_s'])
        m.update(quaternion_world_xyzw=torch.tensor([[0.,0.,0.,1.]]*env.num_envs,dtype=torch.float64),
            velocity_world_mps=torch.zeros(env.num_envs,3,dtype=torch.float64),
            computed_torque_nm=torch.ones(env.num_envs,18,dtype=torch.float64),
            applied_torque_nm=torch.ones(env.num_envs,18,dtype=torch.float64))
        return packet


class Recorder:
    def __init__(self,env):
        self.env=env;self.rows=[];self.active=False;self.control=-1;self.completed_controls=0
    def begin_control(self,index):
        assert index==self.completed_controls;self.active=True;self.control=index
    def end_control(self,row):
        assert self.active and len(self.rows)==8*(self.control+1)
        self.active=False;self.completed_controls+=1


class Env:
    def __init__(self,n=32):
        self.rig=Rig(n);self.num_envs=n;self.device='cpu';self.step_dt=.02
        self.episode_length_buf=torch.zeros(n,dtype=torch.long);self.control=0;self.done_calls=0;self.reset_ids=[]
        self.interior_spike_control=None;self.spike_kind='computed_torque_nm'
        self.reset_terminated=torch.zeros(n,dtype=torch.bool);self.reset_time_outs=self.reset_terminated.clone();self.reset_buf=self.reset_terminated.clone()
        self.reference_residual_controller=self.rig.controller;self.reference_residual_target=self.rig.c
        self._robot=types.SimpleNamespace(data=types.SimpleNamespace(default_joint_pos=self.rig.wave.s['q'].float(),
            soft_joint_pos_limits=self.rig.packet()['measurement']['soft_joint_pos_limits_rad'].float()))
        sensors=[Sensor(n) for _ in range(14)]
        self._feet_contact_sensors=sensors[:6];self._coxa_contact_sensor=sensors[6]
        self._femur_contact_sensors=sensors[7:13];self._base_contact_sensor=sensors[13];self.sensors=sensors
    def set_reference_targets(self,q,v,a,valid):self.pending=(q,v,a,valid)
    def set_evaluation_targets(self,command):self.command=command
    def _get_dones(self):
        self.done_calls+=1;t=torch.zeros(self.num_envs,dtype=torch.bool)
        if self.control==215:t[0]=True
        return t,torch.zeros_like(t)
    def _reset_idx(self,ids):
        self.reset_ids.append(ids.clone());q=self.reference_residual_controller.reference_position.clone()
        q[ids]=self._robot.data.default_joint_pos[ids].double()+.01
        self.reference_residual_controller.reset(q[ids],q[ids],env_ids=ids)
        for s in self.sensors:
            s._timestamp[ids]=0;s._timestamp_last_update[ids]=0;s._is_outdated[ids]=True
        for i in ids.tolist():self.rig.fs[i].qtarget=q[i].numpy().copy()
    def step(self,actions):
        q,v,a,valid=self.pending
        self.rig.c=self.reference_residual_controller.step(q,actions,reference_valid=valid,
            analytic_reference_velocity=v,analytic_reference_acceleration=a)
        self.reference_residual_target=self.rig.c
        for i,f in enumerate(self.rig.fs):
            f.qtarget=self.rig.c['target_position_rad'][i].numpy().copy();f.time+=.02
        self.control+=1;self.episode_length_buf+=1;self.rig.steps+=1
        for substep in range(8):
            for s in self.sensors:s._timestamp+=.0025;s._is_outdated.fill_(True)
            row={'computed_torque_nm':np.ones((self.num_envs,18)),'applied_torque_nm':np.ones((self.num_envs,18))}
            if self.control==self.interior_spike_control and substep==3:row[self.spike_kind][1,4]=1.61
            self.session.recorder.rows.append(row)
        t,tr=self._get_dones()
        self.reset_terminated.copy_(t);self.reset_time_outs.copy_(tr);self.reset_buf.copy_(t|tr)
        # Demonstrate the native loop would reset only returned done flags.
        if (t|tr).any():self._reset_idx(torch.nonzero(t|tr).flatten())
        return {'policy':torch.zeros(self.num_envs,675),'critic':torch.zeros(self.num_envs,678)},torch.zeros(self.num_envs),t,tr,{}


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_tensor_recovery_matches_exact_scalar_knots(self):
        rig=Rig(32);nominal=rig.wave.s['q'];initial=nominal+.029*torch.sin(torch.arange(32*18).reshape(32,18))
        limits=rig.packet()['measurement']['soft_joint_pos_limits_rad']
        a=RecoveryTargets(initial,nominal,limits)
        b=CanonicalStanceStartup(initial.numpy(),nominal.numpy(),limits[...,0].numpy(),limits[...,1].numpy())
        for index in range(202):
            x=a.sample_next();y=b.sample(index+1)
            for k,old in [('q_ref','q_ref'),('v_ref','analytic_velocity_rad_s'),('a_ref','analytic_acceleration_rad_s2')]:
                torch.testing.assert_close(x[k],torch.tensor(y[old]),rtol=0,atol=1e-14)
            a.advanced()

    def test_actual_class_mixed_recovery_history_and_clock_isolation(self):
        env=Env();stance={'joint_positions_rad':dict(zip(env.rig.names,env._robot.data.default_joint_pos[0].tolist()))}
        with tempfile.TemporaryDirectory() as d,patch.object(moving_session,'DeviceTelemetry',Reader),patch.object(moving_session,'PhysicsSubstepRecorder',Recorder):
            s=moving_session.MovingSession(env,{'joint_names_runtime':env.rig.names},None,stance,
                warp_to_torch=lambda x:x,output=Path(d)/'raw');env.session=s;s.quiet_only.fill_(True)
            with defer_training_resets(env,s._capture):
                for _ in range(200):s._advance(s.zero,initializing=True)
                self.assertTrue(s.active.all());self.assertFalse(s.encoder.last_fd_valid.any())
                for _ in range(15):out,r,done,extras=s.step(s.zero)
                self.assertEqual(env.done_calls,215);self.assertEqual(len(env.reset_ids),1)
                self.assertTrue(done[0]);self.assertEqual(float(r[0]),-3.)
                self.assertEqual(int(extras['masked_transition']['final_episode_id'][0]),0)
                self.assertEqual(int(out['episode_id'][0,0]),1);self.assertFalse(out['learning_valid'][0,0])
                self.assertTrue((out['policy'][0]==0).all())
                for _ in range(199):out,*_=s.step(s.zero)
                self.assertFalse(s.active[0]);self.assertTrue(s.active[1:].all())
                out,*_=s.step(s.zero)
                self.assertTrue(s.active.all());self.assertEqual(int(s.age[0]),0)
                self.assertFalse(s.encoder.last_fd_valid[0]);self.assertTrue(s.encoder.last_fd_valid[1:].all())
                self.assertEqual(int(s.encoder.history_valid[0].sum()),1)
                self.assertTrue((s.encoder.history_valid[1:].sum(-1)==5).all())
                self.assertEqual(int(s.age[1]),215)
                self.assertTrue(s.freshness.ready.all());self.assertEqual(env.control,415)

    def test_reset_clock_rebase_requires_real_selected_reset(self):
        env=Env();f=ContactFreshness(env,lambda x:x);mask=torch.zeros(32,dtype=torch.bool);mask[2]=True
        with self.assertRaises(ValueError):rebase_reset_clocks(f,mask)
        for sensor in env.sensors:sensor._is_outdated[2]=True
        result=rebase_reset_clocks(f,mask);self.assertFalse(result['measurement_admitted'])
        env.sensors[0]._timestamp[1]=.1
        with self.assertRaises(ValueError):rebase_reset_clocks(f,mask)

    def test_command_change_enters_next_observation_before_next_reference(self):
        env=Env();stance={'joint_positions_rad':dict(zip(env.rig.names,env._robot.data.default_joint_pos[0].tolist()))}
        with tempfile.TemporaryDirectory() as d,patch.object(moving_session,'DeviceTelemetry',Reader),patch.object(moving_session,'PhysicsSubstepRecorder',Recorder):
            s=moving_session.MovingSession(env,{'joint_names_runtime':env.rig.names},None,stance,
                warp_to_torch=lambda x:x,output=Path(d)/'raw');env.session=s
            with defer_training_resets(env,s._capture):
                for _ in range(200):s._advance(s.zero,initializing=True)
                self.assertEqual(float(s.commands[1,0]),.005)
                first=s.get_observations()['policy']
                self.assertAlmostEqual(float(first[1,4*63+6]/5),.005,8)
                s.quiet_only[1]=True  # Synthetic scheduler steer; current packet remains unchanged.
                s.step(s.zero)
                self.assertEqual(float(s.rows[-1]['requested_command'][1,0]),.005)
                self.assertEqual(float(s.get_observations()['policy'][1,4*63+6]),0.)
                s.step(s.zero)
                self.assertEqual(float(s.rows[-1]['requested_command'][1,0]),0.)
    def test_interior_requested_spike_is_one_row_terminal_and_applied_spike_is_fatal(self):
        for kind in ('computed_torque_nm','applied_torque_nm'):
            env=Env();env.interior_spike_control=205;env.spike_kind=kind
            stance={'joint_positions_rad':dict(zip(env.rig.names,env._robot.data.default_joint_pos[0].tolist()))}
            with tempfile.TemporaryDirectory() as d,patch.object(moving_session,'DeviceTelemetry',Reader),patch.object(moving_session,'PhysicsSubstepRecorder',Recorder):
                s=moving_session.MovingSession(env,{'joint_names_runtime':env.rig.names},None,stance,
                    warp_to_torch=lambda x:x,output=Path(d)/'raw');env.session=s;s.quiet_only.fill_(True)
                with defer_training_resets(env,s._capture):
                    for _ in range(200):s._advance(s.zero,initializing=True)
                    for _ in range(4):s.step(s.zero)
                    if kind=='applied_torque_nm':
                        with self.assertRaises(RuntimeError):s.step(s.zero)
                        self.assertFalse(env.reset_ids)
                        self.assertEqual(len(s.rows),205)
                    else:
                        out,reward,done,extra=s.step(s.zero)
                        self.assertEqual(int(done.sum()),1);self.assertTrue(done[1])
                        self.assertEqual(float(reward[1]),-3.)
                        self.assertEqual(float(s.rows[-1]['computed_torque_nm'][1,4]),1.)
                        self.assertEqual(float(s.recorder.rows[-5][kind][1,4]),1.61)
                        self.assertEqual(env.reset_ids[-1].tolist(),[1])
if __name__=='__main__':unittest.main()
