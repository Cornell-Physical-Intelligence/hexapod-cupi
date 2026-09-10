"""CPU test the actual adapter methods with explicit fake articulation data."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from velocity_action import *

HERE=Path(__file__).parent
REPO=HERE.parents[1]
import sys
sys.path.append(str(REPO/"tools"))
space={'torch':torch}
math_tree=ast.parse((REPO/'tools/omni_flat_math.py').read_text())
history=next(n for n in math_tree.body if isinstance(n,ast.ClassDef) and n.name=='ObservationHistory')
exec(compile(ast.Module(body=[history],type_ignores=[]),'real_history_helper','exec'),space)
ObservationHistory=space['ObservationHistory']
NAMES=tuple(f'joint_{i}' for i in range(18))
def wrapped(value):return SimpleNamespace(torch=value)
class FakeBase:
    def __init__(self,cfg,render_mode=None,*,evaluation=False,**kw):
        self.cfg=cfg;self.num_envs=2;self.device='cpu';self.step_dt=.02;self.omni_step=0
        self._actions=torch.zeros(2,18);self._commands=torch.zeros(2,3)
        data=SimpleNamespace(joint_pos=wrapped(torch.zeros(2,18)),default_joint_pos=wrapped(torch.zeros(2,18)),
            joint_vel=wrapped(torch.zeros(2,18)),soft_joint_pos_limits=wrapped(torch.tensor([-1.,1.]).expand(2,18,2).clone()),
            root_ang_vel_b=wrapped(torch.zeros(2,3)),projected_gravity_b=wrapped(torch.tensor([0.,0.,-1.]).expand(2,3)),
            root_lin_vel_b=wrapped(torch.zeros(2,3)),root_pos_w=wrapped(torch.zeros(2,3)),
            root_quat_w=wrapped(torch.tensor([0.,0.,0.,1.]).expand(2,4)),
            applied_torque=wrapped(torch.ones(2,18)*.5),computed_torque=wrapped(torch.ones(2,18)*.6))
        self._robot=SimpleNamespace(joint_names=NAMES,data=data,set_joint_position_target_index=self.position_drive,set_joint_velocity_target_index=self.velocity_drive)
        self._processed_actions=torch.zeros(2,18);self._previous_processed_joint_target=torch.zeros(2,18)
        self._has_previous_processed_joint_target=torch.zeros(2,dtype=torch.bool)
        self.omni_previous_action=torch.zeros(2,18);self.omni_previous_velocity=torch.zeros(2,18);self.omni_previous_target=torch.zeros(2,18)
        self.omni_noise_step=-1;self.omni_noise=torch.zeros(2,63)
    def _get_rewards(self):
        self.omni_measurement_sample={'saturation':torch.zeros(2),'nonfoot':torch.zeros(2),'requested_torque_max':torch.ones(2)*.6}
        self.omni_diagnostic_sample={'filtered_target_rad':0,'unfiltered_target_rad':0}
        return torch.zeros(2)
    def _pre_physics_step(self,actions):raise AssertionError('Absolute parent mapping must not run')
    def _vector_in_command_frame(self,x):return x
    def _reset_idx(self,ids):
        if ids is None:ids=torch.arange(self.num_envs)
        self._previous_processed_joint_target[ids]=getattr(self.cfg,'test_reset_position',0.)
        self._actions[ids]=0.;self._commands[ids]=0.
        self.omni_history.reset(ids)
    def position_drive(self,*,target):self.last_position=target.clone()
    def velocity_drive(self,*,target):self.last_velocity=target.clone()

ns={'torch':torch,'OmniFlatEnv':FakeBase,'ObservationHistory':ObservationHistory,
    'JointTargetVelocity':JointTargetVelocity,'VelocityActionConfig':VelocityActionConfig,
    'append_executable_state':append_executable_state,'LINEAGE':LINEAGE,'RUNTIME_BINDING':{'mode':'CPU_adapter_test'}}
tree=ast.parse((HERE/'candidate_env.py').read_text())
body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))]
exec(compile(ast.Module(body=body,type_ignores=[]),'actual_candidate_env_methods','exec'),ns)
Env=ns['VelocityCandidateEnv']

def env():
    return Env(SimpleNamespace(candidate_action_config={'profile':'formal_004','max_acceleration_rad_s2':8.},omni_observation_noise_scale=1.))

class AdapterTests(unittest.TestCase):
    def test_constructor_resets_before_exposing_controller_state(self):
        e=Env(SimpleNamespace(candidate_action_config={'profile':'formal_004','max_acceleration_rad_s2':8.},omni_observation_noise_scale=1.,test_reset_position=.3))
        self.assertTrue(torch.allclose(e.target_velocity_controller.position,torch.full((2,18),.3)))
        self.assertTrue((e.target_velocity_controller.velocity==0).all())
    def test_quaternion_heading_converted_from_actual_xyzw(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from omni_quiet_review import quiet_metrics
        for axis,expected in [('z',10.),('x',0.)]:
            e=env();e.omni_diagnostic_enabled=True
            states=[]
            for angle in np.linspace(0,10,20):
                quat=Rotation.from_euler(axis,angle,degrees=True).as_quat()
                e._robot.data.root_quat_w.torch=torch.tensor(np.tile(quat,(2,1)),dtype=torch.float32)
                e._pre_physics_step(torch.zeros(2,18));e._get_rewards()
                states.append(e.omni_diagnostic_sample['quaternion_world_wxyz'])
            zeros=np.zeros((20,2,18));data={key:zeros.copy() for key in ('joint_position_rad','joint_target_rad','joint_velocity_rad_s','computed_torque_nm','applied_torque_nm')}
            data.update(position_world_m=np.zeros((20,2,3)),quaternion_world_wxyz=np.stack(states),terminated=np.zeros((20,2),bool),truncated=np.zeros((20,2),bool))
            score=quiet_metrics(data,0,0,NAMES,.02)
            self.assertAlmostEqual(score['max_heading_excursion_deg'],expected,places=4)
    def test_shape_current_state_and_repeat_idempotency(self):
        e=env();e._pre_physics_step(torch.ones(2,18)*.2)
        a=e._get_observations();b=e._get_observations()
        self.assertEqual(a['policy'].shape,(2,495));self.assertEqual(a['critic'].shape,(2,498))
        self.assertTrue(torch.equal(a['policy'],b['policy']))
        frame=a['policy'].reshape(2,5,99)[:,-1]
        self.assertTrue(torch.equal(frame[:,63:81],e.target_velocity_controller.position))
        self.assertTrue(torch.equal(frame[:,81:],e.target_velocity_controller.velocity/2.))
        self.assertTrue(torch.equal(a['critic'][:,:495],a['policy']))
    def test_apply_action_and_partial_reset_no_stale_state(self):
        e=env();e._pre_physics_step(torch.ones(2,18));e._get_observations();e._apply_action()
        self.assertTrue(torch.equal(e.last_position,e.target_velocity_controller.position))
        self.assertTrue(torch.equal(e.last_velocity,torch.zeros(2,18)))
        other=e.target_velocity_controller.position[1].clone();e._reset_idx(torch.tensor([0]))
        obs=e._get_observations()['policy'].reshape(2,5,99)
        self.assertTrue(torch.equal(obs[0,:,63:],torch.zeros(5,36)))
        self.assertTrue(torch.equal(e.target_velocity_controller.position[1],other))
    def test_history_uses_99_wide_frames(self):
        e=env();old=e._get_observations()['policy'].clone().reshape(2,5,99)
        e._pre_physics_step(torch.ones(2,18)*.1);new=e._get_observations()['policy'].reshape(2,5,99)
        self.assertTrue(torch.equal(new[:,:-1],old[:,1:]))
        self.assertFalse(torch.equal(new[:,-1,63:],old[:,-1,63:]))
    def test_pre_reset_measurements_and_diagnostic_contract(self):
        e=env();e.omni_diagnostic_enabled=True;e.omni_measurements_enabled=True
        e._pre_physics_step(torch.zeros(2,18));e._get_rewards()
        self.assertEqual(e.omni_diagnostic_start_position.shape,(2,3))
        self.assertEqual(e.omni_diagnostic_start_quaternion.shape,(2,4))
        self.assertEqual(e.omni_measurement_sample['max_applied_torque'].tolist(),[.5,.5])
        self.assertNotIn('unfiltered_target_rad',e.omni_diagnostic_sample)
        self.assertEqual(e.omni_diagnostic_sample['executable_target_velocity_rad_s'].shape,(2,18))
        self.assertEqual(e.omni_diagnostic_sample['quaternion_world_wxyz'][0].tolist(),[1.,0.,0.,0.])
        e._robot.data.applied_torque.torch[:]=float('nan')
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):e._get_rewards()
    def test_feedback_moves_targets_at_zero_body_command(self):
        e=env();self.assertTrue((e._commands==0).all())
        e._pre_physics_step(torch.ones(2,18)*.1)
        self.assertTrue((e.target_velocity_controller.position>0).all())

if __name__=='__main__':unittest.main()
