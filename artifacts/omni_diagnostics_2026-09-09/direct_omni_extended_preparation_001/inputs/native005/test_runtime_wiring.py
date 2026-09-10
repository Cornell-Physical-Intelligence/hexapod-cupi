"""Focused new seams; synthetic SDK fixtures, never physical admission."""
import json, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
LEGACY=HERE.parents[1]/'ppo_repair_003_preparation/tools'
sys.path.append(str(LEGACY))
from omni_flat_math import ObservationHistory, slew_commands
from direct_stop_evaluation import evaluate_stop
from direct_training import audited_environment, export_audit
from direct_contract import validate_result, sha, EXPECTED_OVERRIDES, URDF
from direct_config import selection

class FakeStopEnv:
    def __init__(self):
        self.num_envs=48;self.step_dt=.02;self.device='cpu';self._robot=NS(joint_names=['joint'+str(i) for i in range(18)])
        self.episode_length_buf=torch.zeros(48,dtype=torch.long);self.reset()
    def reset(self,seed=None):
        self._commands=torch.zeros(48,3);self.targets=torch.zeros_like(self._commands);self.t=0
        self.history=ObservationHistory(48,5,63,'cpu');self.age=np.zeros(48);return self._get_observations(),{}
    def set_evaluation_targets(self,targets):self.targets=targets.clone()
    def _get_observations(self):
        frame=torch.zeros(48,63);frame[:,5]=-1;frame[:,6:9]=self._commands*torch.tensor([5,5,2.5])
        actor=self.history.observe(frame,self.t)
        return {'policy':actor,'critic':torch.cat((actor,torch.zeros(48,3)),1)}
    def step(self,action):
        self.age+=.02;n=48;j=np.zeros((n,18));v=np.zeros((n,3));gravity=v.copy();gravity[:,2]=-1;quat=np.zeros((n,4));quat[:,3]=1
        term=np.zeros(n,bool);term[0]=self.t==99
        self.omni_diagnostic_sample={'age_s':self.age.copy(),'command':self._commands.numpy().copy(),'target_command':self.targets.numpy().copy(),
            'joint_position_rad':j.copy(),'joint_velocity_rad_s':j.copy(),'joint_target_rad':j.copy(),'computed_torque_nm':j.copy(),'applied_torque_nm':j.copy(),
            'velocity_navigation_mps':v.copy(),'finite_difference_velocity_navigation_mps':v.copy(),'gyro_navigation_rad_s':v.copy(),
            'position_world_m':v.copy(),'quaternion_world_wxyz':quat,'projected_gravity':gravity,'terminated':term,'truncated':np.zeros(n,bool)}
        self._commands=slew_commands(self._commands,self.targets,.02);self.t+=1
        if term[0]:self._commands[0]=0;self.age[0]=0;self.history.reset(torch.tensor([0]))
        return self._get_observations(),torch.zeros(n),torch.from_numpy(term),torch.zeros(n,dtype=torch.bool),{}

class WiringTests(unittest.TestCase):
    def test_complete_stop_rollout_and_observed_command_timing(self):
        env=FakeStopEnv();runner=NS(get_inference_policy=lambda device:lambda obs:torch.zeros(48,18))
        with tempfile.TemporaryDirectory() as folder:
            report=evaluate_stop(env,runner,{'omni':{'overrides':EXPECTED_OVERRIDES}},Path(folder),'c'*64)
            self.assertTrue(report['complete']);self.assertEqual(report['controls'],1600)
            self.assertEqual(report['observation_audit']['max_command_slice_difference'],0.)
            self.assertEqual(report['observation_audit']['max_history_shift_difference'],0.)
            self.assertFalse(report['scenarios'][0]['replicas'][0]['quiet']['pass'])
            trace=np.load(Path(folder)/'stop_trace.npz')
            self.assertEqual(trace['target_command'][199,4,0],0.)
            self.assertAlmostEqual(trace['target_command'][200,4,0],.1)
            self.assertEqual(trace['command'][200,4,0],0.)
            self.assertGreater(trace['command'][201,4,0],0.)
            self.assertGreater(trace['command'][800,4,0],0.)
            self.assertEqual(trace['target_command'][800,4,0],0.)
            self.assertTrue(np.all(trace['command'][1100:]==0))
            self.assertTrue(np.all(trace['quaternion_world_xyzw'][...,3]==1));self.assertTrue(np.all(trace['quaternion_world_wxyz'][...,0]==1))

    def test_actual_audit_hook_keeps_pre_reset_event_and_failed_row(self):
        t=lambda value:NS(torch=value)
        class Parent:
            def __init__(self):
                self.num_envs=4;self.step_dt=.02;self._commands=torch.zeros(4,3)
                self.reset_terminated=torch.tensor([False,True,False,False]);self.reset_time_outs=torch.zeros(4,dtype=torch.bool)
                self._episode_elapsed_s=torch.ones(4);self._processed_actions=torch.full((4,18),.3);self.omni_previous_target=torch.full((4,18),.29);self.omni_raw_policy_action=torch.zeros(4,18)
                d={k:t(torch.zeros(4,18)) for k in ['joint_pos','joint_vel','computed_torque','applied_torque']}
                d.update({k:t(torch.zeros(4,3)) for k in ['root_lin_vel_b','root_ang_vel_b','root_pos_w']});d['root_quat_w']=t(torch.tensor([[0.,0.,0.,1.]]).repeat(4,1))
                self._robot=NS(data=NS(**d),joint_names=['j'+str(i) for i in range(18)])
                self._coxa_contact_sensor=NS(data=NS(net_forces_w_history=t(torch.zeros(4,1,6,3))))
                self._femur_contact_sensors=[NS(data=NS(net_forces_w_history=t(torch.zeros(4,1,1,3)))) for _ in range(6)]
            def _vector_in_command_frame(self,x):return x
            def _get_foot_contact_state(self):return torch.ones(4,6,dtype=torch.bool),torch.zeros(4,6,dtype=torch.bool),torch.zeros(4,6)
            def _get_rewards(self):self._commands+=1;return torch.ones(4)
        env=audited_environment(Parent)()
        with torch.inference_mode():env._get_rewards()
        env._robot.data.joint_pos.torch.fill_(10) # Simulated subsequent reset must not overwrite captured event.
        self.assertTrue(torch.all(env.direct_audit_rows[0]['command']==0));self.assertTrue(torch.all(env.direct_audit_events[0]['fields']['joint_position_rad']==0))
        env._robot.data.applied_torque.torch[3,17]=1.7
        with torch.inference_mode(),self.assertRaisesRegex(RuntimeError,'Applied|applied'):env._get_rewards()
        self.assertEqual(len(env.direct_audit_rows),2)
        with tempfile.TemporaryDirectory() as folder:
            report=export_audit(env,Path(folder));self.assertGreater(report['applied_torque_max_per_row_nm'][3],1.6)
            self.assertEqual(report['terminations_per_row'],[0,2,0,0])

    def test_training_result_rejects_missing_raw_changed_checkpoint_and_hidden_replica(self):
        selected=selection('train','smoke','quiet_priority',None,None)
        identity={'plan_sha256':'p'*64,'source_manifest_sha256':'s'*64,'selection':selected}
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'policy').mkdir();(p/'policy/final.pt').write_bytes(b'actual-fixture');(p/'policy/decision_002.pt').write_bytes(b'decision-fixture')
            for file in ['training_trace.npz','training_joint_trace.npz','training_events.json']:(p/file).write_bytes(b'raw-fixture')
            audit={'controls':48,'replicas':32,'terminations_per_row':[0]*32,'truncations_per_row':[0]*32,'requested_torque_max_per_row_nm':[7.]*32,'applied_torque_max_per_row_nm':[1.6]*32,'requested_saturation_fraction_per_row':[.1]*32,
                   'trace_sha256':sha(p/'training_trace.npz'),'joint_trace_sha256':sha(p/'training_joint_trace.npz'),'event_ledger_sha256':sha(p/'training_events.json')}
            receipt={'selection':selected,'complete':True,'updates_completed':2,'wall_seconds':1.,'audit':audit,
                     'reload':{'passed':True,'exact_actor_critic_normalizer_optimizer':True,'exact_deterministic_action':True,'optimizer_entries':17,'checkpoint_sha256':sha(p/'policy/final.pt')},
                     'final_checkpoint_sha256':sha(p/'policy/final.pt'),'decision_checkpoints':{'2':{'file':'decision_002.pt','completed_updates':2,'sha256':sha(p/'policy/decision_002.pt')}}}
            state={'status':'completed','mode':'train','variant':'f050_t060','urdf_sha256':URDF,'plan_sha256':identity['plan_sha256'],'stance_index':0,'direct_selection':selected,'iterations':2,'checkpoint_sha256':receipt['final_checkpoint_sha256'],'training_receipt':receipt}
            init={'checkpoint_sha256':'1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8','actor_and_critic_preserved_except_std':True,'observation_normalizers_preserved':True,'optimizer_state_entries':0}
            def write():
                for file,data in [('state.json',state),('training_receipt.json',receipt),('repair_initialization.json',init)]: (p/file).write_text(json.dumps(data))
            from test_optimizer_contract import diagnostic_receipt
            receipt.update(diagnostic_receipt(2,32))
            write();self.assertTrue(validate_result(p,'train',identity)['complete'])
            audit['applied_torque_max_per_row_nm'][31]=float('nan');write()
            with self.assertRaises(ValueError):validate_result(p,'train',identity)
            audit['applied_torque_max_per_row_nm'][31]=1.6;write();(p/'training_trace.npz').unlink()
            with self.assertRaises(FileNotFoundError):validate_result(p,'train',identity)
            (p/'training_trace.npz').write_bytes(b'raw-fixture');(p/'policy/final.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_result(p,'train',identity)

if __name__=='__main__':unittest.main()
