"""Actual session class methods with synthetic fixed measurements, no Isaac/GPU.

Only physical stepping/capture/recorder dependencies are replaced. The exact
wave005, residual002 and846/849 encoder execute on CPU. This is timing/history
contract evidence, not support or motor qualification.
"""
import ast,contextlib,sys,types,unittest
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
ROOT=Path(__file__).resolve().parent;TMP=ROOT.parent
sys.path.insert(0,str(TMP/'reference_policy_observation_005_001'))
from fixtures import Rig
from observation import ObservationBuilder
from batch_wave import BatchWave005

def array(v):return v.detach().cpu().numpy().copy()
class Recorder:
    def __init__(self,*a):pass
    def __enter__(self):return self
    def __exit__(self,*a):return False
@contextlib.contextmanager
def capture(*args):yield []
class Startup:
    def __init__(self,initial,*a,**k):self.initial=torch.tensor(initial,dtype=torch.float64)
    def sample(self,k):return {'q_ref':self.initial,'v_ref':torch.zeros_like(self.initial),'a_ref':torch.zeros_like(self.initial),'valid':torch.ones(len(self.initial),dtype=torch.bool)}

def actual_class():
    path=TMP/'reference_residual_ppo_001/rollout_session.py';tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ReferenceResidualSession')
    ns=dict(np=np,torch=torch,TensorDict=TensorDict,ObservationBuilder=ObservationBuilder,BatchWave005=BatchWave005,DeviceTelemetry=lambda *a:None,ContactFreshness=lambda *a:None,PhysicsSubstepRecorder=Recorder,CanonicalStanceStartup=Startup,array=array,capture_before_reset=capture)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns);return ns['ReferenceResidualSession']

class Harness:
    def __init__(self,initial=None):
        self.rig=Rig(1);r=self.rig
        data=types.SimpleNamespace(default_joint_pos=r.wave.s['q'].float(),soft_joint_pos_limits=r.packet()['measurement']['soft_joint_pos_limits_rad'].float())
        env=types.SimpleNamespace(num_envs=1,step_dt=.02,device='cpu',max_episode_length=1000,episode_length_buf=torch.zeros(1,dtype=torch.long),_robot=types.SimpleNamespace(data=data),reference_residual_controller=types.SimpleNamespace(reference_position=r.wave.s['q'].float()),reference_residual_target=r.c)
        stance={'joint_positions_rad':dict(zip(r.names,data.default_joint_pos[0].tolist()))}
        self.session=actual_class()(env,{'joint_names_runtime':r.names},None,stance,warp_to_torch=None,initial_commands=initial)
        self.calls=[];self.throw=False
        def physical(s,reference,actions,*,post_startup):
            if self.throw:raise ValueError('synthetic terminal failure')
            self.calls.append(s.commands.clone());r.c=r.controller.step(reference['q_ref'],actions,reference_valid=reference['valid'])
            f=r.fs[0];f.qtarget=array(r.c['target_position_rad'])[0];f.v=array(r.c['target_velocity_rad_s'])[0];f.time+=.02;r.steps+=1
            env.reference_residual_target=r.c;s.last=r.packet();s.control+=1
            return torch.zeros(1),torch.zeros(1,dtype=torch.bool),{}
        self.session._physical_step=types.MethodType(physical,self.session)
    def enter(self):self.session.__enter__();return self.session

def requested_history(obs):return obs['policy'][:,4*63+6:4*63+9]/torch.tensor([5,5,2.5])

class SessionTests(unittest.TestCase):
    def test_initial_command_is_copied_after_startup_before_first_packet(self):
        commands=torch.tensor([[.005,0,0]],dtype=torch.float64);h=Harness(commands);commands[0,0]=.1
        s=h.enter();self.assertEqual(len(h.calls),200);self.assertTrue(all(not x.any() for x in h.calls))
        torch.testing.assert_close(s.commands,torch.tensor([[.005,0,0]],dtype=torch.float64))
        torch.testing.assert_close(requested_history(s.get_observations()),torch.tensor([[.005,0,0]]))
        s.step(torch.zeros(1,18));self.assertEqual(float(h.calls[-1][0,0]),.005)
    def test_queue_then_stop_preserves_current_action_command_and_history(self):
        h=Harness();s=h.enter();old=s.get_observations();s.queue_commands([[.005,0,0]])
        torch.testing.assert_close(s.get_observations()['policy'],old['policy']);self.assertEqual(s.policy_steps,0)
        nextobs,*_=s.step(torch.zeros(1,18));self.assertFalse(h.calls[-1].any());torch.testing.assert_close(requested_history(nextobs),torch.tensor([[.005,0,0]]))
        s.queue_commands([[0,0,0]]);stopobs,*_=s.step(torch.zeros(1,18));self.assertEqual(float(h.calls[-1][0,0]),.005);self.assertFalse(requested_history(stopobs).any())
        s.step(torch.zeros(1,18));self.assertFalse(h.calls[-1].any());self.assertEqual(s.policy_steps,3)
    def test_invalid_command_rejected_without_rewriting_cached_packet(self):
        h=Harness();s=h.enter();before=s.get_observations()['policy'].clone()
        for c in ([[.006,0,0]],[[-.001,0,0]],[[0,.001,0]],[[0,0,.001]],[[float('nan'),0,0]]):
            with self.subTest(c=c),self.assertRaises(ValueError):s.queue_commands(c)
        self.assertIsNone(s.pending_commands);torch.testing.assert_close(before,s.get_observations()['policy'])
        external=s.get_observations();external['policy'].zero_();torch.testing.assert_close(before,s.get_observations()['policy'])
    def test_failure_retry_does_not_advance_planner_or_emit_a_packet(self):
        h=Harness();s=h.enter();h.throw=True
        with self.assertRaises(ValueError):s.step(torch.zeros(1,18))
        frozen={k:v.clone() for k,v in s.wave.s.items()};count=len(h.calls)
        with self.assertRaises(RuntimeError):s.step(torch.zeros(1,18))
        self.assertEqual(len(h.calls),count)
        for k,v in frozen.items():torch.testing.assert_close(s.wave.s[k],v,equal_nan=True)
        with self.assertRaises(RuntimeError):s.get_observations()
        with self.assertRaises(RuntimeError):s.queue_commands([[0,0,0]])
    def test_optimization_cannot_add_a49th_control_or_change_command(self):
        h=Harness();s=h.enter();s.optimization_steps=0
        for _ in range(48):s.step(torch.zeros(1,18))
        self.assertEqual(s.policy_steps,48);count=len(h.calls)
        with self.assertRaises(RuntimeError):s.step(torch.zeros(1,18))
        self.assertEqual(len(h.calls),count)
        h=Harness();s=h.enter();s.optimization_steps=0;s.queue_commands([[.005,0,0]])
        with self.assertRaises(RuntimeError):s.step(torch.zeros(1,18))
        self.assertEqual(len(h.calls),200)
if __name__=='__main__':torch.set_num_threads(1);unittest.main()
