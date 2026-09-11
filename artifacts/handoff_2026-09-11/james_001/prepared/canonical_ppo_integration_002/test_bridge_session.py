"""Synthetic session contract only; actual native safety/admission remains pending."""
import json
from pathlib import Path
import sys,tempfile,unittest
import numpy as np
import torch
from test_support import ROOT,TEST_DEPS
sys.path.insert(0,str(TEST_DEPS))
from canonical_direct_ppo.native_bridge import CanonicalNativeBridge
from canonical_direct_ppo.runner import learn_smoke

class FakeSession:
 n=32;reset_count=1
 def __init__(self,model,bad_contact_substep=None):
  self.names=[j['name']for j in model['joints']][::-1]
  self.body_names=['body']+[leg+'_'+part for part in ['coxa','femur','tibia']for leg in ['lf','lm','lr','rf','rm','rr']]
  by={j['name']:j for j in model['joints']};limits=np.array([[by[n]['lower'],by[n]['upper']]for n in self.names])
  self.native={'limits':np.broadcast_to(limits,(32,18,2)).copy(),'coms':np.zeros((32,19,7))}
  self.count=8000;self.control=1000;self.failure=None;self.bad_contact_substep=bad_contact_substep;self.last=[]
  self.current={'joint_position_rad':np.zeros((32,18),np.float32),'joint_velocity_rad_s':np.zeros((32,18),np.float32),'joint_target_rad':np.zeros((32,18),np.float32),
    'computed_torque_nm':np.zeros((32,18)),'applied_torque_nm':np.zeros((32,18)),
    'root_pose_xyzw':np.tile([0,0,.2,0,0,0,1.],(32,1)),'root_com_velocity':np.zeros((32,6)),
    'distal_contact':np.ones((32,6),bool),'nonfoot_contact':np.zeros(32,bool),'contact_valid':np.ones(32,bool),'interval_valid':np.ones(32,bool),
    'terminated':np.zeros(32,bool),'truncated':np.zeros(32,bool),'minimum_non_toe_floor_m':np.full(32,.01),'explicit_counter':np.array(8000)}
 def observe(self):return {k:v.copy()for k,v in self.current.items()}
 def step_control(self,target):
  self.last=[]
  for sub in range(8):
   row=self.observe();q=row['joint_position_rad'];new=(q+.05*(target-q)).astype(np.float32)
   row['joint_velocity_rad_s']=(new-q)/.0025;row['joint_position_rad']=new;row['joint_target_rad']=target.copy()
   row['computed_torque_nm']=.3*(target-new);row['applied_torque_nm']=row['computed_torque_nm'].copy()
   row['distal_contact']=np.ones((32,6),bool)
   if sub==self.bad_contact_substep:row['distal_contact'][3,2]=False
   self.count+=1;row['explicit_counter']=np.array(self.count);self.current=row;self.last.append(self.observe())
  self.control+=1;return self.observe()
 def observe_control_substeps(self):return [{k:v.copy()for k,v in r.items()}for r in self.last]
 def flush(self):pass

class SessionBridgeChecks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1)
  cls.model=json.loads((ROOT/'robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json').read_text())
 def test_actual_adapter_bridge_to_real_rsl_two_updates_cpu_only(self):
  session=FakeSession(self.model);bridge=CanonicalNativeBridge(session,self.model,device='cpu')
  with tempfile.TemporaryDirectory()as d:
   r=learn_smoke(bridge,{'CPU_SYNTHETIC_SESSION':True},Path(d)/'run',device='cpu')
   self.assertEqual(r['status'],'completed');self.assertEqual(r['native_audit']['policy_substeps_verified'],384)
   self.assertEqual(session.count,8384);self.assertEqual(bridge.count,48)
   newest=bridge.history.actor_observation()[:,-81:]
   np.testing.assert_allclose(newest[:,63:],bridge.held-session.current['joint_position_rad'],atol=1e-10)
   self.assertEqual(bridge.names,session.names)
 def test_intermediate_contact_loss_rejected_even_when_endpoint_recovers(self):
  session=FakeSession(self.model,bad_contact_substep=2);bridge=CanonicalNativeBridge(session,self.model,device='cpu')
  with self.assertRaisesRegex(ValueError,'six-toe'):bridge.step(torch.zeros(32,18))
  self.assertTrue(session.current['distal_contact'].all());self.assertEqual(bridge.count,0)
  self.assertEqual(bridge.rows[0]['status'],'failed');self.assertEqual(len(session.last),8)
 def test_invalid_initial_contact_is_not_synthesized(self):
  session=FakeSession(self.model);session.current['contact_valid'][:]=False
  with self.assertRaisesRegex(ValueError,'not valid'):CanonicalNativeBridge(session,self.model,device='cpu')
 def test_initial_actual_missing_foot_is_rejected(self):
  session=FakeSession(self.model);session.current['distal_contact'][4,2]=False
  with self.assertRaisesRegex(ValueError,'six-toe'):CanonicalNativeBridge(session,self.model,device='cpu')
 def test_measured_limit_violation_prevents_actor_step(self):
  session=FakeSession(self.model);session.current['joint_position_rad'][0,0]=session.native['limits'][0,0,1]+.001
  with self.assertRaisesRegex(ValueError,'exceeded'):CanonicalNativeBridge(session,self.model,device='cpu')
 def test_native_error_during_control_preserves_endpoint_and_blocks_next_action(self):
  session=FakeSession(self.model);errors=[];bridge=CanonicalNativeBridge(session,self.model,device='cpu',native_errors=errors)
  original=session.step_control
  def step(target):
   result=original(target);errors.append({'native':'error'});return result
  session.step_control=step
  with self.assertRaisesRegex(ValueError,'during policy'):bridge.step(torch.zeros(32,18))
  self.assertIn('native_endpoint',bridge.rows[0]);self.assertEqual(session.count,8008);self.assertEqual(bridge.count,0)
  with self.assertRaisesRegex(RuntimeError,'latched'):bridge.step(torch.zeros(32,18))
  self.assertEqual(session.count,8008)

if __name__=='__main__':unittest.main()
