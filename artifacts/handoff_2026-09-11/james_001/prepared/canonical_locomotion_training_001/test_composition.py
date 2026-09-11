import copy,unittest
import numpy as np
from oracles.adapter import ObservationScales,ObservationHistory,ActorFrameInputs
from oracles.commands import CommandBank
from oracles.objective import reward
from command_history import CommandHistory
from contact_contract import contact_facts,ProposedMovingRule
from reward_packet import pack_reward,material_slip,pack_material_slip,neutral_interval_context,LEGS


def inputs(n,q=0.,held=0.):
 return ActorFrameInputs(np.zeros((n,3)),np.tile([0,0,-1],(n,1)),np.zeros((n,3)),np.full((n,18),q),np.zeros((n,18)),np.zeros((n,18)),np.full((n,18),held))

def history(n):
 scales=ObservationScales(*[np.ones(k)for k in [3,3,3,18,18,18,18,3]])
 return ObservationHistory(np.zeros(18),scales,n)

def rows(n=2):
 pose=np.tile([0,0,.1,0,0,0,1.],(n,1));held=np.zeros((n,18),np.float32)
 before={'explicit_counter':8000,'time_s':20.,'joint_position_rad':np.zeros((n,18))}
 samples=[]
 for i in range(8):
  samples.append({'explicit_counter':8001+i,'time_s':20.+(i+1)*.0025,'substep_index':i,'contact_valid':np.ones(n,bool),'interval_valid':np.ones(n,bool),
   'root_pose_xyzw':pose.copy(),'root_com_velocity':np.zeros((n,6)),'joint_target_rad':held.copy(),'computed_torque_nm':np.zeros((n,18)),
   'applied_torque_nm':np.zeros((n,18)),'distal_contact':np.ones((n,6),bool),'joint_position_rad':np.zeros((n,18)),
   'terminated':np.zeros(n,bool),'truncated':np.zeros(n,bool)})
 kw={'command':np.zeros((n,3)),'previous_held':held,'emitted':held.copy(),'requested':held.copy(),'previous_delta':held.copy(),
     'root_com_local':np.zeros((n,3)),'slip':np.zeros((n,8,6)),'slip_valid':np.ones((n,8),bool)}
 return before,samples,kw

class CommandTests(unittest.TestCase):
 def test_new_command_and_previous_actual_target_visible(self):
  x=CommandHistory(CommandBank(32,0),history(32),inputs(32,.02,.03))
  token=x.begin(np.full((32,18),.03),np.full((32,18),.02))
  np.testing.assert_array_equal(token['reward_requested_command'],x.bank.observe())
  self.assertTrue(np.any(token['reward_requested_command'][:,2]!=0))
  self.assertTrue(np.any(token['reward_requested_command'][:,:2]<0))
  token['reward_requested_command'][:]=777
  x.complete(inputs(32,.04,.05),complete_substeps=8,reward_command=x.bank.observe())
  np.testing.assert_allclose(x.history.history[:,-1,63:81],.01)
 def test_partial_hold_preserves_pending_and_history(self):
  x=CommandHistory(CommandBank(2,0),history(2),inputs(2));old=x.history.history
  t=x.begin(np.zeros((2,18)),np.zeros((2,18)))
  with self.assertRaises(ValueError):x.complete(inputs(2),complete_substeps=7,reward_command=t['reward_requested_command'])
  np.testing.assert_array_equal(old,x.history.history)
  self.assertIsNotNone(x.pending)
  with self.assertRaises(ValueError):x.actor_observation()
 def test_wrong_reward_command_rejected(self):
  x=CommandHistory(CommandBank(2,0),history(2),inputs(2));x.begin(np.zeros((2,18)),np.zeros((2,18)))
  with self.assertRaises(ValueError):x.complete(inputs(2),complete_substeps=8,reward_command=np.ones((2,3)))
 def test_rollover_preserves_history_and_other_row_clock(self):
  b=CommandBank(2,0);b.clocks[:]=[599,249]
  x=CommandHistory(b,history(2),inputs(2,.02,.03));old=x.history.history
  t=x.begin(np.full((2,18),.03),np.full((2,18),.02));r=x.complete(inputs(2,.06,.07),complete_substeps=8,reward_command=t['reward_requested_command'])
  np.testing.assert_array_equal(r['command_program_restarted_rows'],[0]);np.testing.assert_array_equal(x.bank.clocks,[0,250])
  np.testing.assert_array_equal(x.history.history[:,:-1],old[:,1:]);self.assertEqual(r['physical_resets'],0)
  self.assertTrue(np.all(x.history.history[1,-1,6:9]==0))
  self.assertTrue(np.any(old[1,-1,6:9]!=0))
 def test_pending_checkpoint_continuation_is_exact(self):
  x=CommandHistory(CommandBank(2,0),history(2),inputs(2));t=x.begin(np.zeros((2,18)),np.zeros((2,18)))
  y=CommandHistory.from_state(x.state())
  for z in [x,y]:z.complete(inputs(2,.01,.02),complete_substeps=8,reward_command=t['reward_requested_command'])
  np.testing.assert_array_equal(x.actor_observation(),y.actor_observation())
 def test_future_target_does_not_pass_begin(self):
  x=CommandHistory(CommandBank(2,0),history(2),inputs(2))
  with self.assertRaises(ValueError):x.begin(np.ones((2,18))*.03,np.zeros((2,18)))

class ContactTests(unittest.TestCase):
 def fixture(self):return np.ones((1,6),bool),np.ones((1,8,6),bool),np.ones((1,8),bool)
 def test_lift_separate_from_standing(self):
  a,b,v=self.fixture();b[:,2:6,0]=False
  self.assertFalse(contact_facts(a,b,v,phase='neutral_standing')['proposed_count_check'][0])
  # Count5 is a synthetic test contract ONLY, not a native recommendation/admission.
  r=contact_facts(a,b,v,phase='moving_or_stopping',moving_rule=ProposedMovingRule('fixture_only',5))
  self.assertTrue(r['proposed_count_check'][0]);self.assertEqual(r['force_contact_departures'].sum(),1);self.assertEqual(r['force_contact_returns'].sum(),1)
  self.assertIsNone(r['qualified_flight']);self.assertFalse(r['physical_admission'])
 def test_empty_invalid_and_unresolved_not_admitted(self):
  a,b,v=self.fixture()
  with self.assertRaises(ValueError):contact_facts(a,b,v,phase='moving_or_stopping')
  v[0,1]=False
  with self.assertRaises(ValueError):contact_facts(a,b,v,phase='neutral_standing')
  b[:]=False;v[:]=True
  r=contact_facts(a,b,v,phase='moving_or_stopping',moving_rule=ProposedMovingRule('fixture_only',4))
  self.assertFalse(r['proposed_count_check'][0])
  with self.assertRaises(ValueError):ProposedMovingRule('no_empty',0)

class PacketTests(unittest.TestCase):
 def test_exact_objective_zero_packet(self):
  b,s,k=rows();r=pack_reward(b,s,**k)
  np.testing.assert_array_equal(r['reward'],np.ones(2)*.14);self.assertFalse(r['native_admission'])
 def test_root_com_correction_native_axes_translation_yaw(self):
  b,s,k=rows();k['command'][:]=[.03,-.01,.12];k['root_com_local'][:]=[.1,0,0]
  # native body velocity(-.01,-.03,0); vCOM adds omega x COM=(0,.012,0)
  for row in s:row['root_com_velocity'][:]=[-.01,-.018,0,0,0,.12]
  r=pack_reward(b,s,**k)
  np.testing.assert_allclose(r['component_rates']['planar_tracking'],5,rtol=0,atol=1e-12)
  np.testing.assert_allclose(r['component_rates']['yaw_tracking'],2,rtol=0,atol=1e-12)
 def test_all_eight_samples_catch_cancellation_and_torque_spike(self):
  b,s,k=rows();base=pack_reward(b,s,**k)
  for i,row in enumerate(s):row['root_com_velocity'][:,0]=.1*(-1 if i%2 else 1)
  s[3]['computed_torque_nm'][0,0]=9.
  r=pack_reward(b,s,**k)
  self.assertTrue(np.all(r['component_rates']['planar_tracking']<base['component_rates']['planar_tracking']))
  self.assertLess(r['component_rates']['requested_excess'][0],0)
 def test_counter_target_partial_slip_and_earlier_failure(self):
  for mutate in [lambda s,k:s[2].update(explicit_counter=1),lambda s,k:s[5]['joint_target_rad'].__setitem__((0,0),.01),
                 lambda s,k:k['slip_valid'].__setitem__((0,4),False),lambda s,k:s[1]['terminated'].__setitem__(0,True)]:
   b,s,k=rows();mutate(s,k)
   with self.assertRaises(ValueError):pack_reward(b,s,**k)
  b,s,k=rows()
  with self.assertRaises(ValueError):pack_reward(b,s[:7],**k)
 def test_actual_angle_independent_of_sdk(self):
  b,s,k=rows();base=pack_reward(b,s,**k)
  s[-1]['joint_position_rad'][:]=.004
  for row in s:row['joint_velocity_rad_s']=np.zeros((2,18))
  r=pack_reward(b,s,**k);self.assertLess(r['component_rates']['zero_command_angle_rate'][0],0)
  self.assertLess(r['reward'][0],base['reward'][0])

class SlipTests(unittest.TestCase):
 def fixture(self):
  p=np.tile([0.,0,0,0,0,0,1],(1,6,1));a=np.zeros((1,6),bool);a[0,0]=True
  patch={'env':0,'body':'lf_tibia','category':'toe','normal_force_n':2.,'point_world_m':[.1,0,0],
         'normal_world':[0,0,1.0005],'separation_m':0.,'inactive_zero_normal':False}
  return p,a,patch
 def test_patch_motion_on_stationary_body_is_not_material_slip(self):
  p,a,x=self.fixture()
  for position in [[.1,0,0],[.8,.4,0]]:
   x['point_world_m']=position
   r=material_slip(p,p,[x],a,body_names=[l+'_tibia'for l in LEGS],classified_valid=True)
   np.testing.assert_array_equal(r,np.zeros((1,6)))
 def test_body_motion_and_inactive_zeros(self):
  p,a,x=self.fixture();q=p.copy();q[0,0,0]=.00025
  zero=copy.deepcopy(x);zero.update(normal_force_n=0.,normal_world=[0,0,0],inactive_zero_normal=True)
  r=material_slip(p,q,[zero,x],a,body_names=[l+'_tibia'for l in LEGS],classified_valid=True)
  self.assertAlmostEqual(r[0,0],.1,places=12)
 def test_missing_or_unaccepted_evidence_fails(self):
  p,a,x=self.fixture();kwargs={'body_names':[l+'_tibia'for l in LEGS],'classified_valid':True}
  with self.assertRaises(ValueError):material_slip(p,p,[],a,**kwargs)
  kwargs['classified_valid']=False
  with self.assertRaises(ValueError):material_slip(p,p,[x],a,**kwargs)
 def test_accepted_quaternion_normalized_bad_quaternion_rejected(self):
  p,a,x=self.fixture();p[:,:,6]=np.sqrt(1.00001)
  material_slip(p,p,[x],a,body_names=[l+'_tibia'for l in LEGS],classified_valid=True)
  p[:,:,6]=1.01
  with self.assertRaises(ValueError):material_slip(p,p,[x],a,body_names=[l+'_tibia'for l in LEGS],classified_valid=True)

class AdditionalSeamTests(unittest.TestCase):
 def test_world_rotation_and_translation_covariance(self):
  from oracles.frames import rotations_xyzw
  b,s,k=rows();k['command'][:]=[.03,-.01,.12];k['root_com_local'][:]=[.1,0,0]
  for row in s:row['root_com_velocity'][:]=[-.01,-.018,0,0,0,.12]
  baseline=pack_reward(b,s,**k)
  angle=1.1;q=np.array([0,0,np.sin(angle/2),np.cos(angle/2)]);r=rotations_xyzw(q[None])[0]
  for row in s:
   row['root_pose_xyzw'][:,:3]=[10,30,.1];row['root_pose_xyzw'][:,3:]=q
   row['root_com_velocity'][:,:3]=row['root_com_velocity'][:,:3]@r.T
   row['root_com_velocity'][:,3:]=row['root_com_velocity'][:,3:]@r.T
  actual=pack_reward(b,s,**k)
  np.testing.assert_allclose(actual['reward'],baseline['reward'],atol=1e-14,rtol=0)
 def test_complete_patch_accessor_pairing_and_private_clock(self):
  b,s,k=rows(1);pose=np.tile([0.,0,0,0,0,0,1],(1,6,1));b['link_pose_xyzw']=pose.copy();packets=[]
  for i,row in enumerate(s):
   row['sequence']=8000+i;row['link_pose_xyzw']=pose.copy();row['link_pose_xyzw'][0,0,0]=.00025*(i+1)
   row['distal_contact'][:]=False;row['distal_contact'][0,0]=True
   patch={'env':0,'body':'lf_tibia','category':'toe','normal_force_n':2.,'point_world_m':[.1,0,0],
          'normal_world':[0,0,1.],'separation_m':0.,'inactive_zero_normal':False}
   packets.append({'sequence':8000+i,'explicit_counter':8001+i,'classifier_accepted':True,'patches':[patch]})
  names=[l+'_tibia'for l in LEGS]
  slip=pack_material_slip(b,s,packets,body_names=names);np.testing.assert_allclose(slip[0,:,0],.1,atol=1e-14,rtol=0)
  packets[2]['explicit_counter']-=1
  with self.assertRaises(ValueError):pack_material_slip(b,s,packets,body_names=names)

class ReviewCounterexamples(unittest.TestCase):
 def test_first_interval_context_from_actual_constant_hold(self):
  before,s,k=rows(1);last=s[-1];prior=copy.deepcopy(last);prior['explicit_counter']-=8;prior['time_s']-=.02
  last['link_pose_xyzw']=np.tile([0.,0,0,0,0,0,1],(1,6,1))
  result=neutral_interval_context(prior,last);np.testing.assert_array_equal(result['previous_delta'],np.zeros((1,18)))
  last['joint_target_rad'][0,0]=.001
  with self.assertRaises(ValueError):neutral_interval_context(prior,last)
 def test_fixed_slip_denominator_is_preserved_not_silently_repaired(self):
  b,s,k=rows(1);k['slip'][:]=.05
  full=pack_reward(b,s,**k)['component_rates']['contact_slip'][0]
  for row in s:row['distal_contact'][0,0]=False
  five=pack_reward(b,s,**k)['component_rates']['contact_slip'][0]
  self.assertAlmostEqual(five/full,5/6)
  # This demonstrates the known incentive; it is not a physical admission.
 def test_synthetic_bank_restart_is_not_physical_resume(self):
  bank=CommandBank(2,0);bank.clocks[0]=600
  with self.assertRaises(ValueError):bank.observe()
  bank.reset_rows(np.array([0]));self.assertEqual(bank.epochs[0],1)
  self.assertEqual(bank.clocks[1],0)

if __name__=='__main__':unittest.main()
