import unittest
import numpy as np
from objective import reward,material_point_tangent_velocity,terminal_margin,Weights
from commands import program,command_at,CommandBank,held_out_cases
from causal_command import begin_control


def fixture(n=1):
 return dict(command=np.zeros((n,3)),linear_body=np.zeros((n,8,3)),angular_body=np.zeros((n,8,3)),
  gravity_body=np.tile([0,0,-1.],(n,8,1)),computed_torque=np.zeros((n,8,18)),applied_torque=np.zeros((n,8,18)),
  target_delta=np.zeros((n,18)),target_previous_delta=np.zeros((n,18)),requested_minus_executed=np.zeros((n,18)),interval_angle_rate=np.zeros((n,18)),
  contact_tangent_speed=np.zeros((n,8,6)),contact_active=np.ones((n,8,6),bool),valid_interval=np.ones(n,bool),terminated=np.zeros(n,bool),truncated=np.zeros(n,bool))


class RewardChecks(unittest.TestCase):
 def test_all_signs_tracking_beats_stopped_wrong_and_overspeed(self):
  for c in [[.03,0,0],[-.03,0,0],[0,.03,0],[0,-.03,0],[.02,-.02,0],[0,0,.12],[0,0,-.12],[.02,.02,-.12]]:
   values=[]
   for multiplier in [1,0,-1,3]:
    x=fixture();x['command'][0]=c;x['linear_body'][0,:,:2]=np.asarray(c[:2])*multiplier;x['angular_body'][0,:,2]=c[2]*multiplier
    values.append(float(reward(**x)['reward'][0]))
   self.assertGreater(values[0],max(values[1:]),c)
 def test_world_yaw_covariance_after_frame_conversion(self):
  c=np.array([.023,-.015,.11]);v=np.array([.019,-.009,.0]);scores=[]
  for angle in [0,.7,2.4]:
   rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
   x=fixture();x['command'][0]=c;x['linear_body'][0,:,:2]=rot.T@(rot@v[:2]);x['angular_body'][0,:,2]=.10
   scores.append(reward(**x)['reward'][0])
  np.testing.assert_allclose(scores,scores[0],atol=1e-15)
 def test_raw_demand_penalty_cannot_be_hidden_by_applied_clamp(self):
  x=fixture();x['applied_torque'][:]=1.6;x['computed_torque'][:]=1.6;normal=reward(**x)
  x['computed_torque'][:]=3.2;over=reward(**x)
  self.assertLess(over['reward'][0],normal['reward'][0]);self.assertEqual(over['component_rates']['applied_effort'][0],normal['component_rates']['applied_effort'][0])
 def test_intermediate_torque_pulse_cannot_hide_in_endpoint(self):
  x=fixture();baseline=reward(**x)['reward'][0];x['computed_torque'][0,2,7]=3.2
  self.assertLess(reward(**x)['reward'][0],baseline);self.assertTrue((x['computed_torque'][:,-1]==0).all())
 def test_mean_square_tracking_does_not_hide_substep_cancellation(self):
  x=fixture();flat=reward(**x)['reward'][0];x['angular_body'][0,:,2]=[1,-1]*4
  self.assertEqual(x['angular_body'][0,:,2].mean(),0)
  self.assertLess(reward(**x)['reward'][0],flat)
 def test_upside_down_is_not_flat(self):
  x=fixture();upright=reward(**x)['reward'][0];x['gravity_body'][0]=[0,0,1]
  self.assertLess(reward(**x)['reward'][0],upright)
 def test_actual_target_oscillation_and_actual_quiet_motion_cost(self):
  x=fixture();stationary=reward(**x)['reward'][0]
  x['target_delta'][:]=.04;x['target_previous_delta'][:]=-.04;x['interval_angle_rate'][:]=1.
  jitter=reward(**x);self.assertLess(jitter['reward'][0],stationary)
  self.assertLess(jitter['component_rates']['target_second_difference'][0],0)
 def test_request_cannot_use_limiter_as_cost_free_filter(self):
  x=fixture();x['target_delta'][:]=.04;within=reward(**x)['reward'][0];x['requested_minus_executed'][:]=.08
  self.assertLess(reward(**x)['reward'][0],within)
 def test_stationary_tracking_credit_is_not_a_mathematical_optimum(self):
  x=fixture();x['command'][0,0]=.02;base=reward(**x)['reward'][0];x['linear_body'][0,:,0]=1e-6
  self.assertGreater(reward(**x)['reward'][0],base)
 def test_bounded_costs_and_terminal_discount_margin(self):
  margin=terminal_margin();self.assertTrue(margin['sufficient']);self.assertLess(margin['strict_terminal_penalty_lower_bound'],20)
  x=fixture();x['computed_torque'][:]=1e100;x['applied_torque'][:]=1e100;x['requested_minus_executed'][:]=1e100
  r=reward(**x);cost=sum(v[0]for k,v in r['component_rates'].items()if not k.endswith('tracking'))
  self.assertGreaterEqual(cost,-margin['maximum_cost_rate'])
  with self.assertRaisesRegex(ValueError,'Terminal penalty'):reward(**fixture(),weights=Weights(terminal=0.))
 def test_timeout_bootstraps_but_actual_terminal_does_not(self):
  x=fixture(2);x['truncated'][0]=True;x['terminated'][1]=True
  r=reward(**x);np.testing.assert_array_equal(r['bootstrap_allowed'],[True,False]);self.assertEqual(r['termination_penalty'][0],0);self.assertEqual(r['termination_penalty'][1],-20)
 def test_no_invalid_or_nonfinite_interval_silently_becomes_quiet(self):
  x=fixture();x['valid_interval'][0]=False
  with self.assertRaisesRegex(ValueError,'Invalid interval'):reward(**x)
  x['valid_interval'][0]=True;x['linear_body'][0,0]=np.nan
  with self.assertRaisesRegex(ValueError,'nonfinite'):reward(**x)
 def test_inactive_feet_have_no_slip_cost_but_do_not_earn_support(self):
  x=fixture();x['contact_tangent_speed'][:]=2.;x['contact_active'][:]=False
  r=reward(**x);self.assertEqual(r['component_rates']['contact_slip'][0],0);self.assertFalse(r['physics_admission'])


class SlipChecks(unittest.TestCase):
 def test_static_link_changing_patch_location_has_no_fake_slip(self):
  pose=np.eye(4)[None];normal=np.array([[0.,0.,1.]])
  for point in [[0.,0.,0.],[.08,-.04,.001]]:
   np.testing.assert_array_equal(material_point_tangent_velocity(np.array([point]),normal,pose,pose,.0025),np.zeros((1,3)))
 def test_translating_material_point_and_normal_velocity_separation(self):
  old=np.eye(4)[None];new=old.copy();new[0,:3,3]=[.00025,-.0005,.001]
  point=np.array([[.1,.2,0.]])
  np.testing.assert_allclose(material_point_tangent_velocity(point,np.array([[0.,0.,1.]]),old,new,.0025),[[.1,-.2,0]],atol=1e-14)
 def test_rotating_material_point_uses_same_local_point(self):
  old=np.eye(4)[None];new=old.copy();theta=.01
  new[0,:2,:2]=[[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]
  local=np.array([.1,0,0]);point=(new[0,:3,:3]@local)[None]
  expected=(point-local)/.0025
  np.testing.assert_allclose(material_point_tangent_velocity(point,np.array([[0.,0.,1.]]),old,new,.0025),expected,atol=1e-14)
 def test_accepted_raw_normal_is_normalized_at_reward_boundary(self):
  # Existing native1e-3 acceptance is not tightened by the unit-vector helper.
  raw=np.array([[0.,0.,1.0005]]);self.assertLess(abs(np.linalg.norm(raw)-1),1e-3)
  unit=raw/np.linalg.norm(raw,axis=-1,keepdims=True);pose=np.eye(4)[None]
  np.testing.assert_array_equal(material_point_tangent_velocity(np.zeros((1,3)),unit,pose,pose,.0025),np.zeros((1,3)))
  with self.assertRaisesRegex(ValueError,'not active/unit'):material_point_tangent_velocity(np.zeros((1,3)),raw,pose,pose,.0025)

 def test_bad_clock_rejected(self):
  pose=np.eye(4)[None]
  for dt in [0,-.01,float('nan'),float('inf')]:
   with self.assertRaises(ValueError):material_point_tangent_velocity(np.zeros((1,3)),np.array([[0,0,1.]]),pose,pose,dt)


class CommandChecks(unittest.TestCase):
 def test_new_command_precedes_action_and_reward_keeps_that_command(self):
  c=np.array([[.03,0,.12]]);held=np.full((1,18),.025);q=np.full((1,18),.01);frame=np.zeros((1,81));frame[:,6:9]=c;frame[:,63:]=held-q
  transition=begin_control(c,frame,held,q,7);c[:]=0
  np.testing.assert_array_equal(transition['reward_requested_command'],[[.03,0,.12]])
  with self.assertRaisesRegex(ValueError,'visible before'):begin_control(c,frame,held,q,8)
  frame[:,6:9]=c;frame[:,63:]=.09-q
  with self.assertRaisesRegex(ValueError,'future'):begin_control(c,frame,held,q,8)
 def test_exact_zero_time_share_and_all_command_families(self):
  plans=[program(i,0,0)for i in range(25)]
  zero=sum(s.controls for p in plans for s in p['segments']if not any(s.command))
  self.assertEqual(zero/(25*600),.2);self.assertEqual({p['kind']for p in plans},{'quiet','axis','diagonal','pure_yaw','arc'})
  commands=np.array([s.command for p in plans for s in p['segments']]);self.assertTrue(np.all(commands.min(0)<0));self.assertTrue(np.all(commands.max(0)>0))
 def test_yaw_sign_is_not_locked_to_arc_heading(self):
  seen=set()
  for epoch in range(100):
   for row in range(25):
    p=program(row,epoch,0)
    if p['kind']=='arc':
     x,y,w=p['segments'][0].command;seen.add((round(np.arctan2(y,x)/(np.pi/4))%8,int(np.sign(w))))
  self.assertEqual(seen,{(i,s)for i in range(8)for s in [-1,1]})
 def test_selective_reset_does_not_change_neighbours_or_share_rng(self):
  bank=CommandBank(32,0);bank.advance();before=bank.observe();state=bank.state();bank.reset_rows([2,9]);after=bank.observe()
  untouched=[i for i in range(32)if i not in [2,9]]
  np.testing.assert_array_equal(after[untouched],before[untouched]);np.testing.assert_array_equal(bank.clocks[untouched],state['clocks'][untouched]);np.testing.assert_array_equal(bank.epochs[untouched],state['epochs'][untouched])
  self.assertTrue((bank.clocks[[2,9]]==0).all());self.assertTrue((bank.epochs[[2,9]]==1).all())
 def test_explicit_time_boundary_no_silent_new_command(self):
  p=program(1,0,0);self.assertTrue(command_at(p,249).any());self.assertFalse(command_at(p,250).any());self.assertTrue(command_at(p,300).any())
  with self.assertRaises(ValueError):command_at(p,600)
  with self.assertRaises(ValueError):command_at(p,-1)
 def test_mixed_program_boundary_and_checkpoint_continuation(self):
  bank=CommandBank(3,0);bank.clocks[:]=[599,24,17];mask=bank.advance();np.testing.assert_array_equal(mask,[True,False,False])
  with self.assertRaises(ValueError):bank.observe()
  bank.reset_rows([0]);self.assertEqual(bank.clocks.tolist(),[0,25,18]);copy=CommandBank.from_state(bank.state())
  for _ in range(20):
   np.testing.assert_array_equal(bank.observe(),copy.observe());bank.advance();copy.advance()
 def test_direct_reversal_and_heldout_pure_turn_have_correct_goals(self):
  p=program(1,1,0);np.testing.assert_array_equal(command_at(p,250),-command_at(p,249))
  cases=held_out_cases(0);self.assertEqual(len(cases),32)
  for case in cases:
   self.assertEqual(case['stop_controls'],600);self.assertEqual(case['quiet_last_controls'],500)
   self.assertFalse(case['training_distribution_admission'])
   if case['name'].startswith('yaw'):self.assertEqual(case['segments'][1].command[:2],(0.,0.))

if __name__=='__main__':unittest.main()
