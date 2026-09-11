import unittest
import numpy as np
from canonical_direct_ppo.adapter import JointConfig,TargetPipeline
from canonical_direct_ppo.native_bridge import govern,checked_q,quiet_reward,LIMIT_NUMERIC_TOL_RAD

class BridgeMathChecks(unittest.TestCase):
 def test_emitted_float32_targets_never_exceed_actual_point040_budget(self):
  rng=np.random.default_rng(761);held=np.zeros((32,18),np.float32);lo=np.full(18,-.08);hi=np.full(18,1.5);neutral=np.zeros(18)
  for _ in range(80):
   out=govern(rng.normal(size=(32,18))*3,held,lo,hi,neutral);target=out['emitted_target_rad']
   self.assertEqual(target.dtype,np.float32);self.assertLessEqual(np.max(abs(target.astype(float)-held)),.04)
   self.assertTrue(np.all(target>=lo));self.assertTrue(np.all(target<=hi));held=target
 def test_action_math_parity_to_partner_pipeline_with_only_float32_rounding(self):
  config=JointConfig([f'test_{i}'for i in range(18)],np.full(18,-.08),np.full(18,1.5),np.zeros(18),np.full(18,.1))
  ref=TargetPipeline(config,32);actual=np.zeros((32,18),np.float32);rng=np.random.default_rng(447)
  for _ in range(30):
   # Reference restarts at the actual hold; its cleared prior-action field is irrelevant to target arithmetic.
   ref.reset_rows(np.arange(32),actual)
   action=rng.normal(size=(32,18));expected=ref.step(action);out=govern(action,actual,config.lower,config.upper,config.neutral)
   np.testing.assert_array_equal(out['clipped_action'],expected.clipped_action)
   np.testing.assert_allclose(out['requested_target_rad'],expected.requested_target,atol=0,rtol=0)
   np.testing.assert_allclose(out['emitted_target_rad'],expected.executed_target,atol=1e-8,rtol=0)
   actual=out['emitted_target_rad']
 def test_measured_limit_violation_is_rejected_before_helper_clamp(self):
  q=np.zeros((32,18));lo=np.full(18,-.08);hi=np.ones(18)
  q[2,3]=lo[3]-2*LIMIT_NUMERIC_TOL_RAD
  with self.assertRaises(ValueError):checked_q(q,lo,hi)
  q[2,3]=lo[3]-LIMIT_NUMERIC_TOL_RAD/2
  self.assertEqual(checked_q(q,lo,hi)[2,3],q[2,3])
 def test_reward_uses_actual_control_angle_difference_not_sdk_bias(self):
  before={'joint_position_rad':np.zeros((32,18))};after={'joint_position_rad':np.zeros((32,18)),'joint_velocity_rad_s':np.full((32,18),900.),'root_pose_xyzw':np.tile([0,0,.2,0,0,0,1.],(32,1)),'root_com_velocity':np.zeros((32,6))}
  reward,parts=quiet_reward(before,after,np.zeros((32,18)),np.zeros(18))
  np.testing.assert_array_equal(reward,0);np.testing.assert_array_equal(parts['control_interval_rate_cost'],0)
  after['joint_position_rad']+=.002
  _,parts=quiet_reward(before,after,np.zeros((32,18)),np.zeros(18));np.testing.assert_allclose(parts['control_interval_rate_cost'],-.01)

if __name__=='__main__':unittest.main()
