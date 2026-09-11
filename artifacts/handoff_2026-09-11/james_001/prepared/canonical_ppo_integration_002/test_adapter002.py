import unittest
import numpy as np
from canonical_direct_ppo.adapter import JointConfig,TargetPipeline,ObservationScales,ObservationHistory,ActorFrameInputs
from canonical_direct_ppo.frames import world_to_native_body,native_to_navigation,navigation_to_native,body_velocity_at_root_origin

class Adapter002Checks(unittest.TestCase):
 def test_actual_target_error_and_nonzero_command_survive_exact_layout(self):
  scales=ObservationScales(*[np.ones(n)for n in [3,3,3,18,18,18,18,3]])
  history=ObservationHistory(np.zeros(18),scales,2)
  inputs=ActorFrameInputs(np.array([[1.,2.,3.],[4.,5.,6.]]),np.tile([0,0,-1.],(2,1)),np.array([[.2,-.1,.3],[-.4,.5,-.6]]),np.full((2,18),.03),np.full((2,18),.07),np.full((2,18),.2),np.full((2,18),.04))
  history.reset_rows(np.arange(2),inputs);frame=history.actor_observation()[:,-81:]
  np.testing.assert_allclose(frame[:,63:],.01);np.testing.assert_allclose(frame[:,6:9],inputs.command)
  np.testing.assert_allclose(frame[:,:3],[[-2,1,3],[-5,4,6]])
 def test_nonzero_world_rotation_and_com_origin(self):
  angle=.7;q=np.array([[0,0,np.sin(angle/2),np.cos(angle/2)]])
  c,s=np.cos(angle),np.sin(angle);r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
  native=np.array([[.4,-.2,.7]]);world=native@r.T
  np.testing.assert_allclose(world_to_native_body(world,q),native,atol=1e-15)
  nav=native_to_navigation(native);np.testing.assert_allclose(nav,[[.2,.4,.7]])
  np.testing.assert_allclose(navigation_to_native(nav),native)
  pose=np.c_[np.zeros((1,3)),q];local=np.array([.2,0,0]);omega=np.array([[0,0,.6]])
  com=world+np.cross(omega,np.array([r@local]));linear,angular=body_velocity_at_root_origin(np.c_[com,omega],pose,local)
  np.testing.assert_allclose(linear,native);np.testing.assert_allclose(angular,omega)
 def test_selected_history_reset_preserves_unselected_rows(self):
  scales=ObservationScales(*[np.ones(n)for n in [3,3,3,18,18,18,18,3]])
  hist=ObservationHistory(np.zeros(18),scales,3)
  def value(n,v):return ActorFrameInputs(np.zeros((n,3)),np.tile([0,0,-1.],(n,1)),np.full((n,3),v),np.full((n,18),v),np.zeros((n,18)),np.zeros((n,18)),np.full((n,18),v+.01))
  hist.reset_rows(np.arange(3),value(3,.1));hist.push(value(3,.2));before=hist.history
  hist.reset_rows(np.array([1]),value(1,.3));after=hist.history
  np.testing.assert_array_equal(before[[0,2]],after[[0,2]]);np.testing.assert_allclose(after[1,:,6:9],.3)
  np.testing.assert_allclose(after[1,:,63:],.01)

if __name__=='__main__':unittest.main()
