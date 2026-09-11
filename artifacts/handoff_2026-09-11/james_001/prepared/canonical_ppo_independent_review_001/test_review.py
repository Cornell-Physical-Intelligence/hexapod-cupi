"""Independent CPU seam checks; no native physics or training admission."""
from pathlib import Path
import sys,json,tempfile,unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'tmp/canonical_ppo_integration_001'
sys.path[:0]=[str(SOURCE),str(ROOT/'tmp/reference_residual_ppo_001/_deps')]
from canonical_direct_ppo.native_bridge import CanonicalNativeBridge,govern
from canonical_direct_ppo.frames import body_velocity_at_root_origin
from canonical_direct_ppo.runner import learn_smoke,equal_tree
import canonical_direct_ppo.runner as runner
from test_bridge_session import FakeSession
MODEL=json.loads((ROOT/'robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json').read_text())

class Review(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def test_linear_privilege_cannot_enter_actor_at_reset_or_after_step(self):
  a=FakeSession(MODEL);b=FakeSession(MODEL);b.current['root_com_velocity'][:,:3]=[2,-3,4]
  x=CanonicalNativeBridge(a,MODEL,device='cpu');y=CanonicalNativeBridge(b,MODEL,device='cpu')
  for _ in range(3):
   ox=x.get_observations();oy=y.get_observations();self.assertTrue(torch.equal(ox['policy'],oy['policy']));self.assertFalse(torch.equal(ox['critic'][:,-3:],oy['critic'][:,-3:]));np.testing.assert_allclose(oy['critic'][0,-3:],[3,2,4])
   x.step(torch.full((32,18),.5));y.step(torch.full((32,18),.5))
 def test_random_full3d_com_shift_sign(self):
  from scipy.spatial.transform import Rotation
  rng=np.random.default_rng(514);q=Rotation.random(32,random_state=rng).as_quat();R=Rotation.from_quat(q).as_matrix();origin=rng.normal(size=(32,3));omega=rng.normal(size=(32,3));local=rng.normal(size=(32,3))*.03
  com=origin+np.cross(omega,np.einsum('nij,nj->ni',R,local));v,w=body_velocity_at_root_origin(np.c_[com,omega],np.c_[np.zeros((32,3)),q],local)
  np.testing.assert_allclose(v,np.einsum('nji,nj->ni',R,origin),atol=2e-15);np.testing.assert_allclose(w,np.einsum('nji,nj->ni',R,omega),atol=2e-15)
 def test_float32_governor_random_and_fulltravel_boundaries(self):
  named=MODEL['joints'];lo=np.array([j['lower']for j in named]);hi=np.array([j['upper']for j in named]);neutral=np.zeros(18);rng=np.random.default_rng(281)
  held=np.broadcast_to(neutral,(32,18)).copy()
  for _ in range(300):
   action=rng.normal(size=(32,18))*5;r=govern(action,held,lo,hi,neutral);target=r['emitted_target_rad'];self.assertEqual(target.dtype,np.float32)
   self.assertLessEqual(float(abs(target.astype(float)-held).max()),.04);self.assertTrue(np.all(target.astype(float)>=lo));self.assertTrue(np.all(target.astype(float)<=hi));held=target.astype(float)
 def test_actual_hold_history_and_failure_state_are_temporally_separate(self):
  s=FakeSession(MODEL);bridge=CanonicalNativeBridge(s,MODEL,device='cpu');old=bridge.history.history.copy();self.assertTrue((old[:,:,-18:]==0).all())
  bridge.step(torch.ones(32,18));q=s.current['joint_position_rad'];latest=bridge.history.history[:,-1];np.testing.assert_allclose(latest[:,63:],bridge.held-q);self.assertTrue((latest[:,45:63]==1).all());np.testing.assert_array_equal(bridge.history.history[:,:-1],old[:,1:])
  s.bad_contact_substep=1;before=bridge.export_runtime_state()
  with self.assertRaises(ValueError):bridge.step(-torch.ones(32,18))
  after=bridge.export_runtime_state();self.assertTrue(equal_tree(before,after));counter=s.count
  with self.assertRaises(RuntimeError):bridge.step(torch.zeros(32,18))
  self.assertEqual(s.count,counter)
 def test_real_RSL_reload_diagnostics_preserve_learning_and_rng(self):
  lineage={'scope':'independent synthetic CPU only'}
  original=runner.strict_reload
  with tempfile.TemporaryDirectory()as d:
   paths=[];receipts=[]
   for with_reload in (True,False):
    runner.strict_reload=original if with_reload else lambda *a,**k:{'test_disabled':True}
    try:
     env=CanonicalNativeBridge(FakeSession(MODEL),MODEL,device='cpu');path=Path(d)/str(with_reload);r=learn_smoke(env,lineage,path,device='cpu')
     saved=torch.load(path/'decision_002.pt',weights_only=False);paths.append(saved);receipts.append(r)
    finally:runner.strict_reload=original
   self.assertTrue(equal_tree(paths[0]['algorithm'],paths[1]['algorithm']));self.assertTrue(equal_tree(paths[0]['runtime_state'],paths[1]['runtime_state']));self.assertTrue(torch.equal(paths[0]['torch_rng_cpu'],paths[1]['torch_rng_cpu']))
   for saved in paths:
    for which in ['actor_state_dict','critic_state_dict']:self.assertEqual(int(saved['algorithm'][which]['obs_normalizer.count']),1536)
   report={'scope':'CPU synthetic dynamics only','two_runs_each_updates':2,'each_optimizer_minibatches':40,'actor_critic_normalizer_counts':1536,'reload_enabled_disabled_algorithm_exact':True,'runtime_snapshot_exact':True,'next_torch_rng_exact':True,'normalizer_diagnostics_add_no_samples':True,'reload_enabled_receipt':receipts[0]}
   (Path(__file__).parent/'RSL_PARITY.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':unittest.main()
