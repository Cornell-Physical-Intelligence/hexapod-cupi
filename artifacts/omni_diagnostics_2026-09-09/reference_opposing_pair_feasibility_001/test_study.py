"""Focused target/lineage checks; none substitute for measured four-support physics."""
import copy,json,unittest
from unittest.mock import patch
import numpy as np
from study import HERE,load_inputs,evaluate,WaveContactReference,ReferenceResidualTarget

class CandidateTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.plan,cls.names,cls.snapshot=load_inputs()
  cls.report=json.loads((HERE/'report.json').read_text())
  with np.load(HERE/'planned_targets.npz',allow_pickle=False) as z:cls.data={k:z[k].copy() for k in z.files}
 def test_actual_named_q0_and_first_last_knot_continuity(self):
  d=self.data;dt=self.plan['control_dt_s']
  np.testing.assert_array_equal(d['q'][0],self.snapshot['joint_target_rad'][0])
  np.testing.assert_array_equal(d['q'][:101],np.broadcast_to(d['q'][0],(101,18)))
  np.testing.assert_array_equal(d['v'][1:],np.diff(d['q'],axis=0)/dt)
  np.testing.assert_array_equal(d['a'][1:],np.diff(d['v'],axis=0)/dt)
  self.assertGreaterEqual(self.report['final_reference_quiet_duration_s']-2.,10.)
  np.testing.assert_array_equal(d['q'][-500:],np.broadcast_to(d['q'][-1],(500,18)))
  np.testing.assert_array_equal(d['v'][-500:],0.);np.testing.assert_array_equal(d['a'][-500:],0.)
 def test_budget_and_full_residual_reserve_on_every_emitted_knot(self):
  d=self.data;g=WaveContactReference(self.names);g.reset(self.snapshot)
  self.assertEqual(self.plan['fixed_target_limits']['reference_velocity_rad_s'],1.75)
  self.assertEqual(self.plan['fixed_target_limits']['reference_acceleration_rad_s2'],6.)
  self.assertEqual(self.plan['torque_ceiling_nm'],1.6)
  self.assertLessEqual(abs(d['v']).max(),1.75);self.assertLessEqual(abs(d['a']).max(),6.)
  self.assertTrue((d['q']>=g._runtime(g.lower)+.02).all());self.assertTrue((d['q']<=g._runtime(g.upper)-.02).all())
  self.assertEqual(self.report['max_zero_residual_executable_lag_rad'],0.)
 def test_planned_support_partition_and_no_liftoff_after_stop(self):
  d=self.data
  for i,pair in enumerate(self.plan['pair_order']):
   indices=[('lf','lm','lr','rf','rm','rr').index(n) for n in pair]
   mask=np.ones(6,bool);mask[indices]=False
   active=d['active_pair_index']==i
   self.assertTrue(active.any());np.testing.assert_array_equal(d['planned_support_mask'][active],np.broadcast_to(mask,(active.sum(),6)))
  self.assertEqual(self.report['planned_liftoffs_after_stop_request'],0)
  self.assertGreaterEqual(d['projected_margin_m'].min(),.05)
  self.assertEqual(self.report['actual_new_contact_confirmations'],0)
  self.assertFalse(self.report['physics_admitted'])
 def test_runtime_name_permutation_preserves_all_targets(self):
  names=tuple(reversed(self.names));snapshot=copy.deepcopy(self.snapshot);order=list(reversed(range(18)))
  for key,value in snapshot.items():
   if isinstance(value,np.ndarray) and value.shape==(1,18):snapshot[key]=value[:,order]
   elif isinstance(value,np.ndarray) and value.shape==(1,18,2):snapshot[key]=value[:,order,:]
  r,d=evaluate(self.plan,names,snapshot)
  self.assertTrue(r['completed_target_sequence'])
  for key in ('q','v','a'):np.testing.assert_array_equal(d[key][:,order],self.data[key])
  np.testing.assert_array_equal(d['projected_margin_m'],self.data['projected_margin_m'])
 def test_unreachable_target_latches_and_is_never_emitted(self):
  original=WaveContactReference._predict;calls=[];emit=ReferenceResidualTarget.step
  def unreachable(g,target,horizon):
   p,R=original(g,target,horizon);p[0]+=.8;return p,R
  def record(self,*args,**kwargs):calls.append(1);return emit(self,*args,**kwargs)
  with patch.object(WaveContactReference,'_predict',unreachable),patch.object(ReferenceResidualTarget,'step',record):r,d=evaluate(self.plan,self.names,self.snapshot)
  self.assertFalse(r['completed_target_sequence']);self.assertIsNotNone(r['failure'])
  self.assertEqual(len(calls),r['accepted_controls'])
  self.assertEqual(len(d['q']),r['accepted_controls']+1)
  self.assertIn('first_rejected_qva',d);self.assertFalse(np.array_equal(d['first_rejected_qva'][0],d['q'][-1]))
  self.assertLess(r['accepted_controls'],r['planned_controls'])
 def test_strict_json_and_exact_plan_source_lineage(self):
  json.dumps(self.report,allow_nan=False)
  from study import sha
  self.assertEqual(self.report['plan_sha256'],sha(HERE/'PLAN.json'))
  self.assertEqual(self.report['study_source_sha256'],sha(HERE/'study.py'))
  self.assertEqual(self.plan['source_files']['oracle/wave_reference.py'],'8bacf0c61f7fdc6d43d9f9ebdb5adc643b026495bb303ec3c18157d121c78893')
  self.assertEqual(self.plan['source_files']['oracle/reference_residual.py'],'fbf65749c2d1ffc36b22986f71c0972ce7b1ecf82beebef57a0bd3e1be966e40')
if __name__=='__main__':unittest.main()
