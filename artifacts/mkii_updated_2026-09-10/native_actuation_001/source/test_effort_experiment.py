"""Actual experiment class with deterministic raw-view fixtures; no Isaac or CUDA."""
import io,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from effort_experiment import Experiment,evidence_json
from floating_pulse import floating_mass
from inertia_probe import frames
from actuation_contract import EXPERIMENT_STEPS,validate_raw,verify_admission

ROOT=Path(__file__).resolve().parents[2]

class View:
 def __init__(self,model):
  self.model=model;self.names=[j['name']for j in model['joints']];self.links=[x['name']for x in model['links']]
  self.M,self.r=floating_mass(model,self.names);self.q=np.zeros((1,18),dtype=np.float32);self.v=self.q.copy();self.force=self.q.copy();self.target=self.q.copy();self.vtarget=self.q.copy()
  self.root=np.array([[0,0,.08161109,0,0,0,1]],dtype=np.float32);self.rootv=np.zeros((1,6),dtype=np.float32)
  self.writes=[];self.fail_getter=False;self.corrupt_reset=False;self.nan_next=False
 def _set(self,name,x,ids):
  if np.asarray(ids).dtype!=np.uint32:raise ValueError('Wrong index dtype')
  setattr(self,name,np.asarray(x,dtype=np.float32).copy());self.writes.append(name)
 def set_root_transforms(self,x,ids):self._set('root',x,ids)
 def set_root_velocities(self,x,ids):self._set('rootv',x,ids)
 def set_dof_positions(self,x,ids):
  self._set('q',x,ids)
  if self.corrupt_reset:self.q[0,0]+=.01
 def set_dof_velocities(self,x,ids):self._set('v',x,ids)
 def set_dof_position_targets(self,x,ids):self._set('target',x,ids)
 def set_dof_velocity_targets(self,x,ids):self._set('vtarget',x,ids)
 def set_dof_actuation_forces(self,x,ids):self._set('force',x,ids)
 def get_root_transforms(self):return self.root.copy()
 def get_root_velocities(self):return self.rootv.copy()
 def get_dof_positions(self):return self.q.copy()
 def get_dof_velocities(self):return self.v.copy()
 def get_dof_position_targets(self):return self.target.copy()
 def get_dof_velocity_targets(self):return self.vtarget.copy()
 def get_dof_actuation_forces(self):return self.force.copy()
 def get_dof_projected_joint_forces(self):
  if self.fail_getter:raise RuntimeError('intentional uncertain getter')
  return self.force.copy()+.123 # Reaction deliberately differs from actuation input.
 def get_link_incoming_joint_force(self):return np.ones((1,19,6),dtype=np.float32)*.456
 def get_generalized_mass_matrices(self):return self.M[None].copy()

class Sim:
 def __init__(self,view):self.view=view;self.counter=8;self.extra_counter=False
 def get_physics_step_count(self):return self.counter
 def step(self,render=False):
  v=self.view;u=np.r_[np.zeros(6),v.force[0]];acc=np.linalg.solve(v.M,u)[6:]
  v.v=(v.v+.0025*acc).astype(np.float32);v.q=(v.q+.0025*v.v).astype(np.float32)
  if v.nan_next:v.q[0,0]=np.nan
  self.counter+=2 if self.extra_counter else 1

def snap(view,sim,step):
 T,*_=frames(view.model,view.q[0],view.names);poses=[]
 # Fixture's root stays fixed; this tests actual collector logic, not floating native dynamics.
 for name in view.links:
  m=T[name];poses.append([*(m[:3,3]+view.root[0,:3]),*Rotation.from_matrix(m[:3,:3]).as_quat()])
 return {'explicit_step':step,'explicit_step_counter':sim.counter,'joint_position':view.q.tolist(),'joint_velocity_sdk':view.v.tolist(),
  'link_pose_xyzw':[poses],'link_com_velocity':np.zeros((1,19,6)).tolist()}

def save(path,value):Path(path).write_text(evidence_json(value))

class ActualDriver(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.model=json.loads((ROOT/'robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json').read_text())
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.out=Path(self.tmp.name)
  wp=types.SimpleNamespace(array=lambda x,dtype,device:np.array(x,dtype=dtype),uint32=np.uint32,float32=np.float32)
  self.p=patch.dict(sys.modules,{'warp':wp});self.p.start();self.addCleanup(self.p.stop)
  self.mem=patch('effort_experiment.numeric_memory',return_value={'scope':'synthetic'});self.mem.start();self.addCleanup(self.mem.stop)
  self.v=View(self.model);self.sim=Sim(self.v);self.exp=Experiment(self.v,self.sim,self.model,{'joint_names':self.v.names,'body_names':self.v.links},self.out,snap,save,lambda x:np.asarray(x).tolist(),[],device='cpu')
  save(self.out/'native_readback.json',{'joint_names':self.v.names});save(self.out/'samples.json',[{'explicit_step_counter':8}])
  self.addCleanup(lambda:self.exp.stream.close());self.addCleanup(lambda:self.exp.reset_stream.close())

 def test_complete_actual_driver_schedule_and_separate_channels(self):
  r=self.exp.run();self.assertEqual(r['step_count'],EXPERIMENT_STEPS);self.assertEqual(r['case_count'],36);self.assertEqual(r['coordinate_count'],36)
  self.assertEqual(r['actual_physics_steps'],EXPERIMENT_STEPS);self.assertEqual(r['reset_count'],74)
  self.assertEqual(self.sim.counter,8+EXPERIMENT_STEPS);np.testing.assert_array_equal(self.v.force,0)
  rows=[json.loads(x)for x in (self.out/'experiment.jsonl').read_text().splitlines()]
  self.assertEqual(len(rows),EXPERIMENT_STEPS)
  self.assertEqual(sum(x['phase']=='pulse'for x in rows),36*8);self.assertEqual(sum(x['phase']=='coast'for x in rows),36*8)
  for i,x in enumerate(rows):
   self.assertEqual(x['sequence'],i);self.assertEqual(x['explicit_step_counter'],9+i)
   np.testing.assert_array_equal(x['native_input_pre'],x['software_applied_nm'])
  self.assertNotEqual(rows[0]['projected_joint_reaction_nm'],rows[0]['native_input_pre'])
  self.assertLess(r['maximum_joint_excursion_rad'],.01)
  self.assertEqual(validate_raw(self.out)['raw_steps'],EXPERIMENT_STEPS)

 def test_stdlib_raw_gate_rejects_duplicate_clock_and_reset_target(self):
  self.exp.run();path=self.out/'experiment.jsonl';original=path.read_text();rows=original.splitlines()
  a=json.loads(rows[1]);a['explicit_step_counter']=9;rows[1]=json.dumps(a);path.write_text('\n'.join(rows)+'\n')
  with self.assertRaisesRegex(ValueError,'clock/order/phase'):validate_raw(self.out)
  path.write_text(original);rp=self.out/'reset_readbacks.jsonl';rows=rp.read_text().splitlines();a=json.loads(rows[1]);a['position_target'][0][0]=.02;rows[1]=json.dumps(a);rp.write_text('\n'.join(rows)+'\n')
  with self.assertRaisesRegex(ValueError,'Reset state/target'):validate_raw(self.out)

 def test_reset_failure_is_saved_before_rejection(self):
  self.v.corrupt_reset=True
  with self.assertRaises(AssertionError):self.exp.run()
  self.assertEqual(self.sim.counter,8);self.assertEqual(json.loads((self.out/'experiment_summary.json').read_text())['status'],'failed')
  self.assertTrue((self.out/'reset_readbacks.jsonl').read_text())

 def test_native_getter_failure_preserves_partial_raw_and_actual_step(self):
  self.v.fail_getter=True
  with self.assertRaisesRegex(RuntimeError,'uncertain getter'):self.exp.run()
  partial=json.loads((self.out/'failed_partial_step.json').read_text());self.assertEqual(partial['explicit_step_counter'],9)
  self.assertIn('joint_position',partial);self.assertIn('native_input_post',partial)
  summary=json.loads((self.out/'experiment_summary.json').read_text());self.assertEqual(summary['actual_physics_steps'],1);self.assertEqual(summary['status'],'failed')

 def test_counter_skip_fails_after_raw_sample(self):
  self.sim.extra_counter=True
  with self.assertRaisesRegex(ValueError,'exactly once'):self.exp.run()
  raw=json.loads((self.out/'experiment.jsonl').read_text().splitlines()[0]);self.assertEqual(raw['explicit_step_counter'],10)

 def test_nonfinite_serialization_is_explicit_forensic_data(self):
  x=json.loads(evidence_json({'a':[float('nan'),float('inf')]}));self.assertEqual(x['a'][0],{'nonfinite':'nan'})

 def test_force_mismatch_and_reset_getter_preserve_failed_values(self):
  self.v.get_dof_actuation_forces=lambda:np.full((1,18),.0123,dtype=np.float32)
  with self.assertRaisesRegex(ValueError,'differs from software'):self.exp.run()
  evidence=[json.loads(x)for x in(self.out/'failed_force_input.jsonl').read_text().splitlines()]
  self.assertGreaterEqual(len(evidence),2) # initial failure and cleanup both retained
  self.assertEqual(evidence[0]['requested_external_nm'],np.zeros((1,18)).tolist())
  self.assertAlmostEqual(evidence[0]['actual_native_input'][0][0],.0123,places=6)
  self.assertEqual(self.sim.counter,8)

 def test_reset_getter_partial_fields_survive(self):
  def fail():raise RuntimeError('target getter unavailable')
  self.v.get_dof_position_targets=fail
  with self.assertRaisesRegex(RuntimeError,'target getter'):self.exp.run()
  row=json.loads((self.out/'failed_reset_readback.jsonl').read_text().splitlines()[0])
  self.assertIn('joint_position',row);self.assertIn('root_com_velocity',row)
  self.assertEqual(row['active_getter'],'get_dof_position_targets')

 def test_actual_completed_phase_a_and_changed_state_reject(self):
  source=ROOT/'tmp/canonical_native_inspection_terminal_003/run/inspection'
  self.assertEqual(verify_admission(source)['phase_a_state_sha256'],__import__('hashlib').sha256((source/'state.json').read_bytes()).hexdigest())
  import shutil
  target=self.out/'admission';shutil.copytree(source,target)
  (target/'state.json').write_text('{}\n')
  with self.assertRaisesRegex(ValueError,'Changed input'):verify_admission(target)

if __name__=='__main__':unittest.main()
