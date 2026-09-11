"""Source005 composition/readonly diagnostic lifecycle and tamper counterexamples."""
from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
import copy,hashlib,json,sys,tempfile,textwrap,unittest
from canonical_direct_ppo import native_entry_adapter as entry,source005_solver_diagnostics as diag
from canonical_direct_ppo.binding_contract import verify_admissions,sha
from canonical_direct_ppo.native_contract import validate_result
from test_support import ROOT,STANDING
from test_solver_fixture import solver_record,write_member_fixture
import test_native_result as result_fixture
from test_native_result import write

HERE=Path(__file__).resolve().parent
PARENT_MAP=json.loads((HERE/'history/ppo002/FREEZE_SHA256.json').read_text())

class Standing005Checks(unittest.TestCase):
 def test_exact_source_and_actor_servo_preservation(self):
  self.assertEqual(entry.STANDING_FREEZE,'c87f2d340c935cd947c7aed3f3dcd7c9a9965b6e19d6726546abb560b27ab131')
  self.assertEqual(sha(STANDING/'FREEZE_SHA256.json'),entry.STANDING_FREEZE)
  for name in ['adapter.py','frames.py','native_bridge.py','runner.py','run_bound_smoke.py','smoke_config.py','binding_contract.py','rsl_binding.py']:
   self.assertEqual(sha(HERE/'canonical_direct_ppo'/name),PARENT_MAP['canonical_direct_ppo/'+name],name)
  self.assertEqual(sha(HERE/'run_native_smoke.py'),PARENT_MAP['run_native_smoke.py'])
  self.assertEqual((HERE/'canonical_direct_ppo/source005_solver_diagnostics.py').read_bytes(),(STANDING/'solver_diagnostics.py').read_bytes())
  self.assertEqual(sha(HERE/'history/ppo002/FREEZE_SHA256.json'),'1e173946fe2548a82207791528f503ac6d12766fd5b50f25746e6edf92704617')
 def test_earlier_source_refused_and_32_still_disabled(self):
  with tempfile.TemporaryDirectory()as td:
   old=Path(td);(old/'FREEZE_SHA256.json').write_bytes((HERE/'STANDING003_FREEZE_SHA256.json').read_bytes())
   self.assertEqual(sha(old/'FREEZE_SHA256.json'),'e922ff1312b5106f957b384874ca5eb6ed4a9ed6566c0ef9a0602247a49670f0')
   with self.assertRaisesRegex(ValueError,'Wrong frozen source'):entry.adapted_source(old)
  b=json.loads((HERE/'BINDINGS.json').read_text())
  self.assertEqual(b['standing1_state_sha256'],'33920a4af296ed124643fe61a16f4840c19949b5f6e434327d7cd7ea9953cd3f')
  self.assertIsNone(b['standing32_state_sha256']);self.assertFalse(b['ready_for_native_dispatch'])
  with self.assertRaisesRegex(ValueError,'Pending/invalid'):verify_admissions(STANDING,Path('/notread1'),Path('/notread32'),b)
 def test_neutral_and_final_solver_reads_enclose_48_policy_controls(self):
  events=[];saved=[];session=SimpleNamespace(count=8000)
  session.close=lambda:events.append(('close',session.count))
  def verify(stage,roots,schema):events.append(('verify',session.count));return solver_record()['after_authoring']
  def save(path,value):saved.append(copy.deepcopy(value));events.append(('save',session.count))
  def callback(s,*args):
   self.assertEqual(s.count,8000);self.assertIn('after_neutral_steps',saved[-1]);self.assertNotIn('after_controlled_steps',saved[-1]);events.append(('learner',s.count));s.count+=384
  record=solver_record(final=False);record.pop('after_neutral_steps')
  env={'solver_record':record,'verify_solver':verify,'stage':None,'roots':None,'PhysxSchema':None,'save':save,'out':Path('/CPU_ONLY'),'state':{'checks':{}},'learner_callback':callback,'session':session,'model':None,'errors':[],'a':None,'time':SimpleNamespace(monotonic=lambda:1.),'started':0.,'print':lambda *a,**k:None}
  exec(textwrap.dedent(entry.LEARNER_TAIL),env)
  self.assertEqual(events,[('verify',8000),('save',8000),('learner',8000),('close',8384),('verify',8384),('save',8384)])
  self.assertEqual(env['state']['explicit_steps_completed'],8384);self.assertEqual(env['state']['status'],'completed')
 def test_solver_failure_before_callback_stops_actor(self):
  def fail(*a):raise ValueError('changed32/4')
  env={'verify_solver':fail,'solver_record':{},'stage':None,'roots':None,'PhysxSchema':None,'learner_callback':lambda *a:self.fail('actor reached')}
  with self.assertRaisesRegex(ValueError,'changed32/4'):exec(textwrap.dedent(entry.LEARNER_TAIL),env)
 def test_new_helpers_reject_foreign_cached_modules(self):
  for name in ['solver_recipe','solver_diagnostics']:
   m=ModuleType(name);m.__file__='/foreign/'+name+'.py'
   with patch.dict(sys.modules,{name:m}):
    with self.assertRaisesRegex(ValueError,'Foreign cached'):
     with entry.exact_helper_imports(STANDING):self.fail('accepted')

class DiagnosticResultChecks(unittest.TestCase):
 def fixture(self,d):return result_fixture.ResultChecks().fixture(d)
 def seal(self,d,state):
  prefix=d/'neutral_prefix';write(prefix/'SHA256.json',{p.name:sha(p)for p in prefix.iterdir()if p.name!='SHA256.json'})
  result_fixture.ResultChecks().seal(d,state)
 def test_wrong_solver_missing_root_phase_or_unsealed_copy_rejected(self):
  changes=[lambda v:v['after_neutral_steps']['/Robot'].update(velocity_iterations=4),lambda v:v.pop('after_controlled_steps'),lambda v:v['after_controlled_steps'].pop('/Robot_031')]
  for change in changes:
   with self.subTest(change=change),tempfile.TemporaryDirectory()as td:
    d=Path(td);identity,state,_=self.fixture(d);v=json.loads((d/'solver_readback.json').read_text());change(v);write(d/'solver_readback.json',v);self.seal(d,state)
    with self.assertRaises(ValueError):validate_result(d,identity)
  with tempfile.TemporaryDirectory()as td:
   d=Path(td);identity,state,_=self.fixture(d);state['outputs'].pop('neutral_prefix/solver_readback.json');write(d/'state.json',state)
   with self.assertRaisesRegex(ValueError,'not sealed'):validate_result(d,identity)
 def test_declaration_counters_and_missing_npz_channel_rejected(self):
  for where in ['','neutral_prefix']:
   with self.subTest(where=where),tempfile.TemporaryDirectory()as td:
    d=Path(td);identity,state,_=self.fixture(d);p=d/where/'session.json';v=json.loads(p.read_text());v['solver_diagnostics']['captured_steps']-=1;write(p,v);self.seal(d,state)
    with self.assertRaisesRegex(ValueError,'declaration'):validate_result(d,identity)
   with tempfile.TemporaryDirectory()as td:
    d=Path(td);identity,state,_=self.fixture(d);write_member_fixture(d/where/'substeps_000.npz',omit=diag.FLOOR_FIELD);self.seal(d,state)
    with self.assertRaisesRegex(ValueError,'Missing raw diagnostic'):validate_result(d,identity)
 def test_nonzero_legacy_allowed_changed_or_nonfinite_prefix_rejected(self):
  with tempfile.TemporaryDirectory()as td:
   d=Path(td);identity,state,_=self.fixture(d)
   self.assertEqual(json.loads((d/'legacy_friction_readback.json').read_text())['coefficients'][0][0],.125)
   self.assertEqual(validate_result(d,identity)['status'],'completed')
   p=d/'neutral_prefix/legacy_friction_readback.json';v=json.loads(p.read_text());v['coefficients'][0][0]=.25;write(p,v);self.seal(d,state)
   with self.assertRaisesRegex(ValueError,'Changed native prefix'):validate_result(d,identity)
   v['coefficients'][0][0]=float('nan');write(p,v);self.seal(d,state)
   with self.assertRaisesRegex(ValueError,'Nonfinite legacy'):validate_result(d,identity)

if __name__=='__main__':unittest.main()
