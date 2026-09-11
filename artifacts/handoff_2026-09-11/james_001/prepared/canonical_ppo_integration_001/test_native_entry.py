"""No native/GPU import: exact source seams and live-prefix lifecycle fixtures."""
from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
import ast,hashlib,json,sys,tempfile,unittest
import numpy as np
from canonical_direct_ppo import native_entry_adapter as entry
from canonical_direct_ppo.neutral_prefix import snapshot

from test_support import ROOT,STANDING


def save(path,data):Path(path).write_text(json.dumps(data))


class EntryChecks(unittest.TestCase):
 def test_exact_frozen_main_native_prefix_and_finally_unchanged(self):
  actual,receipt=entry.adapted_source(STANDING);old=(STANDING/'run_standing.py').read_text()
  self.assertEqual(actual.replace(entry.LEARNER_TAIL,entry.ORIGINAL_TAIL),old)
  before=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)and n.name=='main')
  after=next(n for n in ast.parse(actual).body if isinstance(n,ast.FunctionDef)and n.name=='main')
  t1=next(n for n in before.body if isinstance(n,ast.Try));t2=next(n for n in after.body if isinstance(n,ast.Try))
  self.assertEqual(ast.dump(ast.Module(body=t1.finalbody,type_ignores=[])),ast.dump(ast.Module(body=t2.finalbody,type_ignores=[])))
  self.assertEqual(ast.dump(ast.Module(body=t1.handlers,type_ignores=[])),ast.dump(ast.Module(body=t2.handlers,type_ignores=[])))
  prefix=old.split(entry.ORIGINAL_TAIL)[0]
  self.assertEqual(actual.split(entry.LEARNER_TAIL)[0],prefix)
  self.assertTrue(receipt['neutral_physics_statements_unchanged'])
 def test_lazy_native_import_is_in_scope_after_load_returns(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);(root/'standing_math.py').write_text('VALUE=37\n')
   source='def main(argv=None):\n import standing_math\n return standing_math.VALUE\n'
   previous=sys.modules.pop('standing_math',None)
   try:
    with patch.object(entry,'adapted_source',return_value=(source,{})):
     run,_=entry.load(root,None,None,None,'test')
    self.assertNotIn(str(root),sys.path);self.assertEqual(run(),37);self.assertNotIn(str(root),sys.path)
   finally:
    sys.modules.pop('standing_math',None)
    if previous is not None:sys.modules['standing_math']=previous
 def test_foreign_lazy_helper_inserted_after_load_is_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);previous=sys.modules.pop('standing_math',None)
   try:
    with patch.object(entry,'adapted_source',return_value=('def main(argv=None):\n return 9\n',{})):
     run,_=entry.load(root,None,None,None,'test')
    foreign=ModuleType('standing_math');foreign.__file__='/unrelated/standing_math.py';sys.modules['standing_math']=foreign
    with self.assertRaisesRegex(ValueError,'Foreign cached'):run()
   finally:
    sys.modules.pop('standing_math',None)
    if previous is not None:sys.modules['standing_math']=previous
 def test_changed_standing_source_fails_before_compile(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);(p/'FREEZE_SHA256.json').write_text('{}')
   with self.assertRaisesRegex(ValueError,'Wrong frozen source'):entry.adapted_source(p)


class PrefixChecks(unittest.TestCase):
 def fixture(self,root):
  for n in ['initial_reset.json','native_readback.json']:(root/n).write_text('{}')
  (root/'contacts.jsonl').write_text('prefix\n');(root/'substeps_000.npz').write_bytes(b'IMMUTABLE FIXTURE')
  s=SimpleNamespace(n=32,count=8000,captured_count=8000,control=1000,reset_count=1,failure=None,output=root,
    controls=[{'control_index':np.array(i)}for i in range(1000)],files=['substeps_000.npz'],body_names=['body'],names=['q'],roots=['robot']*32)
  s.flush=lambda:None
  return s
 def test_snapshot_does_not_close_reset_or_alias_active_contact_stream(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);s=self.fixture(root)
   def score(p):
    self.assertEqual(json.loads((p/'session.json').read_text())['steps'],8000)
    with np.load(p/'control_trace.npz')as z:self.assertEqual(z['control_index'][-1],999)
    return {'all_pass':True,'num_envs':32,'replicas':[{'pass':True}]*32}
   result=snapshot(s,root/'prefix',score,save)
   (root/'contacts.jsonl').write_text('prefix\nPOLICY\n');s.controls[0]['control_index'][...]=999
   self.assertEqual((root/'prefix/contacts.jsonl').read_text(),'prefix\n')
   with np.load(root/'prefix/control_trace.npz')as z:self.assertEqual(z['control_index'][0],0)
   self.assertEqual(s.count,8000);self.assertEqual(s.reset_count,1);self.assertFalse(result['session_closed'])
 def test_rejected_prefix_is_preserved_and_stops_before_learner(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);s=self.fixture(root)
   with self.assertRaisesRegex(ValueError,'actor was not constructed'):
    snapshot(s,root/'prefix',lambda p:{'all_pass':False,'num_envs':32,'replicas':[{'pass':False}]*32},save)
   self.assertTrue((root/'prefix/SHA256.json').is_file());self.assertEqual(s.control,1000)
 def test_incomplete_native_capture_cannot_score(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);s=self.fixture(root);s.captured_count=7999
   with self.assertRaisesRegex(ValueError,'complete32'):snapshot(s,root/'prefix',None,save)
   self.assertFalse((root/'prefix').exists())

if __name__=='__main__':unittest.main()
