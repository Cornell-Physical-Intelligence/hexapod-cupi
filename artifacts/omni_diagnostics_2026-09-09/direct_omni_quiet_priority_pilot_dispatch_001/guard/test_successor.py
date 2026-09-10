"""Narrow selection delta and exact native smoke-validation delegation."""
import ast, importlib.util, json, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'direct_omni_train_pilot_curriculum_guard_002/launch_guarded_remote.py'
spec=importlib.util.spec_from_file_location('quiet_pilot_guard',HERE/'launch_guarded_remote.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

class SuccessorTests(unittest.TestCase):
 def test_entire_main_preserved_except_declared_branch_pause_and_reason(self):
  old=ast.parse(PARENT.read_text());new=ast.parse((HERE/'launch_guarded_remote.py').read_text())
  old_main=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  new_main=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  class Normalize(ast.NodeTransformer):
   def visit_Constant(self,node):
    if isinstance(node.value,str):
     node.value=node.value.replace('quiet-priority pilot','curriculum pilot').replace('restore-060','restore-056').replace('35af5100e39980f55a3f5e19654e4550b3c3fb6da114b64fbe0d143817242cf3','22c615d67c6c4dc14ab370b291f8eddd72f8475ad34118b91a4237a34ee05fab')
     if node.value=='quiet_priority':node.value='curriculum'
    return node
  self.assertEqual(ast.dump(old_main,include_attributes=False),ast.dump(Normalize().visit(new_main),include_attributes=False))
  for name in ('call','sha','verify_frozen'):
   a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
   b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
   self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))

 def test_native_rejected_smoke_propagates_without_any_mutation(self):
  host=SimpleNamespace(selected_phases=lambda args:('standing','initial_constant','initial_stop','train','final_constant','final_stop'),
                       verify_inputs=Mock(side_effect=ValueError('Exact smoke checkpoint or diagnostic receipt rejected')))
  hashes={g.SUPERVISOR_SOURCE/'campaign_source_hashes.json':g.SUPERVISOR_SHA256,
          g.SOURCE/'campaign_source_hashes.json':g.SOURCE_SHA256,g.CHECKPOINT:g.CHECKPOINT_SHA256,g.HOST:g.HOST_SHA256}
  with patch.object(g,'sha',side_effect=lambda p:hashes[p]),patch.object(g,'verify_frozen'),patch.object(g.subprocess,'run') as run:
   with self.assertRaisesRegex(ValueError,'smoke checkpoint or diagnostic'):g.validate_train_inputs(host)
   run.assert_not_called()
  args=host.verify_inputs.call_args.args[0]
  self.assertEqual((args.allocation,args.branch,args.smoke),('pilot','quiet_priority',g.SMOKE))

 def test_current_frozen_native_requires_all_four_actual_accepted_results(self):
  # Execute the actual stdlib native smoke aggregator. Its phase validator is
  # substituted only here so no checkpoint/NPZ fixtures or simulator are needed.
  # Native004's source-bound tests and host preflight own that deeper validation.
  import tempfile,hashlib
  source=HERE.parent/'direct_omni_recovery_001/native_004/direct_contract.py'
  fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='verify_smoke_campaign')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);identity={'schema':'synthetic-test-only'}
   phases=('standing','train','final_constant','final_stop')
   expected={phase:{'phase':phase,'checkpoint_sha256':'test-final'} for phase in phases}
   data={'status':'completed','terminal_inputs_unchanged':True,'allocation':'smoke','branch':'quiet_priority',
         'identity':identity,'planned_phases':list(phases),'accepted_phases':expected}
   def write(): (p/'campaign.json').write_text(json.dumps(data))
   write();calls=[]
   def validate(folder,phase,actual,checkpoint=None):
    calls.append((phase,actual,checkpoint));return expected[phase]
   scope={'Path':Path,'SMOKE_BRANCH':'quiet_priority','read':lambda q:json.loads(q.read_text()),
          'sha':lambda q:hashlib.sha256(q.read_bytes()).hexdigest(),'validate_result':validate}
   exec(compile(ast.Module(body=[fn],type_ignores=[]),'<exact native004 smoke gate>','exec'),scope)
   scope['verify_smoke_campaign'](p,identity)
   self.assertEqual([x[0] for x in calls],['train',*phases])
   self.assertTrue(all(x[2]=='test-final' for x in calls[1:]))
   for change in ({'status':'failed'},{'terminal_inputs_unchanged':False},{'branch':'caps'},
                  {'identity':{'wrong_source':True}},{'accepted_phases':{k:v for k,v in expected.items() if k!='final_stop'}}):
    original=dict(data);data.update(change);write()
    with self.assertRaises(ValueError):scope['verify_smoke_campaign'](p,identity)
    data=original

if __name__=='__main__':unittest.main()
