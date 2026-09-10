import importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('host',Path(__file__).with_name('launch_residual_ppo_spark.py'));h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
class HostTest(unittest.TestCase):
 def args(self,base):
  return SimpleNamespace(**{k:base/k for k in ['source','run','device_run','bridge','consumer','observation','output']})
 def test_readonly_dependency_mounts_and_phase_boundaries(self):
  args=self.args(Path('/private/test'))
  for phase in h.PHASES[1:]:
   c=h.command(args,'owned',phase);mounts=[c[i+1] for i,v in enumerate(c) if v=='-v']
   self.assertEqual(sum(v.endswith(':rw') for v in mounts),1)
   self.assertIn('/private/test/consumer:/consumer:ro',mounts)
   self.assertIn('/private/test/output/inputs/study:/study:ro',mounts)
   self.assertEqual(c[c.index('--mode')+1],phase)
   self.assertEqual(c[c.index('--output')+1],'/outputs/'+phase)
   self.assertEqual('--smoke-output' in c,phase.startswith('evaluate_'))
   if phase.startswith('evaluate_'):self.assertIn('/private/test/output/smoke:/outputs/smoke:ro',mounts)
  with self.assertRaises(ValueError):h.command(args,'owned','train_forever')
 def test_digest_helper_alias(self):
  self.assertIs(h.digest,h.sha)
 def test_parent_owns_unchanged_standing_command(self):
  parent=SimpleNamespace(command=lambda *args:list(args));args=self.args(Path('/test'))
  with patch.object(h,'PARENT',parent):self.assertEqual(h.command(args,'name','standing'),[args.source,args.output,'name','standing'])
 def exercise(self,fail=None,mutate=False,wrong_checkpoint=False,wrong_mode=False):
  with tempfile.TemporaryDirectory() as d:
   args=self.args(Path(d));args.output.mkdir();calls=[]
   for phase in h.PHASES:(args.output/phase).mkdir()
   (args.output/'standing/admission.json').write_text('{}')
   (args.output/'smoke/final.pt').write_bytes(b'exact final checkpoint')
   (args.output/'smoke/initial.pt').write_bytes(b'exact initial checkpoint')
   def run(a,phase):
    calls.append(phase)
    if mutate and phase=='evaluate_initial':(args.output/'smoke/final.pt').write_bytes(b'bad')
    which='initial' if phase=='evaluate_initial' else 'final'
    return {'checkpoint_sha256':'wrong' if wrong_checkpoint else h.sha(args.output/'smoke'/(which+'.pt'))}
   def validate(p,identity):
    if p.name==fail:raise ValueError('rejected result')
    return {'passed':True,'mode':'wrong' if wrong_mode else p.name}
   consumer=SimpleNamespace(verify_standing=lambda *a:None,validate_result=validate)
   with patch.object(h,'validate_inputs',return_value={'bound':True}),patch.object(h,'run_owned',side_effect=run),patch.object(h,'PARENT',SimpleNamespace(check_source=lambda p:None)),patch.object(h,'CONSUMER',consumer):
    if fail or mutate or wrong_checkpoint or wrong_mode:
     with self.assertRaises(ValueError):h.run_phases(args,{'bound':True},{})
    else:h.run_phases(args,{'bound':True},{})
   return calls
 def test_four_serial_phases(self):self.assertEqual(self.exercise(),list(h.PHASES))
 def test_rejected_smoke_never_evaluates(self):self.assertEqual(self.exercise(fail='smoke'),['standing','smoke'])
 def test_failed_initial_retention_prevents_final(self):self.assertEqual(self.exercise(fail='evaluate_initial'),list(h.PHASES[:3]))
 def test_smoke_mutation_prevents_next_allocation(self):self.assertEqual(self.exercise(mutate=True),list(h.PHASES[:3]))
 def test_wrong_checkpoint_stops_retention(self):self.assertEqual(self.exercise(wrong_checkpoint=True),list(h.PHASES[:3]))
 def test_wrong_phase_stops_smoke(self):self.assertEqual(self.exercise(wrong_mode=True),list(h.PHASES[:2]))
 def test_owned_supervisor_preserved_except_declared_two_edits(self):
  import ast
  p=Path(__file__).resolve().parents[1]/'reference_physics_adapter_009/source_009/tools/launch_reference_physics_spark.py'
  original=ast.parse(p.read_text());actual=ast.parse(Path(h.__file__).read_text())
  find=lambda tree,name:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
  self.assertEqual(ast.dump(find(original,'owned_container')),ast.dump(find(actual,'owned_container')))
  src=ast.get_source_segment(p.read_text(),find(original,'run_owned')).replace('no_policy_loaded=True','no_policy_loaded=phase == "standing"').replace('command(args.source, args.output, name, phase)','command(args, name, phase)')
  self.assertEqual(ast.dump(ast.parse(src).body[0]),ast.dump(find(actual,'run_owned')))
if __name__=='__main__':unittest.main()
