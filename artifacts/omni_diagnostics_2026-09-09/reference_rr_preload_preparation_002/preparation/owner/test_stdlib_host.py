"""Run real host import/preflight without site packages or NumPy."""
from pathlib import Path
import ast,json,os,subprocess,sys,unittest
HERE=Path(__file__).resolve().parent;SOURCE=HERE/'source_rr_preload_002';PARENT=HERE.parent/'reference_rr_preload_diagnostic_001/source_rr_preload_001'
class HostImportTests(unittest.TestCase):
 def run_host(self,source):
  code='''import sys,importlib.util,json
from pathlib import Path
assert importlib.util.find_spec('numpy') is None, 'Fixture unexpectedly has NumPy'
sys.path.insert(0,sys.argv[1]+'/tools')
import launch_directional_physics_spark as host
from directional_contract import CASES,PROTOCOL
r=host.check_source(Path(sys.argv[1]))
assert list(CASES)==['left_strafe']
assert 'numpy' not in sys.modules
print(json.dumps({'host_import_without_numpy':True,'actual_no_App_host_check_source':r,'source_manifest_sha256':host.digest(Path(sys.argv[1])/'campaign_source_hashes.json'),'proposal':PROTOCOL['rr_preload_diagnostic']}))
'''
  return subprocess.run([sys.executable,'-S','-c',code,str(source)],capture_output=True,text=True,timeout=40,env={**os.environ,'PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1'})
 def test_exact_old_import_failure_and_new_full_host_preflight(self):
  old=self.run_host(PARENT);self.assertNotEqual(old.returncode,0);self.assertIn("No module named 'numpy'",old.stderr)
  new=self.run_host(SOURCE);self.assertEqual(new.returncode,0,new.stderr);r=json.loads(new.stdout);self.assertTrue(r['host_import_without_numpy']);self.assertEqual(r['actual_no_App_host_check_source']['python_files_verified'],16)
 def test_PROPOSAL_exact_and_controller_class_AST_identical(self):
  old=ast.parse((PARENT/'tools/rr_preload_diagnostic.py').read_text());new=ast.parse((SOURCE/'tools/rr_preload_diagnostic.py').read_text());meta=ast.parse((SOURCE/'tools/rr_preload_contract.py').read_text())
  def proposal(t):return ast.literal_eval(next(n.value for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='PROPOSAL' for x in n.targets)))
  self.assertEqual(proposal(old),proposal(meta))
  self.assertEqual(ast.dump(next(n for n in old.body if isinstance(n,ast.ClassDef))),ast.dump(next(n for n in new.body if isinstance(n,ast.ClassDef))))
  self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(meta)))
 def test_exact_narrow_source_delta_and_hash_bindings(self):
  import hashlib
  sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
  old=json.loads((PARENT/'campaign_source_hashes.json').read_text());new=json.loads((SOURCE/'campaign_source_hashes.json').read_text())
  for f,h in new.items():self.assertEqual(sha(SOURCE/f),h,f)
  self.assertEqual(set(f for f in new if new[f]!=old.get(f)),{'source_origin.json','tools/rr_preload_diagnostic.py','tools/directional_contract.py','tools/rr_preload_contract.py'})
  for f in ['wave_reference.py','run_directional_physics.py','screen_contract.py','screen_metrics.py','directional_metrics.py','solver_comparison.py','launch_directional_physics_spark.py','reference_residual.py','reference_residual_env.py']:
   self.assertEqual((PARENT/'tools'/f).read_bytes(),(SOURCE/'tools'/f).read_bytes(),f)
if __name__=='__main__':unittest.main()
