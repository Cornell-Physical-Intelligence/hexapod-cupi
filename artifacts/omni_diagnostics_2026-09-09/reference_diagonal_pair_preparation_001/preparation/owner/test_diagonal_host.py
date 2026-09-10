import ast,importlib.util,sys,unittest,json,tempfile,copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source_diagonal_pair_001/tools'));sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('pair_host',HERE/'launch_pair_physics_spark.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
import pair_screen_contract as c
class PairHostTests(unittest.TestCase):
 def test_exact_three_phases_and_named_cases(self):
  for phase,count in [('standing','1000'),('lf_rr','1300'),('lr_rf','1300')]:
   a=h.command(Path('/source'),Path('/out'),'owned',phase);self.assertEqual(a[a.index('--steps')+1],count)
   self.assertIn('/source:/workspace/hexapod:ro',a);self.assertIn('/out/inputs/study:/study:ro',a)
   self.assertEqual('--admission' in a,phase!='standing')
  for phase in ['wave','training','origin_a','pair','lm_rm']:
   with self.assertRaises(ValueError):h.command(Path('/source'),Path('/out'),'owned',phase)
 def test_cleanup_and_limits_unchanged(self):
  old=ast.parse((HERE.parent/'reference_pair_physics_adapter_001/source_pair_001/tools/launch_pair_physics_spark.py').read_text());new=ast.parse((HERE/'launch_pair_physics_spark.py').read_text())
  for name in ('run_owned','owned_container','tree_hashes'):
   a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name);b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name)
   self.assertEqual(ast.dump(a),ast.dump(b))
 def test_no_origin_reset_or_wave_dispatch(self):
  s=(HERE/'run_pair_physics.py').read_text();self.assertNotIn('patched_factory',s);self.assertNotIn('WaveContactReference',s)
  host=(HERE/'launch_pair_physics_spark.py').read_text();self.assertNotIn('run_owned(args, "wave")',host);self.assertIn('run_owned(args, case)',host);self.assertEqual(c.PAIR_PROTOCOL['case_order'],['lf_rr','lr_rf'])
 def test_fresh_exact32_quiet_required(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'admission.json';a=SimpleNamespace(mode='pair',pair_case='lf_rr',num_envs=1,steps=1300,admission=p);identity={'source':'diagonal001'}
   receipt={'mode':'standing','identity':identity,'status':'completed','gate':{'passed':True,'num_envs':32,'control_steps':1000,'all_replica_quiet':{'passed':True}}}
   with patch.object(c,'standing_preflight',return_value=(identity,{})):
    p.write_text(json.dumps(receipt));self.assertEqual(c.preflight(a,Path(td))[0]['standing_identity'],identity)
    for kind in ['oldsource','quiet','replicas']:
     bad=copy.deepcopy(receipt)
     if kind=='oldsource':bad['identity']={'source':'009'}
     elif kind=='quiet':bad['gate']['all_replica_quiet']['passed']=False
     else:bad['gate']['num_envs']=1
     p.write_text(json.dumps(bad))
     with self.assertRaises(ValueError):c.preflight(a,Path(td))
if __name__=='__main__':unittest.main()
