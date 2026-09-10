from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,importlib.util,json,sys,tempfile,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import launch_cold_spark as h
import run_cold_entry as e
ROOT=HERE.parents[1]
CONTRACT=ROOT/'tmp/direct_omni_recovery_001/baseline'
PARENT=ROOT/'tmp/reference_physics_adapter_009/source_009'

def module(path):
 s=importlib.util.spec_from_file_location('test_cold_contract',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def args(out):return SimpleNamespace(source=Path('/source-real'),checkpoint=Path('/original.pt'),contract=CONTRACT,supervisor_source=PARENT,output=out,isaaclab=Path('/IsaacLab'),host_freeze_sha256='host')
class HostTests(unittest.TestCase):
 def setUp(self):self.old=h.CONTRACT;h.CONTRACT=module(CONTRACT/'cold_contract.py')
 def tearDown(self):h.CONTRACT=self.old
 def test_exact_two_phase_cli_ro_aliases(self):
  a=args(Path('/out'))
  for phase in h.PHASES:
   c=h.command(a,'owned',phase)
   self.assertIn('/source-real:/source:ro',c);self.assertIn('/original.pt:/checkpoint/original.pt:ro',c)
   self.assertIn('/out:/output:rw',c);self.assertIn('/cold-host/run_cold_entry.py',c)
   k=c.index('--');self.assertEqual(c[k+1:],h.CONTRACT.runtime_arguments(phase))
  c=h.command(a,'owned','baseline');self.assertIn('/out/standing:/admission:ro',c)
  self.assertGreater(c.index('/out/standing:/output/standing:ro'),c.index('/out:/output:rw'))
  with self.assertRaises(ValueError):h.command(a,'owned','train')
 def test_parent_runtime_method_parity(self):
  m=json.loads((HERE/'LEGACY_RUNTIME_SHA256.json').read_text())
  n={k.removeprefix('isaaclab/hexapod_rl/'):v for k,v in m.items()}
  self.assertEqual(len(m),16);self.assertEqual(hashlib.sha256(json.dumps(n,sort_keys=True,separators=(',',':')).encode()).hexdigest(),h.LEGACY_RUNTIME_TREE)
 def test_supervisor_functions_unchanged(self):
  a=args(Path('/unused'));parent=h.load_supervisor(a)
  original=ast.parse((PARENT/'tools/launch_reference_physics_spark.py').read_text())
  import inspect
  for name in ['owned_container','run_owned']:
   orig=next(x for x in original.body if isinstance(x,ast.FunctionDef) and x.name==name)
   self.assertEqual(ast.dump(orig),ast.dump(ast.parse(inspect.getsource(getattr(parent,name))).body[0]))
  self.assertEqual(parent.RUNTIME_TREE,h.LEGACY_RUNTIME_TREE)
 def test_actual_supervisor_uses_cold_precontainer_validation(self):
  with tempfile.TemporaryDirectory() as t:
   a=args(Path(t));(a.output/'jobs').mkdir();(a.output/'logs').mkdir()
   coordination=a.output/'coordination.md';coordination.write_text('test');a.coordination_sha256=h.sha(coordination)
   parent=h.load_supervisor(a);called=[]
   actual=parent.verified_source
   def stop_after_validation(source):
    actual(source);called.append(source);raise ValueError('validated cold before container')
   with patch.object(parent,'COORDINATION',coordination),patch.object(parent,'preflight',return_value={}),patch.object(parent,'verified_source',side_effect=stop_after_validation),patch.object(h,'verify_tree') as vt,patch.object(h,'verify_legacy_runtime') as vr,patch.object(parent.os,'open',return_value=5),patch.object(parent.os,'close'),patch.object(parent.fcntl,'flock'),patch.object(parent.subprocess,'Popen') as popen:
    with self.assertRaisesRegex(ValueError,'validated cold'):parent.run_owned(a,'standing')
    vt.assert_called_once_with(a.source,'campaign_source_hashes.json',h.SOURCE_MAP);vr.assert_called_once_with(a.source);popen.assert_not_called()
   self.assertEqual(called,[a.source]);self.assertTrue(h.read(a.output/'jobs/standing.json')['cleanup_checked'])
 def test_truthful_policy_metadata(self):
  with tempfile.TemporaryDirectory() as t:
   a=args(Path(t));(a.output/'jobs').mkdir();p=h.load_supervisor(a)
   p.save(a.output/'jobs/baseline.json',{'phase':'baseline','no_policy_loaded':True})
   row=h.read(a.output/'jobs/baseline.json');self.assertIsNone(row['no_policy_loaded']);self.assertTrue(row['legacy_supervisor_no_policy_loaded']);self.assertTrue(row['policy_phase_requested'])
   (a.output/'baseline').mkdir();(a.output/'baseline/state.json').write_text(json.dumps({'status':'completed','checkpoint_sha256':'1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8'}))
   p.save(a.output/'jobs/baseline.json',{'phase':'baseline','no_policy_loaded':True})
   self.assertFalse(h.read(a.output/'jobs/baseline.json')['no_policy_loaded'])
 def test_readiness_after_construct_only(self):
  text='app = AppLauncher(args).app\ndef save(path,data):\n pass\n'
  tree=e.instrument(text);events=[]
  class App:
   def __init__(self,a):events.append('construct');self.app=1
  exec(compile(tree,'test','exec'),{'AppLauncher':App,'args':None,'_cold_app_ready':lambda:events.append('ready'),'_cold_bind_save':lambda x:x})
  self.assertEqual(events,['construct','ready'])
  def broken(a):raise RuntimeError('appfailed')
  events=[]
  with self.assertRaises(RuntimeError):exec(compile(tree,'test','exec'),{'AppLauncher':broken,'args':None,'_cold_app_ready':lambda:events.append('ready'),'_cold_bind_save':lambda x:x})
  self.assertEqual(events,[])
 def test_original_all_function_AST_preserved(self):
  text=(CONTRACT/'inputs/old_entry.py').read_text();orig=ast.parse(text);new=e.instrument(text)
  for node in orig.body:
   if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
    match=next(x for x in new.body if isinstance(x,type(node)) and x.name==node.name);self.assertEqual(ast.dump(node),ast.dump(match))
  self.assertEqual(len(new.body),len(orig.body)+2)
 def test_wrong_import_origin_rejects(self):
  with tempfile.TemporaryDirectory() as t:
   with patch.dict(sys.modules,{'hexapod_rl.env':SimpleNamespace(__file__='/wrong/env.py')}):
    with self.assertRaisesRegex(ValueError,'Wrong legacy module origin'):e.binding(Path(t),{})
 def test_noncanonical_runtime_inventory_rejects(self):
  with tempfile.TemporaryDirectory() as t:
   (Path(t)/'isaaclab/hexapod_rl').mkdir(parents=True)
   with self.assertRaisesRegex(ValueError,'inventory'):e.verify_runtime(Path(t))
 def _campaign(self,fail=None,alter=False):
  temp=tempfile.TemporaryDirectory();out=Path(temp.name);a=args(out);identity={'x':1};calls=[]
  class Contract:
   def validate_result(self,path,phase,i):
    if phase==fail:raise ValueError('gate failure')
    return {'phase':phase,'passed':True}
  def run(a,phase):
   calls.append(phase);(out/phase).mkdir();(out/phase/'state.json').write_text('{}')
   if phase=='baseline' and alter:(out/'standing/state.json').write_text('{"changed":true}')
  with patch.object(h,'CONTRACT',Contract()),patch.object(h,'PARENT',SimpleNamespace(run_owned=run)),patch.object(h,'verify_inputs',return_value=identity):
   try:result=h.run_campaign(a,identity)
   except Exception as ex:result=ex
  data=h.read(out/'campaign.json');temp.cleanup();return calls,result,data
 def test_failed_standing_never_allocates_baseline(self):
  calls,result,data=self._campaign(fail='standing');self.assertEqual(calls,['standing']);self.assertIsInstance(result,ValueError);self.assertEqual(data['status'],'failed')
 def test_two_phases_no_automatic_train(self):
  calls,result,data=self._campaign();self.assertEqual(calls,list(h.PHASES));self.assertTrue(data['baseline_complete']);self.assertFalse(data['policy_training_started']);self.assertFalse(data['Stage2_complete'])
 def test_standing_mutation_fails_final_campaign(self):
  calls,result,data=self._campaign(alter=True);self.assertEqual(calls,list(h.PHASES));self.assertEqual(data['status'],'failed');self.assertFalse(data['terminal_inputs_unchanged'])
if __name__=='__main__':unittest.main()
