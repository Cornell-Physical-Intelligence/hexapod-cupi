"""Exercise actual adapted run_owned with a fake clock and explicit Docker ownership."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import ast,hashlib,importlib.util,json,os,subprocess,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import launch_ppo_spark as h
import deadline_adapter as d
SUP=HERE.parent/'reference_physics_adapter_009/source_009'
class FakeClock:
 def __init__(self):self.now=0.
 def time(self):return 1000000+self.now
 def monotonic(self):return self.now
 def sleep(self,s):self.now+=s
class DeadlineTests(unittest.TestCase):
 def fixture(self,duration=10000,marker=True,stop=False,inspect_error=False,competitor=False):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td)
   for name in ('jobs','logs','canonical_ppo_smoke'):(out/name).mkdir()
   coord=out/'coordination.md';coord.write_text('exact fixture')
   a=NS(source=out/'source',supervisor_source=SUP,isaaclab=out,output=out,coordination_sha256=h.sha(coord),num_envs=32,standing_one=out/'prior',asset=out/'asset',admission=out/'admission')
   identity={'runtime_binding':{'runtime_tree_sha256':h.SOURCE_FREEZE}}
   parent=h.load_supervisor(a,identity);clock=FakeClock();calls=[];proc=NS(returncode=None);alive=[True]
   def poll():
    if proc.returncode is None and clock.now>=duration:proc.returncode=0;alive[0]=False
    return proc.returncode
   def terminate():calls.append('terminate');proc.returncode=-15
   proc.poll=poll;proc.terminate=terminate;proc.wait=lambda timeout:proc.returncode;proc.kill=lambda:None
   def popen(*args,stdout=None,**kwargs):
    if marker:stdout.write('REFERENCE_SCREEN_APP_READY\n');stdout.flush()
    if stop:(out/'stop.request').touch()
    return proc
   def owned(name,identity=None):
    if inspect_error:raise RuntimeError('unknown Docker ownership')
    return ('immutable-owned-id',alive[0])
   def subrun(command,**kwargs):
    calls.append(command)
    if command[1]=='stop':alive[0]=False
    return NS(returncode=0,stdout='PID\n5\n',stderr='')
   fake_os=NS(open=lambda *a:len([x for x in calls if isinstance(x,tuple)])+10,close=lambda fd:calls.append(('closed',fd)),O_RDONLY=os.O_RDONLY)
   (out/'canonical_ppo_smoke/state.json').write_text(json.dumps({'status':'completed','runtime_binding':identity['runtime_binding']}))
   with patch.object(parent,'time',clock),patch.object(parent,'COORDINATION',coord),patch.object(parent,'os',fake_os),patch.object(parent,'fcntl',NS(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None)),patch.object(parent,'preflight',return_value={}),patch.object(parent,'verified_source'),patch.object(parent,'command',return_value=['fake-owned']),patch.object(parent,'owned_container',side_effect=owned),patch.object(parent,'resources',return_value=([],64*1024**3)),patch.object(parent,'live_competitors',return_value=['competitor'] if competitor else []),patch.object(parent,'audit_contact_log',return_value={'passed':True}),patch.object(parent,'subprocess',NS(Popen=popen,run=subrun,STDOUT=-2,TimeoutExpired=subprocess.TimeoutExpired)):
    try:r=parent.run_owned(a,'canonical_ppo_smoke')
    except Exception as e:r=e
   return r,json.loads((out/'jobs/canonical_ppo_smoke.json').read_text()),clock.now,calls
 def test_complete700_seconds_does_not_hit_old600_bound(self):
  r,j,t,c=self.fixture(duration=700);self.assertIsInstance(r,dict);self.assertEqual(t,700);self.assertEqual(j['deadline_seconds'],1200);self.assertEqual(j['app_ready_deadline_seconds'],90);self.assertTrue(j['cleanup_checked']);self.assertEqual(j['supervisor_runtime_adapter']['schema'],d.SCHEMA)
 def test_timeout1200_retains_exact_owned_cleanup_and_locks(self):
  r,j,t,c=self.fixture();self.assertIsInstance(r,TimeoutError);self.assertIn('canonical_ppo_smoke phase exceeded 1200-second',str(r));self.assertEqual(t,1200);self.assertTrue(j['cleanup_checked']);self.assertIn(['docker','stop','--time','20','immutable-owned-id'],c);self.assertEqual(sum(isinstance(x,tuple) and x[0]=='closed' for x in c),2)
 def test_AppReady_still90(self):
  r,j,t,c=self.fixture(marker=False);self.assertIsInstance(r,TimeoutError);self.assertEqual(t,90);self.assertEqual(j['startup_failure_kind'],'no_reference_AppReady_by90s');self.assertTrue(j['cleanup_checked'])
 def test_stop_unknown_owner_and_competitor_still_fail_closed(self):
  r,j,t,c=self.fixture(stop=True);self.assertIsInstance(r,InterruptedError);self.assertEqual(j['status'],'stopped');self.assertTrue(j['cleanup_checked'])
  r,j,t,c=self.fixture(inspect_error=True);self.assertIsInstance(r,RuntimeError);self.assertFalse(j['cleanup_checked']);self.assertTrue(j['cleanup_requires_owner_review'])
  r,j,t,c=self.fixture(competitor=True);self.assertIsInstance(r,RuntimeError);self.assertIn('Unrelated CUDA',str(r));self.assertTrue(j['cleanup_checked'])
 def test_only_fixed_PPO_phase(self):
  self.assertEqual(d.phase_deadline_seconds(NS(),'canonical_ppo_smoke'),1200)
  for phase in ('standing','train','inspection',None):
   with self.assertRaises(ValueError):d.phase_deadline_seconds(NS(),phase)
 def test_original_cleanup_and_all_other_host_functions_unchanged(self):
  p=d.inspect_supervisor(SUP/'tools/launch_reference_physics_spark.py');s=p['adapted_function_source']
  for before,after in reversed(d.SEAMS):s=s.replace(after,before,1)
  self.assertEqual(s,p['original_function_source']);self.assertTrue(p['cleanup_AST_unchanged']);self.assertIn('app_ready_deadline = time.monotonic() + 90',s)
  def funcs(path):return {n.name:ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
  old=funcs(HERE/'inputs/PARENT_HOST002.py');new=funcs(HERE/'launch_ppo_spark.py')
  self.assertEqual(set(new)-set(old),{'load_deadline_adapter'})
  for k in old:
   if k!='load_supervisor':self.assertEqual(old[k],new[k],k)
 def test_changed_module_code_or_source_rejected(self):
  path=SUP/'tools/launch_reference_physics_spark.py';sys.path.insert(0,str(path.parent));parent=h.load_module('_original_for_mutation',path)
  original=parent.run_owned.__code__;parent.run_owned.__code__=original.replace(co_consts=tuple(601 if type(v)is int and v==600 else v for v in original.co_consts))
  with self.assertRaisesRegex(ValueError,'Loaded run_owned differs'):d.install(parent,path)
  self.assertNotIn(d.HELPER_NAME,vars(parent))
  with tempfile.TemporaryDirectory() as td:
   wrong=Path(td)/'supervisor.py';wrong.write_bytes(path.read_bytes()+b'\n')
   with self.assertRaisesRegex(ValueError,'exact frozen'):d.inspect_supervisor(wrong)
 def test_full_module_compilation_without_executing_output(self):
  path=SUP/'tools/launch_reference_physics_spark.py';sys.path.insert(0,str(path.parent));parent=h.load_module('_original_for_compile',path);owned=parent.owned_container.__code__;calls=[];original=compile
  def spy(source,filename,mode,*args,**kwargs):calls.append((source,kwargs));return original(source,filename,mode,*args,**kwargs)
  with patch('builtins.compile',side_effect=spy):m=d.install(parent,path)
  self.assertEqual(sum(isinstance(s,str) and s==path.read_text() and k.get('dont_inherit')is True for s,k in calls),1);self.assertEqual(parent.owned_container.__code__,owned);self.assertIs(parent.run_owned.__globals__,vars(parent));self.assertNotIn('_original_module_source',m)
  with self.assertRaises(ValueError):d.install(parent,path)
 def test_file_relative_guard_import_no_path_injection(self):
  code="import importlib.util,sys;s=importlib.util.spec_from_file_location('h',"+repr(str(HERE/'launch_ppo_spark.py'))+");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);d=m.load_deadline_adapter();assert d.SCHEMA=='canonical_ppo_supervisor_deadline_v1';assert not any(k in sys.modules for k in ('numpy','torch','isaaclab'))"
  with tempfile.TemporaryDirectory() as td:subprocess.run([sys.executable,'-B','-S','-c',code],cwd=td,check=True,capture_output=True)
if __name__=='__main__':unittest.main()
