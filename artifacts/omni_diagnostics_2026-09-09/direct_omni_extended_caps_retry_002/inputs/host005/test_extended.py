from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast,copy,hashlib,importlib.util,json,os,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import launch_train_spark as h
import deadline_adapter as d
SUP=HERE.parents[1]/'tmp/reference_physics_adapter_009/source_009'
NATIVE=HERE.parents[1]/'tmp/direct_omni_recovery_001/native_005'

class FakeClock:
 def __init__(self):self.now=0.
 def time(self):return 1000000+self.now
 def monotonic(self):return self.now
 def sleep(self,seconds):self.now+=seconds

class ExtendedTests(unittest.TestCase):
 def fixture(self,allocation='extended',phase='train',duration=10000,marker=True,stop=False,inspect_error=False):
  with tempfile.TemporaryDirectory() as td:
   out=Path(td);(out/'jobs').mkdir();(out/'logs').mkdir();(out/phase).mkdir()
   coord=out/'coordination.md';coord.write_text('exact fixture')
   a=SimpleNamespace(allocation=allocation,branch='caps' if allocation!='smoke' else 'quiet_priority',smoke=None if allocation=='smoke' else out/'prior',source=out/'source',supervisor_source=SUP,isaaclab=out,output=out,coordination_sha256=h.sha(coord))
   parent=h.load_supervisor(a);clock=FakeClock();calls=[];proc=SimpleNamespace(returncode=None);alive=[True]
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
   def subprocess_run(command,**kwargs):
    calls.append(command)
    if command[1]=='stop':alive[0]=False
    return SimpleNamespace(returncode=0,stdout='PID\n5\n',stderr='')
   fake_os=SimpleNamespace(open=lambda path,flags:len([c for c in calls if isinstance(c,tuple)])+10,
                          close=lambda fd:calls.append(('closed',fd)),O_RDONLY=os.O_RDONLY)
   report_state={'status':'completed','runtime_binding':{'runtime_tree_sha256':h.LEGACY_RUNTIME_TREE}}
   (out/phase/'state.json').write_text(json.dumps(report_state))
   with patch.object(parent,'time',clock),patch.object(parent,'COORDINATION',coord),patch.object(parent,'os',fake_os),patch.object(parent,'fcntl',SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *x:None)),patch.object(parent,'preflight',return_value={}),patch.object(parent,'verified_source'),patch.object(parent,'command',return_value=['fake-owned']),patch.object(parent,'owned_container',side_effect=owned),patch.object(parent,'resources',return_value=([],64*1024**3)),patch.object(parent,'live_competitors',return_value=[]),patch.object(parent,'audit_contact_log',return_value={'passed':True}),patch.object(parent,'subprocess',SimpleNamespace(Popen=popen,run=subprocess_run,STDOUT=-2,TimeoutExpired=__import__('subprocess').TimeoutExpired)):
    try:result=parent.run_owned(a,phase)
    except Exception as e:result=e
   receipt=json.loads((out/'jobs'/(phase+'.json')).read_text())
   return result,receipt,clock.now,calls

 def test_extended_train_can_run_beyond600_and_has1800receipt(self):
  result,r,t,c=self.fixture(duration=700);self.assertIsInstance(result,dict);self.assertEqual(r['deadline_seconds'],1800);self.assertEqual(r['app_ready_deadline_seconds'],90);self.assertTrue(r['cleanup_checked']);self.assertEqual(t,700);self.assertEqual(r['supervisor_runtime_adapter']['schema'],d.SCHEMA)
 def test_extended_train_timeout1800_and_exact_owned_cleanup(self):
  result,r,t,c=self.fixture();self.assertIsInstance(result,TimeoutError);self.assertIn('train phase exceeded 1800-second',str(result));self.assertEqual(t,1800);self.assertTrue(r['cleanup_checked']);self.assertIn(['docker','stop','--time','20','immutable-owned-id'],c);self.assertEqual(len([x for x in c if isinstance(x,tuple) and x[0]=='closed']),2)
 def test_extended_all_five_nontraining_phases_timeout600(self):
  for phase in d.PHASES:
   if phase=='train':continue
   result,r,t,c=self.fixture(phase=phase);self.assertIsInstance(result,TimeoutError);self.assertEqual(r['deadline_seconds'],600);self.assertEqual(t,600);self.assertIn(phase+' phase exceeded 600-second',str(result))
 def test_smoke_and_pilot_keep_original600and_original_message(self):
  for allocation in ('smoke','pilot'):
   result,r,t,c=self.fixture(allocation=allocation);self.assertIsInstance(result,TimeoutError);self.assertEqual(t,600);self.assertEqual(r['deadline_seconds'],600);self.assertEqual(str(result),'Standing phase exceeded ten-minute bound');self.assertNotIn('supervisor_runtime_adapter',r)
 def test_AppReady_still90_for_extended(self):
  result,r,t,c=self.fixture(marker=False);self.assertIsInstance(result,TimeoutError);self.assertEqual(t,90);self.assertEqual(r['startup_failure_kind'],'no_reference_AppReady_by90s');self.assertTrue(r['cleanup_checked'])
 def test_stop_request_and_unknown_ownership_still_fail_closed(self):
  result,r,t,c=self.fixture(stop=True);self.assertIsInstance(result,InterruptedError);self.assertEqual(r['status'],'stopped');self.assertTrue(r['cleanup_checked'])
  result,r,t,c=self.fixture(inspect_error=True);self.assertIsInstance(result,RuntimeError);self.assertFalse(r['cleanup_checked']);self.assertTrue(r['cleanup_requires_owner_review'])
 def test_original_and_changed_source_rejection(self):
  source=SUP/'tools/launch_reference_physics_spark.py';bound=hashlib.sha256(source.read_bytes()).hexdigest();proof=d.inspect_supervisor(source)
  self.assertEqual(proof['source_substitutions'],3);self.assertEqual(proof['phase_deadline_seconds']['train'],1800);self.assertTrue(proof['cleanup_AST_unchanged'])
  with tempfile.TemporaryDirectory() as td:
   wrong=Path(td)/'supervisor.py';wrong.write_bytes(source.read_bytes()+b'\n')
   with self.assertRaisesRegex(ValueError,'exact frozen'):d.inspect_supervisor(wrong)
  self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),bound)
 def test_missing_or_duplicate_seam_rejected_even_under_test_hash(self):
  source=(SUP/'tools/launch_reference_physics_spark.py').read_text()
  for changed in [source.replace('deadline_seconds=600','deadline_seconds=601'),source.replace('deadline_seconds=600','deadline_seconds=600, other_deadline_seconds=600')]:
   with tempfile.TemporaryDirectory() as td:
    p=Path(td)/'s.py';p.write_text(changed)
    with patch.object(d,'SUPERVISOR_SHA256',hashlib.sha256(p.read_bytes()).hexdigest()),self.assertRaisesRegex(ValueError,'seam'):d.inspect_supervisor(p)
 def test_wrong_loaded_code_and_duplicate_adapter_rejected(self):
  a=SimpleNamespace(allocation='pilot',supervisor_source=SUP);parent=h.load_supervisor(a)
  parent.run_owned=lambda *a:None
  with self.assertRaisesRegex(ValueError,'own module globals'):d.install(parent,SUP/'tools/launch_reference_physics_spark.py')
  parent=h.load_supervisor(a);d.install(parent,SUP/'tools/launch_reference_physics_spark.py')
  with self.assertRaisesRegex(ValueError,'differs'):d.install(parent,SUP/'tools/launch_reference_physics_spark.py')
 def test_no_adaptation_allowed_for_unknown_phase_or_other_allocation(self):
  for allocation,phase in [('pilot','train'),('smoke','train'),('extended','unknown')]:
   with self.assertRaises(ValueError):d.phase_deadline_seconds(SimpleNamespace(allocation=allocation),phase)
 def test_actual_finally_AST_identical_and_only3_source_seams(self):
  proof=d.inspect_supervisor(SUP/'tools/launch_reference_physics_spark.py');adapted=proof['adapted_function_source']
  for old,new in reversed(d.SEAMS):adapted=adapted.replace(new,old,1)
  self.assertEqual(adapted,proof['original_function_source'])
  self.assertIn('app_ready_deadline = time.monotonic() + 90',adapted)
 def test_extended_selected_phases_and_readonly_smoke_mount(self):
  sys.path.insert(0,str(NATIVE));spec=importlib.util.spec_from_file_location('extended_test_contract',NATIVE/'direct_contract.py');contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
  a=SimpleNamespace(allocation='extended',branch='caps',smoke=Path('/proof-smoke'),source=Path('/source'),checkpoint=Path('/original.pt'),output=Path('/new-output'))
  self.assertEqual(h.selected_phases(a),h.PHASES['pilot'])
  with patch.object(h,'CONTRACT',contract):
   c=h.command(a,'only-owned','train');self.assertIn('/proof-smoke:/smoke:ro',c);self.assertEqual(c[c.index('--iterations')+1],'500');self.assertIn('/source:/source:ro',c)
  a.smoke=None
  with self.assertRaisesRegex(ValueError,'smoke'):h.selected_phases(a)
 def test_actual_guard_style_import_without_host_sys_path(self):
  import subprocess
  code='import importlib.util,sys; p='+repr(str(HERE/'launch_train_spark.py'))+'; s=importlib.util.spec_from_file_location("host_guard_style",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert "deadline_adapter" not in sys.modules; d=m.load_deadline_adapter(); assert d.SCHEMA=="direct315_extended_supervisor_deadlines_v1"; print("guard-style import passed")'
  with tempfile.TemporaryDirectory() as cwd:
   result=subprocess.run([sys.executable,'-S','-B','-c',code],cwd=cwd,text=True,capture_output=True)
  self.assertEqual(result.returncode,0,result.stderr);self.assertIn('passed',result.stdout)

 def test_count_from_selection_receipt_and_accepted_result(self):
  with tempfile.TemporaryDirectory() as td:
   a=SimpleNamespace(allocation='extended',branch='caps',output=Path(td));(a.output/'train').mkdir();r=a.output/'train/training_receipt.json'
   identity={'selection':{'allocation':'extended','branch':'caps','updates':500}};accepted={'phase':'train','updates_completed':500};r.write_text(json.dumps({'complete':True,'updates_completed':500}))
   self.assertEqual(h.validated_completed_updates(a,identity,accepted),500)
   for bad in [50,True,None]:
    with self.assertRaises(ValueError):h.validated_completed_updates(a,identity,{**accepted,'updates_completed':bad})
   r.write_text(json.dumps({'complete':True,'updates_completed':50}))
   with self.assertRaises(ValueError):h.validated_completed_updates(a,identity,accepted)

if __name__=='__main__':unittest.main()
