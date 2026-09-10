from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,importlib.util,json,sys,tempfile,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import launch_train_spark as h
import run_train_entry as e
ROOT=HERE.parents[1]
CONTRACT=ROOT/'tmp/direct_omni_recovery_001/native_005'
OLD=ROOT/'tmp/direct_omni_recovery_001/baseline'
PARENT=ROOT/'tmp/reference_physics_adapter_009/source_009'

def module(path):
 s=importlib.util.spec_from_file_location('test_cold_contract',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def args(out):return SimpleNamespace(source=Path('/source-real'),checkpoint=Path('/original.pt'),contract=CONTRACT,supervisor_source=PARENT,output=out,isaaclab=Path('/IsaacLab'),host_freeze_sha256='host',allocation='smoke',branch='quiet_priority',smoke=None)
class HostTests(unittest.TestCase):
 def setUp(self):
  self.old=h.CONTRACT;sys.path.insert(0,str(CONTRACT));h.CONTRACT=module(CONTRACT/'direct_contract.py')
 def tearDown(self):h.CONTRACT=self.old
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
  text=(OLD/'inputs/old_entry.py').read_text();orig=ast.parse(text);new=e.instrument(text)
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

 def test_exact_allocations_and_completed_ro_aliases(self):
  with tempfile.TemporaryDirectory() as t:
   out=Path(t);a=args(out);a.allocation='pilot';a.branch='curriculum';a.smoke=Path('/old-smoke')
   (out/'train/policy').mkdir(parents=True);checkpoint=out/'train/policy/final.pt';checkpoint.write_bytes(b'fake_test_weights')
   (out/'train/training_receipt.json').write_text(json.dumps({'complete':True,'final_checkpoint_sha256':h.sha(checkpoint)}))
   for phase in h.selected_phases(a):
    c=h.command(a,'owned',phase)
    self.assertIn('/source-real:/source:ro',c);self.assertIn('/original.pt:/checkpoint/original.pt:ro',c)
    self.assertIn('/old-smoke:/smoke:ro',c);self.assertIn('/train-host/run_train_entry.py',c)
    self.assertEqual(c[c.index('--')+1:],h.CONTRACT.runtime_arguments(phase,'pilot','curriculum'))
    for previous in h.selected_phases(a)[:h.selected_phases(a).index(phase)]:
     ro=str(out/previous)+':/output/'+previous+':ro'
     self.assertGreater(c.index(ro),c.index(str(out)+':/output:rw'))
    if phase.startswith('final_'):self.assertIn(str(checkpoint)+':/checkpoint/evaluated.pt:ro',c)
    if phase.startswith('initial_'):self.assertIn('/original.pt:/checkpoint/evaluated.pt:ro',c)
   with self.assertRaises(ValueError):h.command(a,'owned','unallocated')

 def test_smoke_keeps_final_diagnostics_and_requires_quiet_priority(self):
  a=args(Path('/out'));self.assertEqual(h.selected_phases(a),('standing','train','final_constant','final_stop'))
  a.branch='curriculum'
  with self.assertRaises(ValueError):h.selected_phases(a)
  a.branch='quiet_priority';a.allocation='pilot'
  with self.assertRaisesRegex(ValueError,'smoke'):h.selected_phases(a)

 def test_final_checkpoint_must_match_receipt(self):
  with tempfile.TemporaryDirectory() as t:
   a=args(Path(t));(a.output/'train/policy').mkdir(parents=True);(a.output/'train/policy/final.pt').write_bytes(b'bad')
   (a.output/'train/training_receipt.json').write_text(json.dumps({'complete':True,'final_checkpoint_sha256':'0'*64}))
   with self.assertRaisesRegex(ValueError,'differs'):h.phase_checkpoint(a,'final_constant')

 def test_truthful_train_loading_metadata(self):
  with tempfile.TemporaryDirectory() as t:
   a=args(Path(t));(a.output/'jobs').mkdir();(a.output/'train').mkdir();p=h.load_supervisor(a)
   a.phase_expected_checkpoints={'train':h.ORIGINAL_CHECKPOINT}
   p.save(a.output/'jobs/train.json',{'phase':'train','no_policy_loaded':True})
   self.assertIsNone(h.read(a.output/'jobs/train.json')['no_policy_loaded'])
   (a.output/'train/state.json').write_text(json.dumps({'status':'completed','checkpoint_sha256':'new trained SHA'}))
   (a.output/'train/repair_initialization.json').write_text(json.dumps({'checkpoint_sha256':h.ORIGINAL_CHECKPOINT}))
   p.save(a.output/'jobs/train.json',{'phase':'train','no_policy_loaded':True})
   self.assertFalse(h.read(a.output/'jobs/train.json')['no_policy_loaded'])
   (a.output/'train/state.json').write_text('broken')
   p.save(a.output/'jobs/train.json',{'phase':'train','no_policy_loaded':True})
   self.assertIn('policy_metadata_read_error',h.read(a.output/'jobs/train.json'))

 def _campaign(self,fail=None,alter=False,allocation='smoke',mutate_smoke=False,bad_accepted_count=False):
  temp=tempfile.TemporaryDirectory();out=Path(temp.name);a=args(out);a.allocation=allocation
  if allocation in ('pilot','extended'):a.smoke=out/'prior_smoke';a.smoke.mkdir();(a.smoke/'receipt.json').write_text('{}')
  a.checkpoint=out/'original.pt';a.checkpoint.write_bytes(b'original');identity={'x':1,'selection':{'allocation':allocation,'branch':a.branch,'updates':{'smoke':2,'pilot':50,'extended':500}[allocation]}};calls=[]
  class Contract:
   def validate_result(self,path,phase,i,expected_checkpoint_sha256=None):
    if phase==fail:raise ValueError('gate failure')
    result={'phase':phase,'complete':True}
    if phase=='train':result['updates_completed']=50 if bad_accepted_count else {'smoke':2,'pilot':50,'extended':500}[allocation]
    if phase=='train':result['optimizer_diagnostics']={'schema':'direct315_actor_gradients_v1','updates':{'smoke':2,'pilot':50,'extended':500}[allocation],'minibatches':40 if allocation=='smoke' else 1000,'sparse_actor_gradient_rows':2 if allocation=='smoke' else 8}
    return result
  def run(a,phase):
   calls.append(phase);(out/phase).mkdir();(out/phase/'state.json').write_text('{}')
   if phase=='train':
    (out/phase/'policy').mkdir();cp=out/phase/'policy/final.pt';cp.write_bytes(b'trained')
    (out/phase/'training_receipt.json').write_text(json.dumps({'complete':True,'final_checkpoint_sha256':h.sha(cp),'updates_completed':{'smoke':2,'pilot':50,'extended':500}[allocation]}))
   if phase=='train' and mutate_smoke:(a.smoke/'receipt.json').write_text('{"changed":true}')
   if phase=='final_constant' and alter:(out/'train/state.json').write_text('{"changed":true}')
  with patch.object(h,'CONTRACT',Contract()),patch.object(h,'PARENT',SimpleNamespace(run_owned=run)),patch.object(h,'verify_inputs',return_value=identity),patch.object(h,'ORIGINAL_CHECKPOINT',h.sha(a.checkpoint)):
   try:result=h.run_campaign(a,identity)
   except Exception as ex:result=ex
  data=h.read(out/'campaign.json');temp.cleanup();return calls,result,data

 def test_failed_standing_never_trains(self):
  calls,result,data=self._campaign(fail='standing');self.assertEqual(calls,['standing']);self.assertIsInstance(result,ValueError);self.assertEqual(data['status'],'failed');self.assertFalse(data['training_attempted'])

 def test_complete_smoke_no_automatic_pilot(self):
  calls,result,data=self._campaign();self.assertEqual(calls,list(h.PHASES['smoke']));self.assertEqual(data['PPO_updates_completed'],2);self.assertFalse(data['automatic_continuation']);self.assertFalse(data['Stage2_complete'])

 def test_final_failure_preserves_training_count(self):
  calls,result,data=self._campaign(fail='final_constant');self.assertEqual(calls,['standing','train','final_constant']);self.assertEqual(data['PPO_updates_completed'],2);self.assertEqual(data['observed_training']['observed_updates_completed'],2);self.assertEqual(data['status'],'failed')

 def test_completed_train_readonly_integrity(self):
  calls,result,data=self._campaign(alter=True);self.assertEqual(data['status'],'failed');self.assertFalse(data['terminal_inputs_unchanged'])

 def test_pilot_exact_six_phase_order(self):
  calls,result,data=self._campaign(allocation='pilot');self.assertEqual(calls,list(h.PHASES['pilot']));self.assertEqual(data['PPO_updates_completed'],50)

 def test_extended_exact_six_phases_and500accepted_updates(self):
  calls,result,data=self._campaign(allocation='extended');self.assertEqual(calls,list(h.PHASES['extended']));self.assertEqual(data['PPO_updates_completed'],500);self.assertEqual(data['observed_training']['observed_updates_completed'],500);self.assertFalse(data['Stage2_complete']);self.assertFalse(data['automatic_continuation'])

 def test_extended_changed_smoke_aborts_and_never_accepts_training(self):
  calls,result,data=self._campaign(allocation='extended',mutate_smoke=True);self.assertIsInstance(result,ValueError);self.assertEqual(calls,['standing','initial_constant','initial_stop','train']);self.assertEqual(data['PPO_updates_completed'],0);self.assertEqual(data['observed_training']['observed_updates_completed'],500);self.assertFalse(data['terminal_inputs_unchanged']);self.assertNotIn('train',data['accepted_phases'])

 def test_extended_wrong_accepted_count_is_not500success(self):
  calls,result,data=self._campaign(allocation='extended',bad_accepted_count=True);self.assertIsInstance(result,ValueError);self.assertEqual(data['PPO_updates_completed'],0);self.assertEqual(data['observed_training']['observed_updates_completed'],500);self.assertNotIn('train',data['accepted_phases']);self.assertEqual(data['status'],'failed')

 def test_compact_optimizer_summary_preserved_in_campaign(self):
  for allocation,updates,minibatches,sparse in [('smoke',2,40,2),('pilot',50,1000,8)]:
   calls,result,data=self._campaign(allocation=allocation)
   self.assertEqual(data['accepted_phases']['train']['optimizer_diagnostics'],{'schema':'direct315_actor_gradients_v1','updates':updates,'minibatches':minibatches,'sparse_actor_gradient_rows':sparse})
   self.assertFalse(data['Stage2_complete'])

 def test_quiet_priority_native_training_argv_for_both_allocations(self):
  for allocation,updates in [('smoke',2),('pilot',50)]:
   a=args(Path('/out'));a.allocation=allocation;a.smoke=Path('/same-source-quiet-smoke') if allocation=='pilot' else None
   c=h.command(a,'exact-owned-training','train')
   argv=c[c.index('--')+1:]
   self.assertEqual(argv,h.CONTRACT.runtime_arguments('train',allocation,'quiet_priority'))
   self.assertEqual(argv[argv.index('--direct-branch')+1],'quiet_priority')
   self.assertEqual(argv[argv.index('--iterations')+1],str(updates))
   self.assertIn('/source-real:/source:ro',c)

 def test_actual_native_optimizer_diagnostic_summary_contract(self):
  for allocation in ('smoke','pilot'):
   selected=h.CONTRACT.selection('train',allocation,'quiet_priority',None,None)
   updates=[]
   for update in range(1,selected['updates']+1):
    rows=[]
    for mb in range(1,21):
     g=None
     if update in (1,10,25,50) and mb in (1,20):
      g={'norms':dict.fromkeys(('ppo_actor','quiet_temporal','moving_temporal','spatial'),0.),'component_sum_norm':0.,'combined_actor_before_clip':0.,'combined_actor_after_clip':0.,'ppo_quiet_cosine':None}
     rows.append({'update':update,'minibatch':mb,'kl_mean':0.,'learning_rate_before':5e-5,'learning_rate_after':5e-5,'pair_counts':{'valid_pairs':10,'quiet_pairs':4,'moving_pairs':6},'gradient':g})
    updates.append({'completed_update':update,'minibatches':rows,'learning_rate':5e-5})
   receipt={'optimizer_diagnostics_schema':'direct315_actor_gradients_v1','optimizer_updates':updates}
   summary=h.CONTRACT.validate_optimizer_diagnostics(receipt,selected)
   self.assertEqual(summary,{'schema':'direct315_actor_gradients_v1','updates':selected['updates'],'minibatches':20*selected['updates'],'sparse_actor_gradient_rows':2 if allocation=='smoke' else 8})
   receipt['optimizer_diagnostics_schema']='historical receipt without native004 diagnostics'
   with self.assertRaises(ValueError):h.CONTRACT.validate_optimizer_diagnostics(receipt,selected)

 def test_actual_native_entry_only_metadata_insertions(self):
  builder=module(CONTRACT/'build_source.py');text=builder.patch_entry((OLD/'inputs/old_entry.py').read_text())
  old=ast.parse(text);new=e.instrument(text)
  for node in old.body:
   if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
    transformed=next(x for x in new.body if isinstance(x,type(node)) and x.name==node.name)
    self.assertEqual(ast.dump(node),ast.dump(transformed))
  self.assertEqual(len(new.body),len(old.body)+2)

 def test_stdlib_only_host_wrapper_and_contract_import(self):
  import subprocess
  code='import sys;sys.path[:0]='+repr([str(HERE),str(CONTRACT)])+';import launch_train_spark,run_train_entry,direct_contract;assert "numpy" not in sys.modules;assert "torch" not in sys.modules;print("stdlib host import passed")'
  result=subprocess.run([sys.executable,'-S','-B','-c',code],text=True,capture_output=True)
  self.assertEqual(result.returncode,0,result.stderr);self.assertIn('passed',result.stdout)

if __name__=='__main__':unittest.main()
