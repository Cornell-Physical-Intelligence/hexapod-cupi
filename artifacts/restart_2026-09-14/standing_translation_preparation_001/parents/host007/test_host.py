import importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch,Mock
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('canonical_host',HERE/'launch_standing_spark.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
class HostTests(unittest.TestCase):
 def args(self,r):return NS(source=r/'source',asset=r/'asset',admission=r/'admission',supervisor_source=r/'supervisor',output=r/'output',host_freeze_sha256='a'*64,isaaclab=r/'isaaclab',num_envs=1,standing_one=None)
 def identity(self):return {'schema':'canonical_native_standing_v1','inspector_freeze_sha256':'b'*64,'runtime_binding':{'runtime_tree_sha256':'b'*64,'scope':'canonical_provisional_native_standing_only'},'training_allowed':False,'physical_admission':False,'steps':8000,'controls':1000,'settle_controls':200,'num_envs':1,'dt':.0025,'gravity':[0.,0.,-9.81],'actuation_state_sha256':h.ADMISSION_STATE,'ground':True,'control_dt':.02,'substeps_per_control':8}
 def test_pending_source_fails_before_own_bundle_or_process(self):
  with patch.object(h,'SOURCE_FREEZE','PENDING_TEST_BINDING'),patch.object(h,'verify_own_bundle') as own,patch.object(h.subprocess if hasattr(h,'subprocess') else subprocess,'run') as run:
   with self.assertRaisesRegex(ValueError,'pending'):h.verify_inputs(self.args(Path('/a')))
   own.assert_not_called();run.assert_not_called()
 def test_stdlib_import(self):
  subprocess.run([sys.executable,'-S','-B','-c',"import launch_standing_spark,sys;assert not any(k in sys.modules for k in ('torch','numpy','isaaclab'))"],cwd=HERE,check=True,capture_output=True)
 def test_only_standing_no_checkpoint_asset_and_source_readonly(self):
  args=self.args(Path('/a'));c=h.command(args,'exact-owned','standing');mounts=[c[i+1] for i,v in enumerate(c) if v=='-v']
  self.assertEqual(mounts,['/a/output:/output:rw','/a/source:/standing:ro','/a/asset:/asset:ro','/a/admission:/admission:ro'])
  self.assertIn('/standing/run_standing.py',c);self.assertNotIn('--checkpoint',c);self.assertEqual(c[c.index('--num-envs')+1],'1')
  for phase in ('query','train','recording','wave'):
   with self.assertRaises(ValueError):h.command(args,'exact-owned',phase)
 def test_reused_overlapping_and_dangling_output_reject(self):
  with tempfile.TemporaryDirectory() as t:
   a=self.args(Path(t));a.output.mkdir()
   with self.assertRaises(ValueError):h.require_fresh_output(a)
   a.output=Path(t)/'link';a.output.symlink_to(Path(t)/'missing')
   with self.assertRaises(ValueError):h.require_fresh_output(a)
   a.output=a.asset/'under'
   with self.assertRaises(ValueError):h.require_fresh_output(a)
   a.output=a.admission/'under'
   with self.assertRaises(ValueError):h.require_fresh_output(a)
 def test_exact_canonical_selection_and_nine_asset_map(self):
  selector=h.read(HERE/'inputs/active_model.json');m=h.read(HERE/'inputs/ASSET_SHA256.json');self.assertEqual(h.sha(HERE/'inputs/active_model.json'),h.SELECTOR_SHA256)
  self.assertEqual(len(m),9);self.assertEqual(m['robot.usda'],selector['usd_sha256']);self.assertEqual(m['source/source.urdf'],selector['urdf']['sha256']);self.assertEqual(m['source/model.json'],selector['model']['sha256'])
  self.assertEqual(selector['model']['sha256'],'7126935b35398d2a7f03271335a8a1b8f2059354af2b19d9474f22b8e73e4881')
 def test_real_contract_identity_shape_and_phase_only(self):
  a=self.args(Path('/a'));i=self.identity();m=NS(__file__=str(a.source/'standing_contract.py'),verify_inputs=lambda args:i)
  selector=h.read(HERE/'inputs/active_model.json')
  def digest(p):
   p=Path(p)
   if p.name=='active_model.json':return h.SELECTOR_SHA256
   if p.name=='launch_reference_physics_spark.py':return h.SUPERVISOR_CODE
   return {'robot.usda':selector['usd_sha256'],'model.json':selector['model']['sha256'],'source.urdf':selector['urdf']['sha256']}[p.name]
  with patch.object(h,'SOURCE_FREEZE','b'*64),patch.object(h,'verify_own_bundle'),patch.object(h,'verify_tree'),patch.object(h,'sha',side_effect=digest),patch.object(h,'CONTRACT',m):
   self.assertEqual(h.verify_inputs(a),i)
   for key,value in [('training_allowed',True),('physical_admission',True),('steps',1000),('schema','old_C_study'),('actuation_state_sha256','f'*64),('ground',False),('num_envs',32)]:
    before=i[key];i[key]=value
    with self.assertRaises(ValueError):h.verify_inputs(a)
    i[key]=before
 def test_runtime_old_C_identity_rejects(self):
  self.assertNotIn('abe4e354',(HERE/'launch_standing_spark.py').read_text())
  self.assertIn("identity.get('runtime_binding')!={'runtime_tree_sha256':SOURCE_FREEZE,'scope':'canonical_provisional_native_standing_only'}",(HERE/'launch_standing_spark.py').read_text())
 def test_canonical_verifier_and_explicit_deadline_installer(self):
  a=self.args(Path('/a'));parent=NS(save=lambda *args:None,run_owned=Mock(),owned_container=Mock());run=parent.run_owned;owned=parent.owned_container;i=self.identity()
  with patch.object(h,'load_module',return_value=parent),patch.object(h,'load_deadline_adapter',return_value=NS(install=Mock())) as adapter,patch.object(sys,'path',list(sys.path)),patch.object(h,'verify_tree') as verify:
   got=h.load_supervisor(a,i);self.assertIs(got.run_owned,run);self.assertIs(got.owned_container,owned);self.assertEqual(got.RUNTIME_TREE,i['runtime_binding']['runtime_tree_sha256']);got.verified_source(a.source);verify.assert_called_once_with(a.source,'FREEZE_SHA256.json',h.SOURCE_FREEZE)
   adapter.return_value.install.assert_called_once_with(parent,a.supervisor_source/'tools/launch_reference_physics_spark.py')
   with self.assertRaises(ValueError):got.verified_source(a.asset)
 def test_failed_standing_preserved_and_terminal_checked(self):
  with tempfile.TemporaryDirectory() as t:
   a=self.args(Path(t));a.output.mkdir();i=self.identity()
   with patch.object(h,'PARENT',NS(run_owned=Mock(side_effect=RuntimeError('SDF cooking failed')))),patch.object(h,'verify_inputs',return_value=i) as verify:
    with self.assertRaisesRegex(RuntimeError,'SDF'):h.run_campaign(a,i)
    verify.assert_called_once()
   d=h.read(a.output/'campaign.json');self.assertEqual(d['status'],'failed');self.assertTrue(d['terminal_inputs_unchanged']);self.assertFalse(d['physical_admission'])
 def test_success_requires_native_contract_and_post_exit_input_recheck(self):
  with tempfile.TemporaryDirectory() as t:
   a=self.args(Path(t));a.output.mkdir();i=self.identity();result={'phase':'standing','status':'completed','physical_admission':False,'standing_pass':True,'num_envs':1}
   c=NS(validate_result=Mock(return_value=result))
   with patch.object(h,'PARENT',NS(run_owned=Mock())),patch.object(h,'CONTRACT',c),patch.object(h,'verify_inputs',return_value=i) as verify:
    d=h.run_campaign(a,i);self.assertEqual(verify.call_count,2);c.validate_result.assert_called_once_with(a.output/'standing',i)
   self.assertEqual(d['status'],'completed');self.assertTrue(d['post_exit_original_inputs_reverified']);self.assertFalse(d['training_allowed']);self.assertTrue(d['post_exit_all_standing_payloads_inventoried']);self.assertEqual(h.read(a.output/'standing_immutable.sha256.json'),{})
 def test_changed_final_inputs_force_failed_campaign(self):
  with tempfile.TemporaryDirectory() as t:
   a=self.args(Path(t));a.output.mkdir();i=self.identity()
   with patch.object(h,'PARENT',NS(run_owned=Mock())),patch.object(h,'CONTRACT',NS(validate_result=Mock(return_value={}))),patch.object(h,'verify_inputs',side_effect=[i,{'changed':True}]):
    with self.assertRaisesRegex(RuntimeError,'terminal'):h.run_campaign(a,i)
   self.assertEqual(h.read(a.output/'campaign.json')['status'],'failed')
 def test_original_supervisor_600_and_90_preserved(self):
  self.assertEqual(h.sha(HERE/'inputs/source009_supervisor.py'),h.SUPERVISOR_CODE)
  t=(HERE/'inputs/source009_supervisor.py').read_text();self.assertIn('deadline = time.monotonic() + 600',t);self.assertIn('app_ready_deadline = time.monotonic() + 90',t)
 def test_explicit32_requires_and_mounts_same_source1_only(self):
  a=self.args(Path('/a'));a.num_envs=32
  with self.assertRaisesRegex(ValueError,'requires'):h.verify_inputs(a)
  a.standing_one=Path('/accepted/standing');c=h.command(a,'owned','standing')
  self.assertIn('/accepted/standing:/standing_one:ro',c);self.assertEqual(c[c.index('--num-envs')+1],'32');self.assertEqual(c[c.index('--standing-one')+1],'/standing_one')
  a.num_envs=1
  with self.assertRaisesRegex(ValueError,'consumes none'):h.verify_inputs(a)
  a.num_envs=2
  with self.assertRaisesRegex(ValueError,'Explicit'):h.verify_inputs(a)
 def test_standing1_input_overlap_rejects(self):
  a=self.args(Path('/a'));a.standing_one=Path('/accepted/standing');a.output=a.standing_one/'overwrite'
  with self.assertRaises(ValueError):h.require_fresh_output(a)
 def test_failed_standing_score_never_host_completes(self):
  with tempfile.TemporaryDirectory()as t:
   a=self.args(Path(t));a.output.mkdir();i=self.identity()
   with patch.object(h,'PARENT',NS(run_owned=Mock())),patch.object(h,'CONTRACT',NS(validate_result=Mock(side_effect=ValueError('Standing physics/quiet rejected')))),patch.object(h,'verify_inputs',return_value=i):
    with self.assertRaisesRegex(ValueError,'quiet'):h.run_campaign(a,i)
   d=h.read(a.output/'campaign.json');self.assertEqual(d['status'],'failed');self.assertTrue(d['terminal_inputs_unchanged']);self.assertFalse(d['training_allowed'])
if __name__=='__main__':unittest.main()
