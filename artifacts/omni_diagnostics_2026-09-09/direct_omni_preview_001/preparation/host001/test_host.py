import ast,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch,Mock
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('previewhost',HERE/'launch_preview_spark.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
class Tests(unittest.TestCase):
 def args(self,root):
  return NS(source=root/'source',contract=root/'contract',supervisor_source=root/'supervisor',adapter=root/'adapter',pilot=root/'pilot',checkpoint=root/'pilot/train/policy/final.pt',output=root/'output',checkpoint_sha256='a'*64,campaign_sha256='b'*64,host_freeze_sha256='c'*64,isaaclab=root/'isaaclab')
 def test_stdlib_only_import(self):
  subprocess.run([sys.executable,'-S','-B','-c',"import launch_preview_spark,sys; assert 'torch' not in sys.modules and 'numpy' not in sys.modules"],cwd=HERE,check=True)
 def test_only_recording_and_all_inputs_readonly(self):
  a=self.args(Path('/base'));c=h.command(a,'owned','recording');mounts=[c[i+1] for i,x in enumerate(c) if x=='-v']
  self.assertEqual(sum(m.endswith(':rw') for m in mounts),1)
  for key in ('source','contract','adapter','pilot','checkpoint'):self.assertTrue(any(m.startswith(str(getattr(a,key))+':') and m.endswith(':ro') for m in mounts),key)
  self.assertIn(str(a.pilot/'standing')+':/admission:ro',mounts)
  self.assertEqual(c[c.index('--checkpoint-sha256')+1],a.checkpoint_sha256)
  self.assertIn('/recording/record_direct_preview.py',c)
  with self.assertRaises(ValueError):h.command(a,'owned','train')
 def test_reused_and_overlapping_output_reject(self):
  with tempfile.TemporaryDirectory() as t:
   a=self.args(Path(t));h.require_fresh_output(a);a.output.mkdir()
   with self.assertRaisesRegex(ValueError,'Fresh'):h.require_fresh_output(a)
   a.output=a.source/'out'
   with self.assertRaisesRegex(ValueError,'overlaps'):h.require_fresh_output(a)
 def test_shared_ownership_functions_remain_exact(self):
  text=(HERE/'inputs/source009_supervisor.py').read_text();self.assertEqual(h.sha(HERE/'inputs/source009_supervisor.py'),h.SUPERVISOR_CODE)
  source=ast.parse((HERE/'launch_preview_spark.py').read_text())
  names={n.name for n in source.body if isinstance(n,ast.FunctionDef)}
  self.assertNotIn('run_owned',names);self.assertNotIn('owned_container',names)
  a=self.args(Path('/base'));module=NS(command=None,verified_source=None,RUNTIME_TREE=None,save=lambda *a:None)
  original_run=Mock();original_owned=Mock();module.run_owned=original_run;module.owned_container=original_owned
  with patch.object(h,'load_module',return_value=module),patch.object(sys,'path',list(sys.path)),patch.object(h,'verify_tree') as verify,patch.object(h,'verify_legacy_runtime') as legacy:
   parent=h.load_supervisor(a);self.assertIs(parent.run_owned,original_run);self.assertIs(parent.owned_container,original_owned)
   parent.verified_source(a.source);verify.assert_called_once_with(a.source,'campaign_source_hashes.json',h.SOURCE_MAP);legacy.assert_called_once_with(a.source)
 def test_complete_and_partial_recording_result(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t);identity={k:k for k in ('source_manifest_sha256','plan_sha256','native_contract_freeze_sha256','adapter_freeze_sha256','checkpoint_sha256')};identity.update(pilot={'completed_updates':50},rendering_overrides={'num_envs':1})
   v={**identity,'complete':True,'frames':950,'fps':25,'recorded_control_steps':1900,'planned_control_steps':1900,'planned_frames':950,'physics_duration_s':38.,'playback_duration_s':38.,'playback_rate':1.,'dt_s':.02,'source_and_inputs_reverified_after_recording':True,'stage2_complete':False,'policy_training_started':False,'pose_forcing':False,'qualification_performed':False,'terminal_event':None,'error':None}
   for when in ('before','after'):v['checkpoint_readback_'+when]={'passed':True,'actor_and_critic_including_normalizers_exact':True,'checkpoint_sha256':identity['checkpoint_sha256']}
   for file,key in (('rollout.mp4','video_sha256'),('trace.npz','trace_sha256')):(d/file).write_bytes(file.encode());v[key]=h.sha(d/file)
   h.save(d/'state.json',dict(status='completed',checkpoint_sha256=identity['checkpoint_sha256'],runtime_binding={'runtime_tree_sha256':h.LEGACY_RUNTIME_TREE},recording_provenance=identity));h.save(d/'final_integrity.json',{'passed':True,'provenance':identity});h.save(d/'video.json',v)
   self.assertTrue(h.validate_result(d,identity)['complete'])
   v['recorded_control_steps']=7;v['complete']=False;h.save(d/'video.json',v)
   with self.assertRaisesRegex(ValueError,'Incomplete'):h.validate_result(d,identity)
 def test_failed_render_keeps_campaign_and_checks_inputs(self):
  with tempfile.TemporaryDirectory() as t:
   a=self.args(Path(t));a.output.mkdir();identity={'x':1}
   with patch.object(h,'PARENT',NS(run_owned=Mock(side_effect=RuntimeError('first terminal')))),patch.object(h,'verify_inputs',return_value=identity) as verify:
    with self.assertRaisesRegex(RuntimeError,'terminal'):h.run_campaign(a,identity)
   d=h.read(a.output/'campaign.json');self.assertEqual(d['status'],'failed');self.assertTrue(d['terminal_inputs_unchanged']);verify.assert_called_once()
 def test_adapter_is_the_pilot_authority_not_host_reinterpretation(self):
  a=self.args(Path('/base'));module=NS(__file__=str(a.adapter/'preview_contract.py'),verify_inputs=Mock(side_effect=ValueError('smoke or partial pilot')))
  with patch.object(h,'ADAPTER',module),patch.object(h,'verify_own_bundle'),patch.object(h,'verify_tree'),patch.object(h,'verify_legacy_runtime'),patch.object(h,'sha',side_effect=lambda p:h.SUPERVISOR_CODE if p.name.endswith('.py') else a.campaign_sha256):
   with self.assertRaisesRegex(ValueError,'smoke or partial'):h.verify_inputs(a)
  self.assertEqual(module.verify_inputs.call_args.args[0].checkpoint_sha256,a.checkpoint_sha256)
 def test_wrong_selected_campaign_stops_before_adapter(self):
  a=self.args(Path('/base'));module=NS(verify_inputs=Mock())
  with patch.object(h,'ADAPTER',module),patch.object(h,'verify_own_bundle'),patch.object(h,'verify_tree'),patch.object(h,'verify_legacy_runtime'),patch.object(h,'sha',side_effect=lambda p:h.SUPERVISOR_CODE if p.name.endswith('.py') else '0'*64):
   with self.assertRaisesRegex(ValueError,'campaign SHA'):h.verify_inputs(a)
  module.verify_inputs.assert_not_called()
if __name__=='__main__':unittest.main()
