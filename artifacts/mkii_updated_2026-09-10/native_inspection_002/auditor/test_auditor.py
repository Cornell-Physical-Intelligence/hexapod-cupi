import ast,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('audit',HERE/'audit_remote.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
class AuditTests(unittest.TestCase):
 def test_stdlib_import_does_not_call_process(self):
  subprocess.run([sys.executable,'-B','-S','-c','import audit_remote,sys;assert not any(x in sys.modules for x in ("torch","numpy","isaaclab"))'],cwd=HERE,check=True,capture_output=True)
 def test_terminal_exact_or_collected_invocation_requires_journal(self):
  unit={'MainPID':'0','ActiveState':'failed','InvocationID':''};rows=[{'_SYSTEMD_INVOCATION_ID':a.INV}]
  self.assertEqual(a.terminal_unit(unit,rows),rows)
  for delta,journal in [({'MainPID':'2'},rows),({'ActiveState':'active'},rows),({'InvocationID':'f'*32},rows),({},[])]:
   with self.assertRaises(ValueError):a.terminal_unit({**unit,**delta},journal)
 def test_failed_outcome_is_authentic_not_completed(self):
  c={'status':'failed','planned_phases':['inspection'],'training_allowed':False,'physical_admission':False,'error':'Missing installed API source'}
  self.assertEqual(a.classify(c,{'status':'failed'},{'status':'failed'},{'Result':'exit-code'},{'passed':False}),'authentic_terminal_failure')
  c.pop('error')
  with self.assertRaises(ValueError):a.classify(c,{}, {},{}, {})
 def test_completed_needs_native_postexit_receipt_and_zero_exit(self):
  receipt={'scope':'passive_import_cooking_inspection_only'}
  c={'status':'completed','planned_phases':['inspection'],'training_allowed':False,'physical_admission':False,'inspection':receipt,'terminal_inputs_unchanged':True,'post_exit_original_inputs_reverified':True,'post_exit_all_inspection_payloads_inventoried':True}
  job={'status':'completed','exit_code':0,'container_id':'a'*64};unit={'ExecMainStatus':'0','Result':'success'};state={'status':'completed'}
  self.assertEqual(a.classify(c,job,state,unit,{'passed':True,'receipt':receipt}),'authentic_completed_inspection')
  for bad in ({'passed':False},{'passed':True,'receipt':{'changed':True}}):
   with self.assertRaises(ValueError):a.classify(c,job,state,unit,bad)
  with self.assertRaises(ValueError):a.classify(c,job,state,{**unit,'ExecMainStatus':'1'},{'passed':True,'receipt':receipt})
 def test_exact_two_absences_unknown_docker_rejects_no_fabricated_id(self):
  job={'phase':'inspection','cleanup_checked':True,'container_name':'hexapod-reference-physics-'+'a'*32,'container_id':'b'*64}
  with patch.object(a.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','No such object')) as r:
   d=a.owned_absence(job);self.assertEqual(len(d['identifiers']),2);self.assertEqual(r.call_count,2)
   job['container_id']=None;d=a.owned_absence(job);self.assertEqual(len(d['identifiers']),1);self.assertTrue(d['unavailable_id_not_claimed_absent'])
   r.return_value=subprocess.CompletedProcess([],1,'','daemon unavailable')
   with self.assertRaises(ValueError):a.owned_absence(job)
 def test_restore_exact_timer_and_launch_bindings(self):
  pause={'unit':a.UNIT,'output':str(a.RUN),'source':str(a.SOURCE),'coordination_sha256':'35af5100e39980f55a3f5e19654e4550b3c3fb6da114b64fbe0d143817242cf3','units':{'a.timer':'ActiveState=active','b.timer':'ActiveState=inactive'},'created_unix':10}
  restored={'timers':['a.timer'],'restored_unix':12}
  launch={'native_source_freeze_sha256':a.PINS[0][3],'command':a.expected_launch_command()}
  self.assertTrue(a.restoration(pause,restored,launch)['passed'])
  for bad in ({'timers':['a.timer','b.timer']},{'timers':['a.timer','a.timer']},{'restored_unix':float('nan')}):
   with self.assertRaises(ValueError):a.restoration(pause,{**restored,**bad},launch)
  with self.assertRaises(ValueError):a.restoration(pause,restored,{**launch,'native_source_freeze_sha256':'f'*64})
 def test_exact_full_inventory_rejects_added_or_changed(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);(r/'one').write_bytes(b'abc');m={'one':a.sha(r/'one')};(r/'map.json').write_text(json.dumps(m));bound=a.sha(r/'map.json')
   self.assertTrue(a.checked(r,'map.json',bound,1)['passed']);self.assertEqual(a.inventory(r)['one']['size_bytes'],3)
   (r/'one').write_bytes(b'xyz')
   with self.assertRaises(ValueError):a.checked(r,'map.json',bound,1)
   (r/'one').write_bytes(b'abc');(r/'extra').write_text('extra')
   with self.assertRaises(ValueError):a.checked(r,'map.json',bound,1)
 def test_no_mutation_api_or_simulator_import(self):
  tree=ast.parse((HERE/'audit_remote.py').read_text());forbidden={'write_text','write_bytes','mkdir','unlink','rmdir','rename','replace','touch','Popen','kill','terminate'}
  for n in ast.walk(tree):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):self.assertNotIn(n.func.attr,forbidden)
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='open':self.assertEqual(n.args[0].value,'rb')
if __name__=='__main__':unittest.main()
