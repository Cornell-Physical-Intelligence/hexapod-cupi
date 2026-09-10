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
  c={'status':'failed','planned_phases':['standing'],'training_allowed':False,'physical_admission':False,'error':'Standing physics/quiet rejected'}
  self.assertEqual(a.classify(c,{'status':'completed','exit_code':0},{'status':'completed','standing_pass':False},{'Result':'exit-code'},{'passed':False}),'authentic_terminal_failure')
  c.pop('error')
  with self.assertRaises(ValueError):a.classify(c,{}, {},{}, {})
 def test_completed_needs_native_postexit_receipt_and_zero_exit(self):
  receipt={'scope':'provisional_standing_only','standing_pass':True,'num_envs':32}
  c={'status':'completed','planned_phases':['standing'],'training_allowed':False,'physical_admission':False,'standing':receipt,'terminal_inputs_unchanged':True,'post_exit_original_inputs_reverified':True,'post_exit_all_standing_payloads_inventoried':True}
  job={'status':'completed','exit_code':0,'container_id':'a'*64};unit={'ExecMainStatus':'0','Result':'success'};state={'status':'completed'}
  self.assertEqual(a.classify(c,job,state,unit,{'passed':True,'receipt':receipt}),'authentic_completed_standing')
  for bad in ({'passed':False},{'passed':True,'receipt':{'changed':True}}):
   with self.assertRaises(ValueError):a.classify(c,job,state,unit,bad)
  with self.assertRaises(ValueError):a.classify(c,job,state,{**unit,'ExecMainStatus':'1'},{'passed':True,'receipt':receipt})
 def test_exact_two_absences_unknown_docker_rejects_no_fabricated_id(self):
  job={'phase':'standing','cleanup_checked':True,'container_name':'hexapod-reference-physics-'+'a'*32,'container_id':'b'*64}
  with patch.object(a.subprocess,'run',return_value=subprocess.CompletedProcess([],1,'','No such object')) as r:
   d=a.owned_absence(job);self.assertEqual(len(d['identifiers']),2);self.assertEqual(r.call_count,2)
   job['container_id']=None;d=a.owned_absence(job);self.assertEqual(len(d['identifiers']),1);self.assertTrue(d['unavailable_id_not_claimed_absent'])
   r.return_value=subprocess.CompletedProcess([],1,'','daemon unavailable')
   with self.assertRaises(ValueError):a.owned_absence(job)
 def test_restore_exact_timer_and_launch_bindings(self):
  pause={'unit':a.UNIT,'output':str(a.RUN),'source':str(a.SOURCE),'admission':str(a.ADMISSION),'standing_one':str(a.STANDING_ONE),'coordination_sha256':'649ccda1bdef20bc967ae78f68a992cc8001f4ed7009f78cab5133cc67b29d8f','units':{'a.timer':'ActiveState=active','b.timer':'ActiveState=inactive'},'created_unix':10,'reservation_path':str(a.RESERVATION),'restoration_scope':'per_job_snapshot_only','persistent_reservation_release_attempted':False}
  restored={'timers':['a.timer'],'restored_unix':12,'scope':'per_job_snapshot_only','persistent_reservation_release_attempted':False,'reservation_path':str(a.RESERVATION)}
  launch={'native_source_freeze_sha256':a.PINS[0][3],'command':a.expected_launch_command()}
  self.assertTrue(a.restoration(pause,restored,launch)['passed'])
  for bad in ({'timers':['a.timer','b.timer']},{'timers':['a.timer','a.timer']},{'restored_unix':float('nan')},{'scope':'released_reservation'},{'persistent_reservation_release_attempted':True}):
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
 def test_actual32_launch_and_same_source_one_are_explicit(self):
  cmd=a.expected_launch_command();self.assertEqual(cmd[cmd.index('--num-envs')+1],'32');self.assertEqual(cmd[cmd.index('--standing-one')+1],str(a.STANDING_ONE))
  self.assertEqual(a.PINS[2][4],62);self.assertEqual(a.INV,'1a38495bda9f4fefa4e5585574aabc56')
 def test_standing1_map_change_rejects_and_large_files_are_not_copied(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);stand=root/'standing';guard=root/'guard';stand.mkdir();(guard/'inputs').mkdir(parents=True)
   (stand/'state.json').write_text('{}');mapping={'state.json':a.sha(stand/'state.json')};(guard/'inputs/STANDING_ONE_SHA256.json').write_text(json.dumps(mapping))
   import hashlib
   digest=hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   with patch.multiple(a,STANDING_ONE=stand,GUARD=guard,STANDING_ONE_MAP_SHA256=digest,STANDING_ONE_STATE_SHA256=mapping['state.json']):
    self.assertEqual(a.standing_one_check()['state.json']['sha256'],mapping['state.json'])
    (stand/'extra.npz').write_bytes(b'new')
    with self.assertRaises(ValueError):a.standing_one_check()
 def test_hash_reads_fixed_blocks_not_entire_large_raw(self):
  import hashlib
  class Stream:
   def __init__(self):self.parts=iter([b'abc',b'def',b'']);self.sizes=[]
   def __enter__(self):return self
   def __exit__(self,*args):pass
   def read(self,size):self.sizes.append(size);return next(self.parts)
  stream=Stream()
  with patch.object(Path,'open',return_value=stream):self.assertEqual(a.sha(Path('/readonly/large')),hashlib.sha256(b'abcdef').hexdigest())
  self.assertEqual(stream.sizes,[8<<20]*3)
 def test_no_mutation_api_or_simulator_import(self):
  tree=ast.parse((HERE/'audit_remote.py').read_text());forbidden={'write_text','write_bytes','mkdir','unlink','rmdir','rename','replace','touch','Popen','kill','terminate'}
  for n in ast.walk(tree):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):self.assertNotIn(n.func.attr,forbidden)
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='open':self.assertEqual(n.args[0].value,'rb')
if __name__=='__main__':unittest.main()
