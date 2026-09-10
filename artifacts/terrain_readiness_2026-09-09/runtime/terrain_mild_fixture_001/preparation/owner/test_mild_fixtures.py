"""CPU source/result/ownership regressions; synthetic receipts confer no physics admission."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast,copy,json,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent;SOURCE=HERE/'source_mild_fixtures_001_final'
sys.path.insert(0,str(SOURCE/'tools'))
import launch_mild_fixtures_spark as host
import mild_fixture_contract as contract


def synthetic_result():
 state=json.loads((HERE/'cpu_audit/validation.json').read_text())
 state.update(status='completed',cpu_only=False,all_fixture_smokes_passed=True,simulated_seconds=2.5,runtime_rows=[])
 for i,name in enumerate(contract.EXPECTED_IDS):
  queries=[]
  for j,xy in enumerate(([-1.,.11],[0.,.031],[.51,-.21],[1.75,0.])):
   queries.append(dict(xy_local_m=xy,expected_height_m=0. if j<3 else None,height_m=0. if j<3 else None,
                       collision=f'/World/Fixtures/f_{i:03d}/Terrain' if j<3 else '',passed=True))
  state['runtime_rows'].append(dict(id=name,passed=True,ray_caster_passed=True,ray_caster_rays=81,ray_caster_max_height_error_m=0.,physx_queries=queries,
     probes=[dict(passed=True,sphere_bottom_gap_m=0.,contact_fraction_last_40_steps=1.,mean_contact_force_n=.49,final_position_m=[0,0,.01],surface_height_m=0.) for _ in range(3)]))
 return state

class ContractTests(unittest.TestCase):
 def test_source105_catalog97_and_actual_cpu48(self):
  result=host.check_source(SOURCE);self.assertEqual(result['source_files'],105);self.assertEqual(result['catalog_geometry_files'],97)
  state=json.loads((HERE/'cpu_audit/validation.json').read_text());self.assertEqual(len(state['cpu_rows']),48)
  self.assertTrue(all(r['cpu_passed'] for r in state['cpu_rows']));self.assertFalse(state['robot_validation_performed'])
  with self.assertRaises(ValueError):contract.validate_result(state,SOURCE)
 def test_complete_synthetic48_strict_payload(self):
  state=synthetic_result();gate=contract.validate_result(state,SOURCE)
  self.assertEqual((gate['fixture_count'],gate['ray_caster_rays'],gate['physx_queries'],gate['physical_probes']),(48,3888,192,144))
  json.dumps({'gate':gate,'state':state},allow_nan=False)
 def test_missing_duplicate_subset_cpu_only_and_wrong_geometry_rejected(self):
  for kind in ['missing','duplicate','cpu_only','npz','harness','collider','ray_count','query_count','probe_count','false_pass','short_time']:
   r=synthetic_result()
   if kind=='missing':r['runtime_rows'].pop()
   elif kind=='duplicate':r['runtime_rows'][1]=r['runtime_rows'][0]
   elif kind=='cpu_only':r['cpu_only']=True
   elif kind=='npz':r['cpu_rows'][0]['npz_sha256']='wrong'
   elif kind=='harness':r['harness_sha256']='wrong'
   elif kind=='collider':r['runtime_rows'][0]['physx_queries'][0]['collision']='/World/GroundPlane'
   elif kind=='ray_count':r['runtime_rows'][0]['ray_caster_rays']=80
   elif kind=='query_count':r['runtime_rows'][0]['physx_queries'].pop()
   elif kind=='probe_count':r['runtime_rows'][0]['probes'].pop()
   elif kind=='false_pass':r['runtime_rows'][0]['probes'][0]['passed']=False
   elif kind=='short_time':r['simulated_seconds']=2.
   with self.assertRaises(ValueError,msg=kind):contract.validate_result(r,SOURCE)
 def test_recorded_numeric_failures_not_hidden_by_pass_flags(self):
  for kind in ['ray','contact_fraction','gap','nan','outside_hit']:
   r=synthetic_result();row=r['runtime_rows'][0]
   if kind=='ray':row['ray_caster_max_height_error_m']=.00021
   elif kind=='contact_fraction':row['probes'][0]['contact_fraction_last_40_steps']=.79
   elif kind=='gap':row['probes'][0]['sphere_bottom_gap_m']=.0031
   elif kind=='nan':row['probes'][0]['mean_contact_force_n']=float('nan')
   elif kind=='outside_hit':row['physx_queries'][3]['height_m']=0.
   with self.assertRaises(ValueError,msg=kind):contract.validate_result(r,SOURCE)
 def test_source_extra_path_and_escape_fail_closed(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);(p/'good.py').write_text('x=1');m={'good.py':host.digest(p/'good.py')};(p/'campaign_source_hashes.json').write_text(json.dumps(m))
   host.verified_source(p);(p/'extra.py').write_text('x=2')
   with self.assertRaises(ValueError):host.verified_source(p)
 def test_exact_readonly_phase_and_marker_no_actor(self):
  cmd=host.command(Path('/source'),Path('/output'),'owned','fixtures')
  self.assertIn('/source:/workspace/terrain-source:ro',cmd);self.assertNotIn('--cpu-only',cmd);self.assertNotIn('--fixtures',cmd)
  self.assertEqual(cmd[cmd.index('--steps')+1],'500');self.assertEqual(cmd[cmd.index('--dt')+1],'0.005')
  self.assertIn('/workspace/terrain-source/'+contract.CATALOG,cmd)
  for phase in ['standing','wave','train']:
   with self.assertRaises(ValueError):host.command(Path('/source'),Path('/output'),'owned',phase)
 def test_geometry_functions_and_cleanup_are_exact_parent_AST(self):
  oldpath=HERE.parent/'reference_physics_adapter_009/source_009/tools/launch_reference_physics_spark.py'
  old=ast.parse(oldpath.read_text());new=ast.parse((SOURCE/'tools/launch_mild_fixtures_spark.py').read_text())
  get=lambda t,name:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
  self.assertEqual(ast.dump(get(old,'owned_container')),ast.dump(get(new,'owned_container')))
  a=next(n for n in get(old,'run_owned').body if isinstance(n,ast.Try));b=next(n for n in get(new,'run_owned').body if isinstance(n,ast.Try))
  self.assertEqual(ast.dump(ast.Module(body=a.finalbody,type_ignores=[])),ast.dump(ast.Module(body=b.finalbody,type_ignores=[])))
  import subprocess
  oldh=ast.parse(subprocess.check_output(['git','show','22957c2062d532f47400f2adb30ef04e257b4b2b:tools/validate_terrain_fixtures.py'],text=True))
  newh=ast.parse((SOURCE/'tools/validate_terrain_fixtures.py').read_text())
  for name in ['run_isaac','audit','load_geometry_helpers']:self.assertEqual(ast.dump(get(oldh,name)),ast.dump(get(newh,name)))

class OwnershipTests(unittest.TestCase):
 def test_output_inside_source_is_rejected_before_preflight(self):
  with patch.object(host.sys,'argv',['host','--source',str(SOURCE),'--output',str(SOURCE/'forbidden')]),patch.object(host,'check_source') as check:
   with self.assertRaises(SystemExit):host.main()
   check.assert_not_called();self.assertFalse((SOURCE/'forbidden').exists())
 def test_terminal_integrity_or_contact_failure_cannot_exit_success(self):
  for kind in ['source','contact']:
   with tempfile.TemporaryDirectory() as td:
    root=Path(td);coord=root/'coordination';coord.write_text('owner');out=root/'fresh_output'
    identity={'source_manifest_sha256':'synthetic','source_files':105}
    def run(args,phase):
     (args.output/'fixtures').mkdir();(args.output/'fixtures/validation.json').write_text('{"status":"completed"}')
     (args.output/'logs/fixtures.log').write_text('incomplete contact data' if kind=='contact' else 'complete log')
     return {'status':'completed'}
    sequence=[identity,identity,RuntimeError('source mutation')] if kind=='source' else [identity]*3
    with patch.object(host.sys,'argv',['host','--source',str(root/'source'),'--output',str(out)]),patch.object(host,'COORDINATION',coord),patch.object(host.signal,'signal'),patch.object(host,'check_source',side_effect=sequence),patch.object(host,'catalog_identity',return_value=([],{'catalog':'sha'})),patch.object(host,'run_owned',run),patch.object(host,'validate_result',return_value={'passed':True}):
     with self.assertRaisesRegex(RuntimeError,'Terminal source/fixture/contact'):host.main()
    result=json.loads((out/'campaign.json').read_text());self.assertEqual(result['status'],'failed')
    self.assertFalse(result['terminal_source_integrity']['passed'] if kind=='source' else result['terminal_contact_log_audit']['passed'])
 def test_unknown_inspect_error_is_not_absence(self):
  with patch.object(host.subprocess,'run',return_value=SimpleNamespace(returncode=1,stderr='daemon permission denied',stdout='')):
   with self.assertRaisesRegex(RuntimeError,'unknown'):host.owned_container('owned')
  with patch.object(host.subprocess,'run',return_value=SimpleNamespace(returncode=1,stderr='Error: No such object: owned',stdout='')):self.assertIsNone(host.owned_container('owned'))
 def execute(self,*,inspection=None,wait_for_ready=False):
  td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup);out=Path(td.name)
  for d in ['jobs','logs','fixtures']:(out/d).mkdir()
  (out/'fixtures/validation.json').write_text(json.dumps(synthetic_result()))
  coord=out/'coordination.txt';coord.write_text('exclusive fixture owner');args=SimpleNamespace(source=SOURCE,output=out,isaaclab=out,coordination_sha256=host.digest(coord))
  class Process:
   returncode=None if wait_for_ready else 0
   def poll(self):return self.returncode
   def terminate(self):self.returncode=0
   def wait(self,timeout=None):self.returncode=0;return 0
   def kill(self):self.returncode=-9
  process=Process()
  mocks=[patch.object(host,'COORDINATION',coord),patch.object(host,'preflight',return_value={'synthetic_CPU_test':True}),
   patch.object(host.os,'open',side_effect=[101,102]),patch.object(host.os,'close'),patch.object(host.fcntl,'flock'),
   patch.object(host.subprocess,'Popen',return_value=process),patch.object(host,'owned_container',side_effect=inspection or [None,None]),
   patch.object(host.subprocess,'run',return_value=SimpleNamespace(returncode=0)),patch.object(host,'resources',return_value=('',32*1024**3))]
  if wait_for_ready:mocks.append(patch.object(host.time,'monotonic',side_effect=[0,0,91,91]))
  active=[m.start() for m in mocks]
  try:
   error=None;state=None
   try:state=host.run_owned(args,'fixtures')
   except Exception as exc:error=exc
   stops=[c for c in active[7].call_args_list if c.args and c.args[0][:2]==['docker','stop']]
   return state,error,json.loads((out/'jobs/fixtures.json').read_text()),stops
  finally:
   for m in reversed(mocks):m.stop()
 def test_exited_client_still_stops_only_owned_container(self):
  state,error,report,stops=self.execute(inspection=[('exact-owned-id',True),None])
  self.assertIsNone(error);self.assertEqual(state['status'],'completed');self.assertTrue(report['cleanup_checked'])
  self.assertEqual(len(stops),1);self.assertEqual(stops[0].args[0][-1],'exact-owned-id')
 def test_cleanup_inspection_uncertainty_propagates(self):
  state,error,report,stops=self.execute(inspection=[RuntimeError('daemon unavailable')])
  self.assertIsInstance(error,RuntimeError);self.assertFalse(report['cleanup_checked']);self.assertTrue(report['cleanup_requires_owner_review']);self.assertEqual(stops,[])
 def test_appready90s_timeout_preserves_cleanup(self):
  state,error,report,stops=self.execute(inspection=[None,None,None],wait_for_ready=True)
  self.assertIsInstance(error,TimeoutError);self.assertEqual(report['startup_failure_kind'],'no_terrain_AppReady_by90s');self.assertTrue(report['cleanup_checked'])
if __name__=='__main__':unittest.main()
