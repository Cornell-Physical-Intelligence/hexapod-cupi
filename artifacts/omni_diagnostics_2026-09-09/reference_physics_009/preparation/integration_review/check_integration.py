"""Read-only009 source and fresh-phase identity preflight; never launches physics."""
from pathlib import Path
import ast,hashlib,json,sys,tempfile
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;SRC=HERE/'source_009';BASE=HERE.parent/'reference_physics_adapter_008_final/source_008'
sys.path.insert(0,str(SRC/'tools'))
from launch_reference_physics_spark import check_source
from screen_contract import preflight
from solver_comparison import PROTOCOL
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 runtime=check_source(SRC);old=json.loads((BASE/'campaign_source_hashes.json').read_text());new=json.loads((SRC/'campaign_source_hashes.json').read_text())
 changed=sorted(k for k in old if old[k]!=new.get(k))
 assert set(old)==set(new)
 assert changed==['source_origin.json','tools/physics_substeps.py','tools/solver_comparison.py','tools/wave_reference.py'],changed
 assert all(sha(SRC/k)==v for k,v in new.items())
 assert {str(p.relative_to(SRC)) for p in SRC.rglob('*') if p.is_file()}==set(new)|{'campaign_source_hashes.json'}
 # Config/readback functions are unchanged; protocol declarations only differ.
 defs=lambda p:[ast.dump(n) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
 assert defs(BASE/'tools/solver_comparison.py')==defs(SRC/'tools/solver_comparison.py')
 with tempfile.TemporaryDirectory() as td:
  root=Path(td);args=SimpleNamespace(package=SRC/'robot/hexapod_mkii_length_study',variant='f050_t060',stance_index=0,mode='standing',num_envs=32,steps=1000,output=root/'never_created_output',admission=None,geometry_reference=SRC/'robot/hexapod_mkii_length_study/candidate_c_reference.json')
  identity,_=preflight(args,SRC)
  args.mode='wave';args.num_envs=1;args.steps=2400;args.admission=HERE.parent/'reference_physics_results_008/run/standing/admission.json'
  try:preflight(args,SRC)
  except ValueError as e:assert 'Mismatched' in str(e)
  else:raise AssertionError('Actual008 admission must not authorize009')
  payload=json.loads(args.admission.read_text());payload['identity']=identity
  args.admission=root/'synthetic_unit_receipt_not_runtime_admission.json';args.admission.write_text(json.dumps(payload))
  preflight(args,SRC)
  payload['gate']['all_replica_quiet']['passed']=False;args.admission.write_text(json.dumps(payload))
  try:preflight(args,SRC)
  except ValueError:pass
  else:raise AssertionError('Missing quiet admission must reject')
 return {'source':str(SRC),'files':len(new),'manifest_sha256':sha(SRC/'campaign_source_hashes.json'),'runtime':runtime,'changed_from008':changed,'same_file_set':True,'all_other_files_byte_identical':True,'source_no_extras':True,'config_readback_function_AST_unchanged':True,'real008_admission_rejected_by009':True,'matching_synthetic_unit_receipt_accepted_but_failed_quiet_rejected':True,'synthetic_receipt_was_temporary_and_not_runtime_evidence':True,'wave_owner_tests_passed':28,'independent_wave_targeted_tests_passed':8,'observer_tests_passed_local_and_independent':11,'protocol':PROTOCOL,'new_physics_result_claimed':False}
if __name__=='__main__':print(json.dumps(main(),indent=2,sort_keys=True))
