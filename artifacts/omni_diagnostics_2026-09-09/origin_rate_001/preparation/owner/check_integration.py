"""Read-only exact frozen-source preflight; synthetic receipts never admit physics."""
import hashlib,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;source=HERE/'source_origin_001'
sys.path.insert(0,str(source/'tools'))
import launch_origin_physics_spark as host
from origin_contract import preflight,CASES
from screen_contract import preflight as standing
runtime=host.check_source(source)
args=SimpleNamespace(package=source/'robot/hexapod_mkii_length_study',variant='f050_t060',stance_index=0,output=HERE/'uncreated-integration-preflight',mode='standing',num_envs=32,steps=1000,admission=None,geometry_reference=source/host.GEOMETRY_REFERENCE)
identity,reference=standing(args,source)
args.mode='origin';args.num_envs=1;args.case='origin_a';args.admission=HERE.parent/'reference_physics_results_009/run/standing/admission.json'
try:preflight(args,source)
except ValueError:old_rejected=True
else:raise AssertionError('Actual009 admission cannot authorize a fresh origin source')
with tempfile.TemporaryDirectory() as td:
 args.admission=Path(td)/'receipt.json'
 receipt={'mode':'standing','identity':identity,'status':'completed','gate':{'passed':True,'num_envs':32,'control_steps':1000,'all_replica_quiet':{'passed':True}}}
 args.admission.write_text(json.dumps(receipt));checks={}
 for case in CASES:
  args.case=case;checks[case]=preflight(args,source)[0]['translation_xy_m']
 receipt['gate']['all_replica_quiet']['passed']=False;args.admission.write_text(json.dumps(receipt))
 try:preflight(args,source)
 except ValueError:failed_quiet_rejected=True
 else:raise AssertionError('Failed quiet receipt accepted')
result={'runtime':runtime,'source_manifest_sha256':host.digest(source/'campaign_source_hashes.json'),'source_files':len(json.loads((source/'campaign_source_hashes.json').read_text())),
 'actual009_old_admission_rejected':old_rejected,'failed_quiet_synthetic_receipt_rejected':failed_quiet_rejected,'declared_cases':checks,
 'synthetic_receipts_used_for_CPU_logic_only':True,'physics_admitted':False,'all_original009_runtime_files_byte_identical_except_source_origin_metadata':True}
path=HERE/'INTEGRATION_PREFLIGHT.json'
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
