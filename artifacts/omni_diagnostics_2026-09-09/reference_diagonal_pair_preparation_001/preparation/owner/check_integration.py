"""Exact assembled-source preflight. Synthetic receipts exercise wiring, never physics admission."""
import hashlib,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;source=HERE/'source_diagonal_pair_001';sys.path.insert(0,str(source/'tools'))
import launch_pair_physics_spark as host
from pair_screen_contract import preflight,PAIR_PROTOCOL,PAIR_CASES
from screen_contract import preflight as standing
runtime=host.check_source(source)
args=SimpleNamespace(package=source/'robot/hexapod_mkii_length_study',variant='f050_t060',stance_index=0,output=HERE/'uncreated-integration-preflight',mode='standing',num_envs=32,steps=1000,admission=None,geometry_reference=source/host.GEOMETRY_REFERENCE)
identity,reference=standing(args,source)
args.mode='pair';args.pair_case='lf_rr';args.num_envs=1;args.steps=1300;args.admission=HERE.parent/'reference_pair_results_001/run/standing/admission.json'
try:preflight(args,source)
except ValueError:old_rejected=True
else:raise AssertionError('Actual middlepair001 admission cannot authorize a new diagonal source')
with tempfile.TemporaryDirectory() as td:
 args.admission=Path(td)/'receipt.json'
 receipt={'mode':'standing','identity':identity,'status':'completed','gate':{'passed':True,'num_envs':32,'control_steps':1000,'all_replica_quiet':{'passed':True}}}
 args.admission.write_text(json.dumps(receipt));pair_identity,_=preflight(args,source)
 assert pair_identity['pair_protocol']==PAIR_PROTOCOL and pair_identity['pair_case']=='lf_rr' and pair_identity['pair_partition']==PAIR_CASES['lf_rr']
 args.pair_case='lr_rf';second_identity,_=preflight(args,source);assert second_identity['pair_case']=='lr_rf' and second_identity['pair_partition']==PAIR_CASES['lr_rf']
 args.pair_case='lm_rm'
 try:preflight(args,source)
 except ValueError:middle_repeat_rejected=True
 else:raise AssertionError('Middle pair repeated under diagonal contract')
 args.pair_case='lr_rf'
 receipt['gate']['all_replica_quiet']['passed']=False;args.admission.write_text(json.dumps(receipt))
 try:preflight(args,source)
 except ValueError:failed_quiet_rejected=True
 else:raise AssertionError('Failed quiet receipt accepted')
result={'runtime':runtime,'source_manifest_sha256':host.digest(source/'campaign_source_hashes.json'),'source_files':len(json.loads((source/'campaign_source_hashes.json').read_text())),
 'actual_middlepair001_old_admission_rejected':old_rejected,'failed_quiet_synthetic_receipt_rejected':failed_quiet_rejected,'pair_protocol':PAIR_PROTOCOL,
 'synthetic_receipts_used_for_CPU_logic_only':True,'physics_admitted':False,'unchanged_parent_payloads':928,'middle_pair_repeat_rejected':middle_repeat_rejected,'no_origin_reset_adapter_in_source':not (source/'tools/matched_origin.py').exists()}
path=HERE/'INTEGRATION_PREFLIGHT.json'
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
