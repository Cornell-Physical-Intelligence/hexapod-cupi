"""Exact assembled-source preflight; synthetic receipts grant no physical admission."""
from pathlib import Path
from types import SimpleNamespace
import json,sys,tempfile
HERE=Path(__file__).resolve().parent;SOURCE=HERE/'source_directional_002';sys.path.insert(0,str(SOURCE/'tools'))
import launch_directional_physics_spark as host
from directional_contract import CASES,preflight
from screen_contract import preflight as standing
runtime=host.check_source(SOURCE)
a=SimpleNamespace(package=SOURCE/'robot/hexapod_mkii_length_study',variant='f050_t060',stance_index=0,output=HERE/'uncreated-preflight',mode='standing',num_envs=32,steps=1000,admission=None,geometry_reference=SOURCE/host.GEOMETRY_REFERENCE)
identity,geometry=standing(a,SOURCE);a.mode='directional';a.num_envs=1;a.steps=2400;a.case='left_strafe';a.admission=HERE.parent/'reference_directional_results_001/run/standing/admission.json'
try:preflight(a,SOURCE)
except ValueError:old_rejected=True
else:raise AssertionError('Actual directional001 admission accepted for new source')
with tempfile.TemporaryDirectory() as td:
 a.admission=Path(td)/'receipt.json';receipt={'mode':'standing','status':'completed','identity':identity,'gate':{'passed':True,'num_envs':32,'control_steps':1000,'all_replica_quiet':{'passed':True}}};a.admission.write_text(json.dumps(receipt));cases={}
 for name in CASES:a.case=name;cases[name]=preflight(a,SOURCE)[0]['requested_forward_left_yaw']
 receipt['gate']['all_replica_quiet']['passed']=False;a.admission.write_text(json.dumps(receipt))
 try:preflight(a,SOURCE)
 except ValueError:quiet_rejected=True
 else:raise AssertionError('Failed quiet accepted')
r={'runtime':runtime,'source_manifest_sha256':host.digest(SOURCE/'campaign_source_hashes.json'),'source_files':930,'actual_directional001_old_admission_rejected':old_rejected,'failed_quiet_synthetic_receipt_rejected':quiet_rejected,'all_named_cases_preflight':cases,'synthetic_receipts_are_CPU_only':True,'physics_admitted':False}
p=HERE/'INTEGRATION_PREFLIGHT.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
