"""Source/phase preflight only; synthetic receipts grant no physical admission."""
from pathlib import Path
from types import SimpleNamespace
import json,sys,tempfile
H=Path(__file__).resolve().parent;SOURCE=H/'source_pair_motion_001';sys.path.insert(0,str(SOURCE/'tools'))
import launch_pair_motion_spark as host
from pair_motion_contract import preflight
from screen_contract import preflight as standing
runtime=host.check_source(SOURCE)
a=SimpleNamespace(package=SOURCE/'robot/hexapod_mkii_length_study',variant='f050_t060',stance_index=0,output=H/'uncreated-preflight',mode='standing',num_envs=32,steps=1000,admission=None,geometry_reference=SOURCE/host.GEOMETRY_REFERENCE)
identity,geometry=standing(a,SOURCE);a.mode='paired_motion';a.num_envs=1;a.steps=2400;a.case='paired_forward';a.admission=H.parent/'reference_physics_results_009/run/standing/admission.json'
try:preflight(a,SOURCE)
except ValueError:old_rejected=True
else:raise AssertionError('Actual source009 admission accepted fornewpairsource')
with tempfile.TemporaryDirectory() as td:
 a.admission=Path(td)/'receipt.json';receipt={'mode':'standing','status':'completed','identity':identity,'gate':{'passed':True,'num_envs':32,'control_steps':1000,'all_replica_quiet':{'passed':True}}};a.admission.write_text(json.dumps(receipt));pair=preflight(a,SOURCE)[0]
 receipt['gate']['all_replica_quiet']['passed']=False;a.admission.write_text(json.dumps(receipt))
 try:preflight(a,SOURCE)
 except ValueError:quiet_rejected=True
 else:raise AssertionError('Failedquietadmitted')
r={'runtime':runtime,'source_manifest_sha256':host.digest(SOURCE/'campaign_source_hashes.json'),
 'source_files':len(json.loads((SOURCE/'campaign_source_hashes.json').read_text())),
 'actual009standing_admission_rejected':old_rejected,'failedquiet_synthetic_receipt_rejected':quiet_rejected,
 'paired_request':pair['requested_forward_left_yaw'],'protocol':pair['paired_motion_protocol'],
 'synthetic_receipts_CPU_only':True,'physics_admitted':False}
p=H/'INTEGRATION_PREFLIGHT.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
